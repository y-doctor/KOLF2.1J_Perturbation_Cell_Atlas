import anndata as ad
import psp.qc as qc
from sklearn.ensemble import IsolationForest
import scanpy as sc
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import numpy as np
import requests
import psp.utils as utils
import psp.pl as pl
from tqdm.contrib.concurrent import process_map
import dcor
from typing import Tuple, Dict, List
from sklearn.svm import OneClassSVM
import os
from functools import partial

######################################
# NTC Processing Utilities
######################################

def get_NTCs_from_whitelist(adata: ad.AnnData, whitelist_path: str) -> ad.AnnData:
    """
    Isolate data to contain only the NTCs from the whitelist.

    Parameters:
    adata (AnnData): The AnnData object to be modified.

    Returns:
    ad.AnnData: The NTCs from the whitelist.
    """
    with open(whitelist_path, 'r') as f:
        sgRNA_whitelist = f.read().splitlines()
    ntc_adata = utils.get_ntc_view(adata)
    ntc_adata = ntc_adata[ntc_adata.obs.gRNA.isin(sgRNA_whitelist)].copy()
    return ntc_adata


def _get_cell_cycle_genes() -> tuple:
    """
    Fetches and returns the canonical list of cell cycle genes from Regev lab.

    The function retrieves the list of cell cycle genes,
    splits them into S phase and G2/M phase genes, and returns them as two separate lists.

    Returns:
    - tuple: A tuple containing two lists:
        - s_genes: List of genes associated with the S phase of the cell cycle.
        - g2m_genes: List of genes associated with the G2/M phase of the cell cycle.
    """
    url = "https://raw.githubusercontent.com/scverse/scanpy_usage/master/180209_cell_cycle/data/regev_lab_cell_cycle_genes.txt"
    cell_cycle_genes = requests.get(url).text.split("\n")[:-1]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    return s_genes, g2m_genes


def _scrub_ntc_pca(adata: ad.AnnData, batch_key: str = None) -> None:
    """
    Performs PCA on the AnnData object after identifying highly variable genes and scoring cell cycle phases.

    This function identifies the top 2000 highly variable genes, scores the cell cycle phases,
    applies log transformation, scales the data, and performs PCA to reduce dimensionality.

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing single-cell data.

    Returns:
    - None
    """
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False, flavor='seurat_v3', layer='counts', batch_key="batch")
    
    # Get cell cycle genes
    s_genes, g2m_genes = _get_cell_cycle_genes()
    
    # Score cell cycle phases
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes, use_raw=False)
    
    # Log transform and scale the data
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    
    # Perform PCA using top 50 PCs
    sc.pp.pca(adata, n_comps=50)

    if batch_key is not None:
        sc.external.pp.harmony_integrate(adata, batch_key)

    
    # Plot the variance explained by each principal component
    plt.plot(100 * np.cumsum(adata.uns["pca"]["variance_ratio"]), '.')
    plt.xlabel("Number of PCs")
    plt.ylabel("Total % Variance Explained")
    plt.show()



def _scrub_ntc_isolation_forest(adata: ad.AnnData, contamination_threshold: float,  batch_key: str = 'batch', sgRNA_column: str = "gRNA", min_cells_per_NTC_guide: int = 25) -> list:
    """
    Identifies outliers in the AnnData object using Isolation Forest and visualizes the results.

    This function uses PCA-reduced data to fit an Isolation Forest model, classifies cells as inliers or outliers,
    and visualizes the classification using UMAP before and after filtering out outliers.

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing single-cell data.
    - contamination_threshold (float): The proportion of outliers in the data.

    Returns:
    - list: A list of indices of the inlier cells.
    """
    # Use PCA representation for outlier detection
    if "X_pca_harmony" in adata.obsm:
        rep = "X_pca_harmony"
    else:
        rep = "X_pca"
    data = adata.obsm[rep]
    
    # Fit the Isolation Forest model
    clf = IsolationForest(contamination=contamination_threshold)
    clf.fit(data)
    
    # Classify points as outliers (-1) and inliers (1)
    labels = clf.predict(data)
    adata.obs['is_outlier'] = labels
    adata.obs['is_outlier'] = adata.obs['is_outlier'].replace({-1: 'outlier', 1: 'inlier'})
    
    # Visualize the outlier classification Pre-Filtering
    sc.pp.neighbors(adata, use_rep=rep)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["is_outlier", "phase", "batch"], title="Pre-Filtering", frameon=False)
    sc.tl.embedding_density(adata)
    sc.pl.embedding_density(adata, frameon=False)
    
    # Filter out outliers and visualize Post-Filtering
    adata_filtered = adata[adata.obs.is_outlier == "inlier", :]

    # Filter out NTCs with too few cells
    if batch_key is not None:
        batch_counts = adata_filtered.obs.groupby([sgRNA_column, batch_key]).size().unstack(fill_value=0)
        guides_to_keep = batch_counts[batch_counts.min(axis=1) >= min_cells_per_NTC_guide].index.tolist()
        print(f"Removing {len(batch_counts) - len(guides_to_keep)} NTC guides due to having less than {min_cells_per_NTC_guide} cells in each batch")
        adata_filtered = adata_filtered[adata_filtered.obs[sgRNA_column].isin(guides_to_keep)]
    
    else:
        counts = adata_filtered.obs[sgRNA_column].value_counts()
        guides_to_keep = counts[counts >= min_cells_per_NTC_guide].index.tolist()
        print(f"Removing {len(counts) - len(guides_to_keep)} NTC guides due to having less than {min_cells_per_NTC_guide} cells")
        adata_filtered = adata_filtered[adata_filtered.obs[sgRNA_column].isin(guides_to_keep)]

    sc.pp.neighbors(adata_filtered, use_rep=rep)
    sc.tl.umap(adata_filtered, )
    sc.pl.umap(adata_filtered, color=["is_outlier", "phase", "batch"], title="Post-Filtering", frameon=False)
    sc.tl.embedding_density(adata_filtered)
    sc.pl.embedding_density(adata_filtered, frameon=False)
    
    return list(adata_filtered.obs.index)


def clean_ntc_cells(adata: ad.AnnData, contamination_threshold: float = 0.3, NTC_whitelist_path: str = None, batch_key: str = None, sgRNA_column: str = "gRNA", min_cells_per_NTC_guide: int = 25) -> ad.AnnData:
    """
    Scrubs non-targeting control (NTC) cells to identify valid cells using PCA and Isolation Forest.

    This function performs PCA to prepare the data and then uses an Isolation Forest to identify
    and filter out outlier cells, returning the indices of valid NTC cells.

    Parameters:
    - adata(anndata.AnnData): The AnnData object you wish to clean.
    - contamination_threshold (float): The proportion of outliers in the data (default is 0.3).
    - batch_key (str, optional): The key in adata.obs that indicates the batch the cell belongs to.
    - NTC_whitelist_path (str, optional): The path to a file containing NTC sgRNAs to keep.

    Returns:
    - anndata.AnnData: The cleaned AnnData object.
    """
    # Assertions to ensure required fields are present
    assert 'counts' in adata.layers, "The AnnData object must have a 'counts' layer."
    assert 'perturbed' in adata.obs, "The AnnData object must have a 'perturbed' column in obs which indicates whether the cell is perturbed or not."
    if batch_key is not None:
        assert batch_key in adata.obs, f"The AnnData object must have a {batch_key} column in obs which indicates the batch the cell belongs to."

    adata_ntc = utils.get_ntc_view(adata).copy()
    print(f"Initial number of NTC Cells: {len(adata_ntc)}")

    if NTC_whitelist_path is not None:
        adata_ntc = get_NTCs_from_whitelist(adata_ntc, NTC_whitelist_path)
        print(f"Number of NTC Cells after whitelist filtering: {len(adata_ntc)}")
    
    # Perform PCA on the data
    if batch_key is not None:
        _scrub_ntc_pca(adata_ntc, batch_key = batch_key)
    else:
        _scrub_ntc_pca(adata_ntc)
    
    # Identify valid NTC cells using Isolation Forest
    valid_ntc_cells = _scrub_ntc_isolation_forest(adata_ntc, contamination_threshold, batch_key, sgRNA_column, min_cells_per_NTC_guide)
    print(f"Number of NTC Cells after Isolation Forest filtering: {len(valid_ntc_cells)}")
    perturbed_mask = adata.obs.perturbed == "True"
    valid_ntc_mask = adata.obs.index.isin(valid_ntc_cells)
    valid_cells_mask = perturbed_mask | valid_ntc_mask
    adata._inplace_subset_obs(valid_cells_mask)
    print(f"Total number of cells after NTC cleaning: {len(adata.obs)}")
    print(f"Number of NTC Cells per batch \n: {utils.get_ntc_view(adata).obs.batch.value_counts()}")
    return adata

######################################
# Isolating sgRNA that cause sufficient target gene knockdown
######################################

def evaluate_per_sgRNA_knockdown(
    adata: ad.AnnData,
    repression_threshold: float = 30.0,
    cells_per_gRNA_threshold: int = 25,
    label_interval: int = 100,
    batch_aware: bool = True,
    NTC_prefix: str = "Non-Targeting"
) -> ad.AnnData:
    """
    Evaluates sgRNA knockdown efficiency and filters cells/guides based on quality thresholds.
    
    Processes data in batches if batch information exists and batch_aware is True.
    
    Parameters:
    - adata: AnnData object containing:
        - 'gRNA' in obs: sgRNA assignments per cell
        - 'target_knockdown' in obs: knockdown efficiency values
        - 'perturbed' in obs: boolean indicating perturbation status
        - 'batch' in obs (optional): batch information
    - repression_threshold: Minimum required median knockdown percentage
    - cells_per_gRNA_threshold: Minimum cells required per sgRNA
    - label_interval: X-axis label display interval for plots
    - batch_aware: Process batches separately if batch information exists
    - NTC_prefix: Prefix for NTC sgRNAs
    
    Returns:
    - Filtered AnnData object with quality-controlled cells and guides
    """
    # Validate input structure
    utils.validate_anndata(adata, required_obs=['gRNA', 'target_knockdown', 'perturbed'])
    
    # Split and process batches if applicable
    if batch_aware and 'batch' in adata.obs:
        batches = utils.split_by_batch(adata, copy=False)
        invalid_cells = []
        low_count_guides = []
        no_repression_guides = []
        for batch_name, batch_adata in batches.items():
            print(f"Processing batch: {batch_name}")
            nr_guides, nr_cells, lc_guides = _process_sgRNA_knockdown_batch(
                    batch_adata,
                    repression_threshold,
                    cells_per_gRNA_threshold,
                    label_interval,
                    batch_name,
                    NTC_prefix
                )
            no_repression_guides.extend(nr_guides)
            invalid_cells.extend(nr_cells)
            low_count_guides.extend(lc_guides)
        adata = _filter_cells_and_guides(adata, no_repression_guides, invalid_cells, low_count_guides, repression_threshold, cells_per_gRNA_threshold)
    else:
        nr_guides, nr_cells, lc_guides  = _process_sgRNA_knockdown_batch(adata, repression_threshold, 
                             cells_per_gRNA_threshold, label_interval, batch_name, NTC_prefix)
        adata = _filter_cells_and_guides(adata, nr_guides, nr_cells, lc_guides, repression_threshold, cells_per_gRNA_threshold)
        

    # Final filtering and visualization
    adata_perturbed = utils.get_perturbed_view(adata)
    pl.plotting.plot_sorted_bars(
        adata_perturbed.obs.gRNA.value_counts(),
        ylabel="Number of Cells per gRNA",
        title="Final Cell Counts per Guide",
        cells_threshold=cells_per_gRNA_threshold,
        label_interval=label_interval,
        vmax=300
    )
    
    return adata

def _process_sgRNA_knockdown_batch(
    batch_adata: ad.AnnData,
    repression_threshold: float,
    cells_threshold: int,
    label_interval: int,
    batch_name: str,
    NTC_prefix: str
) -> ad.AnnData:
    """Process a single batch of data"""
    # Calculate median knockdown percentages
    mean_knockdown = 100 * batch_adata.obs.groupby("gRNA")["target_knockdown"].mean()
    non_ntc = mean_knockdown[~mean_knockdown.index.str.contains(NTC_prefix)]
    
    # Plot knockdown distribution
    pl.plotting.plot_sorted_bars(
        non_ntc,
        ylabel="Mean Knockdown per Cell (%)",
        title=f"sgRNA Knockdown Efficiency for batch {batch_name}",
        repression_threshold=repression_threshold,
        invert_y=True,
        label_interval=label_interval,
        vmin=0,
        vmax=100
    )
    
    # Filter low-efficiency guides
    no_repression_guides = [
        g for g, val in mean_knockdown.items()
        if NTC_prefix not in g and val <= repression_threshold
    ]
    batch_adata = batch_adata[~batch_adata.obs.gRNA.isin(no_repression_guides)]
    
    # Filter cells with negative knockdown
    no_repression_mask = (batch_adata.obs["target_knockdown"] < 0) & (batch_adata.obs.perturbed == "True")
    no_repression_cells = batch_adata.obs.index[no_repression_mask].tolist()  # convert boolean mask to list of cell IDs
    batch_adata = batch_adata[~no_repression_mask]
    
    # Filter guides with low cell counts
    guide_counts = batch_adata[batch_adata.obs.perturbed == "True"].obs.gRNA.value_counts()
    low_count_guides = guide_counts[guide_counts <= cells_threshold].index.tolist()

    return no_repression_guides, no_repression_cells, low_count_guides

def _filter_cells_and_guides(adata: ad.AnnData, no_repression_guides: list, no_repression_cells: list, low_count_guides: list, repression_threshold: float, cells_threshold: int) -> ad.AnnData:
    
    print(f"Removing {len(no_repression_guides)} guides due to low repression (less than {repression_threshold}%)")
    print(f"Removing {len(no_repression_cells)} cells due to no repression (Target knockdown < 0)")
    print(f"Removing {len(low_count_guides)} guides due to low cell count per guide (less than {cells_threshold} cells per guide)")

    mask = adata.obs.gRNA.isin(no_repression_guides) | adata.obs.gRNA.isin(low_count_guides)
    mask = mask | adata.obs.index.isin(no_repression_cells)
    adata._inplace_subset_obs(~mask)
    return adata

def knockdown_qc(
    adata: ad.AnnData,
    obs_key: str = "gene_target_ensembl_id",
    var_key: str = "gene_ids",
    gene_target_expr_col: str = "gene_target_expression (CPM)",
    ntc_target_expr_col: str = "NTC_target_gene_expression (CPM)",
    knockdown_col: str = "target_knockdown",
    ntc_label: str = "NTC",
    layer: str = "counts",
    normalized_layer: str = "normalized_counts",
    batch_aware: bool = True
) -> ad.AnnData:
    """
    Calculate knockdown metrics with batch-aware processing.

    Parameters:
    - adata: AnnData object with single-cell data
    - obs_key: Column in obs containing target gene IDs
    - var_key: Column in var containing ENSEMBL IDs
    - gene_target_expr_col: Name for target expression column
    - ntc_target_expr_col: Name for NTC expression column  
    - knockdown_col: Name for knockdown efficiency column
    - ntc_label: Label indicating NTC guides
    - layer: Layer containing raw counts
    - normalized_layer: Layer to store normalized counts
    - batch_aware: Process batches separately if batch column exists

    Returns:
    - AnnData object with calculated metrics and quality controls
    """
    # Validate input structure
    utils.validate_anndata(adata, required_obs=[obs_key, 'perturbed'], required_var=[var_key])

    if batch_aware and 'batch' in adata.obs:
        print("Batch-aware processing enabled")
        batches = utils.split_by_batch(adata, copy=True)
        for batch_name, batch_adata in batches.items():
            print(f"Processing batch: {batch_name}")
            processed_batch = _process_batch_knockdown(
                batch_adata,
                obs_key,
                var_key,
                gene_target_expr_col,
                ntc_target_expr_col,
                knockdown_col,
                ntc_label,
            )
            # Identify cells belonging to the current batch
            batch_mask = adata.obs['batch'] == batch_name
            # Transfer the computed columns from the processed batch to the master AnnData object
            adata.obs.loc[batch_mask, gene_target_expr_col] = processed_batch.obs[gene_target_expr_col]
            adata.obs.loc[batch_mask, ntc_target_expr_col] = processed_batch.obs[ntc_target_expr_col]
            adata.obs.loc[batch_mask, knockdown_col] = processed_batch.obs[knockdown_col]
            del processed_batch  # Free memory associated with the processed batch
        del batches  # Remove the batches dictionary to free memory
    else:
        adata = _process_batch_knockdown(
            adata,
            obs_key,
            var_key,
            gene_target_expr_col,
            ntc_target_expr_col,
            knockdown_col,
            ntc_label,
        )

    return adata

def _process_batch_knockdown(
    batch_adata: ad.AnnData,
    obs_key: str,
    var_key: str,
    gene_target_expr_col: str,
    ntc_target_expr_col: str,
    knockdown_col: str,
    ntc_label: str,
) -> ad.AnnData:
    """Process knockdown metrics for a single batch"""
    # Normalize counts
    sc.pp.normalize_total(batch_adata, target_sum=1e6)
    
    # Calculate target expressions
    data = batch_adata.X
    var_dict = {gene: idx for idx, gene in enumerate(batch_adata.var[var_key].values)} # Map gene expression to index
    var_indices = batch_adata.obs[obs_key].astype(str).map(var_dict).fillna(-1).astype(int).values # Map gene target to index for each cell
    
    # Initialize expression arrays
    gene_expr = np.zeros(batch_adata.n_obs)
    ntc_expr = np.zeros(batch_adata.n_obs)
    
    # Calculate NTC statistics
    is_ntc = (batch_adata.obs[obs_key] == ntc_label).values
    if np.any(is_ntc):
        ntc_data = data[is_ntc]
        ntc_mean = np.array(ntc_data.mean(axis=0)).flatten()
    else:
        ntc_mean = np.zeros(batch_adata.shape[1])
    
    # Assign expression values
    valid_mask = var_indices >= 0
    gene_expr[valid_mask] = np.array(data[np.arange(batch_adata.n_obs)[valid_mask], var_indices[valid_mask]]).flatten()
    ntc_expr[valid_mask] = ntc_mean[var_indices[valid_mask]]
    
    # Calculate knockdown efficiency
    with np.errstate(divide='ignore', invalid='ignore'):
        knockdown = 1 - np.divide(gene_expr, ntc_expr, out=np.ones_like(gene_expr), where=ntc_expr != 0)
    
    # Store results
    batch_adata.obs[gene_target_expr_col] = gene_expr
    batch_adata.obs[ntc_target_expr_col] = ntc_expr
    batch_adata.obs[knockdown_col] = knockdown
    
    return batch_adata

######################################
# Assesing sgRNA which induce a Transcriptional Phenotype via Energy Distance
######################################

def normalize_log_scale(adata: ad.AnnData, batch_sensitive: bool = True, scale: bool = True) -> ad.AnnData:
    """
    Normalize and log-scale the expression data for each batch in the AnnData object.

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing the data to normalize.
    - batch_sensitive (bool): If True, process batches separately; otherwise, normalize across all data. Default is True.
    - scale (bool): If True, scale the data; otherwise, only normalize. Default is True.
    Returns:
    - adata (anndata.AnnData): The AnnData object with normalized and log-scaled data.
    """
    if batch_sensitive:
        # Validate input structure
        utils.validate_anndata(adata, required_obs=["perturbed", "batch"])

        adata.X = adata.layers['counts'].copy()

        # Split the AnnData object by batch
        batches = utils.split_by_batch(adata, copy=True)
        processed_batches = []

        for batch_name, batch_adata in batches.items():
            # Calculate median NTC counts for the current batch
            median_NTC = np.median(list(utils.get_ntc_view(batch_adata).obs.n_UMI_counts))
            
            # Normalize total counts
            sc.pp.normalize_total(batch_adata, target_sum=median_NTC)
            
            # Log transform and scale the data
            sc.pp.log1p(batch_adata)
            if scale:
                sc.pp.scale(batch_adata)

            processed_batches.append(batch_adata)
        
        # Merge processed batches back into a single AnnData object
        adata = ad.concat(processed_batches, merge='same', join='inner')
    else:
        # Validate input structure
        utils.validate_anndata(adata, required_obs=["perturbed"])

        adata.X = adata.layers['counts'].copy()

        # Calculate median NTC counts for the entire dataset
        median_NTC = np.median(list(utils.get_ntc_view(adata).obs.n_UMI_counts))
        
        # Normalize total counts
        sc.pp.normalize_total(adata, target_sum=median_NTC)
        
        # Log transform and scale the data
        sc.pp.log1p(adata)
        if scale:
            sc.pp.scale(adata)

    return adata


def __subsample_anndata_for_energy_distance(
    adata: ad.AnnData,
    category: str,
    n_min: int,
    ref_size: int = 3000,
    control_string: str = "NTC",
    seed: int = None
) -> ad.AnnData:
    """Subsample cells while maintaining biological diversity through stratified sampling.
    
    Performs stratified subsampling of cells with special handling for control groups:
    1. For control groups (containing control_string), deisgnates as set of these as reference cells of size ref_size and labels remaining as NTC.
    2. For non-control groups, subsamples to ensure minimum representation per category
    3. Adds metadata tracking sample origins in 'ed_category' observation field

    Parameters:
        adata: Input AnnData object containing single-cell data
        category: Observation column name used for stratification
        n_min: Minimum number of cells to retain per non-control category
        ref_size: Number of control cells to designate as reference population
        control_string: Substring identifying control groups in category column
        seed: Random seed for reproducible sampling (default: None)

    Returns:
        Subsampled AnnData copy containing:
        - ref_size control cells labeled 'reference'
        - Remaining control cells labeled 'NTC'
        - min(n_min, group_size) cells per non-control category
        - 'ed_category' observation field tracking sample stratification
        
    Raises:
        ValueError: If input validation fails or insufficient control cells
    """
    # Validate input structure
    if category not in adata.obs:
        raise ValueError(f"Stratification category '{category}' not found in adata.obs")
    
    # Set random seed if provided
    rng = np.random.default_rng(seed)

    # Initialize tracking metadata and storage
    adata.obs["ed_category"] = "None"
    keep_indices = []

    # Process control groups
    control_mask = adata.obs[category].str.contains(control_string)
    control_indices = adata.obs.index[control_mask].tolist()
    
    if len(control_indices) < ref_size:
        raise ValueError(f"Control group(s) contain {len(control_indices)} cells, below ref_size ({ref_size})")
    
    # Sample reference population from controls
    reference_samples = rng.choice(control_indices, ref_size, replace=False)
    adata.obs.loc[reference_samples, "ed_category"] = "reference"
    keep_indices.extend(reference_samples.tolist())
    
    # Label remaining controls as NTC
    ntc_samples = np.setdiff1d(control_indices, reference_samples)
    adata.obs.loc[ntc_samples, "ed_category"] = "NTC"
    keep_indices.extend(ntc_samples.tolist())

    # Process experimental groups
    for group_name, group_indices in adata.obs.groupby(category).groups.items():
        if control_string not in group_name:
            # Subsample if group exceeds minimum size
            if len(group_indices) >= n_min:
                sampled = rng.choice(group_indices, n_min, replace=False)  # Use the RNG to sample
                adata.obs.loc[sampled, "ed_category"] = group_name
                keep_indices.extend(sampled.tolist())

    return adata[keep_indices].copy()


def __preprocess_for_ed(adata_subsampled: ad.AnnData, seed: int = None) -> ad.AnnData:
    """Preprocess subsampled data with seed control"""
    if seed is not None:
        np.random.seed(seed)  # SET GLOBAL SEED FOR SEURAT
        
    n_var_max = 2000  # Maximum number of features to select
    
    # Calculate HVGs using class-balanced subset (exclude NTC cells)
    adata_class_balance: ad.AnnData = adata_subsampled[
        adata_subsampled.obs.ed_category != "NTC"
    ].copy()
    
    sc.pp.highly_variable_genes(
        adata_class_balance,
        n_top_genes=n_var_max,
        subset=False,
        flavor='seurat_v3',
        layer='counts'
    )
    
    # Apply HVG selection to full dataset
    adata_subsampled.var["highly_variable"] = adata_class_balance.var["highly_variable"].copy()
    del adata_class_balance
    
    # Dimensionality reduction and neighborhood graph
    sc.pp.pca(adata_subsampled, use_highly_variable=True)
    sc.pp.neighbors(adata_subsampled)

    return adata_subsampled


def single_null_computation(rng, control_data, reference_data, sample_size):
    """
    Compute the energy distance for a single replicate by subsampling from control data.

    Parameters:
        rng : np.random.Generator
            Random number generator.
        control_data : np.ndarray
            The control data from which to sample.
        reference_data : np.ndarray
            The reference data for computing energy distance.
        sample_size : int
            Number of samples to draw from control_data.
            
    Returns:
        float: The computed energy distance.
    """
    indices = rng.choice(control_data.shape[0], sample_size, replace=False)
    return dcor.energy_distance(control_data[indices], reference_data)


def compute_energy_distance(
    adata: ad.AnnData,
    category: str = "ed_category",
    reference_name: str = "reference",
    control_name: str = "NTC",
    n_replicates: int = 10000,
    ref_sample_size: int = 20,
    use_rep: str = "X_pca",
    n_jobs: int = -1,
    threshold: float = 0.75,
    seed: int = None
) -> Tuple[np.ndarray, List[float], Dict[str, float]]:
    """
    Compute energy distances between experimental groups and reference population.
    
    Parameters:
        adata: AnnData object containing single-cell data
        category: Observation column for group stratification
        reference_name: Name of reference population in category column
        control_name: Name of control population in category column
        n_replicates: Number of null distribution samples
        ref_sample_size: Number of cells to sample from control population
        use_rep: Data representation to use (X_pca, X, etc.)
        n_jobs: Number of parallel jobs (-1 for all cores)
        threshold: Probability threshold for perturbation detection
    
    Returns:
        Tuple containing:
        - Valid sgRNA names that pass threshold
        - Null distribution energy distances
        - Experimental group energy distances
    """
    # Data preparation
    data = _get_energy_distance_data(adata, use_rep)
    reference_data, control_data = _extract_reference_control_data(adata, data, category, reference_name, control_name)

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    # Null distribution computation
    null_distances = _compute_null_distribution(
        control_data, 
        reference_data,
        ref_sample_size,
        n_replicates,
        n_jobs,
        seed
    )

    # Experimental group computation
    experimental_group_distances = _compute_group_distances(
        adata,
        data,
        category,
        reference_name,
        n_jobs,
        seed
    )

    # Visualization and thresholding
    threshold = pl.plot_energy_distance_threshold(
        null_distances,
        experimental_group_distances,
        threshold=threshold
    )
    valid_gRNA = [gRNA for gRNA in experimental_group_distances.keys() if experimental_group_distances[gRNA] > threshold]

    return valid_gRNA, null_distances, experimental_group_distances

def _get_energy_distance_data(adata: ad.AnnData, use_rep: str) -> np.ndarray:
    """Helper to get and validate energy distance input data"""
    if use_rep is None:
        data = adata.X
    else:
        if use_rep not in adata.obsm:
            raise ValueError(f"Representation '{use_rep}' not found in adata.obsm")
        data = adata.obsm[use_rep]
    
    return data.toarray() if hasattr(data, "toarray") else data

def _extract_reference_control_data(
    adata: ad.AnnData,
    data: np.ndarray,
    category: str,
    reference_name: str,
    control_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract reference and control population data"""
    reference_mask = adata.obs[category] == reference_name
    control_mask = adata.obs[category] == control_name
    return data[reference_mask], data[control_mask]

def _compute_null_distribution(
    control_data: np.ndarray,
    reference_data: np.ndarray,
    sample_size: int,
    n_replicates: int,
    n_jobs: int,
    seed: int = None
) -> List[float]:
    """Compute null distribution through parallel sampling using a picklable function."""
    # Create independent RNG states using seed
    master_rng = np.random.default_rng(seed)
    seeds = master_rng.integers(0, 2**32-1, size=n_replicates)
    rngs = [np.random.default_rng(s) for s in seeds]
    
    # Ensure n_jobs is positive, convert -1 to available CPU cores
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    # Compute an optimal chunksize (you can adjust this heuristic as needed)
    chunksize = 64
    
    # Use partial to bind additional arguments to the global function
    fn = partial(single_null_computation, control_data=control_data,
                 reference_data=reference_data,
                 sample_size=sample_size)
    
    return process_map(fn, rngs, max_workers=n_jobs, chunksize=chunksize, desc="Null distribution")


def group_distance_computation(indices, seed, data, pos_mapping, ref_bool_idx):
    """
    Compute the energy distance for a given group of cell indices using precomputed constants.

    Parameters:
        indices (array-like): Group cell identifiers (labels) for the experimental group.
        seed: A seed value (unused here, but retained for API compatibility).
        data (np.ndarray): Data representation (e.g., PCA coordinates) with rows corresponding 
                           to the order in pos_mapping.
        pos_mapping (pd.Index): Precomputed mapping from cell labels to row positions (e.g., adata.obs.index).
        ref_bool_idx (np.ndarray): Precomputed boolean mask for the reference population.
        
    Returns:
        float: The computed energy distance.
    """
    # Convert cell labels (indices) to positional indices using the precomputed mapping.
    pos_indices = pos_mapping.get_indexer(indices)
    
    # Compute and return the energy distance between the experimental group and the reference group.
    return dcor.energy_distance(data[pos_indices], data[ref_bool_idx])


def _compute_group_distances(
    adata: ad.AnnData,
    data: np.ndarray,
    category: str,
    reference_name: str,
    n_jobs: int,
    seed: int = None
) -> dict:
    """Calculate energy distances for all experimental groups with reproducibility."""
    # Create dictionary mapping group names (except for reference_name) to indices
    groups = {
        g: idx 
        for g, idx in adata.obs.groupby(category).groups.items()
        if g != reference_name
    }
    
    # Sort groups for deterministic behavior
    sorted_groups = sorted(groups.items(), key=lambda x: x[0])
    group_names = [g[0] for g in sorted_groups]
    group_indices = [g[1] for g in sorted_groups]

    # Create RNG seeds for each group using master seed
    master_rng = np.random.default_rng(seed)
    seeds = master_rng.integers(0, 2**32-1, size=len(group_indices))

    # Ensure n_jobs is positive
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # Set a chunksize. For example, for lightweight tasks you might set:
    chunksize =  64  # or compute a recommended value, e.g., max(1, len(group_indices) // (n_jobs * 4))
    
    # Precompute the mapping and reference boolean mask outside the parallel loop.
    pos_mapping = adata.obs.index
    ref_bool_idx = (adata.obs[category] == reference_name).to_numpy()

    # Then update the partial binding:
    fn = partial(group_distance_computation,
                 data=data,
                 pos_mapping=pos_mapping,
                 ref_bool_idx=ref_bool_idx)
    
    # Compute energy distances in parallel using process_map
    distances = process_map(fn, group_indices, seeds, max_workers=n_jobs, chunksize=chunksize, desc="Experimental groups")
    
    return dict(zip(group_names, distances))


def _process_single_batch_energy_distance(
    adata: ad.AnnData,
    sgRNA_column: str,
    n_min: int,
    control_string: str,
    seed: int = None,
    **kwargs
) -> Tuple[ad.AnnData, List[float], Dict[str, float]]:
    """Process individual batch with energy distance filtering"""
    try:
        subsampled = __subsample_anndata_for_energy_distance(
            adata,
            category=sgRNA_column,
            n_min=n_min,
            control_string=control_string,
            seed=seed,
            **kwargs
        )
        preprocessed = __preprocess_for_ed(subsampled, seed=seed)
        valid_sgRNA, null_distances, experimental_group_distances = compute_energy_distance(
            preprocessed, 
            seed=seed,
            **kwargs
        )
        return valid_sgRNA, null_distances, experimental_group_distances
    except ValueError as e:
        print(f"Skipping batch due to error: {str(e)}")
        return [], [], {}



def filter_sgRNA_energy_distance(
    adata: ad.AnnData,
    sgRNA_column: str = "gRNA",
    batch_key: str = "batch",
    n_min: int = 20,
    control_string: str = "Non-Targeting",
    verbose: bool = True,
    seed: int = None,
    **kwargs
) -> Tuple[ad.AnnData, List[float], Dict[str, float]]:
    """
    Batch-aware sgRNA filtering using energy distance analysis.
    
    Parameters:
        adata: Input AnnData object
        sgRNA_column: Observation column containing sgRNA information
        batch_key: Column name for batch information (if processing multiple batches)
        n_min: Minimum cells per sgRNA for inclusion
        control_string: Identifier for control sgRNAs
        verbose: Whether to print progress updates
        seed: Random seed for reproducible results
        **kwargs: Additional arguments for compute_energy_distance
    
    Returns:
        Tuple containing:
        - Filtered AnnData object
        - Aggregated null distribution distances
        - Combined experimental group results
    """
    if batch_key is None:
        return _process_single_batch_energy_distance(
            adata, sgRNA_column, n_min, control_string, seed=seed, **kwargs
        )
    else:
        utils.validate_anndata(adata, required_obs=[batch_key])

    # Split by batches and process individually
    batches = adata.obs[batch_key].unique()
    all_valid = set() # set of valid sgRNAs 
    combined_null = [] # list of null distribution distances
    combined_results = {} # dictionary of experimental group results

    for batch in batches:
        if verbose:
            print(f"Processing batch: {batch}")
            
        batch_adata = adata[adata.obs[batch_key] == batch].copy()
        valid_sgRNA, null_distances, experimental_group_distances = _process_single_batch_energy_distance(
            batch_adata, sgRNA_column, n_min, control_string, seed=seed, **kwargs
        )
        
        # Aggregate results
        all_valid.update(valid_sgRNA)
        combined_null.extend(null_distances)
        combined_results.update(experimental_group_distances)

        if verbose:
            print(f"Retained {len(valid_sgRNA)} sgRNAs from batch {batch}") 

    # Apply combined filtering to original data
    valid_mask = adata.obs[sgRNA_column].isin(all_valid) | (adata.obs.perturbed == "False")
    adata_filtered = adata[valid_mask].copy()
    del adata

    if verbose:
        print(f"Retained {valid_mask.sum()} cells after batch-aware filtering")
    
    return adata_filtered, combined_null, combined_results

######################################
# Removing cells that do not have an altered transcriptional phenotype
######################################

def _plot_anomaly_scores(control_scores, perturbed_scores, threshold, batch_name=None):
    """
    Helper function to plot the distribution of anomaly scores for control and perturbed cells.

    Parameters:
    - control_scores (list): Anomaly scores for control cells.
    - perturbed_scores (list): Anomaly scores for perturbed cells.
    - threshold (float): Decision threshold for classifying cells as perturbed.
    - batch_name (str, optional): Name of the batch being processed (for title clarity).

    Returns:
    - None: Displays a histogram plot of the anomaly scores.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram for control cells
    plt.hist(control_scores, bins=100, density=True, alpha=0.4, color='blue', label="Control Cells")
    
    # Plot histogram for perturbed cells
    plt.hist(perturbed_scores, bins=100, density=True, alpha=0.4, color='green', label="Perturbed Cells")
    
    # Add decision threshold line
    plt.axvline(x=0, color='red', linestyle='--', label=f"{threshold * 100:.1f}% Decision Threshold")
    
    # Set plot title
    title = "Anomaly Score Distribution"
    if batch_name:
        title += f" - Batch: {batch_name}"
    plt.title(title, fontsize=14)
    
    # Set axis labels
    plt.xlabel("Anomaly Score (-decision_function)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    # Add legend and grid
    plt.legend(fontsize=10)
    plt.grid(False)
    
    # Show the plot
    plt.tight_layout()
    plt.show()



def remove_unperturbed_cells_SVM(
    adata: ad.AnnData,
    threshold: float = 0.75,
    batch_key: str = None,
    ntc_identifier: str = "NTC",
    verbose: bool = True
) -> Tuple[ad.AnnData, np.ndarray, np.ndarray]:
    """
    Identify and remove cells without transcriptional perturbation using One-Class SVM.

    Processes data in batches if batch information is provided. For each batch:
    1. Recomputes PCA representation
    2. Trains One-Class SVM on NTC cells
    3. Identifies perturbed cells with anomaly scores above threshold

    Parameters:
        adata: AnnData object containing:
            - layers['counts']: Raw count data
            - obs['gene_target']: Column containing NTC identifiers
            - obs['perturbed']: Boolean-like column indicating perturbation status
        threshold: Probability threshold for considering cells perturbed (0-1)
        batch_key: Column name for batch information (None for single batch)
        ntc_identifier: String identifying NTC cells in gene_target column
        verbose: Whether to print progress information

    Returns:
        Tuple containing:
        - Filtered AnnData object
        - Array of control cell anomaly scores
        - Array of perturbed cell anomaly scores

    Raises:
        ValueError: If required columns/layers are missing or no NTC cells found
    """
    # Validate input structure
    utils.validate_anndata(adata, required_obs=['gene_target', 'perturbed'], required_layers=['counts'])
    
    # Prepare storage for results
    all_control_scores = []
    all_perturbed_scores = []
    valid_cells = []

    def _process_batch(batch_name: str, batch_adata: ad.AnnData) -> Tuple[list, list, list]:
        """Process a single batch of data"""
        # Recompute PCA for current batch
        batch_adata.X = batch_adata.layers["counts"].copy()
        sc.pp.highly_variable_genes(batch_adata, flavor='seurat_v3', n_top_genes=2000, layer='counts') #TODO: Does this need to be class balanced like in the energy distance filtering? Interesting debate to have.
        sc.pp.normalize_total(batch_adata)
        sc.pp.log1p(batch_adata)
        sc.pp.scale(batch_adata)
        sc.pp.pca(batch_adata)

        # Split cells into NTC and perturbed views
        adata_ntc = utils.get_ntc_view(batch_adata)
        adata_perturbed = utils.get_perturbed_view(batch_adata)

        # Train One-Class SVM on NTC cells
        clf = OneClassSVM(kernel='rbf', nu=1-threshold).fit(adata_ntc.obsm["X_pca"])

        # Calculate anomaly scores
        control_scores = -clf.decision_function(adata_ntc.obsm["X_pca"])
        perturbed_scores = -clf.decision_function(adata_perturbed.obsm["X_pca"])
        
        # Store results
        batch_valid = [
            *adata_ntc.obs.index.tolist(),
            *adata_perturbed.obs.index[perturbed_scores > 0].tolist()
        ]

        _plot_anomaly_scores(control_scores, perturbed_scores, threshold, batch_name=batch_name)
        
        return batch_valid, control_scores, perturbed_scores

    # Batch processing logic
    if batch_key:
        batches = utils.split_by_batch(adata, batch_key=batch_key, copy=True)
        for batch_name, batch_data in batches.items():
            if verbose:
                print(f"Processing batch: {batch_name}")
            try:
                batch_valid, control, perturbed = _process_batch(batch_name, batch_data)
                valid_cells.extend(batch_valid)
                all_control_scores.extend(control)
                all_perturbed_scores.extend(perturbed)
                if verbose:
                    n_perturbed = len(perturbed)
                    n_valid = sum(np.array(batch_valid > 0))
                    print(f"Retained {n_valid}/{n_perturbed} perturbed cells ({n_valid/n_perturbed:.1%}) above threshold {threshold} in batch {batch_name}")
            except ValueError as e:
                print(f"Skipping batch {batch_name}: {str(e)}")
    else:
        valid_cells, all_control_scores, all_perturbed_scores = _process_batch(adata)

    # Filter and create final dataset
    adata_filtered = adata[adata.obs.index.isin(valid_cells)].copy()
    del adata

    if verbose:
        n_perturbed = len(all_perturbed_scores)
        n_valid = sum(np.array(all_perturbed_scores) > 0)
        print(f"Retained {n_valid}/{n_perturbed} total perturbed cells ({n_valid/n_perturbed:.1%}) above threshold {threshold}")

    return adata_filtered

def remove_perturbations_by_cell_threshold(
    adata: ad.AnnData,
    cell_threshold: int = 25,
    batch_key: str = None,
    perturbation_key: str = "gene_target",
    verbose: bool = True
) -> ad.AnnData:
    """
    Filter perturbations based on minimum cell count requirements, with batch-aware processing.

    Parameters:
        adata: AnnData object containing single-cell data
        cell_threshold: Minimum number of cells required per perturbation (per batch if batch_key provided)
        batch_key: Optional column name in obs for batch information. If provided, 
                   requires threshold to be met in ALL batches where the perturbation exists
        perturbation_key: Observation column containing perturbation identifiers
        verbose: Whether to print filtering statistics

    Returns:
        Filtered AnnData object containing only perturbations that meet cell count requirements

    Example:
        # Keep only perturbations with ≥50 cells in any batch they appear
        adata_filtered = remove_perturbations_by_cell_threshold(adata, cell_threshold=50, batch_key="batch")
    """
    utils.validate_anndata(adata, obs_keys=[perturbation_key])
    initial_cells = adata.n_obs
    initial_perturbations = adata.obs[perturbation_key].nunique()

    if batch_key:
        # Batch-aware filtering: perturbation must meet threshold in ALL batches where it exists
        batches = utils.split_by_batch(adata, batch_key=batch_key, copy=False)
        
        # Get valid perturbations that meet threshold in any batches they appear
        valid_perturbations = set()
        for batch_name, batch_data in batches.items():
            batch_counts = batch_data.obs[perturbation_key].value_counts()
            batch_valid = batch_counts[batch_counts >= cell_threshold].index.tolist()
            valid_perturbations.update(batch_valid)
            if verbose:
                print(f"Batch '{batch_name}': {len(batch_valid)} perturbations meet threshold")
    else:
        # Global filtering
        perturbation_counts = adata.obs[perturbation_key].value_counts()
        valid_perturbations = perturbation_counts[perturbation_counts >= cell_threshold].index.tolist()

    # Apply filtering
    adata_filtered = adata[adata.obs[perturbation_key].isin(valid_perturbations)].copy()
    
    if verbose:
        final_cells = adata_filtered.n_obs
        final_perturbations = len(valid_perturbations)
        removed = initial_perturbations - final_perturbations
        
        print(f"Initial perturbations: {initial_perturbations}")
        print(f"Removed {removed} perturbations below {cell_threshold} cells")
        print(f"Remaining perturbations: {final_perturbations}")
        print(f"Cells kept: {final_cells}/{initial_cells} ({final_cells/initial_cells:.1%})")

    return adata_filtered

def compute_neighbor_corrected_expression(
    adata: ad.AnnData,
    perturbation_key: str = "gene_target",
    control_string: str = "NTC",
    n_neighbors: int = 20,
    n_jobs: int = -1,
    batch_key: str = None,
    use_rep: str = "X_pca",
    n_pcs: int = 50,
    chunked_pca: bool = False,
    chunk_size: int = 5000
) -> ad.AnnData:
    """
    Compute neighbor-corrected expression profiles by subtracting the mean expression
    of k nearest control neighbors for each cell.
    
    Parameters:
        adata: AnnData object containing single-cell data
        perturbation_key: Observation column identifying perturbation types
        control_string: String identifier for control cells in perturbation_key column
        n_neighbors: Number of nearest control neighbors to average
        n_jobs: Number of parallel jobs for neighbor computation (-1 for all cores)
        batch_key: Optional batch key for batch-aware processing
        use_rep: Representation to use for neighbor search after preprocessing
        n_pcs: Number of principal components to compute
        chunked_pca: Whether to compute PCA in chunks to reduce memory usage
        chunk_size: Number of cells per chunk when using chunked PCA
        
    Returns:
        AnnData object with neighbor-corrected expression in 'neighbor_corrected' layer
    """
    try:
        import pynndescent
    except ImportError:
        raise ImportError("The PyNNDescent package is required. Install with: pip install pynndescent")
    
    # Ensure counts layer exists
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
    
    def process_batch(batch_adata):
        """Process a single batch of cells"""
        # Standard preprocessing pipeline (applied in-place)
        original_X = batch_adata.X.copy()  # Only necessary copy
        batch_adata.X = batch_adata.layers['counts'].copy()
        
        # Preprocess data
        sc.pp.normalize_total(batch_adata, target_sum=1e6, inplace=True)
        sc.pp.log1p(batch_adata, copy=False)
        sc.pp.scale(batch_adata, copy=False)
        if chunked_pca:
            sc.pp.pca(batch_adata, n_comps=n_pcs, chunked=True, chunk_size=chunk_size)
        else:
            sc.pp.pca(batch_adata, n_comps=n_pcs)
        
        # Identify control cells
        control_mask = batch_adata.obs[perturbation_key].str.contains(control_string).fillna(False)
        control_indices = np.where(control_mask)[0]
        
        if len(control_indices) < n_neighbors:
            raise ValueError(f"Not enough control cells ({len(control_indices)}) for {n_neighbors} neighbors")
        
        # Build efficient nearest neighbor index using PyNNDescent
        control_pca = batch_adata.obsm[use_rep][control_indices]
        index = pynndescent.NNDescent(
            control_pca, 
            n_neighbors=n_neighbors,
            metric='euclidean',
            n_jobs=n_jobs,
            random_state=42
        )
        
        # Query for nearest neighbors (returns indices and distances)
        query_pca = batch_adata.obsm[use_rep]
        indices, _ = index.query(query_pca, k=n_neighbors)
        
        # Map indices back to global control cell indices
        global_indices = np.array([[control_indices[i] for i in nn_indices] for nn_indices in indices])
        
        # Allocate space for corrected expression in correct format
        from scipy import sparse
        if sparse.issparse(batch_adata.X):
            corrected_expr = sparse.lil_matrix(batch_adata.X.shape, dtype=batch_adata.X.dtype)
        else:
            corrected_expr = np.zeros_like(batch_adata.X)
        
        # Process in batches to avoid memory issues
        process_batch_size = min(1000, batch_adata.n_obs)  # Smaller chunks for processing
        
        for i in range(0, batch_adata.n_obs, process_batch_size):
            end_idx = min(i + process_batch_size, batch_adata.n_obs)
            
            # Get raw expression values
            raw_X = batch_adata.layers['counts'][i:end_idx]
            if hasattr(raw_X, 'toarray'):
                raw_X = raw_X.toarray()
            
            # Process each cell in the current batch
            for j in range(end_idx - i):
                # Get neighbor indices for this cell
                nn_indices = global_indices[i + j]
                
                # Get neighbor expressions (from raw counts layer)
                nn_expr = batch_adata.layers['counts'][nn_indices]
                if hasattr(nn_expr, 'toarray'):
                    nn_expr = nn_expr.toarray()
                
                # Compute mean expression of neighbors
                mean_nn_expr = np.mean(nn_expr, axis=0)
                
                # Calculate difference (avoiding temporary arrays)
                cell_expr = raw_X[j]
                cell_diff = cell_expr - mean_nn_expr
                
                # Store the result
                if sparse.issparse(corrected_expr):
                    for k in range(cell_diff.shape[0]):
                        if cell_diff[k] != 0:
                            corrected_expr[i + j, k] = cell_diff[k]
                else:
                    corrected_expr[i + j] = cell_diff
        
        # Store the corrected expression
        batch_adata.layers['neighbor_corrected'] = corrected_expr
        
        # Restore original X
        batch_adata.X = original_X
        
        return batch_adata
    
    # Apply batch-aware or global processing
    if batch_key is not None and batch_key in adata.obs:
        print(f"Performing batch-aware neighbor correction using '{batch_key}'")
        
        # Get unique batches
        batches = adata.obs[batch_key].unique()
        
        # Process each batch separately without full copies
        for batch_name in batches:
            print(f"Processing batch: {batch_name}")
            batch_mask = adata.obs[batch_key] == batch_name
            batch_indices = np.where(batch_mask)[0]
            
            # Create a view for the current batch
            batch_view = adata[batch_indices]
            process_batch(batch_view)
            
            # If we returned from processing without error, the corrected data
            # is already stored in the original object's layers
    else:
        print("Performing global neighbor correction")
        process_batch(adata)
    
    print("Neighbor-corrected expression computed and stored in 'neighbor_corrected' layer")
    return adata

