import anndata as ad
import psp.qc as qc
from sklearn.ensemble import IsolationForest
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import requests
import psp.utils as utils
import psp.pl as pl


def get_NTCs_from_whitelist(adata: ad.AnnData, whitelist_path: str) -> ad.AnnData:
    """
    Isolate data to the 

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


def _scrub_ntc_pca(adata: ad.AnnData) -> None:
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
    
    # Perform PCA using top 100 PCs
    sc.pp.pca(adata, n_comps=100)
    
    # Plot the variance explained by each principal component
    plt.plot(100 * np.cumsum(adata.uns["pca"]["variance_ratio"]), '.')
    plt.xlabel("Number of PCs")
    plt.ylabel("Total % Variance Explained")
    plt.show()


def _scrub_ntc_isolation_forest(adata: ad.AnnData, contamination_threshold: float) -> list:
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
    sc.pl.umap(adata, color=["is_outlier", "phase", "batch"], title="Pre-Filtering")
    sc.tl.embedding_density(adata)
    sc.pl.embedding_density(adata)
    
    # Filter out outliers and visualize Post-Filtering
    adata_filtered = adata[adata.obs.is_outlier == "inlier", :]
    sc.pp.neighbors(adata_filtered, use_rep=rep)
    sc.tl.umap(adata_filtered)
    sc.pl.umap(adata_filtered, color=["is_outlier", "phase", "batch"], title="Post-Filtering")
    sc.tl.embedding_density(adata_filtered)
    sc.pl.embedding_density(adata_filtered)
    
    return list(adata_filtered.obs.index)


def clean_ntc_cells(adata: ad.AnnData, contamination_threshold: float = 0.3, NTC_whitelist_path: str = None) -> ad.AnnData:
    """
    Scrubs non-targeting control (NTC) cells to identify valid cells using PCA and Isolation Forest.

    This function performs PCA to prepare the data and then uses an Isolation Forest to identify
    and filter out outlier cells, returning the indices of valid NTC cells.

    Parameters:
    - adata(anndata.AnnData): The AnnData object you wish to clean.
    - contamination_threshold (float): The proportion of outliers in the data (default is 0.3).
    - NTC_whitelist_path (str, optional): The path to a file containing NTC sgRNAs to keep.

    Returns:
    - anndata.AnnData: The cleaned AnnData object.
    """
    # Assertions to ensure required fields are present
    assert 'X' in adata.layers, "The AnnData object must have a 'counts' layer."
    assert 'perturbed' in adata.obs, "The AnnData object must have a 'perturbed' column in obs which indicates whether the cell is perturbed or not."
    assert 'batch' in adata.obs, "The AnnData object must have a 'batch' column in obs which indicates the batch the cell belongs to."

    adata_ntc = utils.get_ntc_view(adata).copy()
    print(f"Initial number of NTC Cells: {len(adata_ntc)}")

    if NTC_whitelist_path is not None:
        adata_ntc = get_NTCs_from_whitelist(adata_ntc, NTC_whitelist_path)
        print(f"Number of NTC Cells after whitelist filtering: {len(adata_ntc)}")
    
    # Perform PCA on the data
    _scrub_ntc_pca(adata_ntc)
    
    # Identify valid NTC cells using Isolation Forest
    valid_ntc_cells = _scrub_ntc_isolation_forest(adata_ntc, contamination_threshold)
    print(f"Number of NTC Cells after Isolation Forest filtering: {len(valid_ntc_cells)}")
    print(f"Number of NTC Cells per batch: {adata_ntc.obs.batch.value_counts()}")

    # Filter the original AnnData object to keep only the valid NTC cells
    perturbed_mask = adata.obs.perturbed == "True"
    valid_ntc_mask = adata.obs.index.isin(valid_ntc_cells)
    adata = adata[perturbed_mask & valid_ntc_mask].copy()
    print(f"Total number of cells after NTC cleaning: {len(adata.obs)}")
    return adata


def evaluate_per_sgRNA_knockdown(
    adata: ad.AnnData,
    repression_threshold: float = 30.0,
    cells_per_gRNA_threshold: int = 25,
    label_interval: int = 100,
    batch_aware: bool = True
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
    
    Returns:
    - Filtered AnnData object with quality-controlled cells and guides
    """
    # Validate input structure
    utils.validate_anndata(adata, required_obs=['gRNA', 'target_knockdown', 'perturbed'])
    
    # Split and process batches if applicable
    if batch_aware and 'batch' in adata.obs:
        batches = utils.split_by_batch(adata, copy=True)
        processed = []
        
        for batch_name, batch_adata in batches.items():
            print(f"Processing batch: {batch_name}")
            processed.append(
                _process_batch(
                    batch_adata,
                    repression_threshold,
                    cells_per_gRNA_threshold,
                    label_interval
                )
            )
            
        adata = ad.concat(processed, merge='same', join='inner')
    else:
        adata = _process_batch(adata, repression_threshold, 
                             cells_per_gRNA_threshold, label_interval)

    # Final filtering and visualization
    adata_perturbed = utils.get_perturbed_view(adata)
    pl.plotting.plot_sorted_bars(
        adata_perturbed.obs.gRNA.value_counts(),
        ylabel="Number of Cells per gRNA",
        title="Final Cell Counts per Guide",
        cells_threshold=cells_per_gRNA_threshold,
        label_interval=label_interval
    )
    
    return adata

def _process_batch(
    batch_adata: ad.AnnData,
    repression_threshold: float,
    cells_threshold: int,
    label_interval: int
) -> ad.AnnData:
    """Process a single batch of data"""
    # Calculate median knockdown percentages
    median_knockdown = 100 * batch_adata.obs.groupby("gRNA")["target_knockdown"].mean()
    non_ntc = median_knockdown[~median_knockdown.index.str.contains("Non-Targeting")]
    
    # Plot knockdown distribution
    pl.plotting.plot_sorted_bars(
        median_knockdown,
        ylabel="Mean Knockdown per Cell (%)",
        title="sgRNA Knockdown Efficiency",
        repression_threshold=repression_threshold,
        invert_y=True,
        label_interval=label_interval
    )
    
    # Filter low-efficiency guides
    invalid_guides = [
        g for g, val in median_knockdown.items()
        if "Non-Targeting" not in g and val <= repression_threshold
    ]
    batch_adata = _filter_guides(batch_adata, invalid_guides, "repression")
    
    # Filter cells with negative knockdown
    invalid_cells = (batch_adata.obs["target_knockdown"] < 0) & (batch_adata.obs.perturbed == "True")
    batch_adata = batch_adata[~invalid_cells].copy()
    
    # Filter guides with low cell counts
    guide_counts = batch_adata[batch_adata.obs.perturbed == "True"].obs.gRNA.value_counts()
    low_count_guides = guide_counts[guide_counts <= cells_threshold].index.tolist()
    return _filter_guides(batch_adata, low_count_guides, "cell count")

def _filter_guides(adata: ad.AnnData, guides: list, filter_type: str) -> ad.AnnData:
    """Helper function to filter guides with logging"""
    if not guides:
        return adata
        
    print(f"Removing {len(guides)} guides due to low {filter_type}")
    mask = adata.obs.gRNA.isin(guides)
    print(f"Guides before filtering: {len(adata.obs.gRNA.unique())}")
    adata = adata[~mask].copy()
    print(f"Guides after filtering: {len(adata.obs.gRNA.unique())}")
    return adata

def knockdown_qc(
    adata: ad.AnnData,
    obs_key: str = "gene_ids",
    var_key: str = "gene_target_ensembl_id",
    gene_target_expr_col: str = "gene_target_expression (CPM)",
    ntc_target_expr_col: str = "NTC_target_gene_expression (CPM)",
    knockdown_col: str = "target_knockdown",
    zscore_col: str = "target_knockdown_z_score",
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
    - zscore_col: Name for z-score column
    - ntc_label: Label indicating NTC guides
    - layer: Layer containing raw counts
    - normalized_layer: Layer to store normalized counts
    - batch_aware: Process batches separately if batch column exists
    
    Returns:
    - AnnData object with calculated metrics and quality controls
    """
    # Validate input structure
    utils.validate_anndata(adata, required_obs=[obs_key, 'perturbed'], required_var=[var_key])

    # Batch processing logic
    if batch_aware and 'batch' in adata.obs:
        batches = utils.split_by_batch(adata, copy=True)
        processed = []
        
        for batch_name, batch_adata in batches.items():
            print(f"Processing batch: {batch_name}")
            processed.append(
                _process_batch_knockdown(
                    batch_adata,
                    obs_key,
                    var_key,
                    gene_target_expr_col,
                    ntc_target_expr_col,
                    knockdown_col,
                    zscore_col,
                    ntc_label,
                    layer,
                    normalized_layer
                )
            )
            
        adata = ad.concat(processed, merge='same', join='inner')
    else:
        adata = _process_batch_knockdown(
            adata,
            obs_key,
            var_key,
            gene_target_expr_col,
            ntc_target_expr_col,
            knockdown_col,
            zscore_col,
            ntc_label,
            layer,
            normalized_layer
        )

    return evaluate_per_sgRNA_knockdown(adata, batch_aware=batch_aware)

def _process_batch_knockdown(
    batch_adata: ad.AnnData,
    obs_key: str,
    var_key: str,
    gene_target_expr_col: str,
    ntc_target_expr_col: str,
    knockdown_col: str,
    zscore_col: str,
    ntc_label: str,
    layer: str,
    normalized_layer: str
) -> ad.AnnData:
    """Process knockdown metrics for a single batch"""
    # Copy and normalize counts
    batch_adata.layers[normalized_layer] = batch_adata.layers[layer].copy()
    sc.pp.normalize_total(batch_adata, target_sum=1e6, layer=normalized_layer)
    
    # Calculate target expressions
    data = batch_adata.layers[normalized_layer]
    var_dict = {gene: idx for idx, gene in enumerate(batch_adata.var[var_key].values)}
    var_indices = batch_adata.obs[obs_key].map(var_dict).fillna(-1).astype(int).values
    
    # Initialize expression arrays
    gene_expr = np.zeros(batch_adata.n_obs)
    ntc_expr = np.zeros(batch_adata.n_obs)
    
    # Calculate NTC statistics
    is_ntc = (batch_adata.obs[obs_key] == ntc_label).values
    if np.any(is_ntc):
        ntc_data = data[is_ntc]
        ntc_mean = np.array(ntc_data.mean(axis=0)).flatten()
        ntc_var = np.array(ntc_data.var(axis=0)).flatten()
    else:
        ntc_mean = np.zeros(batch_adata.shape[1])
        ntc_var = np.ones(batch_adata.shape[1])
    
    # Assign expression values
    valid_mask = var_indices >= 0
    gene_expr[valid_mask] = np.array(data[np.arange(batch_adata.n_obs)[valid_mask], var_indices[valid_mask]]).flatten()
    ntc_expr[valid_mask] = ntc_mean[var_indices[valid_mask]]
    
    # Calculate knockdown efficiency
    with np.errstate(divide='ignore', invalid='ignore'):
        knockdown = 1 - np.divide(gene_expr, ntc_expr, out=np.ones_like(gene_expr), where=ntc_expr != 0)
    
    # Calculate z-scores
    zscores = np.zeros(batch_adata.n_obs)
    with np.errstate(divide='ignore', invalid='ignore'):
        expr_diff = gene_expr[valid_mask] - ntc_expr[var_indices[valid_mask]]
        var_mask = ntc_var[var_indices[valid_mask]] != 0
        zscores[valid_mask] = np.where(var_mask, expr_diff / np.sqrt(ntc_var[var_indices[valid_mask]]), 0)
    
    # Store results
    batch_adata.obs[gene_target_expr_col] = gene_expr
    batch_adata.obs[ntc_target_expr_col] = ntc_expr
    batch_adata.obs[knockdown_col] = knockdown
    batch_adata.obs[zscore_col] = zscores
    
    return batch_adata