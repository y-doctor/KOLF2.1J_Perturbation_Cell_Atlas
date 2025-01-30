import anndata as ad
import psp.qc as qc
from sklearn.ensemble import IsolationForest
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import requests

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
    ntc_adata = qc._get_ntc_view(adata)
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

    adata_ntc = qc._get_ntc_view(adata).copy()
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



