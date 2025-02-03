import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map
from functools import partial
import anndata as ad
import scanpy as sc
from scipy.cluster import hierarchy
from psp.utils import _get_ntc_view, validate_anndata
# Define the function to compute the mean normalized profile
def __compute_mean_profile(adata: ad.AnnData, group_indices: np.ndarray) -> np.ndarray:
    """
    Compute mean expression profile for a group of cells.
    
    Parameters:
        adata: AnnData object containing single-cell data
        group_indices: Array of cell indices for the target group
        
    Returns:
        1D array of mean expression values across all genes
    """
    # Use .X directly since we handle layer selection in parent function
    mean_vector = adata[group_indices].X.mean(axis=0)
    return mean_vector.A1 if hasattr(mean_vector, 'A1') else mean_vector.flatten()

def compute_perturbation_correlation(
    adata: ad.AnnData,
    n_jobs: int = -1,
    key_added: str = "perturbation_correlation",
    progress: bool = True,
    cluster: bool = True
) -> None:
    """
    Compute and store gene target correlation matrix with optional hierarchical clustering.
    
    Parameters:
        adata: AnnData object containing single-cell data
        n_jobs: Number of parallel jobs (-1 = all available cores)
        key_added: Key for storing results in adata.uns
        progress: Show progress bar during computation
        cluster: Whether to perform hierarchical clustering and store ordered data
        
    Stores in AnnData:
        .uns[key_added]: Dictionary containing:
            - 'raw_matrix': Original correlation matrix
            - 'clustered_matrix': Reordered matrix (if clustered)
            - 'labels': Original gene target labels
            - 'clustered_labels': Reordered labels (if clustered)
            - 'dataframe': Original DataFrame
            - 'clustered_dataframe': Clustered DataFrame (if clustered)
    """
    # Validate input
    if 'gene_target' not in adata.obs:
        raise ValueError("adata.obs must contain 'gene_target' column")
        
    
    # Get unique gene targets and their indices
    gene_targets = adata.obs['gene_target'].astype('category')
    categories = gene_targets.cat.categories
    group_indices = [np.where(gene_targets == gt)[0] for gt in categories]

    # Create partial function for parallel processing
    compute_fn = partial(__compute_mean_profile, adata)
    
    # Compute mean profiles in parallel with progress bar
    mean_profiles = process_map(
        compute_fn,
        group_indices,
        desc="Computing mean profiles" if progress else None,
        max_workers=n_jobs,
        disable=not progress
    )
    
    # Convert to numpy array and store as layer
    mean_profiles = np.vstack(mean_profiles)
    adata.layers['psuedobulk_perturbation_profiles'] = mean_profiles

    # Compute correlation matrix and store as DataFrame
    corr_df = pd.DataFrame(
        np.corrcoef(mean_profiles),
        index=categories,
        columns=categories
    )
    
    # Store base results
    result_dict = {
        'raw_matrix': corr_df.to_numpy(),
        'labels': corr_df.columns.tolist(),
        'dataframe': corr_df
    }

    # Perform clustering if requested
    if cluster:
        # Compute distance matrix and clustering
        distance_matrix = 1 - corr_df  # Convert correlation to distance
        condensed_dist = distance_matrix.values[np.triu_indices_from(distance_matrix, k=1)]
        linkage_matrix = hierarchy.linkage(condensed_dist, method='average')
        
        # Get clustered order
        dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)
        ordered_labels = corr_df.columns[dendro['leaves']]
        
        # Store clustered data
        clustered_df = corr_df.loc[ordered_labels, ordered_labels]
        result_dict.update({
            'clustered_matrix': clustered_df.to_numpy(),
            'clustered_labels': ordered_labels.tolist(),
            'clustered_dataframe': clustered_df,
            'linkage_matrix': linkage_matrix
        })

    adata.uns[key_added] = result_dict


def compute_UMAP(
    adata: ad.AnnData,
    min_dist: float = 0.9,
    perturbation_key: str = "perturbed",
    batch_key: str = "batch"
) -> ad.AnnData:
    """
    Compute UMAP embeddings and visualize them.

    This function computes UMAP embeddings for the provided AnnData object and generates visualizations
    for perturbations and batch information. It also computes and plots the density of embeddings.

    Parameters:
    ----------
    adata : ad.AnnData
        The AnnData object containing the data.
    min_dist : float, optional
        The minimum distance between points in the UMAP embedding. Default is 0.9.
    perturbation_key : str, optional
        The key in the AnnData object that indicates perturbation status. Default is "perturbed".
    batch_key : str, optional
        The key in the AnnData object that indicates batch information. Default is "batch".

    Returns:
    -------
    adata : ad.AnnData
        The AnnData object with UMAP embeddings and density plots added.
    """
    # Store a copy of the raw data before UMAP processing
    adata.layers['pre_UMAP'] = adata.X.copy()

    # Validate input structure
    validate_anndata(adata, required_obs=[perturbation_key, batch_key])
    
    # Preprocessing steps
    sc.pp.scale(adata)  # Scale the data
    sc.pp.pca(adata)    # Perform PCA
    sc.pp.neighbors(adata)  # Compute the neighborhood graph
    
    # Compute UMAP
    sc.tl.umap(adata, min_dist=min_dist)
    
    # Plot UMAP for perturbations and batches
    sc.pl.umap(adata, color=[perturbation_key, batch_key], title=["Perturbations", "Perturbations - Batch"], frameon=False)
    sc.pl.umap(_get_ntc_view(adata), color=[perturbation_key, batch_key], title=["NTC Cells", "NTC Cells - Batch"], frameon=False)
    
    # Compute and plot Leiden clusters
    sc.tl.leiden(adata, n_iterations=2)
    sc.pl.umap(adata, color=["leiden"], title="Leiden Clusters", frameon=False)
    
    # Compute and plot embedding density for perturbations
    sc.tl.embedding_density(adata, groupby="perturbed")
    sc.pl.embedding_density(adata, groupby="perturbed", fg_dotsize=100, color_map='Reds', title=["NTC Cell Density", "Perturbed Cell Density"], frameon=False)
    
    return adata