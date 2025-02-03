import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map
from functools import partial
import anndata as ad
import scanpy as sc
from scipy.cluster import hierarchy
from psp.utils import _get_ntc_view, validate_anndata
from typing import Tuple, Dict, List
import pymde 
import plotly.io as pio 
import igraph as ig 
import plotly.express as px 
import leidenalg 
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from collections import defaultdict

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


def compute_MDE_map(
    adata: ad.AnnData,
    n_jobs: int = -1,
    random_state: int = 42,
    progress: bool = True,
    # MDE parameters
    preserve: str = "neighbors",
    spectral: bool = True,
    n_components: int = 30,
    mde_repulsive_fraction: float = 0.15,
    mde_n_neighbors: int = 15,
    spectral_n_neighbors: int = 10,
    # Clustering parameters
    leiden_resolution: float = 1.0,
    leiden_neighbors: int = 5,
    # Visualization parameters
    save: bool = True,
    save_dir_stem: str = None,
    plot_size: tuple = (1200, 1000),
    marker_size: int = 7,
    **kwargs
) -> Tuple[ad.AnnData, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deterministic MDE embedding for perturbation profiles with configurable parameters.
    
    Parameters:
        adata: AnnData object containing single-cell data
        n_jobs: Number of parallel jobs (-1 = all cores)
        random_state: Seed for reproducibility
        progress: Show progress bar during computation
        preserve: MDE preservation method ('neighbors' or 'distances')
        spectral: Whether to use spectral embedding initialization (default: True)
        n_components: Dimension for spectral embedding initialization
        mde_repulsive_fraction: Ratio of repulsive edges in MDE
        mde_n_neighbors: Number of neighbors for MDE preservation
        spectral_n_neighbors: Number of neighbors for spectral embedding
        leiden_resolution: Resolution parameter for Leiden clustering
        leiden_neighbors: Number of neighbors for KNN graph construction
        save: Whether to save results
        save_dir_stem: Base path for saving results
        plot_size: Dimensions for output plot (width, height)
        marker_size: Size of markers in scatter plot
        **kwargs: Additional arguments passed to pymde.preserve_neighbors
    
    Returns:
        Tuple containing:
        - AnnData object with embedding results
        - MDE embedding coordinates
        - Cluster labels
        - Mean perturbation profiles
    """
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    ig.set_random_number_generator(random_state)
    
    # Compute mean perturbation profiles
    gene_target_groups = adata.obs.groupby('gene_target').indices
    compute_fn = partial(_compute_mean_normalized_profile, adata)
    
    # Compute mean profiles with process_map
    mean_profiles = process_map(
        compute_fn,
        gene_target_groups.values(),
        desc="Computing mean profiles" if progress else None,
        max_workers=n_jobs if n_jobs != -1 else None,
        disable=not progress
    )
    
    mean_profiles_array = np.array(mean_profiles)
    
    # Compute MDE embedding
    if preserve == "neighbors" and spectral:
        embedder = SpectralEmbedding(
            n_components=n_components,
            affinity='nearest_neighbors',
            n_neighbors=spectral_n_neighbors,
            eigen_solver='arpack',
            random_state=random_state
        )
        initial_embedding = embedder.fit_transform(mean_profiles_array)
        mde = pymde.preserve_neighbors(
            initial_embedding,
            repulsive_fraction=mde_repulsive_fraction,
            n_neighbors=mde_n_neighbors,
            **kwargs
        )
    elif preserve == "neighbors":
        mde = pymde.preserve_neighbors(
            mean_profiles_array,
            repulsive_fraction=mde_repulsive_fraction,
            n_neighbors=mde_n_neighbors,
            **kwargs
        )
    else:
        mde = pymde.preserve_distances(mean_profiles_array, **kwargs)
    
    embedding = mde.embed(
        max_iter=4000 if spectral else 2000,
        print_every=100,
        verbose=True,
        random_state=random_state
    )
    
    # Leiden clustering with KNN graph
    knn_graph = kneighbors_graph(
        embedding, 
        n_neighbors=leiden_neighbors,
        include_self=False
    )
    sources, targets = knn_graph.nonzero()
    
    g = ig.Graph(directed=False)
    g.add_vertices(embedding.shape[0])
    g.add_edges(zip(sources, targets))
    
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution=leiden_resolution,
        seed=random_state
    )
    
    # Create cluster mapping
    cluster_labels = np.array(partition.membership)
    gene_targets = list(gene_target_groups.keys())
    
    # Get existing Perturbation_Stats DataFrame
    if "Perturbation_Stats" in adata.uns:
        pert_stats = adata.uns["Perturbation_Stats"]
    else:
        # Create empty DataFrame if not exists
        pert_stats = pd.DataFrame(index=gene_targets)
    
    # Merge cluster information
    cluster_df = pd.DataFrame({
        'gene_target': gene_targets,
        'mde_cluster': cluster_labels
    }).set_index('gene_target')
    
    # Join clusters to existing stats
    pert_stats = pert_stats.join(cluster_df, how='outer')
    
    # Update AnnData storage
    adata.uns["Perturbation_Stats"] = pert_stats
    
    # Create and save visualization
    embedding_df = _create_embedding_dataframe(
        embedding, 
        gene_target_groups,
        partition.membership
    )
    
    if save and save_dir_stem:
        _save_embedding_results(embedding_df, save_dir_stem, plot_size, marker_size)
    
    return adata, embedding, np.array(partition.membership), mean_profiles_array

def _compute_mean_normalized_profile(
    adata: ad.AnnData, 
    group_indices: np.ndarray
) -> np.ndarray:
    """Helper function to compute normalized mean profiles"""
    mean_vector = adata[group_indices, :].X.mean(axis=0)
    normalized = (mean_vector - mean_vector.mean()) / mean_vector.std()
    return normalized.A1 if hasattr(normalized, 'A1') else normalized

def _create_embedding_dataframe(embedding, gene_groups, clusters):
    """Create formatted DataFrame for visualization"""
    return pd.DataFrame(embedding, columns=['x', 'y']).assign(
        gene_target=list(gene_groups.keys()),
        cluster=clusters.astype(str)
    )

def _save_embedding_results(df, save_stem, plot_size, marker_size):
    """Save visualization and cluster results"""
    fig = px.scatter(
        df, x='x', y='y', text='gene_target', color='cluster',
        hover_data={'x': True, 'y': True, 'gene_target': True},
        title='MDE Embedding of Mean Normalized Profiles',
        color_discrete_sequence=px.colors.qualitative.Bold
    ).update_traces(
        marker=dict(size=marker_size, opacity=0.7),
        textposition='middle center',
        textfont=dict(size=4)
    ).update_layout(
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        width=plot_size[0],
        height=plot_size[1],
        legend_title_text='Cluster'
    )
    
    pio.write_html(fig, f"{save_stem}_MDE.html")
    df.to_excel(f"{save_stem}_clusters.xlsx", index=False)
    fig.update_traces(text=None).write_image(f"{save_stem}_MDE.svg")

def map_complexes(
    cluster_file: str,
    corum_complex_path: str,
    gene_id_mapping_path: str,
    entrez_mapping_path: str,
    min_overlap_ratio: float = 0.66
) -> Dict[str, List[str]]:
    """
    Map protein complexes to perturbation clusters using CORUM database.
    
    Parameters:
        cluster_file: Path to cluster XLSX file
        corum_complex_path: Path to CORUM complexes JSON file
        gene_id_mapping_path: Path to gene ID mapping file
        entrez_mapping_path: Path to Entrez-ENSG mapping file
        min_overlap_ratio: Minimum overlap ratio required for association
        
    Returns:
        Dictionary mapping clusters to associated complexes
    """
    # Helper functions
    def _load_gene_ids(filepath: str) -> Dict[str, str]:
        return pd.read_csv(filepath, sep='\t', header=None, 
                          names=['gene', 'ensg']).set_index('gene')['ensg'].to_dict()

    def _check_overlap(A: set, B: set, ratio: float) -> bool:
        return len(A & B) >= ratio * len(B)

    # Load mappings
    ensg_to_entrez = pd.read_csv(entrez_mapping_path, sep='\t', header=None,
                                names=['entrez', 'ensg']).set_index('ensg')['entrez'].to_dict()
    gene_to_ensg = _load_gene_ids(gene_id_mapping_path)
    
    # Load CORUM complexes
    corum_df = pd.read_json(corum_complex_path)
    complexes = {
        row['ComplexName']: set(row['subunits(Entrez IDs)'].split(';'))
        for _, row in corum_df[~corum_df['ComplexName'].str.contains('homo')].iterrows()
    }
    
    # Load clusters and convert genes
    clusters_df = pd.read_excel(cluster_file)
    clusters = clusters_df.groupby('cluster')['gene_target'].apply(list).to_dict()
    
    # Find associated complexes
    associated = defaultdict(list)
    for cluster, genes in clusters.items():
        cluster_entrez = {
            ensg_to_entrez.get(gene_to_ensg.get(g, ''), '')
            for g in genes
        } - {''}  # Remove failed lookups
        
        for comp, subunits in complexes.items():
            if _check_overlap(cluster_entrez, subunits, min_overlap_ratio):
                associated[cluster].append(comp)
    
    return dict(associated)