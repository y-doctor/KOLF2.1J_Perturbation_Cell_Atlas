import anndata as ad
import os
import pandas as pd
import numpy as np
import scanpy as sc

def split_by_batch(adata: ad.AnnData, copy: bool = True) -> dict[str, ad.AnnData]:
    """
    Split the AnnData object by batch.

    Parameters:
    - adata (anndata.AnnData): The AnnData object to split.
    - copy (bool): If True, return a copy of the split data. If False, return a view. Default is True.

    Returns:
    - dict[str, anndata.AnnData]: A dictionary where keys are batch identifiers and values are the corresponding AnnData objects.
    """
    return {batch: adata[adata.obs.batch == batch].copy() if copy else adata[adata.obs.batch == batch] for batch in adata.obs.batch.unique()}

def split_by_channel(adata: ad.AnnData, copy: bool = True) -> dict[str, ad.AnnData]:
    """
    Split the AnnData object by channel.

    Parameters:
    - adata (anndata.AnnData): The AnnData object to split.
    - copy (bool): If True, return a copy of the split data. If False, return a view. Default is True.

    Returns:
    - dict[str, anndata.AnnData]: A dictionary where keys are channel identifiers and values are the corresponding AnnData objects.
    """
    return {channel: adata[adata.obs.channel == channel].copy() if copy else adata[adata.obs.channel == channel] for channel in adata.obs.channel.unique()}

def split_by_treatment(adata: ad.AnnData, copy: bool = True) -> dict[str, ad.AnnData]:
    """
    Split the AnnData object by treatment.

    Parameters:
    - adata (anndata.AnnData): The AnnData object to split.
    - copy (bool): If True, return a copy of the split data. If False, return a view. Default is True.

    Returns:
    - dict[str, anndata.AnnData]: A dictionary where keys are treatment identifiers and values are the corresponding AnnData objects.
    """
    return {treatment: adata[adata.obs.treatment == treatment].copy() if copy else adata[adata.obs.treatment == treatment] for treatment in adata.obs.treatment.unique()}

def split_by_cell_type(adata: ad.AnnData, copy: bool = True) -> dict[str, ad.AnnData]:
    """
    Split the AnnData object by cell type.

    Parameters:
    - adata (anndata.AnnData): The AnnData object to split.
    - copy (bool): If True, return a copy of the split data. If False, return a view. Default is True.

    Returns:
    - dict[str, anndata.AnnData]: A dictionary where keys are cell type identifiers and values are the corresponding AnnData objects.
    """
    return {cell_type: adata[adata.obs.cell_type == cell_type].copy() if copy else adata[adata.obs.cell_type == cell_type] for cell_type in adata.obs.cell_type.unique()}

def split_by_metadata(adata: ad.AnnData, metadata_key: str, copy: bool = True) -> dict[str, ad.AnnData]:
    """
    Split the AnnData object by a metadata key.

    Parameters:
    - adata (anndata.AnnData): The AnnData object to split.
    - metadata_key (str): The key in `adata.obs` to split by.
    - copy (bool): If True, return a copy of the split data. If False, return a view. Default is True.

    Returns:
    - dict[str, anndata.AnnData]: A dictionary where keys are unique values of the specified metadata key and values are the corresponding AnnData objects.
    """
    return {metadata: adata[adata.obs[metadata_key] == metadata].copy() if copy else adata[adata.obs[metadata_key] == metadata] for metadata in adata.obs[metadata_key].unique()}

def __save_by_dict(adata_dict: dict[str, ad.AnnData], save_path: str) -> None:
    """
    Save the AnnData objects in the dictionary to separate files.
    """
    for key, adata in adata_dict.items():
        adata.write(os.path.join(save_path, f"{key}.h5ad")) 

def validate_anndata(adata: ad.AnnData, required_obs: list[str] = None, required_varm: list[str] = None) -> None:
    """
    Validate an AnnData object's structure.
    
    Parameters:
    - adata: AnnData object to validate
    - required_obs: List of required obs columns
    - required_varm: List of required varm entries
    
    Raises:
    - ValueError if any required elements are missing
    """
    if required_obs:
        missing_obs = [col for col in required_obs if col not in adata.obs]
        if missing_obs:
            raise ValueError(f"Missing required obs columns: {missing_obs}")
            
    if required_varm:
        missing_varm = [key for key in required_varm if key not in adata.varm]
        if missing_varm:
            raise ValueError(f"Missing required varm entries: {missing_varm}")

def get_perturbed_view(adata: ad.AnnData) -> ad.AnnData:
    """
    Get a view of the AnnData object containing only perturbed cells.
    
    Parameters:
    - adata: Input AnnData object
    - copy: Whether to return a copy (default) or view
    
    Returns:
    - Subset AnnData containing only perturbed cells
    """
    if 'perturbed' not in adata.obs:
        raise ValueError("Missing required 'perturbed' column in obs")
    return adata[adata.obs.perturbed == "True"]

def get_ntc_view(adata: ad.AnnData) -> ad.AnnData:
    """
    Get a view of the AnnData object containing only non-targeting control (NTC) cells.
    
    Parameters:
    - adata: Input AnnData object
    - copy: Whether to return a copy (default) or view
    
    Returns:
    - Subset AnnData containing only NTC cells
    """
    if 'perturbed' not in adata.obs:
        raise ValueError("Missing required 'perturbed' column in obs")
    return adata[adata.obs.perturbed == "False"]

def read_gtf(gtf_path: str) -> pd.DataFrame:
    """
    Read and parse GTF file into gene information DataFrame.
    
    Parameters:
    - gtf_path: Path to GTF file
    - gene_type: Type of genes to filter (e.g. 'protein_coding')
    
    Returns:
    - DataFrame with columns: ['seqname', 'gene_name', 'gene_type', 'gene_id']
    """
    # ... existing _read_gtf implementation from lines 450-483 ...
    
def identify_coding_genes(
    adata: ad.AnnData, 
    gtf_path: str,
    subset: bool = False
) -> ad.AnnData:
    """
    Annotate and optionally subset to protein-coding genes.
    
    Parameters:
    - adata: AnnData object with gene_ids in var
    - gtf_path: Path to GTF file with gene annotations
    - subset: Whether to subset to protein-coding genes
    
    Returns:
    - AnnData with gene_type annotation and optionally subset
    """
    # ... existing identify_coding_genes implementation from lines 430-441 ...


def isolate_strong_perturbations(adata: ad.AnnData, min_total_deg: int = 10, gene_target_key: str = "gene_target") -> ad.AnnData:
    """
    Isolate strong perturbations based on the number of differentially expressed genes (DEGs).

    This function filters the AnnData object to include only perturbations with a total number of DEGs
    greater than or equal to the specified minimum threshold.

    Parameters:
    - adata: AnnData object containing DEG results in `adata.uns`
    - min_total_deg: Minimum number of DEGs required to include a perturbation

    Returns:
    - AnnData object containing only strong perturbations
    """

    # Validate input structure
    validate_anndata(adata, required_obs=[gene_target_key])

    if 'Pertubation_Stats' not in adata.uns:
        raise ValueError("Missing required 'Pertubation_Stats' in adata.uns")
    

    # Extract DEG summary from adata.uns
    deg_summary = adata.uns['Pertubation_Stats']
    
    # Filter perturbations with total DEGs >= min_total_deg
    strong_perturbations = deg_summary[deg_summary['Total_DEGs'] >= min_total_deg]
    
    # Create a view of the strong perturbations
    strong_view = adata[adata.obs[gene_target_key].isin(strong_perturbations.index)]
    
    return strong_view


def relative_z_normalization(
    adata: ad.AnnData,
    batch_key: str = "batch",
    clip_min: float = -10,
    clip_max: float = 10,
    ntc_identifier: str = "False",
    layer_name: str = "pre_z_normalization"
) -> None:
    """
    Perform batch-aware Z-normalization using control cells as reference.
    
    Computes normalization parameters (mean, std) from non-targeting control (NTC) cells
    and applies to all cells. When batch_key is provided, calculates separate parameters
    for each batch.

    Parameters:
    - adata: AnnData object containing single-cell data
    - batch_key: Optional column in adata.obs defining batches for batch-specific normalization
    - clip_min: Minimum value for clipping normalized data
    - clip_max: Maximum value for clipping normalized data
    - ntc_identifier: Value in adata.obs['perturbed'] indicating control cells
    - layer_name: Name of layer to store pre-normalized data

    Returns:
    - Modifies adata in-place with normalized values in adata.X
    
    Examples:
    >>> # Global normalization
    >>> relative_z_normalization(adata)
    
    >>> # Batch-specific normalization
    >>> relative_z_normalization(adata, batch_key='batch')
    """
    # Store original data in layer
    adata.layers[layer_name] = adata.X.copy()
    
    if batch_key:
        # Batch-aware processing
        batches = split_by_batch(adata, copy=False)
        
        for batch_name, batch_data in batches.items():
            # Get NTC cells for this batch
            ntc_mask = batch_data.obs['perturbed'] == ntc_identifier
            if not ntc_mask.any():
                raise ValueError(f"Batch {batch_name} has no NTC cells")
                
            ntc_adata = batch_data[ntc_mask]
            X_ntc = ntc_adata.X.toarray() if hasattr(ntc_adata.X, 'toarray') else ntc_adata.X
            
            # Compute batch-specific parameters
            mu = X_ntc.mean(axis=0)
            std = X_ntc.std(axis=0)
            std[std == 0] = 1  # Prevent division by zero
            
            # Normalize batch data
            X_batch = batch_data.X
            X_norm = np.clip((X_batch - mu) / std, clip_min, clip_max)
            
            # Update original adata with normalized values
            adata.X[batch_data.obs.index] = X_norm
    else:
        # Global normalization
        ntc_adata = adata[adata.obs['perturbed'] == ntc_identifier]
        X_ntc = ntc_adata.X.toarray() if hasattr(ntc_adata.X, 'toarray') else ntc_adata.X
        
        mu = X_ntc.mean(axis=0)
        std = X_ntc.std(axis=0)
        std[std == 0] = 1
        
        adata.X = np.clip((adata.X - mu) / std, clip_min, clip_max)


def compute_hvg(adata: ad.AnnData, layer: str = "counts", n_HVGs: int = 2000, batch_key: str = "batch") -> None:
    """
    Compute highly variable genes (HVGs) using the scanpy.pp.highly_variable_genes function.

    This function performs HVG selection on the specified layer of the AnnData object.

    Parameters:
    - adata: AnnData object containing single-cell data
    - layer: Name of layer to use for HVG selection
    - n_HVGs: Number of top HVGs to select
    - batch_key: Optional column in adata.obs defining batches for batch-specific HVG selection

    Returns:
    - None
    """
    # Validate input structure
    validate_anndata(adata, required_obs=[batch_key])

    if batch_key:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_HVGs, subset=False, flavor='seurat_v3', layer='counts', batch_key='run')
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_HVGs, subset=False, flavor='seurat_v3', layer='counts')



def count_sgRNAs_per_perturbation(adata: ad.AnnData, gRNA_key: str = "gRNA", gene_target_key: str = "gene_target") -> None:
    """
    Count the number of unique sgRNAs assigned to each perturbation.

    This function groups the data by the specified gene target key and counts the unique sgRNAs 
    associated with each perturbation. The results are stored in the 'Perturbation_Stats' 
    section of the AnnData object.

    Parameters:
    ----------
    adata : ad.AnnData
        The AnnData object containing the single-cell data and relevant observations.
    gRNA_key : str, optional
        The key in adata.obs that contains the sgRNA identifiers, by default "gRNA".
    gene_target_key : str, optional
        The key in adata.obs that contains the gene target identifiers, by default "gene_target".

    Raises:
    ------
    ValueError
        If 'Perturbation_Stats' is missing from adata.uns or if the required observation keys 
        are not present in the AnnData object.

    Examples:
    --------
    >>> count_sgRNAs_per_perturbation(adata)
    """
    
    # Validate input structure
    validate_anndata(adata, required_obs=[gene_target_key, gRNA_key])

    if adata.uns['Perturbation_Stats'] is None:
        raise ValueError("Missing required 'Perturbation_Stats' in adata.uns")

    # Count the number of sgRNA assigned to each perturbation
    gRNA_counts = adata.obs.groupby(gene_target_key)[gRNA_key].nunique()
    
    # Ensure the counts map correctly to the "Perturbation" column in Perturbation_Stats
    adata.uns["Perturbation_Stats"]["sgRNAs_per_perturbation"] = gRNA_counts.reindex(adata.uns["Perturbation_Stats"].index, fill_value=0)

        
       