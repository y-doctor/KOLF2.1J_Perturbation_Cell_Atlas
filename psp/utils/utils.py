import anndata as ad
import os
import pandas as pd

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

