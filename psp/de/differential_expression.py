import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
from psp.utils import *
from psp.pl.plotting import plot_filtered_genes_inverted 
import scipy.sparse
import re
import os
import psutil
import time
import sys
import tempfile
import shutil
import anndata
from typing import Optional, Dict

# Configure logging only if no handlers are present
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)

# Set global plotting parameters
plt.rcParams['font.family'] = 'Arial'

def generate_pseudo_bulk_replicates_for_de(
    adata, 
    target_value, 
    ntc_cells, 
    n_replicates=3, 
    sample_fraction=0.85, 
    layer="counts",  # Default to counts layer
    target_column="gene_target",
    random_seed=42,
    output_dir=None
):
    """
    Generate pseudo-bulk replicates for differential expression analysis.
    Optimized version using anndata views and vectorized operations.
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create views for target and NTC cells
    target_mask = adata.obs[target_column] == target_value
    target_view = adata[target_mask]
    ntc_view = adata[ntc_cells]
    
    # Get data matrices from counts layer
    target_data = target_view.layers[layer]
    ntc_data = ntc_view.layers[layer]
    
    # Convert to dense if sparse
    if scipy.sparse.issparse(target_data):
        target_data = target_data.toarray()
    if scipy.sparse.issparse(ntc_data):
        ntc_data = ntc_data.toarray()
    
    # Calculate number of cells to sample
    n_target = target_data.shape[0]
    n_ntc = ntc_data.shape[0]
    n_sample_target = int(n_target * sample_fraction)
    n_sample_ntc = min(n_ntc, n_sample_target)

    
    # Pre-allocate arrays for results
    target_bulk = np.zeros((n_replicates, target_data.shape[1]), dtype=np.int64)
    ntc_bulk = np.zeros((n_replicates, ntc_data.shape[1]), dtype=np.int64)
    
    # Generate replicates using vectorized operations
    for i in range(n_replicates):

        # Sample indices
        target_indices = np.random.choice(n_target, n_sample_target, replace=False)
        ntc_indices = np.random.choice(n_ntc, n_sample_ntc, replace=False)
        
        # Calculate sums using vectorized operations
        target_bulk[i] = np.sum(target_data[target_indices], axis=0).astype(np.int64)
        ntc_bulk[i] = np.sum(ntc_data[ntc_indices], axis=0).astype(np.int64)        
    
    # Create sample names
    if "_" in target_value:
        target_value = target_value.replace("_", "-")
    sample_names = [f"{target_value}-rep{i+1}" for i in range(n_replicates)]
    control_names = [f"NTC-rep{i+1}" for i in range(n_replicates)]
    
    # Combine data
    combined_data = np.vstack([target_bulk, ntc_bulk])
    combined_names = sample_names + control_names
    
    # Create metadata
    metadata = pd.DataFrame({
        'condition': [target_value] * n_replicates + ['NTC'] * n_replicates
    }, index=combined_names)
    
    # Create count matrix
    count_matrix = pd.DataFrame(
        combined_data,
        index=combined_names,
        columns=adata.var_names
    )
    
    # Save to temporary files if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_target = target_value.replace('/', '_').replace('\\', '_')
        count_path = os.path.join(output_dir, f"{safe_target}_counts.csv")
        metadata_path = os.path.join(output_dir, f"{safe_target}_metadata.csv")
        
        count_matrix.to_csv(count_path)
        metadata.to_csv(metadata_path)
    
    return count_matrix, metadata


def generate_pseudo_bulk_files(
    adata: anndata.AnnData,
    gene_target_obs_column: str = 'perturbation',
    ntc_cells_delimiter: str = 'NTC',
    batch_key: Optional[str] = None,
    n_replicates: int = 3,
    sample_fraction: float = 0.7,
    layer: str = 'counts',
    random_seed: int = 42,
    output_dir: str = './pseudo_bulk_output',
    n_jobs: int = -1,  # Default to using all available cores
    verbose: bool = False  # Default to no logging
) -> None:
    """
    Generate pseudo-bulk replicates and save them to files for later DESeq2 analysis.
    
    Args:
        adata: AnnData object containing single-cell data
        gene_target_obs_column: Column in adata.obs containing gene targets
        ntc_cells_delimiter: String used to identify NTC cells
        batch_key: Column in adata.obs containing batch information
        n_replicates: Number of pseudo-bulk replicates to generate
        sample_fraction: Fraction of cells to sample for each replicate
        layer: Layer in adata to use for counts
        random_seed: Random seed for reproducibility
        output_dir: Directory to save intermediate files
        n_jobs: Number of parallel jobs to use (-1 for all available cores)
        verbose: Whether to print progress information
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    def process_perturbation(perturbation, batch=None):
        try:
            # Use a unique seed for this perturbation
            target_seed = random_seed + hash(f"{perturbation}_{batch}" if batch else perturbation) % 10000
            
            # Get the appropriate adata and ntc cells
            if batch:
                batch_mask = adata.obs[batch_key] == batch
                batch_adata = adata[batch_mask]
                ntc_cells = batch_adata.obs[batch_adata.obs[gene_target_obs_column] == ntc_cells_delimiter].index
                working_adata = batch_adata
            else:
                ntc_cells = adata.obs[adata.obs[gene_target_obs_column] == ntc_cells_delimiter].index
                working_adata = adata
            
            # Generate pseudo-bulk replicates
            pseudo_bulk_df, metadata_df = generate_pseudo_bulk_replicates_for_de(
                working_adata,
                perturbation,
                ntc_cells,
                n_replicates,
                sample_fraction,
                layer,
                target_column=gene_target_obs_column,
                random_seed=target_seed
            )
            
            # Save to files
            target_dir = os.path.join(output_dir, f"{perturbation}_{batch}" if batch else perturbation)
            os.makedirs(target_dir, exist_ok=True)
            
            pseudo_bulk_df.to_csv(os.path.join(target_dir, "pseudo_bulk.csv"))
            metadata_df.to_csv(os.path.join(target_dir, "metadata.csv"))
            
            # Clean up memory
            del pseudo_bulk_df
            del metadata_df
            if batch:
                del batch_adata
                del ntc_cells
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"Error processing {perturbation} in batch {batch}: {str(e)}")
            return False
    
    if batch_key is not None:
        # Get unique batches
        batches = adata.obs[batch_key].unique()
        
        # Process each batch with a progress bar
        for batch in tqdm(batches, desc="Processing batches"):
            # Get perturbations that exist in this batch
            batch_mask = adata.obs[batch_key] == batch
            batch_adata = adata[batch_mask]
            batch_perturbations = batch_adata.obs[gene_target_obs_column].unique()
            batch_perturbations = batch_perturbations[batch_perturbations != ntc_cells_delimiter]
            
            # Process perturbations in parallel
            with Parallel(n_jobs=n_jobs) as parallel:
                results = parallel(
                    delayed(process_perturbation)(perturbation, batch)
                    for perturbation in tqdm(batch_perturbations, desc=f"Batch {batch}", leave=False)
                )
            
            # Clean up batch data
            del batch_adata
            del batch_perturbations
            
    else:
        # Process without batch awareness
        perturbations = adata.obs[gene_target_obs_column].unique()
        perturbations = perturbations[perturbations != ntc_cells_delimiter]
        
        # Process perturbations in parallel
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                delayed(process_perturbation)(perturbation)
                for perturbation in tqdm(perturbations, desc="Processing perturbations")
            )


def differential_expression(
    adata,
    adata_deg_test_column="perturbation",
    ntc_cells_delimiter="NTC",
    batch_key=None,
    n_replicates=3,
    sample_fraction=0.7,
    layer="counts",
    alpha=0.05,
    random_seed=42,
    save_path=None,
    plot_degs=True,
    fig_save_path=None,
    plot_genes_interval=100,
    debug=False,
    save_all_results=False
):
    """
    Perform differential expression analysis sequentially without precomputed files.
    
    Args:
        adata: AnnData object containing single-cell data
        adata_deg_test_column: Column in adata.obs to be tested for differential expression
        ntc_cells_delimiter: String used to identify NTC cells
        batch_key: Column in adata.obs containing batch information
        n_replicates: Number of pseudo-bulk replicates to generate
        sample_fraction: Fraction of cells to sample for each replicate
        layer: Layer in adata to use for counts
        alpha: Significance threshold for DESeq2
        random_seed: Random seed for reproducibility
        save_path: Path to save results
        plot_degs: Whether to plot DEGs
        fig_save_path: Path to save plots
        plot_genes_interval: Number of genes to plot at a time
        debug: Whether to run in debug mode

    Returns:
        Dictionary of results DataFrames
    """
    start_time = time.time()
    results_dict = {}
    
    # Get unique perturbations from the anndata object
    if adata_deg_test_column not in adata.obs.columns:
        raise ValueError(f"Column '{adata_deg_test_column}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

    # Get unique gene targets
    perturbations = list(adata.obs[adata_deg_test_column].unique())
    perturbations.remove(ntc_cells_delimiter)

    if debug:
        print(f"Debug mode: running on 2 targets")
        perturbations = perturbations[:2]
    
    if batch_key is not None:
        # Process with batch awareness
        batches = adata.obs[batch_key].unique()
        
        for batch in tqdm(batches, desc="Processing batches"):
            batch_mask = adata.obs[batch_key] == batch
            batch_adata = adata[batch_mask]
            
            if ntc_cells_delimiter not in batch_adata.obs[adata_deg_test_column].unique():
                logger.warning(f"No {ntc_cells_delimiter} cells in batch {batch} - skipping")
                continue
            
            batch_ntc_cells = np.where(batch_adata.obs[adata_deg_test_column] == ntc_cells_delimiter)[0]
            batch_targets = [t for t in perturbations if t in batch_adata.obs[adata_deg_test_column].unique()]
            
            for perturbation in tqdm(batch_targets, desc=f"Batch {batch}", leave=False):
                try:
                    target_seed = random_seed + hash(f"{perturbation}_{batch}") % 10000
                    
                    # Generate pseudo-bulk replicates
                    pseudo_bulk_df, metadata_df = generate_pseudo_bulk_replicates_for_de(
                        batch_adata,
                        perturbation,
                        batch_ntc_cells,
                        n_replicates,
                        sample_fraction,
                        layer,
                        target_column=adata_deg_test_column,
                        random_seed=target_seed
                    )
                    

                    # Run DESeq2
                    np.random.seed(random_seed)
                    dds = DeseqDataSet(
                        counts=pseudo_bulk_df,
                        metadata=metadata_df,
                        refit_cooks=True,
                        quiet = True
                    )

                    dds.deseq2()
                    
                    # Get results
                    perturbation_hyphenated = perturbation.replace('_', '-')
                    res = DeseqStats(dds, contrast=['condition', perturbation_hyphenated, 'NTC'], quiet=True)
                    res.summary()
                    degs = res.results_df
                    
                    # Filter by significance
                    # Filter by significance
                    if save_all_results:
                        results_dict[perturbation] = degs
                    else:
                        degs = degs[degs['padj'] < alpha]
                        results_dict[perturbation] = degs
                    

                    
                    # Clean up memory
                    del pseudo_bulk_df
                    del metadata_df
                    del dds
                    del res
                    
                except Exception as e:
                    logger.error(f"Error processing {perturbation} in batch {batch}: {str(e)}")
                    continue
                
            # Clean up batch data
            del batch_adata
            del batch_ntc_cells
            
    else:
        # Process without batch awareness
        ntc_cells = np.where(adata.obs[adata_deg_test_column] == ntc_cells_delimiter)[0]
        
        for perturbation in tqdm(perturbations, desc="Processing perturbations"):
            try:
                target_seed = random_seed + hash(perturbation) % 10000
                
                # Generate pseudo-bulk replicates
                pseudo_bulk_df, metadata_df = generate_pseudo_bulk_replicates_for_de(
                    adata,
                    perturbation,
                    ntc_cells,
                    n_replicates,
                    sample_fraction,
                    layer,
                    target_column=adata_deg_test_column,
                    random_seed=target_seed
                )
                

                # Run DESeq2
                np.random.seed(random_seed)
                dds = DeseqDataSet(
                    counts=pseudo_bulk_df,
                    metadata=metadata_df,
                    refit_cooks=True,
                    quiet = True
                )
                dds.deseq2()
                
                # Get results
                perturbation_hyphenated = perturbation.replace('_', '-')
                res = DeseqStats(dds, contrast=['condition', perturbation_hyphenated, 'NTC'], quiet=True)
                res.summary()
                degs = res.results_df
                
                # Filter by significance
                if save_all_results:
                    results_dict[perturbation] = degs
                else:
                    degs = degs[degs['padj'] < alpha]
                    results_dict[perturbation] = degs


                # Clean up memory
                del pseudo_bulk_df
                del metadata_df
                del dds
                del res
                
            except Exception as e:
                logger.error(f"Error processing {perturbation}: {str(e)}")
                continue
    
    # Process and save results
    if save_path:
        _save_DEG_df(results_dict, p_threshold=alpha, save=True, filepath=save_path,save_all_results=save_all_results)
    
    if plot_degs:
        fig, _ = plot_filtered_genes_inverted(results_dict, p_value_threshold=alpha, ytick_step=plot_genes_interval)
        fig.savefig(fig_save_path)
    # Update adata object with DEG counts
    print("\nUpdating adata object with DEG counts")

    if f'n_DEGs_{adata_deg_test_column}' not in adata.obs.columns:
        adata.obs[f'n_DEGs_{adata_deg_test_column}'] = np.zeros(adata.n_obs)
    
    if batch_key is not None:
        for key, degs_df in results_dict.items():
            target = '_'.join(key.split('_')[:-1])
            batch = key.split('_')[-1]
            mask = (adata.obs[adata_deg_test_column] == target) & (adata.obs[batch_key] == batch)
            if mask.sum() > 0:
                adata.obs.loc[mask, f'n_DEGs_{adata_deg_test_column}'] = len(degs_df)
    else:
        for key, degs_df in results_dict.items():
            mask = adata.obs[adata_deg_test_column] == key
            adata.obs.loc[mask, f'n_DEGs_{adata_deg_test_column}'] = len(degs_df)

    logger.info(f"Completed differential expression analysis in {time.time() - start_time:.2f} seconds")
    return results_dict


def differential_expression_from_files(
    adata,
    pseudo_bulk_dir: str = './pseudo_bulk_output',
    adata_deg_test_column: str = 'perturbation',
    batch_key: Optional[str] = None,
    ntc_cells_delimiter: str = 'NTC',
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    plot_degs: bool = True,
    fig_save_path: Optional[str] = None,
    plot_genes_interval: int = 500,
    debug: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Run DESeq2 analysis on pre-generated pseudo-bulk files and update adata object.
    
    Args:
        adata: AnnData object to update with DEG counts
        pseudo_bulk_dir: Directory containing pre-generated pseudo-bulk files
        adata_deg_test_column: Column in adata.obs to be tested for differential expression
        batch_key: Column in adata.obs containing batch information
        ntc_cells_delimiter: String used to identify NTC cells
        alpha: Significance threshold for DESeq2
        save_path: Path to save results
        plot_degs: Whether to plot DEGs
        fig_save_path: Path to save plots
        plot_genes_interval: Number of genes to plot at a time
        debug: Whether to run in debug mode
    Returns:
        Dictionary of results DataFrames
    """
    start_time = time.time()
    results_dict = {}
    
    print(f"Starting differential expression analysis from files in {pseudo_bulk_dir}")
    
    # Get unique perturbations from the anndata object
    if adata_deg_test_column not in adata.obs.columns:
        raise ValueError(f"Column '{adata_deg_test_column}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    
    # Get unique gene targets
    perturbations = list(adata.obs[adata_deg_test_column].unique())
    perturbations.remove(ntc_cells_delimiter)
    print(f"Found {len(perturbations)} unique perturbations in the anndata object")

    if debug:
        print(f"Debug mode: running on 2 perturbations")
        perturbations = perturbations[:2]
    
    # Check for batch information
    if batch_key and batch_key in adata.obs.columns:
        batches = adata.obs[batch_key].unique()
        print(f"Found {len(batches)} batches: {', '.join(batches[:5])}{'...' if len(batches) > 5 else ''}")
        
        # For each perturbation, check for batch-specific files
        for perturbation in tqdm(perturbations, desc="Computing DEGs per perturbation"):
            for batch in batches:
                # Check if this perturbation exists in this batch
                pert_batch_mask = (adata.obs[adata_deg_test_column] == perturbation) & (adata.obs[batch_key] == batch)
                if pert_batch_mask.sum() == 0:
                    continue  # Skip if no cells with this perturbation in this batch
                
                # Check if files exist for this perturbation-batch combination
                target_dir = f"{perturbation}_{batch}"
                target_dir_path = os.path.join(pseudo_bulk_dir, target_dir)
                
                # Try to find files for this target_dir
                if not os.path.isdir(target_dir_path):
                    print(f"Directory not found for {perturbation} in batch {batch}, skipping")
                    continue
                
                pseudo_bulk_path = os.path.join(target_dir_path, "pseudo_bulk.csv")
                metadata_path = os.path.join(target_dir_path, "metadata.csv")
                
                if not os.path.exists(pseudo_bulk_path) or not os.path.exists(metadata_path):
                    print(f"Missing files for {perturbation} in batch {batch}, skipping")
                    continue
                
                try:
                    # Read the files
                    pseudo_bulk_df = pd.read_csv(pseudo_bulk_path, index_col=0)
                    metadata_df = pd.read_csv(metadata_path, index_col=0)
                    
                    # Run DESeq2
                    np.random.seed(42)
                    dds = DeseqDataSet(
                        counts=pseudo_bulk_df,
                        metadata=metadata_df,
                        refit_cooks=True,
                        ref_level=["condition", "NTC"],
                        quiet=True
                    )
                    
                    dds.deseq2()
                    
                    # Get results
                    perturbation_hyphenated = perturbation.replace('_', '-')
                    res = DeseqStats(dds, contrast=['condition', perturbation_hyphenated, 'NTC'], quiet=True, alpha=alpha)
                    res.summary()
                    degs = res.results_df
                    
                    # Filter by significance, making a copy to avoid SettingWithCopyWarning
                    degs = degs[degs['padj'] < alpha]
                    print(f"Found {len(degs)} significant DEGs for {perturbation} in batch {batch}")
                    
                    
                    # Store results
                    results_dict[f"{perturbation}_{batch}"] = degs
                    
                    # Clean up memory
                    del pseudo_bulk_df
                    del metadata_df
                    del dds
                    del res
                    
                except Exception as e:
                    print(f"Error processing {perturbation} in batch {batch}: {str(e)}")
                    continue
    else:
        # No batch processing
        for perturbation in tqdm(perturbations, desc="Analyzing perturbations"):
            # Check if files exist for this perturbation
            target_dir = perturbation
            target_dir_path = os.path.join(pseudo_bulk_dir, target_dir)
            
            # Try to find files for this target_dir
            if not os.path.isdir(target_dir_path):
                print(f"Directory not found for {perturbation}, skipping")
                continue
            
            pseudo_bulk_path = os.path.join(target_dir_path, "pseudo_bulk.csv")
            metadata_path = os.path.join(target_dir_path, "metadata.csv")
            
            if not os.path.exists(pseudo_bulk_path) or not os.path.exists(metadata_path):
                print(f"Missing files for {perturbation}, skipping")
                continue
            
            try:
                # Read the files
                pseudo_bulk_df = pd.read_csv(pseudo_bulk_path, index_col=0)
                metadata_df = pd.read_csv(metadata_path, index_col=0)
                
                # Run DESeq2
                np.random.seed(42)
                dds = DeseqDataSet(
                    counts=pseudo_bulk_df,
                    metadata=metadata_df,
                    refit_cooks=True,
                    ref_level=["condition", "NTC"],
                    quiet=True
                )
                
                dds.deseq2()
                
                # Get results
                perturbation_hyphenated = perturbation.replace('_', '-')
                res = DeseqStats(dds, contrast=['condition', perturbation_hyphenated, 'NTC'], quiet=True, alpha=alpha)
                res.summary()
                degs = res.results_df
                
                # Filter by significance, making a copy to avoid SettingWithCopyWarning
                degs = degs[degs['padj'] < alpha]
                print(f"Found {len(degs)} significant DEGs for {perturbation} in batch {batch}")
                
                
                # Store results
                results_dict[perturbation] = degs
                
                # Clean up memory
                del pseudo_bulk_df
                del metadata_df
                del dds
                del res
                    
                
            except Exception as e:
                print(f"Error processing {perturbation}: {str(e)}")
                continue
    
    # Process and save results
    if save_path:
        _save_DEG_df(results_dict, p_threshold=alpha, save=True, filepath=save_path, save_all_results=save_all_results)
    
    if plot_degs:
        fig, _ = plot_filtered_genes_inverted(results_dict, p_value_threshold=alpha, ytick_step=plot_genes_interval)
        fig.savefig(fig_save_path)
    # Update adata object with DEG counts
    print("\nUpdating adata object with DEG counts")

    if f'n_DEGs_{adata_deg_test_column}' not in adata.obs.columns:
        adata.obs[f'n_DEGs_{adata_deg_test_column}'] = np.zeros(adata.n_obs)
    
    if batch_key is not None:
        for key, degs_df in results_dict.items():
            target = '_'.join(key.split('_')[:-1])
            batch = key.split('_')[-1]
            mask = (adata.obs[adata_deg_test_column] == target) & (adata.obs[batch_key] == batch)
            if mask.sum() > 0:
                adata.obs.loc[mask, f'n_DEGs_{adata_deg_test_column}'] = len(degs_df)
    else:
        for key, degs_df in results_dict.items():
            mask = adata.obs[adata_deg_test_column] == key
            adata.obs.loc[mask, f'n_DEGs_{adata_deg_test_column}'] = len(degs_df)

    logger.info(f"Completed differential expression analysis in {time.time() - start_time:.2f} seconds")
    return results_dict


def _save_DEG_df(
    results_dict, 
    p_threshold=0.05, 
    save=True, 
    filepath=None,
    save_all_results=False
):
    """
    Save and display differentially expressed genes.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of DESeq2 results
    p_threshold : float, optional
        P-value threshold, by default 0.05
    save : bool, optional
        Whether to save results, by default True
    filepath : str, optional
        Path to save results, by default None
        
    Returns
    -------
    pd.DataFrame
        Combined results DataFrame
    """
    # Process and collect all data first
    data_dict = {}
    
    # Process each gene target's results and collect data
    for gene_target, df in results_dict.items():
        if df is not None:
            if save_all_results:
                filtered_df = df.sort_values('log2FoldChange')
            else:
                filtered_df = df[df['padj'] < p_threshold].sort_values('log2FoldChange')
            data_dict[gene_target] = {
                'DEGs': filtered_df.index.tolist(),
                'L2FC': filtered_df['log2FoldChange'].tolist(),
                'Adj_P': filtered_df['padj'].tolist()
            }
    
    if not data_dict:
        logger.warning("No significant DEGs found with the current threshold.")
        return pd.DataFrame()
    
    # Sort gene targets by number of DEGs
    gene_targets = sorted(
        data_dict.keys(), 
        key=lambda gt: len(data_dict[gt]['DEGs']), 
        reverse=True
    )
    
    # Find maximum array length needed
    max_length = max(len(data_dict[gene_target]['DEGs']) for gene_target in gene_targets)
    
    # Build the final dictionary with all arrays padded to the same length
    final_data = {}
    for gene_target in gene_targets:
        for suffix in ['DEGs', 'L2FC', 'Adj_P']:
            key = f'{gene_target}_{suffix}'
            values = data_dict[gene_target][suffix].copy()
            # Pad with None to ensure all arrays have the same length
            values.extend([None] * (max_length - len(values)))
            final_data[key] = values
    
    # Create the DataFrame in a single operation to avoid fragmentation
    combined_df = pd.DataFrame(final_data)
    
    if save and filepath:
        try:
            # Write out as CSV by default, converting .xls/.xlsx to .csv
            save_path = filepath
            if filepath.lower().endswith(('.xlsx', '.xls')):
                save_path = filepath.rsplit('.', 1)[0] + '.csv'
            combined_df.to_csv(save_path, index=False)
        except Exception as e:
            logger.warning(f"Error saving results to {filepath}: {str(e)}")
    
    return combined_df

def benchmark_NTC_FDR(
    adata,
    gRNA_column="gRNA",
    ntc_cells_delimiter="Non-Targeting",
    batch_key=None,
    n_replicates=3,
    sample_fraction=0.7,
    layer="counts",
    alpha=0.05,
    fdr_column_name="exceeds_ntc_fdr",
    deg_count_column=None,
    random_seed=42,
    save_path=None,
    fig_save_path=None,
    debug=False
):
    """
    Benchmark FDR control by comparing NTC sgRNAs against each other.
    
    This function:
    1. Subsets the AnnData to only NTC cells
    2. For each NTC sgRNA, runs differential expression against all other NTC sgRNAs
    3. Calculates the distribution of DEGs per NTC sgRNA
    4. Determines the threshold at which 95% of NTC sgRNAs have fewer DEGs (FDR 0.05)
    5. Adds a column to adata.obs indicating if perturbations exceed the NTC FDR threshold
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix
    gRNA_column : str, optional
        Column containing sgRNA identifiers, by default "gRNA"
    batch_key : str, optional
        Column containing batch information, by default None.
        If provided, comparisons will be batch-aware.
    n_replicates : int, optional
        Number of pseudo-bulk replicates, by default 3
    sample_fraction : float, optional
        Fraction of cells to sample, by default 0.7
    layer : str, optional
        Layer to use for analysis, by default "counts"
    alpha : float, optional
        Significance threshold for DESeq2, by default 0.05
    fdr_column_name : str, optional
        Name for the column added to adata.obs indicating if perturbations 
        exceed the NTC FDR threshold, by default "exceeds_ntc_fdr".
        Set to None to disable column creation.
    deg_count_column : str, optional
        Column in adata.obs containing DEG counts for each perturbation, by default None.
        If None, will try to infer as 'n_DEGs_' + gRNA_column or 'n_DEGs_gene_target'
    random_seed : int, optional
        Random seed for reproducibility, by default 42
    save_path : str, optional
        Path to save results DataFrame, by default None
    fig_save_path : str, optional
        Path to save the figure, by default None
    debug : bool, optional
        Run in debug mode with only 2 NTC sgRNAs, by default False
        
    Returns
    -------
    tuple
        (deg_threshold, benchmark_results, fig)
        - deg_threshold: The number of DEGs corresponding to FDR 0.05
        - benchmark_results: DataFrame with DEG counts for each NTC sgRNA
        - fig: The matplotlib figure object
    """
    # Set seed for reproducibility
    np.random.seed(random_seed)
    
    # Check if layer exists in adata
    if layer not in adata.layers and layer is not None:
        available_layers = list(adata.layers.keys())
        logger.warning(f"Layer '{layer}' not found in adata. Available layers: {available_layers}")
        if len(available_layers) > 0:
            layer = available_layers[0]
            logger.warning(f"Using '{layer}' layer instead")
        else:
            logger.warning("No layers found, using .X matrix")
            layer = None
    
    # Get AnnData with only NTC cells based on gRNA_column and delimiter
    ntc_mask = adata.obs[gRNA_column].astype(str).str.contains(ntc_cells_delimiter)
    if not ntc_mask.any():
        raise ValueError(f"No NTC cells found in {gRNA_column} containing '{ntc_cells_delimiter}'")
    ntc_adata = adata[ntc_mask].copy()
    logger.info(f"Using {ntc_adata.n_obs} non-targeting control cells for benchmarking")
    
    # Get unique NTC sgRNAs
    if gRNA_column not in ntc_adata.obs.columns:
        raise ValueError(f"Column '{gRNA_column}' not found in NTC cells. Available columns: {list(ntc_adata.obs.columns)}")
    
    ntc_sgRNAs = list(ntc_adata.obs[gRNA_column].unique())
    logger.info(f"Found {len(ntc_sgRNAs)} unique NTC sgRNAs")
    
    if len(ntc_sgRNAs) < 3:
        logger.error(f"Not enough unique NTC sgRNAs for benchmarking (need at least 3, found {len(ntc_sgRNAs)})")
        return None, None, None
    
    # If in debug mode, limit to a few sgRNAs
    if debug:
        logger.warning(f"Running in DEBUG mode with only {min(3, len(ntc_sgRNAs))} NTC sgRNAs")
        ntc_sgRNAs = ntc_sgRNAs[:min(3, len(ntc_sgRNAs))]
    
    # Check if we're doing batch-aware processing
    has_batches = batch_key is not None and batch_key in ntc_adata.obs.columns
    
    if has_batches:
        logger.info(f"Performing batch-aware analysis using batch key: {batch_key}")
        batches = ntc_adata.obs[batch_key].unique()
        logger.info(f"Found {len(batches)} unique batches")
    
    # Store results
    results_dict = {}
    
    # Set up outer progress bar (indicate batch-aware if applicable)
    outer_desc = "Benchmarking NTC sgRNAs (batch-aware)" if has_batches else "Benchmarking NTC sgRNAs"
    # Process each NTC sgRNA separately
    for target_gRNA in tqdm(ntc_sgRNAs, desc=outer_desc):
        try:
            # Use a unique seed for each sgRNA
            target_seed = random_seed + hash(target_gRNA) % 10000
            np.random.seed(target_seed)
            
            if has_batches:
                # Process this sgRNA across all batches it appears in
                gRNA_batches = ntc_adata[ntc_adata.obs[gRNA_column] == target_gRNA].obs[batch_key].unique()
                
                # Show a nested progress bar for batches of this sgRNA
                for batch in gRNA_batches:
                    # Create a batch-specific key
                    batch_key_name = f"{target_gRNA}_{batch}"
                    
                    # Get batch-filtered data
                    batch_mask = ntc_adata.obs[batch_key] == batch
                    batch_adata = ntc_adata[batch_mask].copy()
                    
                    # Check if there are enough target cells in this batch
                    target_cells = batch_adata[batch_adata.obs[gRNA_column] == target_gRNA].obs.index
                    if len(target_cells) < 10:
                        logger.warning(f"Skipping {target_gRNA} in batch {batch}: too few cells ({len(target_cells)})")
                        continue
                    
                    # Get other NTC cells in this batch (excluding the target gRNA)
                    other_ntc_cells = batch_adata[batch_adata.obs[gRNA_column] != target_gRNA].obs.index
                    if len(other_ntc_cells) < 10:
                        logger.warning(f"Skipping {target_gRNA} in batch {batch}: too few other NTC cells ({len(other_ntc_cells)})")
                        continue
                    
                    # Trick: Temporarily create a "perturbation" column to use with generate_pseudo_bulk_replicates_for_de
                    # This labels target gRNA cells as the "perturbation" and other NTC cells as "NTC"
                    batch_adata.obs['temp_pert'] = "NTC"
                    batch_adata.obs.loc[batch_adata.obs[gRNA_column] == target_gRNA, 'temp_pert'] = target_gRNA
                    
                    try:
                        # Use the existing generate_pseudo_bulk_replicates_for_de function
                        # We pass target_gRNA as the "perturbation" and the other NTC cells as "NTC"
                        pseudo_bulk_df, metadata_df = generate_pseudo_bulk_replicates_for_de(
                            batch_adata,
                            target_gRNA,  # The "perturbation" to compare
                            other_ntc_cells,  # Other NTC cells as "control"
                            n_replicates,
                            sample_fraction,
                            layer,
                            target_column='temp_pert',  # Use our temporary column
                            random_seed=target_seed
                        )
                        
                        # Run DESeq2
                        target_hyphenated = target_gRNA.replace('_', '-')
                        dds = DeseqDataSet(
                            counts=pseudo_bulk_df,
                            metadata=metadata_df,
                            refit_cooks=True,
                            ref_level=["condition", "NTC"],
                            quiet=True
                        )
                        
                        dds.deseq2()
                        
                        # Get results
                        res = DeseqStats(dds, contrast=["condition", target_hyphenated, "NTC"], quiet=True, alpha=alpha)
                        res.summary()
                        degs = res.results_df
                        
                        # Add batch information
                        degs['batch'] = batch
                        results_dict[batch_key_name] = degs
                        
                        # Clean up memory
                        del pseudo_bulk_df, metadata_df, dds, res, batch_adata
                    
                    except Exception as e:
                        logger.error(f"Error processing {target_gRNA} in batch {batch}: {str(e)}")
                        continue
            else:
                # Process without batch awareness
                # Check if there are enough target cells
                target_cells = ntc_adata[ntc_adata.obs[gRNA_column] == target_gRNA].obs.index
                if len(target_cells) < 10:
                    logger.warning(f"Skipping {target_gRNA}: too few cells ({len(target_cells)})")
                    continue
                
                # Get other NTC cells (excluding the target gRNA)
                other_ntc_cells = ntc_adata[ntc_adata.obs[gRNA_column] != target_gRNA].obs.index
                if len(other_ntc_cells) < 10:
                    logger.warning(f"Skipping {target_gRNA}: too few other NTC cells ({len(other_ntc_cells)})")
                    continue
                
                # Trick: Temporarily create a "perturbation" column to use with generate_pseudo_bulk_replicates_for_de
                # This labels target gRNA cells as the "perturbation" and other NTC cells as "NTC"
                ntc_adata_temp = ntc_adata.copy()
                ntc_adata_temp.obs['temp_pert'] = "NTC"
                ntc_adata_temp.obs.loc[ntc_adata_temp.obs[gRNA_column] == target_gRNA, 'temp_pert'] = target_gRNA
                
                try:
                    # Use the existing generate_pseudo_bulk_replicates_for_de function
                    pseudo_bulk_df, metadata_df = generate_pseudo_bulk_replicates_for_de(
                        ntc_adata_temp,
                        target_gRNA,  # The "perturbation" to compare
                        other_ntc_cells,  # Other NTC cells as "control"
                        n_replicates,
                        sample_fraction,
                        layer,
                        target_column='temp_pert',  # Use our temporary column
                        random_seed=target_seed
                    )
                    
                    # Run DESeq2
                    target_hyphenated = target_gRNA.replace('_', '-')
                    dds = DeseqDataSet(
                        counts=pseudo_bulk_df,
                        metadata=metadata_df,
                        refit_cooks=True,
                        ref_level=["condition", "NTC"],
                        quiet=True
                    )
                    
                    dds.deseq2()
                    
                    # Get results
                    res = DeseqStats(dds, contrast=["condition", target_hyphenated, "NTC"], quiet=True, alpha=alpha)
                    res.summary()
                    degs = res.results_df
                    
                    results_dict[target_gRNA] = degs
                    
                    # Clean up memory
                    del pseudo_bulk_df, metadata_df, dds, res, ntc_adata_temp
                
                except Exception as e:
                    logger.error(f"Error processing {target_gRNA}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing {target_gRNA}: {str(e)}")
            if has_batches:
                logger.error(f"Error details: {e}")
            
        # Force garbage collection after each target
        import gc
        gc.collect()
    
    # Check if we have any successful comparisons
    if not results_dict:
        logger.error("No successful comparisons. Check if the data layer contains integer counts.")
        return None, pd.DataFrame(), None
    
    # Calculate DEGs per sgRNA-batch (handles sgRNA names with underscores)
    deg_counts = {}
    for key, result_df in results_dict.items():
        if result_df is None:
            continue
        if has_batches:
            # key == "<sgRNA>_<batch>", split on last underscore
            sgRNA, batch_str = key.rsplit('_', 1)
            count_key = f"{sgRNA}_{batch_str}"
        else:
            count_key = key
        # Count DEGs for this comparison
        deg_count = sum((result_df['padj'] < alpha) & pd.notna(result_df['padj']))
        deg_counts[count_key] = deg_count

    # Check if we have any DEG counts
    if not deg_counts:
        logger.error("No DEGs found in any comparison.")
        return 0, pd.DataFrame(columns=['n_DEGs', 'sgRNA']), None

    # Create results DataFrame
    benchmark_results = pd.DataFrame.from_dict(deg_counts, orient='index', columns=['n_DEGs'])
    # Parse batch information if available
    if has_batches:
        # split index on last underscore to recover sgRNA and batch
        benchmark_results['sgRNA'] = benchmark_results.index.map(lambda x: x.rsplit('_', 1)[0])
        benchmark_results['batch'] = benchmark_results.index.map(lambda x: x.rsplit('_', 1)[1] if '_' in x else None)
    else:
        benchmark_results['sgRNA'] = benchmark_results.index

    benchmark_results.index.name = 'key'
    benchmark_results = benchmark_results.sort_values('n_DEGs', ascending=False)
    
    # Calculate 95th percentile threshold (FDR 0.05)
    if len(benchmark_results) > 0:
        deg_threshold = np.percentile(benchmark_results['n_DEGs'].values, 95)
    else:
        logger.error("No results to calculate percentile.")
        return 0, benchmark_results, None
    
    # Store threshold in adata.uns
    adata.uns['ntc_fdr_threshold'] = int(deg_threshold)
    logger.info(f"Stored NTC FDR threshold ({int(deg_threshold)}) in adata.uns['ntc_fdr_threshold']")
    
    # Plot results with improved aesthetics
    # Set style parameters
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('white')

    # Prepare bar plot data: discrete DEG counts
    deg_counts = benchmark_results['n_DEGs'].value_counts().sort_index()

    # Create bar plot
    ax.bar(
        deg_counts.index,
        deg_counts.values,
        alpha=0.8,
        color='#4682B4',   # Steel blue
        edgecolor='white',
        linewidth=1.5
    )

    # Add threshold line with improved styling
    ax.axvline(
        deg_threshold,
        color='#B22222',  # Firebrick
        linestyle='-',
        linewidth=2,
        alpha=0.8
    )

    # Labels and title with improved styling
    batch_text = f" (batch-aware: {batch_key})" if has_batches else ""
    ax.set_xlabel('Number of DEGs', fontsize=12, labelpad=10)
    ax.set_ylabel('Number of NTC sgRNA comparisons', fontsize=12, labelpad=10)
    ax.set_title(f'Distribution of DEGs in NTC comparisons{batch_text}', fontsize=14, pad=20)

    # Turn off grid
    ax.grid(False)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, pad=5)

    # Add some padding to the axes
    ax.margins(x=0.02)

    # Print statistics instead of annotating the plot
    logger.info(f"NTC Benchmark Statistics:")
    logger.info(f"  • FDR 0.05 threshold: {int(deg_threshold)} DEGs")
    logger.info(f"  • NTC comparisons analyzed: {len(benchmark_results)}")
    logger.info(f"  • Mean DEGs per comparison: {benchmark_results['n_DEGs'].mean():.1f}")
    logger.info(f"  • Median DEGs per comparison: {benchmark_results['n_DEGs'].median():.1f}")
    logger.info(f"  • 95% of comparisons have < {int(deg_threshold)} DEGs")

    # Add custom legend-like text at bottom
    threshold_text = f"FDR 0.05 threshold: {int(deg_threshold)} DEGs"
    fig.text(0.5, 0.01, threshold_text, ha='center', fontsize=11, color='#B22222', weight='bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.show()

    
    # Save figure if path provided
    if fig_save_path:
        fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved benchmark figure to {fig_save_path}")
    
    # Save results if path provided
    if save_path:
        try:
            benchmark_results.to_csv(save_path)
            logger.info(f"Saved benchmark results to {save_path}")
        except Exception as e:
            logger.warning(f"Error saving results to {save_path}: {str(e)}")
    
    # Add column to adata.obs indicating if perturbations exceed NTC FDR threshold
    if fdr_column_name is not None:
        # Determine which DEG count column to use
        if deg_count_column is None:
            # Try to infer the column
            possible_columns = [
                'n_degs',
                f'n_DEGs_{gRNA_column}',
                'n_DEGs_gene_target',
                'n_DEGs_perturbation'
            ]
            
            for col in possible_columns:
                if col in adata.obs.columns:
                    deg_count_column = col
                    logger.info(f"Using {deg_count_column} for FDR significance")
                    break
            
            if deg_count_column is None:
                logger.warning("Could not determine DEG count column. FDR significance column not added.")
                return int(deg_threshold), benchmark_results, fig
        
        # Validate that the column exists
        if deg_count_column not in adata.obs.columns:
            logger.warning(f"Column '{deg_count_column}' not found in adata.obs. FDR significance column not added.")
            return int(deg_threshold), benchmark_results, fig
        
        # Create the new column
        adata.obs[fdr_column_name] = adata.obs[deg_count_column] > deg_threshold
        
        # Handle batch-specific thresholds if needed
        if has_batches and batch_key in adata.obs.columns:
            # Check if we have batch-specific thresholds
            batch_thresholds = {}
            
            # Group results by batch and calculate thresholds
            if 'batch' in benchmark_results.columns:
                for batch, batch_df in benchmark_results.groupby('batch'):
                    if len(batch_df) > 0:
                        batch_threshold = np.percentile(batch_df['n_DEGs'].values, 95)
                        batch_thresholds[batch] = int(batch_threshold)
                        logger.info(f"Batch {batch} FDR threshold: {int(batch_threshold)} DEGs")
            
            # If we have batch-specific thresholds, apply them
            if batch_thresholds:
                # Store in adata.uns
                adata.uns['ntc_fdr_threshold_by_batch'] = batch_thresholds
                
                # Create a temporary copy to avoid SettingWithCopyWarning
                temp_col = adata.obs[fdr_column_name].copy()
                
                # Apply batch-specific thresholds
                for batch, threshold in batch_thresholds.items():
                    batch_mask = adata.obs[batch_key] == batch
                    temp_col.loc[batch_mask] = adata.obs.loc[batch_mask, deg_count_column] > threshold
                
                # Assign back to adata.obs
                adata.obs[fdr_column_name] = temp_col
        
        logger.info(f"Added FDR significance column to adata.obs['{fdr_column_name}']")
        
        # Calculate percentage of significant perturbations
        # Use perturbed column to identify real perturbations
        if 'perturbed' in adata.obs.columns and 'perturbation' in adata.obs.columns:
            # Filter out NTC cells
            perturbed_mask = adata.obs['perturbed'] == "True"
            perturbed_obs = adata.obs[perturbed_mask]
            
            # Calculate the overall percentage of perturbed cells exceeding the NTC FDR threshold
            if len(perturbed_obs) > 0:
                sig_count = perturbed_obs.groupby('perturbation')[fdr_column_name].any().sum()
                total_count = perturbed_obs['perturbation'].nunique()
                
                sig_pct = 100 * sig_count / total_count
                logger.info(f"Perturbations exceeding NTC FDR threshold: {sig_count}/{total_count} ({sig_pct:.1f}%)")
                
                # Store in adata.uns for reference
                adata.uns['ntc_fdr_significant_targets'] = {
                    'total': total_count,
                    'significant': int(sig_count),
                    'percentage': sig_pct,
                    'cell_ids': perturbed_obs.loc[perturbed_obs[fdr_column_name]].index.tolist()
                }
    
    return int(deg_threshold), benchmark_results, fig

def differential_expression_grouped_by_batch(
    adata,
    adata_deg_test_column="gene_target",
    ntc_cells_delimiter="NTC",
    batch_key=None,
    n_replicates=3,
    sample_fraction=0.7,
    layer="counts",
    alpha=0.05,
    random_seed=42,
    save_path=None,
    plot_degs=True,
    fig_save_path=None,
    plot_genes_interval=100,
    debug=False,
    n_NTCs=165
):
    """
    Perform differential expression analysis by grouping replicates per batch.
    For each batch, generate pseudo-bulk replicates for every perturbation (including NTC)
    into a single counts and metadata matrix, then run DESeq2 and contrasts of each perturbation vs NTC.
    """
    import scipy  # ensure sparse detection
    start_time = time.time()
    results_dict = {}
    # Determine batches
    if batch_key and batch_key in adata.obs.columns:
        batches = adata.obs[batch_key].unique()
    else:
        batches = [None]
    for batch in tqdm(batches, desc="Processing batches"):
        # Subset per batch
        if batch:
            mask = adata.obs[batch_key] == batch
            adata_batch = adata[mask]
        else:
            adata_batch = adata
        # Unique perturbations including NTC
        perturbations = list(adata_batch.obs[adata_deg_test_column].unique())
        if ntc_cells_delimiter not in perturbations:
            raise ValueError(f"No {ntc_cells_delimiter} in batch {batch}")
        if debug:
            perturbations = perturbations[:4]
        # Generate pseudo-bulk for all groups
        count_list, meta_list = [], []
        rng = np.random.RandomState(random_seed)
        for pert in perturbations:
            group_mask = adata_batch.obs[adata_deg_test_column] == pert
            group_adata = adata_batch[group_mask]
            data = group_adata.layers.get(layer, group_adata.X)
            if scipy.sparse.issparse(data):
                data = data.toarray()
            n_cells = data.shape[0]
            n_sample = int(n_cells * sample_fraction)
            if pert == ntc_cells_delimiter:
                n_sample = n_NTCs
            for i in range(n_replicates):
                indices = rng.choice(n_cells, n_sample, replace=False)
                bulk = np.sum(data[indices], axis=0).astype(np.int64)
                count_list.append(bulk)
                sample_name = f"{pert.replace('_','-')}-rep{i+1}"
                meta_list.append({'sample': sample_name, 'condition': pert})
        # Build DataFrames
        count_matrix = pd.DataFrame(
            np.vstack(count_list),
            index=[m['sample'] for m in meta_list],
            columns=adata.var_names
        )
        metadata_df = pd.DataFrame(meta_list).set_index('sample')
        # Run DESeq2
        np.random.seed(random_seed)
        dds = DeseqDataSet(counts=count_matrix, metadata=metadata_df, refit_cooks=True, quiet=True)
        dds.deseq2()
        # Contrast each perturbation vs NTC
        for pert in perturbations:
            if pert == ntc_cells_delimiter:
                continue
            pert_hyph = pert.replace('_','-')
            res = DeseqStats(dds, contrast=['condition', pert_hyph, ntc_cells_delimiter], quiet=True)
            res.summary()
            degs = res.results_df[res.results_df['padj'] < alpha]
            key = f"{pert}_{batch}" if batch else pert
            results_dict[key] = degs
    # Save and plot
    if save_path:
        _save_DEG_df(results_dict, p_threshold=alpha, save=True, filepath=save_path)
    if plot_degs:
        fig, _ = plot_filtered_genes_inverted(results_dict, p_value_threshold=alpha, ytick_step=plot_genes_interval)
        fig.savefig(fig_save_path)
    # Update adata.obs
    col_name = f'n_DEGs_{adata_deg_test_column}'
    if col_name not in adata.obs.columns:
        adata.obs[col_name] = np.zeros(adata.n_obs, dtype=int)
    for key, df in results_dict.items():
        if batch_key and '_' in key and batch_key in adata.obs.columns:
            pert, batch_val = key.rsplit('_', 1)
            mask = (adata.obs[adata_deg_test_column] == pert) & (adata.obs[batch_key] == batch_val)
        else:
            pert = key
            mask = adata.obs[adata_deg_test_column] == pert
        if mask.sum() > 0:
            adata.obs.loc[mask, col_name] = len(df)
    logger.info(f"Completed grouped-by-batch DE in {time.time() - start_time:.2f} seconds")
    return results_dict

