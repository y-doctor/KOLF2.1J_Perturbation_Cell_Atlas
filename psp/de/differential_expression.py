import psp.utils as utils
import psp.pl as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from tqdm.contrib.concurrent import process_map
from typing import Tuple, Dict, List
import warnings
import anndata as ad


def _generate_pseudo_bulk_replicates_for_de(
    adata, gene_target: str, ntc_cells: np.ndarray, n_replicates: int = 3, 
    sample_fraction: float = 0.85, layer: str = None, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate pseudo-bulk replicates for a given gene target and matched NTC cells.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - gene_target: The gene target for which to generate replicates.
    - ntc_cells: Indices of NTC cells.
    - n_replicates: Number of replicates to generate.
    - sample_fraction: Fraction of cells to sample for each pseudo-bulk pseudoreplicate.
    - layer: Layer in AnnData to use for data matrix.
    - seed: Seed for random number generator.

    Returns:
    - Tuple of DataFrames: (pseudo_bulk_df, metadata_df)
    """
    data_matrix = adata.layers[layer] if layer else adata.X
    target_indices = np.where(adata.obs['gene_target'] == gene_target)[0]
    n_target = len(target_indices)
    target_bulk = []
    ntc_bulk = []
    sample_names = []
    control_names = []
    np.random.seed(seed)

    for i in range(n_replicates):
        # Sample from gene_target
        sampled_target_indices = np.random.choice(target_indices, int(n_target * sample_fraction), replace=False)
        target_profile = data_matrix[sampled_target_indices].sum(axis=0)
        target_bulk.append(target_profile)
        sample_names.append(f"{gene_target}_rep_{i+1}")

        # Sample from NTC
        sampled_ntc_indices = np.random.choice(ntc_cells, int(n_target * sample_fraction), replace=False) #Sample the same number of NTC cells as the number of perturbed cells
        ntc_profile = data_matrix[sampled_ntc_indices].sum(axis=0)
        ntc_bulk.append(ntc_profile)
        control_names.append(f"NTC_rep_{i+1}")
        
    
    # Convert to DataFrame
    sample_names.extend(control_names)
    metadata_records = [{'condition': sample.split('_')[0]} for sample in sample_names]
    pseudo_bulk_df = pd.DataFrame(np.vstack(target_bulk + ntc_bulk), index=sample_names, columns=adata.var_names)
    pseudo_bulk_df = pseudo_bulk_df[pseudo_bulk_df.columns[pseudo_bulk_df.sum(axis=0)>=1]] #Remove any samples with 0s in both NTC and Perturbed Sample
    metadata_df = pd.DataFrame(metadata_records, index=sample_names)
    
    return pseudo_bulk_df, metadata_df


def deseq2(data: pd.DataFrame, metadata: pd.DataFrame, contrast: List[str], alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform differential expression analysis using DESeq2.

    Parameters:
    - data: DataFrame of counts.
    - metadata: DataFrame of metadata.
    - contrast: List specifying the contrast for DESeq2.
    - alpha: Significance level for adjusted p-values.

    Returns:
    - DataFrame of DESeq2 results.
    """
    dds = DeseqDataSet(
        counts=data,
        metadata=metadata,
        design_factors="condition",
        refit_cooks=True,
        ref_level=["condition", "NTC"],
        quiet = True
    )

    #Suppress warnings about dispersion trend curve fitting
    warnings.filterwarnings("ignore", category=UserWarning, message="The dispersion trend curve fitting did not converge.")
    #Run DESeq2
    dds.deseq2()
    #Run DESeq2 stats
    stat_res = DeseqStats(dds, contrast=contrast, quiet=True, alpha=alpha)
    stat_res.summary()
    results = stat_res.results_df
    
    return results

def save_DEG_df(
    results_dict: Dict[str, pd.DataFrame], 
    p_threshold: float = 0.05, 
    save: bool = True, 
    filepath: str = None, 
    preserve_batch_info: bool = False,
    adata: ad.AnnData = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save and display differentially expressed genes (DEGs) with batch handling.

    Parameters:
        results_dict: Dictionary of DESeq2 results (keys may contain batch names)
        p_threshold: Adjusted p-value threshold for filtering
        save: Whether to save results to file
        filepath: Output path for Excel file
        preserve_batch_info: Maintain batch identifiers in column names if False
        adata: AnnData object to store results in .uns['DEG_summary']

    Returns:
        Tuple of (combined_df - DataFrame of DEG results, summary_df - DataFrame of summary statistics (DEGs per perturbation))
    """
    final_dict = {}

    for full_key, df in results_dict.items():
        # Extract base gene target name if needed
        if preserve_batch_info:
            gene_target = full_key
        else:
            gene_target = full_key.split('_')[0]  # Simple split - modify if using different delimiter

        # Filter and sort
        filtered_df = df[df['padj'] < p_threshold]
        l2fc_sorted = filtered_df.sort_values('log2FoldChange')
        
        # Create or append to existing entries
        if f'{gene_target}_DEGs' in final_dict:
            # Handle multiple entries by extending lists #TODO: This is a hack to handle multiple entries for the same gene target, really we need to handle this better via sets or something.
            final_dict[f'{gene_target}_DEGs'].extend(l2fc_sorted.index.tolist())
            final_dict[f'{gene_target}_L2FC'].extend(l2fc_sorted['log2FoldChange'].tolist())
            final_dict[f'{gene_target}_Adj_P'].extend(l2fc_sorted['padj'].tolist())
        else:
            # Create new entries
            final_dict[f'{gene_target}_DEGs'] = l2fc_sorted.index.tolist()
            final_dict[f'{gene_target}_L2FC'] = l2fc_sorted['log2FoldChange'].tolist()
            final_dict[f'{gene_target}_Adj_P'] = l2fc_sorted['padj'].tolist()

    # Sort gene targets by number of DEGs
    sorted_targets = sorted(
        [k.replace('_DEGs', '') for k in final_dict if k.endswith('_DEGs')],
        key=lambda x: len(final_dict[f'{x}_DEGs']), 
        reverse=True
    )

    # Create ordered dictionary and pad lists
    ordered_dict = {}
    max_length = 0
    for target in sorted_targets:
        for suffix in ['_DEGs', '_L2FC', '_Adj_P']:
            key = target + suffix
            ordered_dict[key] = final_dict[key]
            max_length = max(max_length, len(final_dict[key]))

    # Pad all lists to equal length
    for key in ordered_dict:
        ordered_dict[key] += [None] * (max_length - len(ordered_dict[key]))

    # Create and save DataFrame
    combined_df = pd.DataFrame(ordered_dict)
    
    # Create summary statistics
    summary_data = []
    for target in sorted_targets:
        # Split target into perturbation and batch components
        if '_' in target:
            perturbation, batch = target.split('_', 1)  # Split on first underscore only
        else:
            perturbation, batch = target, "N/A"
        
        # Get all DEGs and their L2FC values
        degs = final_dict[f'{target}_DEGs']
        l2fcs = final_dict[f'{target}_L2FC']
        
        # Calculate counts
        valid_degs = [deg for deg in degs if deg is not None]
        total_degs = len(valid_degs)
        upregulated = sum(1 for fc in l2fcs if fc is not None and fc > 0)
        downregulated = sum(1 for fc in l2fcs if fc is not None and fc < 0)
        
        summary_data.append({
            'Perturbation': perturbation,
            'Batch': batch,
            'Total_DEGs': total_degs,
            'Total_Upregulated_DEGs': upregulated,
            'Total_Downregulated_DEGs': downregulated
        })
    summary_df = pd.DataFrame(summary_data)
    del summary_data
    
    if save and filepath:
        with pd.ExcelWriter(filepath) as writer:
            combined_df.to_excel(writer, sheet_name='DEG Results', index=False)
            summary_df.to_excel(writer, sheet_name='DEGs Per Perturbation', index=False)
    
    # Store in AnnData if provided
    if adata is not None:
        adata.uns["DEGs_for_each_perturbation"] = combined_df
        adata.uns['Number_of_DEGs_per_perturbation'] = summary_df
    
    return combined_df, summary_df


def run_deseq2_analysis(
    adata: ad.AnnData,
    gene_target_obs_column: str = "gene_target",
    ntc_cells_delimiter: str = "NTC",
    n_replicates: int = 3,
    sample_fraction: float = 0.7,
    layer: str = None,
    alpha: float = 0.05,
    n_jobs: int = -1,
    seed: int = 42,
    batch_key: str = "batch",
    save: bool = False,
    save_filepath: str = None,
    p_threshold: float = 0.05,
    preserve_batch_info: bool = None,
) -> Dict[str, pd.DataFrame]:
    """
    Perform batch-aware differential expression analysis using DESeq2 with pseudo-bulk replicates.
    
    Generates pseudo-bulk samples for each gene target and matched NTC controls, then performs
    differential expression analysis either across all cells or within specified batches.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing single-cell data with gene expression counts
    gene_target_obs_column : str, optional
        Column in adata.obs containing gene target identifiers, by default "gene_target"
    ntc_cells_delimiter : str, optional
        Identifier for non-targeting control cells, by default "NTC"
    n_replicates : int, optional
        Number of pseudo-bulk replicates to generate per condition, by default 3
    sample_fraction : float, optional
        Fraction of cells to sample for each pseudo-bulk replicate, by default 0.7
    layer : str, optional
        Layer in AnnData containing count data to use, by default None (uses .X)
    alpha : float, optional
        Significance threshold for adjusted p-values, by default 0.05
    n_jobs : int, optional
        Number of parallel jobs for processing, by default -1 (all available cores)
    seed : int, optional
        Random seed for reproducibility, by default 42
    batch_key : str, optional
        Column in adata.obs specifying batches for batch-aware analysis, by default "batch"
    save : bool, optional
        Whether to save results using save_DEG_df, by default False
    save_filepath : str, optional
        File path for saving results (required if save=True), by default None
    p_threshold : float, optional
        Adjusted p-value threshold for DEG filtering during saving, by default 0.05
    preserve_batch_info : bool, optional
        Whether to maintain batch identifiers in output columns, by default None
        (auto-detected based on batch_key presence if None)

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of DESeq2 results where keys are either:
        - {gene_target} (when no batch_key)
        - {gene_target}_{batch} (when batch_key provided)

    Examples
    --------
    >>> de_results = run_deseq2_analysis(adata, batch_key="batch")
    >>> de_results = run_deseq2_analysis(adata, save=True, save_filepath="results.xlsx")
    """
    ntc_cells = np.where(adata.obs[gene_target_obs_column] == ntc_cells_delimiter)[0]
    gene_targets = list(adata.obs[gene_target_obs_column].unique())
    gene_targets.remove(ntc_cells_delimiter)

    def process_gene_target(gene_target: str) -> pd.DataFrame:
        pseudo_bulk_df, metadata_df = _generate_pseudo_bulk_replicates_for_de(
            adata, gene_target, ntc_cells, 
            n_replicates=n_replicates, sample_fraction=sample_fraction, 
            layer=layer, seed=seed
        )
        return deseq2(pseudo_bulk_df, metadata_df, 
                    contrast=["condition", gene_target, ntc_cells_delimiter], 
                    alpha=alpha)

    if batch_key:
        # Batch-aware processing - run analysis per batch and concatenate
        batches = utils.split_by_batch(adata, batch_key=batch_key)
        combined_results = {}
        
        for batch_name, batch_data in batches.items():
            try:
                # Get results for this batch
                batch_ntc = np.where(batch_data.obs[gene_target_obs_column] == ntc_cells_delimiter)[0]
                if len(batch_ntc) == 0:
                    print(f"Skipping batch {batch_name} - no NTC cells")
                    continue
                
                batch_targets = [gt for gt in gene_targets if gt in batch_data.obs[gene_target_obs_column].values]
                
                # Process each target in this batch
                batch_results = dict(zip(
                    batch_targets,
                    process_map(
                        process_gene_target, 
                        batch_targets,
                        n_jobs=n_jobs,
                        desc=f"DE analysis - {batch_name}"
                    )
                ))
                
                # Add batch identifier to keys
                combined_results.update({
                    f"{gene_target}_{batch_name}": df
                    for gene_target, df in batch_results.items()
                })
                
            except Exception as e:
                print(f"Error processing batch {batch_name}: {str(e)}")
                continue
                
        final_results = combined_results
    else:
        # Original single-batch processing
        results = process_map(
            process_gene_target, 
            gene_targets, 
            n_jobs=n_jobs,
            desc="Running DE analysis", 
            total=len(gene_targets)
        )
        final_results = dict(zip(gene_targets, results))

    # Handle saving and AnnData storage
    if preserve_batch_info is None:
        preserve_batch_info = bool(batch_key)
    
    # Always store results in AnnData, optionally save to file
    save_DEG_df(
        final_results,
        p_threshold=p_threshold,
        save=save,
        filepath=save_filepath,
        preserve_batch_info=preserve_batch_info,
        adata=adata
    )

    fig = pl.plot_number_of_DEGs(adata)

    return final_results, fig

