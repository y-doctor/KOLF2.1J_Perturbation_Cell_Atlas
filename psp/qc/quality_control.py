# Data analysis
# Import plotting functions
from psp.pl.plotting import (
    plot_gRNA_distribution,
    plot_gRNA_UMI_distribution,
    plot_cells_per_guide_distribution,
    doublet_detection_sanity_check
)

import scanpy as sc
import muon as mu
import anndata as ad
import numpy as np
import pandas as pd
import psp.utils as utils
# Visualization
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import seaborn as sns
from plotnine import (
    ggplot, aes, geom_bar, ggtitle, xlab, ylab,
    scale_fill_manual, geom_histogram, labs, theme,
    element_text, scale_y_continuous
)
from IPython.display import display

# Statistics
from scipy import stats

# Clustering and dimensionality reduction
from sklearn.preprocessing import StandardScaler

# Parallel processing
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# System utilities
import os
import sys
import time

# Configure scanpy settings
sc.settings.verbosity = 4
sc.settings.set_figure_params(dpi=100, facecolor='white')

# Import plotting functions
from psp.pl.plotting import (
    plot_gRNA_distribution,
    plot_gRNA_UMI_distribution,
    plot_cells_per_guide_distribution,
    doublet_detection_sanity_check
)

from psp.utils import get_perturbed_view

# Utility Functions
def _data_statistics(adata_gex, adata_crispr, ntc_delimeter="Non-Targeting") -> None:
    """
    Calculate and print various quality control statistics from gene expression and CRISPR data.

    Parameters:
    - adata_gex: AnnData object containing RNA data.
    - adata_crispr: AnnData object containing CRISPR data.
    - ntc_delimeter: String used to identify non-targeting controls (default is "Non-Targeting").

    Prints:
    - Number of cells detected.
    - Number of unique gRNA perturbed genes.
    - Average number of gRNAs detected per gene.
    - Total GEX and CRISPR UMIs.
    - Total number of cells, genes, gRNAs, and NTC gRNAs detected.
    - Average GEX and CRISPR UMIs per cell.
    """
    # Extract unique genes from CRISPR data, excluding non-targeting controls
    genes = sorted({gene.split('_')[0] for gene in adata_crispr.var["gene_ids"]})
    if ntc_delimeter in genes:
        genes.remove(ntc_delimeter)

    # Identify non-targeting control gRNAs
    ntcs = [gRNA for gRNA in adata_crispr.var["gene_ids"] if ntc_delimeter in gRNA]

    # Calculate average gRNAs detected per gene
    avg_gRNAs_per_gene = (len(adata_crispr.var["gene_ids"]) - len(ntcs)) / len(genes)

    # Print statistics
    print(f"Number of cells detected: {len(adata_crispr.obs)}")
    print(f"Unique gRNA perturbed genes detected: {len(genes)}")
    print(f"Average gRNA detected per gene: {avg_gRNAs_per_gene:.2f}")  # TODO: Investigate this, it seems to be off by 1
    print(f"Total GEX UMIs: {adata_gex.X.sum():,}")
    print(f"Total CRISPR UMIs: {adata_crispr.X.sum():,}")
    print(f"Total cells detected: {len(adata_crispr.obs)}")
    print(f"Total genes detected: {len(adata_gex.var)}")
    print(f"Total gRNAs detected: {len(adata_crispr.var)}")
    print(f"Total NTC gRNAs detected: {len(ntcs)}")
    print(f"Average GEX UMIs per cell: {adata_gex.X.sum() / len(adata_gex.obs):,.2f}")
    print(f"Average CRISPR UMIs per cell: {adata_crispr.X.sum() / len(adata_crispr.obs):,.2f}")


def _make_counts_layer(adata: ad.AnnData) -> None:
    """
    Create a 'counts' layer in the AnnData object by copying the data from adata.X.

    Parameters:
    adata (AnnData): The AnnData object to be modified.

    Returns:
    None
    """
    adata.layers["counts"] = adata.X.copy()


def _assign_batch_id(adata: ad.AnnData) -> None:
    """
    Assign a 'batch' ID to the AnnData object by extracting the first part of each label in adata.obs.channel.

    Parameters:
    adata (AnnData): The AnnData object to be modified.

    Returns:
    None
    """
    adata.obs["batch"] = [label.split('-')[0] for label in adata.obs.channel]


def __gene_ids_to_ensg(filepath: str) -> dict:
        """
        Reads a file and returns a dictionary mapping gene names to Ensembl IDs.

        Parameters:
        - filepath: Path to the file containing gene-to-Ensembl ID mappings.

        Returns:
        - A dictionary with gene names as keys and Ensembl IDs as values.
        """
        ensembl_ids = {}
        with open(filepath, 'r') as file:
            for line in file:
                gene, ensg = line.strip().split('\t')
                ensembl_ids[gene] = ensg
        return ensembl_ids


def __remove_invalid_gene_targets(
    adata: ad.AnnData, 
    obs_key: str, 
    var_key: str, 
    whitelist: list[str] | set[str] = ["NTC"]
) -> ad.AnnData:
    """
    Removes cells from the AnnData object where the gene target in `obs` is not present in `var`,
    except for those in a provided whitelist.

    Parameters:
    - adata (anndata.AnnData): The AnnData object to be modified.
    - obs_key (str): The key in `adata.obs` corresponding to gene target identifiers.
    - var_key (str): The key in `adata.var` corresponding to gene IDs in ENSEMBL format.
    - whitelist (list or set of str): Gene targets that are allowed even if they are not in `var` (default is ["NTC"]).

    Returns:
    - anndata.AnnData: The modified AnnData object with invalid gene targets removed.
    """
    # Ensure obs_key exists in .obs and var_key exists in .var
    if obs_key not in adata.obs.columns:
        raise ValueError(f"{obs_key} not found in .obs.")
    if var_key not in adata.var.columns:
        raise ValueError(f"{var_key} not found in .var.")

    # Create a set of valid gene IDs from .var
    valid_gene_ids = set(adata.var[var_key].values)

    # If a whitelist is provided, add it to the valid_gene_ids set
    if not isinstance(whitelist, (list, set)):
        raise ValueError("Whitelist must be a list or set of gene targets.")
    valid_gene_ids.update(whitelist)

    # Identify invalid gene targets that are not in the valid_gene_ids set
    invalid_mask = ~adata.obs[obs_key].isin(valid_gene_ids)
    invalid_genes = adata.obs.loc[invalid_mask, obs_key].unique()

    # Print a warning if there are any invalid gene targets
    if invalid_genes.size > 0:
        print(f"Warning: The following gene targets were not found in var[{var_key}] and are not in the whitelist. They will be dropped:")
        for gene in invalid_genes:
            print(f"  - {gene}")

    # Drop cells with invalid gene targets
    return adata[~invalid_mask].copy()


# Data Processing Functions
def read_in_10x_mtx(mtx_dir, save_filepath, ntc_delimeter="Non-Targeting") -> ad.AnnData:
    """
    Reads 10x Genomics matrix output, processes gene expression and CRISPR data, 
    and saves the combined data to a specified file path.

    Parameters:
    - mtx_dir: str, directory containing the 10x Genomics matrix files.
    - save_filepath: str, path where the processed data will be saved.
    - ntc_delimeter: str, identifier for non-targeting controls (default is "Non-Targeting").

    Returns:
    - adata_gex: AnnData object containing RNA data.
    """
    # Read the 10x matrix data
    mdata_combined = sc.read_10x_mtx(mtx_dir, gex_only=False, cache=True)
    
    # Separate gene expression and CRISPR data
    adata_gex = mdata_combined[:, mdata_combined.var['feature_types'] == 'Gene Expression'].copy()
    adata_crispr = mdata_combined[:, mdata_combined.var['feature_types'] == 'CRISPR Guide Capture'].copy()
    
    # Reindex CRISPR data to ensure unique identifiers
    reindex = ['-'.join(index.split('-')[:-1]) + "_" + str(index.split('-')[-1]) for index in adata_crispr.var.index]
    adata_crispr.var.index = reindex
    adata_crispr.var.gene_ids = reindex
    
    # Combine RNA and CRISPR data into a MuData object
    mdata = mu.MuData({"rna": adata_gex, "crispr": adata_crispr})
    
    # Save the combined data
    mdata.write(save_filepath)
    
    # Extract the RNA and CRISPR data from the MuData object
    adata_gex = mdata.mod["rna"]
    adata_crispr = mdata.mod["crispr"]
    
    # Calculate and print data statistics
    _data_statistics(adata_gex, adata_crispr, ntc_delimeter=ntc_delimeter)
    
    return adata_gex


def assign_protospacers(adata, protospacer_calls_file_path, NTC_Delimiter="Non-Targeting") -> ad.AnnData:
    """
    Assigns protospacers to the AnnData object based on the provided protospacer calls file.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - protospacer_calls_file_path: Path to the CSV file containing protospacer calls (output from cellranger_count).
    - NTC_Delimiter: String used to identify non-targeting controls within the protospacer calls file (default is "Non-Targeting").

    Returns:
    - Updated AnnData object with protospacer assignments.
    """
    # Read protospacer calls from the CSV file
    protospacer_calls = pd.read_csv(protospacer_calls_file_path, index_col='cell_barcode')
    
    # Create a mask for cells present in the protospacer calls
    mask = adata.obs_names.isin(protospacer_calls.index)
    
    # Initialize gRNA-related columns in the AnnData object
    adata.obs["gRNA"] = "None"
    adata.obs["n_gRNA"] = 0
    adata.obs["n_gRNA_UMIs"] = "-1"
    
    # Assign gRNA values to cells based on the protospacer calls
    adata.obs.loc[mask, "gRNA"] = protospacer_calls.loc[adata.obs_names[mask], "feature_call"]
    
    # Reindex gRNA values for consistency
    adata.obs.gRNA = ['-'.join(guide.split('-')[:-1]) + "_" + str(guide.split('-')[-1]) if '|' not in guide else guide for guide in adata.obs.gRNA]
    
    # Assign gene targets based on gRNA values
    adata.obs["gene_target"] = [guide.split('_')[0] if '|' not in guide else guide for guide in adata.obs.gRNA]
    
    # Replace non-targeting controls with "NTC"
    adata.obs["gene_target"] = ["NTC" if guide == NTC_Delimiter else guide for guide in adata.obs.gene_target]
    
    # Assign the number of gRNAs and UMIs to cells
    adata.obs.loc[mask, "n_gRNA"] = protospacer_calls.loc[adata.obs_names[mask], "num_features"]
    adata.obs.loc[mask, "n_gRNA_UMIs"] = protospacer_calls.loc[adata.obs_names[mask], "num_umis"]
    
    # Plot distributions for quality control
    plot_gRNA_distribution(adata)
    plot_gRNA_UMI_distribution(adata)
    plot_cells_per_guide_distribution(adata)
    
    return adata


def assign_gene_ids(adata: ad.AnnData, gene_id_filepath: str, obs_key: str = "gene_target_ensembl_id", var_key: str = "gene_ids", ntc_label: str = "NTC") -> None:
    """
    Assigns gene IDs to the AnnData object based on a provided file mapping gene names to Ensembl IDs.

    Parameters:
    - adata: AnnData object to be modified.
    - gene_id_filepath: Path to the file containing gene-to-Ensembl ID mappings.
    - obs_key: The key in .obs where the gene_target_ensembl_id will be stored (default is "gene_target_ensembl_id").
    - var_key: The key in .var corresponding to gene_ids in ENSEMBL format (default is "gene_ids").
    - ntc_label: Label for non-targeting controls (default is "NTC").

    Returns:
    - None
    """
    

    # Map gene names to Ensembl IDs
    gene_to_ensembl_ids = __gene_ids_to_ensg(filepath=gene_id_filepath)
    gene_to_ensembl_ids[ntc_label] = ntc_label  # Ensure NTC is included in the mapping

    # Assign Ensembl IDs to the AnnData object
    adata.obs[obs_key] = [gene_to_ensembl_ids.get(gene, ntc_label) for gene in adata.obs["gene_target"]]

    # Remove invalid gene targets
    adata = __remove_invalid_gene_targets(adata, obs_key=obs_key, var_key=var_key)

    return adata


def assign_metadata(
    adata: ad.AnnData, 
    cell_type: str, 
    perturbation_type: str, 
    subset_to_1_gRNA: bool = True, 
    channel_dict: dict = None, 
    treatment_dict: dict = None,
    gene_id_filepath: str = None,
    obs_key: str = "gene_target_ensembl_id",
    var_key: str = "gene_ids",
    ntc_label: str = "NTC",
    ntc_sgRNA_prefix: str = "Non-Targeting"
) -> ad.AnnData:
    """
    Assigns metadata to the AnnData object and assigns gene IDs.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - cell_type: String representing the cell type to assign.
    - perturbation_type: String representing the perturbation type to assign.
    - subset_to_1_gRNA: Boolean indicating whether to subset the data to cells with exactly one gRNA (default is True).
    - channel_dict: Optional dictionary mapping cell barcodes to channels.
    - treatment_dict: Optional dictionary mapping cell barcodes to treatments.
    - gene_id_filepath: Path to the file containing gene-to-Ensembl ID mappings.
    - obs_key: The key in .obs where the gene_target_ensembl_id will be stored (default is "gene_target_ensembl_id").
    - var_key: The key in .var corresponding to gene_ids in ENSEMBL format (default is "gene_ids").
    - ntc_label: Label for non-targeting controls (default is "NTC").
    - ntc_sgRNA_prefix: Prefix for non-targeting controls in the sgRNA column (default is "Non-Targeting").

    Returns:
    - Updated AnnData object with assigned metadata and gene IDs.
    """
    # Assign basic metadata
    adata.obs["celltype"] = cell_type
    adata.obs["perturbation_type"] = perturbation_type

    # Calculate and assign UMI counts and gene counts
    adata.obs["n_UMI_counts"] = adata.X.sum(axis=1)
    adata.obs["n_genes"] = (adata.X != 0).sum(axis=1).A1
    adata.var["n_UMI_counts"] = adata.X.T.sum(axis=1)
    adata.var["n_cells"] = (adata.X.T != 0).sum(axis=1).A1

    # Subset to cells with exactly one gRNA if specified
    if subset_to_1_gRNA:
        adata = adata[adata.obs.n_gRNA == 1, :]
        adata.obs["perturbed"] = ["True" if gene_target != "NTC" else "False" for gene_target in adata.obs["gene_target"]]

    # Assign channel information if provided
    if channel_dict is not None:
        adata.obs["channel"] = [channel_dict[cell.split('-')[1]] for cell in adata.obs.index]
        _assign_batch_id(adata)
    
    # Assign a perturbation column
    adata.obs['perturbation'] = [gRNA if ntc_sgRNA_prefix not in gRNA else "NTC" for gRNA in adata.obs['gRNA']]

    # Assign treatment information if provided
    if treatment_dict is not None:
        adata.obs["treatment"] = [treatment_dict[cell.split('-')[1]] for cell in adata.obs.index]

    # Assign gene IDs
    adata = assign_gene_ids(adata, gene_id_filepath, obs_key, var_key, ntc_label)

    # Make counts layer
    _make_counts_layer(adata)

    return adata


# Quality Control Functions
def general_qc(adata) -> ad.AnnData:
    """
    Perform general quality control on the AnnData object.

    This function generates plots to visualize the distribution of gene expression
    and filters out genes that are not detected in any cells.

    Parameters:
    - adata: AnnData object containing single-cell data.

    Returns:
    - qc_adata: AnnData object after filtering out genes not detected in any cells.
    """
    # Create a copy of the input AnnData object for quality control
    qc_adata = adata.copy()

    # Plot the highest expressed genes
    sc.pl.highest_expr_genes(qc_adata, n_top=20)

    # Create subplots for visualizing gene and cell distributions
    fig, ax = plt.subplots(2, 1, figsize=(6, 10))

    # Plot the number of genes detected per cell
    ax[0].plot(sorted(adata.obs["n_genes"], reverse=True), '.')
    ax[0].set_xlabel("Cell Number")
    ax[0].set_ylabel("Number of Genes Detected")

    # Plot the log1p number of cells in which each gene is detected
    ax[1].plot(np.log1p(sorted(qc_adata.var["n_cells"], reverse=True)), '.')
    ax[1].set_xlabel("Gene Number")
    ax[1].set_ylabel("Log1p Number of Cells Detected")

    # Display the plots
    plt.show()

    # Print the number of genes not detected in any cells and filter them out
    num_genes_removed = sum(qc_adata.var['n_cells'] == 0)
    print(f"Removing {num_genes_removed} genes not detected in any cells.")
    qc_adata = qc_adata[:, qc_adata.var["n_cells"] > 0]

    return qc_adata


def dead_cell_qc(adata: ad.AnnData, count_MADs: int = 5, mt_MADs: int = 3, ribo_MADs: int = 5) -> ad.AnnData:
    """
    Perform quality control to identify and filter out dead cells based on mitochondrial, ribosomal, and hemoglobin gene metrics.

    This function calculates quality control metrics for mitochondrial, ribosomal, and hemoglobin genes, identifies outliers
    based on median absolute deviation (MAD), and filters out these outliers.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - count_MADs: Number of MADs to use for filtering based on total counts and number of genes by counts. Default is 5.
    - mt_MADs: Number of MADs to use for filtering based on mitochondrial gene percentage. Default is 3.
    - ribo_MADs: Number of MADs to use for filtering based on ribosomal gene percentage. Default is 5.

    Returns:
    - adata: AnnData object after filtering out dead cells.
    """
    # Identify mitochondrial, ribosomal, and hemoglobin genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var['hb'] = adata.var_names.str.contains("^HB[^(P)]")

    # Calculate quality control metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo', 'hb'], percent_top=[20], log1p=True, inplace=True)

    def is_outlier(adata: ad.AnnData, metric: str, nmads: int) -> np.ndarray:
        """Identify outliers based on the specified metric and number of MADs."""
        M = adata.obs[metric]
        return (M < np.median(M) - nmads * stats.median_abs_deviation(M)) | (
            M > np.median(M) + nmads * stats.median_abs_deviation(M)
        )

    # Determine outliers based on the specified metrics and MADs
    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", count_MADs)
        | is_outlier(adata, "log1p_n_genes_by_counts", count_MADs)
        | is_outlier(adata, "pct_counts_in_top_20_genes", count_MADs)
        | is_outlier(adata, "pct_counts_mt", mt_MADs)
        | is_outlier(adata, "pct_counts_ribo", ribo_MADs)
    )

    # Display outlier counts
    display(adata.obs.outlier.value_counts()) 

    # Plot data before filtering
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", title="Prior to Filtering")
    sc.pl.violin(adata, "pct_counts_mt")

    # Filter out outliers
    adata = adata[~adata.obs.outlier]

    # Plot data after filtering
    sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", title="Post Filtering")
    sc.pl.violin(adata, "pct_counts_mt")

    return adata

def remove_batch_duplicates(adata, batch_key = 'batch', perturbation_key = 'perturbation', gene_target_key = 'gene_target'):
    """
    Remove perturbations present in multiple batches.

    This function identifies perturbations that are present in multiple batches and removes them.
    It uses the `batch_key` to identify batches and the `perturbation_key` to identify perturbations.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - batch_key: The key in .obs where the batch information is stored.
    - perturbation_key: The key in .obs where the perturbation information is stored.
    - gene_target_key: The key in .obs where the gene_target information is stored.
    Returns:
    - adata: AnnData object after removing perturbations present in multiple batches.
    """
    # Identify perturbations present in multiple batches
    perturbed_view = get_perturbed_view(adata)
    batch_counts = perturbed_view.obs.groupby(gene_target_key)[batch_key].nunique()
    multiple_batches = batch_counts[batch_counts > 1].index.tolist()
    print(f"Number of genes present in multiple batches: {len(multiple_batches)}")
    print(f"({multiple_batches})")
    print(f"Selecting the batch with the most sgRNAs per gene")
    # Subset to gene targets that appear in multiple batches
    multi_view = perturbed_view[perturbed_view.obs[gene_target_key].isin(multiple_batches)]

    # Count unique sgRNAs per gene target and batch
    sgRNA_counts = (
        multi_view.obs
        .groupby([gene_target_key, batch_key])[perturbation_key]
        .nunique()
        .reset_index(name="n_sgRNAs")
    )

    # For each gene target, pick the batch with the most sgRNAs (ties broken arbitrarily by idxmax)
    best = sgRNA_counts.loc[sgRNA_counts.groupby(gene_target_key)["n_sgRNAs"].idxmax()]
    best_batch_per_gene = dict(zip(best[gene_target_key], best[batch_key]))

    # Keep only cells from the selected best batch for each gene target
    mask = multi_view.obs[batch_key] == multi_view.obs[gene_target_key].map(best_batch_per_gene)
    cells_to_remove = multi_view[~mask].obs.index
    adata = adata[~adata.obs.index.isin(cells_to_remove)].copy()
    return adata


def default_qc(input_dict: dict) -> ad.AnnData:
    """
    Performs the default quality control pipeline.
    
    Parameters:
    ----------
    input_dict : dict
        Dictionary containing the following required keys:
        - 'mtx_dir': Path to the filtered_matrix_mex directory
        - 'save_directory': Path where to save the .h5mu file after reading in the 10x matrix
        - 'protospacer_calls_file': Path to protospacer_calls_per_cell.csv
        - 'aggregation_csv': Path to aggregation_csv file
        - 'cell_type': String describing the cell type
        - 'perturbation_type': String describing the perturbation type
        - 'pre_qc_save_path': Path where to save the data after sgRNA assignment but before QC filtering
        - 'final_save_path': Path where to save the final QC-filtered data
        - 'gene_id_filepath': Path to the file containing gene-to-Ensembl ID mappings
        - 'obs_key': The key in .obs where the gene_target_ensembl_id will be stored
        - 'var_key': The key in .var corresponding to gene_ids in ENSEMBL format
        - 'ntc_label': Label for non-targeting control cells in the gene_id_filepath
        - 'ntc_sgRNA_prefix': Prefix for non-targeting controls in the sgRNA column (default is "Non-Targeting")
        - 'remove_batch_duplicates': Boolean indicating whether to remove perturbations across multiple batches (default is False)
        
    Optional parameters in input_dict:
        - 'mt_MADs': Number of MADs for mitochondrial filtering (default: 5)
        - 'count_MADs': Number of MADs for count filtering (default: 5)
        - 'ribo_MADs': Number of MADs for ribosomal filtering (default: 5)
        - 'treatment_dict': Dictionary mapping cell barcodes to treatments
        
    Returns:
    -------
    ad.AnnData
        Processed and quality-controlled AnnData object
    """
    # Validate required inputs
    required_keys = [
        'mtx_dir', 'save_directory', 'protospacer_calls_file',
        'aggregation_csv', 'cell_type', 'perturbation_type',
        'pre_qc_save_path', 'final_save_path',
        'gene_id_filepath', 'obs_key', 'var_key', 'ntc_label'
    ]
    
    for key in required_keys:
            assert key in input_dict, f"Missing required key in input_dict: {key}"
            assert isinstance(input_dict[key], str) and input_dict[key], f"{key} must be a non-empty string."

    # Validate file paths
    assert os.path.exists(input_dict['mtx_dir']), f"mtx_dir does not exist: {input_dict['mtx_dir']}"
    assert os.path.exists(input_dict['protospacer_calls_file']), f"protospacer_calls_file does not exist: {input_dict['protospacer_calls_file']}"
    assert os.path.exists(input_dict['aggregation_csv']), f"aggregation_csv does not exist: {input_dict['aggregation_csv']}"
    assert os.path.exists(input_dict['gene_id_filepath']), f"gene_id_filepath does not exist: {input_dict['gene_id_filepath']}"
    
    # Read in the 10x matrix data
    print("Reading 10x matrix data...")
    adata = read_in_10x_mtx(
        input_dict['mtx_dir'],
        input_dict['save_directory']
    )
    
    # Assign protospacers
    print("Assigning protospacers...")
    adata = assign_protospacers(
        adata,
        protospacer_calls_file_path=input_dict['protospacer_calls_file'],
        NTC_Delimiter=input_dict['ntc_sgRNA_prefix']
    )
    
    # Create channel dictionary from aggregation CSV
    print("Creating channel dictionary...")
    d = pd.read_csv(input_dict['aggregation_csv'])
    channel_dict = {str(i+1): channel for i, channel in enumerate(d["sample_id"])}
    
    # Get optional treatment dictionary
    treatment_dict = input_dict.get('treatment_dict', None)
    
    # Assign metadata
    print("Assigning metadata...")
    adata = assign_metadata(
        adata=adata,
        cell_type=input_dict['cell_type'],
        perturbation_type=input_dict['perturbation_type'],
        channel_dict=channel_dict,
        treatment_dict=treatment_dict,
        gene_id_filepath=input_dict['gene_id_filepath'],
        obs_key=input_dict['obs_key'],
        var_key=input_dict['var_key'],
        ntc_label=input_dict['ntc_label'],
        ntc_sgRNA_prefix=input_dict['ntc_sgRNA_prefix']
    )
    
    # Convert n_gRNA_UMIs to string
    adata.obs['n_gRNA_UMIs'] = adata.obs['n_gRNA_UMIs'].astype(str)

    if 'remove_batch_duplicates' in input_dict:
        if input_dict['remove_batch_duplicates']:
            print("Removing perturbations present in multiple batches...")
            adata = remove_batch_duplicates(adata)
    
    # Save data after sgRNA assignment but before QC
    print(f"Saving pre-QC data to {input_dict['pre_qc_save_path']}...")
    adata.write(input_dict['pre_qc_save_path'])
    
    # Perform general QC
    print("Performing general QC...")
    adata = general_qc(adata)
    
    # Perform dead cell QC
    print("Performing dead cell QC...")
    mt_MADs = input_dict.get('mt_MADs', 5)
    count_MADs = input_dict.get('count_MADs', 5)
    ribo_MADs = input_dict.get('ribo_MADs', 5)
    
    adata = dead_cell_qc(
        adata,
        mt_MADs=mt_MADs,
        count_MADs=count_MADs,
        ribo_MADs=ribo_MADs
    )
    
    # Perform doublet detection sanity check
    print("Performing doublet detection sanity check...")
    doublet_detection_sanity_check(adata)
    
    # Save final file
    print(f"Saving final QC file to {input_dict['final_save_path']}...")
    adata.write(input_dict['final_save_path'])
    
    return adata



   
