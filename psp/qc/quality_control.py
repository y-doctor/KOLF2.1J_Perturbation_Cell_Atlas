# Data analysis
import scanpy as sc
import muon as mu
import anndata as ad
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import (
    ggplot, aes, geom_bar, ggtitle, xlab, ylab,
    scale_fill_manual, geom_histogram, labs, theme,
    element_text, scale_y_continuous
)

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


def _read_gtf(gtf_path: str) -> pd.DataFrame:
    """
    Reads a GTF file and extracts gene information.

    Parameters:
    - gtf_path (str): Path to the GTF file.

    Returns:
    - pd.DataFrame: A DataFrame containing gene information with columns 'seqname', 'gene_name', 'gene_type', and 'gene_id'.
    """
    # Read the GTF file into a DataFrame
    gtf = pd.read_table(
        gtf_path,
        comment="#",
        sep="\t",
        names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    )
    
    # Filter for gene features and select relevant columns
    genes = gtf[gtf.feature == "gene"][['seqname', 'attribute']].copy().reset_index(drop=True)
    
    def _gene_info(attribute: str) -> tuple:
        """
        Extracts gene name, type, and ID from the attribute string.

        Parameters:
        - attribute (str): The attribute string from the GTF file.

        Returns:
        - tuple: A tuple containing gene name, gene type, and gene ID.
        """
        g_name = next(filter(lambda x: 'gene_name' in x, attribute.split(";"))).split(" ")[2].strip('"')
        g_type = next(filter(lambda x: 'gene_type' in x, attribute.split(";"))).split(" ")[2].strip('"')
        g_id = next(filter(lambda x: 'gene_id' in x, attribute.split(";"))).split(" ")[1].strip('"')
        return g_name, g_type, g_id

    # Apply the gene_info function to extract gene details
    genes["gene_name"], genes["gene_type"], genes["gene_id"] = zip(*genes.attribute.apply(_gene_info))
    
    return genes


def _get_ntc_view(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Returns a view of the AnnData object with non-perturbed cells.

    Parameters:
    - adata (anndata.AnnData): The input AnnData object.

    Returns:
    - anndata.AnnData: A view of the AnnData object with non-perturbed cells.
    """
    return adata[adata.obs.perturbed == "False"]


def _get_perturbed_view(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Returns a view of the AnnData object with perturbed cells.

    Parameters:
    - adata (anndata.AnnData): The input AnnData object.

    Returns:
    - anndata.AnnData: A view of the AnnData object with perturbed cells.
    """
    return adata[adata.obs.perturbed == "True"]


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


def assign_metadata(adata, cell_type, perturbation_type, subset_to_1_gRNA=True, channel_dict=None, treatment_dict=None) -> ad.AnnData:
    """
    Assigns metadata to the AnnData object.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - cell_type: String representing the cell type to assign.
    - perturbation_type: String representing the perturbation type to assign.
    - subset_to_1_gRNA: Boolean indicating whether to subset the data to cells with exactly one gRNA (default is True).
    - channel_dict: Optional dictionary mapping cell barcodes to channels (e.g. the original channel they originated from in the 10x Genomics chip).
    - treatment_dict: Optional dictionary mapping cell barcodes to treatments.

    Returns:
    - Updated AnnData object with assigned metadata.
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

    # Assign treatment information if provided
    if treatment_dict is not None:
        adata.obs["treatment"] = [treatment_dict[cell.split('-')[1]] for cell in adata.obs.index]

    return adata


def identify_coding_genes(adata: anndata.AnnData, gtf_path: str, subset_to_coding_genes: bool = False) -> anndata.AnnData:
    """
    Identifies and optionally subsets coding genes in the AnnData object.

    Parameters:
    - adata (anndata.AnnData): The input AnnData object.
    - gtf_path (str): Path to the GTF file. Default is set to a specific path.
    - subset_to_coding_genes (bool): If True, subset the AnnData object to only include protein-coding genes. Default is False.

    Returns:
    - anndata.AnnData: The AnnData object with gene type information added and optionally subsetted to coding genes.
    """
    gtf = _read_gtf(gtf_path)
    name_to_type = {k: v for k, v in zip(gtf['gene_id'], gtf['gene_type'])}
    adata.var["gene_type"] = [name_to_type[idx] for idx in adata.var.gene_ids]
    
    if subset_to_coding_genes:
        adata = adata[:, adata.var.gene_type == "protein_coding"].copy()
    
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
        return (M < np.median(M) - nmads * sp.stats.median_abs_deviation(M)) | (
            M > np.median(M) + nmads * sp.stats.median_abs_deviation(M)
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


def doublet_detection_sanity_check(adata: ad.AnnData) -> None:
    """
    Perform a sanity check for doublet detection by plotting histograms of key metrics.

    Parameters:
    - adata: AnnData object containing single-cell data.

    Returns:
    - None
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(adata.obs["total_counts"], bins=100, ax=axes[0])
    axes[0].set_title('Total Counts')

    sns.histplot(adata.obs["n_genes_by_counts"], bins=100, ax=axes[1])
    axes[1].set_title('Genes by Counts')

    sns.histplot(adata.obs["n_genes"], bins=100, ax=axes[2])
    axes[2].set_title('Number of Genes')

    plt.tight_layout()
    plt.show()


# Visualization Functions
def plot_gRNA_distribution(adata: ad.AnnData) -> None:
    """
    Plots the distribution of the number of gRNAs per cell and calculates statistics 
    related to sgRNA calls and multiplet rates.

    Parameters:
    - adata: AnnData object containing the single-cell data with 'n_gRNA' in the observations.

    This function creates a bar plot showing the distribution of gRNAs per cell, 
    categorizing cells with more than 6 gRNAs as '>6'. It also prints the percentage 
    of cells without confident sgRNA calls and the estimated multiplet rate.
    """
    # Create a DataFrame for the number of gRNAs per cell
    df = pd.DataFrame(adata.obs['n_gRNA'], columns=['n_gRNA'])

    # Categorize cells with more than 6 gRNAs as '>6'
    df['n_gRNA'] = df['n_gRNA'].apply(lambda x: str(x) if x <= 6 else '>6')

    # Define a colormap for the categories
    categories = df['n_gRNA'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = dict(zip(categories, colors[:len(categories)]))

    # Create and display the bar plot
    plot = (ggplot(df, aes(x='n_gRNA', fill='n_gRNA')) +
            geom_bar() +
            scale_fill_manual(values=color_map) +
            ggtitle('Number of gRNA Assigned') +
            xlab('n_gRNA') +
            ylab('Count'))
    plot.show()

    # Calculate and print statistics
    value_counts = df['n_gRNA'].value_counts()
    no_guide = value_counts.get('0', 0)
    one_guide = value_counts.get('1', 0)
    total_guides = sum(value_counts)
    no_call_rate = (no_guide / total_guides) * 100
    multiplet_rate = ((total_guides - (no_guide + one_guide)) / total_guides) * 100

    print(f"Cells without confident sgRNA calls: {no_call_rate:.2f}%")
    print(f"Estimated Multiplet Rate: {multiplet_rate:.2f}%")


def plot_gRNA_UMI_distribution(adata: ad.AnnData) -> None:
    """
    Plots the distribution of gRNA UMI counts for cells with exactly one sgRNA assigned.

    Parameters:
    - adata: AnnData object containing the single-cell data with 'n_gRNA' and 'n_gRNA_UMIs' in the observations.

    This function creates a histogram showing the distribution of gRNA UMI counts for cells that have exactly one sgRNA assigned.
    """
    # Filter the AnnData object to include only cells with exactly one gRNA
    adata_1_gRNA = adata[adata.obs['n_gRNA'] == 1, :]

    # Create a DataFrame for the gRNA UMI counts
    df = pd.DataFrame(adata_1_gRNA.obs['n_gRNA_UMIs'].astype(int), columns=['n_gRNA_UMIs'])

    # Create and display the histogram
    plot = (ggplot(df, aes(x='n_gRNA_UMIs')) +
            geom_histogram(bins=50, fill='#AEC6CF') +
            labs(title='gRNA UMI Counts For Cells with 1 sgRNA Assigned', x='gRNA UMI Counts', y='Counts'))
    plot.show()


def plot_cells_per_guide_distribution(adata: ad.AnnData) -> None:
    """
    Plots the distribution of cells per guide RNA (gRNA) for cells with exactly one gRNA assigned.

    Parameters:
    - adata: AnnData object containing single-cell data with 'n_gRNA' and 'gene_target' in the observations.

    This function creates a bar plot showing the number of cells per perturbation target, excluding the 'NTC' (Non-Targeting Control) category.
    It also prints the number and percentage of perturbations with at least 50 cells assigned to a single guide.
    """
    # Filter the AnnData object to include only cells with exactly one gRNA
    adata_1_gRNA = adata[adata.obs['n_gRNA'] == 1, :]

    # Count the occurrences of each gene target
    gRNA_counts = adata_1_gRNA.obs['gene_target'].value_counts()

    # Convert the counts to a DataFrame for plotting
    gRNA_counts_df = gRNA_counts.reset_index()
    gRNA_counts_df.columns = ['gene_target', 'count']

    # Extract and remove the count for the 'NTC' category
    ntc_count = gRNA_counts_df.loc[gRNA_counts_df['gene_target'] == 'NTC', 'count'].values[0]
    gRNA_counts_df = gRNA_counts_df[gRNA_counts_df['gene_target'] != 'NTC']

    # Plot the data using plotnine
    plot = (ggplot(gRNA_counts_df, aes(x='reorder(gene_target, -count)', y='count')) +
            geom_bar(stat='identity', fill='#FFD1DC') +
            theme(axis_text_x=element_text(rotation=90, hjust=1), figure_size=(20, 6)) +
            scale_y_continuous(breaks=range(0, max(gRNA_counts_df['count']) + 50, 50)) +
            ggtitle(f'Number of Cells per Single Perturbation Target; {ntc_count} NTC Cells') +
            xlab('Perturbation Target') +
            ylab('Number of Cells'))
    plot.show()

    # Calculate and print statistics about perturbations
    total_perts = len(gRNA_counts_df)
    over_50 = len(gRNA_counts_df[gRNA_counts_df['count'] >= 50])
    print(f"Number of perturbations with >= 50 cells with single guide assigned: {over_50}/{total_perts} ({100 * (over_50 / total_perts):.2f}%)")