# Data analysis
import scanpy as sc
import muon as mu
import anndata as ad
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotnine import (
    ggplot, aes, geom_bar, ggtitle, xlab, ylab,
    scale_fill_manual, geom_histogram, labs, theme,
    element_text, scale_y_continuous
)

# Statistics and machine learning
from scipy import stats, spatial
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding

# Quality control
import scrublet as scr

# Clustering and dimensionality reduction
import igraph as ig
import leidenalg
import pymde
import networkx as nx
from pynndescent import NNDescent

# Differential expression
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# Parallel processing
from joblib import Parallel, delayed, parallel_backend
import concurrent.futures
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from numba import jit

# System utilities
import os
import sys
import time
import requests

# Configure scanpy settings
sc.settings.verbosity = 4  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=100, facecolor='white')


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


def dead_cell_qc(adata, count_MADs = 5, mt_MADs = 3, ribo_MADs = 5):
    # mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-') 
    # ribosomal genes
    adata.var['ribo'] = adata.var_names.str.startswith(("RPS","RPL"))
    # hemoglobin genes
    adata.var['hb'] = adata.var_names.str.contains(("^HB[^(P)]"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=[20], log1p=True, inplace=True) #calculates quality control metrics
    def is_outlier(adata, metric: str, nmads: int):
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * sp.stats.median_abs_deviation(M)) | (
            np.median(M) + nmads * sp.stats.median_abs_deviation(M) < M
        )
        return outlier

    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", count_MADs)
        | is_outlier(adata, "log1p_n_genes_by_counts", count_MADs)
        | is_outlier(adata, "pct_counts_in_top_20_genes", count_MADs)
        | is_outlier(adata, "pct_counts_mt", mt_MADs)
        | is_outlier(adata, "pct_counts_ribo", ribo_MADs)
    )
    display(adata.obs.outlier.value_counts()) #type: ignore
    p11 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", title="Prior to Filtering")
    p12 = sc.pl.violin(adata, "pct_counts_mt")
    adata = adata[~adata.obs.outlier]
    p21 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", title="Post Filtering")
    p22 = sc.pl.violin(adata, "pct_counts_mt")
    return adata


def doublet_detection_sanity_check(adata):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(adata.obs["total_counts"], bins=100, ax=axes[0])
    axes[0].set_title('Total Counts')

    sns.histplot(adata.obs["n_genes_by_counts"], bins=100, ax=axes[1])
    axes[1].set_title('Genes by Counts')

    sns.histplot(adata.obs["n_genes"], bins=100, ax=axes[2])
    axes[2].set_title('Number of Genes')

    plt.tight_layout()
    plt.show()


def read_gtf(gtf_path):
    gtf = pd.read_table(gtf_path,comment="#", sep = "\t", names = ['seqname', 'source', 'feature', 'start' , 'end', 'score', 'strand', 'frame', 'attribute'])
    genes = gtf[(gtf.feature == "gene")][['seqname', 'attribute']].copy().reset_index().drop('index', axis=1)
    def gene_info(x):
        g_name = list(filter(lambda x: 'gene_name' in x,  x.split(";")))[0].split(" ")[2].strip('"')
        g_type = list(filter(lambda x: 'gene_type' in x,  x.split(";")))[0].split(" ")[2].strip('"')
        g_id = list(filter(lambda x: 'gene_id' in x,  x.split(";")))[0].split(" ")[1].strip('"')
        return (g_name, g_type, g_id)

    genes["gene_name"], genes["gene_type"], genes["gene_id"] = zip(*genes.attribute.apply(lambda x: gene_info(x)))
    return genes


def identify_coding_genes(adata, gtf_path="/tscc/projects/ps-malilab/utils/cellranger/refdata-gex-GRCh38-2024-A/genes/genes.gtf", subset_to_coding_genes = False):
    gtf = read_gtf(gtf_path)
    name_to_type = {k:v for k,v in zip(gtf['gene_id'],gtf['gene_type'])}
    adata.var["gene_type"] = [name_to_type[idx] for idx in adata.var.gene_ids]
    if subset_to_coding_genes:
        adata = adata[:,adata.var.gene_type == "protein_coding"].copy()
    return adata


def normalize_NTCs(adata, control_treatment=None):
    adata_ntc = adata[adata.obs.perturbed == "False",:].copy()
    if control_treatment is not None:
        adata_ntc = adata_ntc[adata_ntc.obs.treatment==control_treatment,:].copy()
    value_counts = adata_ntc.obs.gRNA.value_counts()
    value_counts = value_counts[value_counts >= 25] #only sgRNA with atleast 20 cells
    adata_ntc = adata_ntc[adata_ntc.obs.gRNA.isin(list(value_counts.index)),:].copy()
    previous_mean_umi_counts = np.mean(adata_ntc.X.sum(axis=0))
    scaling_ratio = 1e6/previous_mean_umi_counts
    sc.pp.normalize_total(adata_ntc, target_sum=1e6) #count-per-million normalization
    means = np.asarray(adata_ntc.X.T.mean(axis=1)).flatten()
    n,bins,_ = plt.hist(means,bins=200)
    plt.loglog(base=2)
    plt.xlabel("Number of UMIs")
    plt.ylabel("Number of Genes");
    total_genes = adata_ntc.X.T.shape[0]
    umi_thresh = bins[2]
    num_genes_filtered = sum(n[0:2])
    plt.axvline(umi_thresh, color="red")
    plt.axvline(scaling_ratio,color="blue")
    print(f"Based on the plot, we will set the UMI threshold for considering a gene to {umi_thresh:.2f} (red line) which corresponds to {total_genes - num_genes_filtered} remaining genes.")
    print(f"Using the Weissman threshold (blue line) of 1 UMI ({scaling_ratio:.2f} scaled UMIs) corresponds to {total_genes - sum(means < scaling_ratio)} remaining genes.")
    return adata_ntc, umi_thresh


def filter_genes_by_UMI_thresh_and_scale(adata, threshold):
    """
    Filter an AnnData object to only include variables (genes) with a mean count greater than the threshold.
    
    Parameters:
    adata (anndata.AnnData): The input AnnData object.
    threshold (float): The threshold for the mean count.
    
    Returns:
    anndata.AnnData: The filtered AnnData object.
    """
    # Calculate the mean count for each gene
    mean_counts = np.mean(adata.X, axis=0).A1  # .A1 to convert to 1D array if sparse matrix
    
    # Filter the genes based on the threshold
    genes_to_keep = mean_counts > threshold
    
    # Subset the AnnData object to keep only the selected genes
    filtered_adata = adata[:, genes_to_keep].copy()

    #Scale the NTCs
    sc.pp.scale(filtered_adata)
    
    return filtered_adata
    
def downsample_ntc(adata,num_sgRNA):
    ntc_adata = adata[adata.obs.perturbed == "False", :]
    ntc_guides = ntc_adata.obs.gRNA.unique()
    choice = np.random.choice(ntc_guides,size=num_sgRNA,replace=False)
    return ntc_adata[ntc_adata.obs.gRNA.isin(choice),:].copy()


def ks_cpu_axis(data1, data2, axis=0, alternative='two-sided'):
    res = ks_2samp(data1, data2, axis=axis, alternative=alternative, method="asymp")
    return res.statistic, res.pvalue

def process_pair(i, j, data, axis, alternative):
    data1 = data[data.obs.gRNA == i].X.toarray()
    data2 = data[data.obs.gRNA == j].X.toarray()
    u, p = ks_cpu_axis(data1, data2, axis, alternative)
    return [i, j, u, p]

def pairwise_ks_test_cpu(data, axis=0, alternative='two-sided'):
    gRNA_unique = list(data.obs.gRNA.unique())
    combos = [combo for combo in combinations(gRNA_unique, 2)]
    results = Parallel(n_jobs=-1)(delayed(process_pair)(i, j, data, axis, alternative) for i, j in tqdm(combos))
    
    # Extract and flatten p-values for FDR correction
    p_values_adjusted = np.array([multipletests(np.array(result[3]),method='fdr_bh')[1] for result in results])
    
    # Append corrected p-values to results
    for idx, result in enumerate(results):
        result.append(p_values_adjusted[idx])
    
    return results

def create_significant_matrix(results, gRNA_unique, alpha=0.05):
    n = len(gRNA_unique)
    significant_matrix = np.zeros((n, n), dtype=int)
    gRNA_to_index = {gRNA: idx for idx, gRNA in enumerate(gRNA_unique)}
    
    for result in results:
        i, j, _, _, p_adj = result
        significant_count = np.sum(np.array(p_adj) < alpha)
        idx_i = gRNA_to_index[i]
        idx_j = gRNA_to_index[j]
        significant_matrix[idx_i, idx_j] = significant_count
        significant_matrix[idx_j, idx_i] = significant_count  # Ensure symmetry
    
    return significant_matrix

# Function to process AnnData input
def pairwise_NTC_DEGs(adata, axis=0, alternative='two-sided', alpha=0.05):
    gRNA_unique = list(adata.obs.gRNA.unique())
    results = pairwise_ks_test_cpu(adata, axis=axis, alternative=alternative)
    
    # Create a matrix of significant comparisons
    significant_matrix = create_significant_matrix(results, gRNA_unique, alpha=alpha)
    sns.clustermap(significant_matrix)
    return significant_matrix, results


def find_and_plot_largest_clique(matrix, max_diff, plot=False):
    n = len(matrix)
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(n))
    
    # Add edges where differences are less than or equal to max_diff
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] <= max_diff:
                G.add_edge(i, j)

    # Finding all maximal cliques
    all_cliques = list(nx.find_cliques_recursive(G))

    # Filter cliques to only include those that meet the minimum size requirement
    cliques = [clique for clique in all_cliques if len(clique)]
    if not cliques:  # check if list is empty
        print("No cliques meet the minimum size requirement.")
        return []

    # Find the largest clique from the filtered list
    largest_clique = max(cliques, key=len)

    if plot:
        # Plotting the graph using a circular layout
        pos = nx.spring_layout(G)  # positions for all nodes using circular layout
        plt.figure(figsize=(12, 12))

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')

        # edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3)

        # labels - optional due to size
        # nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif")

        # highlight the largest clique with a different color
        nx.draw_networkx_nodes(G, pos, nodelist=largest_clique, node_color='red', node_size=50)

        plt.title("Graph of Elements with Max Difference " + str(max_diff))
        plt.axis("off")
        plt.show()

    return list(largest_clique)

def filter_to_core_NTCs(adata, sig_mat, max_num_shared_degs, control_treatment = None, plot = False):
    ntc_set_indices = find_and_plot_largest_clique(sig_mat, max_diff=max_num_shared_degs, plot=plot)
    core_NTCs = None
    if control_treatment is not None:
        adata_condition = adata[adata.obs.treatment==control_treatment,:]
        adata_condition_ntc = adata_condition[adata_condition.obs.perturbed=="False",:]
        core_NTCs = np.array(adata_condition_ntc.obs.gRNA.unique())[ntc_set_indices]
        mask_perturbed = adata.obs['perturbed'] == "True"
        mask_core_ntcs = (adata.obs['perturbed'] == "False") & (adata.obs['gRNA'].isin(core_NTCs))
        combined_mask = mask_perturbed | mask_core_ntcs
        return adata[combined_mask,:].copy()
    else:
        core_NTCs = np.array(adata[adata.obs.perturbed=="False",:].obs.gRNA.unique())[ntc_set_indices]
        mask_perturbed = adata.obs['perturbed'] == "True"
        mask_core_ntcs = (adata.obs['perturbed'] == "False") & (adata.obs['gRNA'].isin(core_NTCs))
        combined_mask = mask_perturbed | mask_core_ntcs
        return adata[combined_mask,:].copy()


def subsample_NTC_cells(adata, num_NTC_cells=1000):
    # Subset the AnnData object to get only the non-perturbed cells
    adata_ntc = adata[adata.obs['perturbed'] == "False"]

    # If the number of non-perturbed cells is less than the desired number, use all available
    if adata_ntc.n_obs < num_NTC_cells:
        num_NTC_cells = adata_ntc.n_obs

    # Randomly choose num_NTC_cells from the non-perturbed cells
    random_indices = np.random.choice(adata_ntc.obs.index, size=num_NTC_cells, replace=False)

    # Subset the AnnData object to get only the perturbed cells
    adata_perturbed = adata[adata.obs['perturbed'] == "True"]

    # Create a boolean index for selecting the desired cells
    selected_indices = adata.obs.index.isin(random_indices) | (adata.obs['perturbed'] == "True")

    # Subset the AnnData object based on the boolean index
    adata_subsampled = adata[selected_indices,:].copy()

    return adata_subsampled

def print_gene_percentages(adata, threshold):
    print(f"Using a total UMI threshold of: {threshold}")
    for i in range(10):
        print(f"{100*i/10}% = {int(len(adata.var)*i/10)} HVGs")

def plot_UMI_threshold(adata, nbins=100, threshold=40, x_min = 0, x_max=2000):
    total_UMIs = adata.X.sum(axis=0).A1
    umi_ct = [umi_ct for umi_ct in total_UMIs if x_min < umi_ct < x_max]
    plt.hist(umi_ct, bins=nbins)
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1)
    plt.xlabel("Total UMI Count")
    plt.ylabel("Number of Genes")
    plt.show()
    adata_high_UMIs = adata[:,adata.X.sum(axis=0).A1>=threshold]
    print_gene_percentages(adata_high_UMIs, threshold)


def HVGs(adata, HVG_UMI_threshold, percent=30):
    percent = percent/100
    adata_high_UMIs = adata[:,adata.X.sum(axis=0).A1>=HVG_UMI_threshold].copy()
    sc.pp.highly_variable_genes(adata_high_UMIs, n_top_genes=int(len(adata_high_UMIs.var)*percent),flavor='seurat_v3')
    hvg = adata_high_UMIs[:,adata_high_UMIs.var['highly_variable']==True].var.index
    adata.var["highly_variable"] = [True if gene in hvg else False for gene in adata.var.index]
    return adata

def scale_to_core_ntcs(adata):
    ntc_adata_median_counts = np.median(adata[adata.obs.perturbed=="False"].X.sum(axis=1), axis=0).item()
    sc.pp.normalize_total(adata,target_sum=ntc_adata_median_counts)
    return adata

def remove_perturbations_by_cell_threshold(adata, cell_threshold=25):
    perturbation_counts = dict(adata.obs.gene_target.value_counts())
    remaining_perturbations = [k for k,v in perturbation_counts.items() if v >= cell_threshold]
    print(f"Removing {len(perturbation_counts) - len(remaining_perturbations)} perturbations for having under {cell_threshold} cells.")
    return adata[adata.obs.gene_target.isin(remaining_perturbations),:].copy()

def gene_ids_to_ensg(filepath):
    ensembl_ids = {}
    with open(filepath, 'r') as file:
        for line in file:
            gene, ensg = line.strip().split('\t')
            ensembl_ids[gene] = ensg
    return ensembl_ids

def calculate_guide_repression(adata, pert, mu_X_control, ensembl_ids_to_gene):
    pert_idx = None
    try:
        pert_idx = list(adata.var["gene_ids"]).index(pert)
    except ValueError:
        print(f"Gene {ensembl_ids_to_gene[pert]}({pert}) not found in the RNA data.")
        return {}
    control_perturbation_target_expression = mu_X_control[pert_idx]
    mean_repressions = {}
    for guide in adata[adata.obs["gene_target_ensembl_id"] == pert].obs["gRNA"].unique():
        guide_mask = adata.obs.gRNA == guide
        X_guide = adata.X[guide_mask]
        guide_perturbation = X_guide.mean(axis=0).A1[pert_idx]
        if control_perturbation_target_expression == 0.0:
            #i.e. if there is literally no expression of the target gene in NTCs
            mean_repressions[guide] = 0.0
        else:
            percent_repression = 100*((guide_perturbation-control_perturbation_target_expression)/control_perturbation_target_expression)
            if percent_repression > 100:
                percent_repression = 100 #clipping for plotting purposes, these will be filtered out anyways
            mean_repressions[guide] = percent_repression
    return mean_repressions

def remove_poor_sgRNA_repression_threshold(adata,gene_id_file_path,repression_threshold = 30):
    gene_to_ensembl_ids = gene_ids_to_ensg(filepath=gene_id_file_path)
    gene_to_ensembl_ids["NTC"] = "NTC" #incase NTC was not included in the file path
    ensembl_ids_to_gene = {v: k for k, v in gene_to_ensembl_ids.items()}
    adata.obs["gene_target_ensembl_id"] = [gene_to_ensembl_ids[gene] for gene in adata.obs["gene_target"]]
    control_mask = adata.obs["perturbed"] == "False"
    X_control = adata.X[control_mask]
    mu_X_control = X_control.mean(axis=0).A1
    perturbed_genes = list(adata.obs["gene_target_ensembl_id"].unique())
    perturbed_genes.remove("NTC")
    results = Parallel(n_jobs=-1)(delayed(calculate_guide_repression)(adata, pert, mu_X_control, ensembl_ids_to_gene) for pert in tqdm(perturbed_genes,"Processing at per-sgRNA Level"))
    mean_repressions = {k: v for res in results for k, v in res.items()}
    for guide in mean_repressions:
        guide_mask = adata.obs.gRNA == guide
        adata.obs.loc[guide_mask,"mean_repressions"] = mean_repressions[guide]
    perturbed_to_remove = adata[adata.obs["mean_repressions"] > -repression_threshold,:].obs.index
    fig, ax = plt.subplots(1,1,figsize=(8,12))
    mean_repressions_array = np.array(sorted(list(mean_repressions.values())))
    print(f"{sum(mean_repressions_array <= -repression_threshold)}/{len(mean_repressions_array)} ({100*(sum(mean_repressions_array <= -repression_threshold)/len(mean_repressions_array)):.2f}%) sgRNA achieve a mean repression of atleast {repression_threshold}%")
    adata_filtered = adata[~adata.obs.index.isin(perturbed_to_remove),:].copy()
    print(f"This corresponds to {len(adata_filtered.obs.gene_target.value_counts())} remaining perturbations.")
    ax.plot(mean_repressions_array,'.')
    ax.set_xlabel("guide")
    ax.set_ylabel("Percent Perturbed Target Expression Relative to NTC")
    ax.axhline(-30,color = 'red')
    ax.axhline(0,color = 'k',linestyle='--')
    return adata_filtered

def relative_z_normalization(adata):
    ntc_adata = adata[adata.obs.perturbed == "False", :]
    
    # Convert sparse matrix to dense if necessary
    X_dense = ntc_adata.X.toarray() if hasattr(ntc_adata.X, 'toarray') else ntc_adata.X
    
    # Compute mean and standard deviation
    mu_ntc = X_dense.mean(axis=0)
    std_ntc = X_dense.std(axis=0)
    
    # Handle division by zero by replacing zeros with ones
    std_ntc[std_ntc == 0] = 1
    
    # Normalize the entire dataset using the computed mean and std from control cells
    X_normalized = (adata.X - mu_ntc) / std_ntc
    
    # Replace the data in adata with the normalized data
    adata.layers["pre_z_normalization"] = adata.X
    adata.X = X_normalized
    return adata


def set_perturbation_target_to_zero(adata):
    zero_pert = adata.copy()
    perturbations = zero_pert.obs.gene_target_ensembl_id.unique()
    perturbations = perturbations[perturbations != "NTC"]  # Remove "NTC"
    
    for perturbation in perturbations:
        if perturbation in zero_pert.var.index:
            indices = zero_pert.obs.gene_target_ensembl_id == perturbation
            gene_index = zero_pert.var.index.get_loc(perturbation)
            zero_pert.X[indices, gene_index] = 0.0
    
    return zero_pert


def mannwhitneyu_cpu_axis(data1, data2, axis=0, alternative='two-sided'):
    # Perform Mann-Whitney U Test
    u_stat, p_value = mannwhitneyu(data1, data2, axis=axis, alternative=alternative,method='asymptotic')

    # Calculate the maximum possible U value
    n1 = data1.shape[axis]
    n2 = data2.shape[axis]
    u_max = n1 * n2

    # Modify the U-statistic based on its comparison with n1 * n2 / 2
    u_stat_modified = np.where(u_stat < (u_max / 2), -1, 1)

    return u_stat_modified, p_value


def ks_2_samp(data1, data2, axis=0, alternative='two-sided'):
    res = ks_2samp(data1, data2, axis=0, alternative=alternative, method='asymp')
    return res

from contextlib import contextmanager
import warnings
import logging



@contextmanager
def suppress_stdout_stderr():
    """Suppresses stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run_deseq2(data1, data2):
    # Combine data1 and data2 into a single DataFrame
    combined_data = np.concatenate([data1, data2], axis=0)
    condition = ['treated'] * data1.shape[0] + ['control'] * data2.shape[0]

    # Create a DataFrame for DESeq2 input
    counts_df = pd.DataFrame(combined_data, columns=[f'gene_{i}' for i in range(combined_data.shape[1])]).astype(int)
    metadata_df = pd.DataFrame({'condition': condition})

    with suppress_stdout_stderr():
        # Prepare the DESeq2 dataset
        dds = DeseqDataSet(counts=counts_df, metadata = metadata_df, design_factors='condition')
        dds.deseq2()

        # Run DESeq2 statistics
        results = DeseqStats(dds, alpha=0.05)
        results.summary()
        res_df = results.results_df

    # Extract p-values and log2 fold changes
    p_values = res_df['pvalue'].values
    adj_p_values = res_df['padj'].values
    log2_fold_changes = res_df['log2FoldChange'].values

    return log2_fold_changes, p_values, adj_p_values

def ks_2_samp_bh_direct(data1, data2, axis=0, alternative='two-sided'):
    res = ks_2samp(data1, data2, axis=0, alternative=alternative, method='asymp')
    # Check where values are greater than zero
    # Check where values are greater than zero
    gt_zero_data1 = data1 > 0
    gt_zero_data2 = data2 > 0

    # Count the number of values greater than zero along the specified axis
    count_gt_zero_data1 = np.sum(gt_zero_data1, axis=axis)
    count_gt_zero_data2 = np.sum(gt_zero_data2, axis=axis)

    # Ensure the shapes are compatible for broadcasting
    if count_gt_zero_data1.shape != count_gt_zero_data2.shape:
        raise ValueError("Shapes of count_gt_zero_data1 and count_gt_zero_data2 are not compatible for broadcasting")

    # Find indices where at most one value in data1 and data2 is greater than zero
    condition = (count_gt_zero_data1 > 0) & (count_gt_zero_data2 > 0)
    p_values = np.array(res.pvalue)
    adj_p = np.ones_like(p_values)
    testable_indices = np.where(condition)[0]
    adj_p[testable_indices] = multipletests(p_values[testable_indices], method='fdr_bh')[1]
    return res.statistic_sign, p_values, adj_p


def process_perturbation(perturbation, control, adata, alternative, test, expr_type, subsample_control):
    data1 = adata[adata.obs.gene_target == perturbation].X.toarray()
    p = None
    stat = None
    if subsample_control:
        n_control = data1.shape[0]
        control = control[np.random.choice(control.shape[0],n_control,replace=False),:]
    if test == "MWU":
        stat, p = mannwhitneyu_cpu_axis(data1, control, axis=0, alternative=alternative)
    elif test == "KS":
        res = ks_2_samp(data1, control, axis=0, alternative=alternative)
        p = res.pvalue
        stat = [-x for x in res.statistic_sign] #see the ks_2samp documentation -- this is for the underlying distribution
    elif test == "KS_BH_direct":
        stat, p, adj_p = ks_2_samp_bh_direct(data1, control, axis=0, alternative=alternative)
        stat = [-x for x in stat]
        return[perturbation, stat, p, adj_p, stat]
    elif test == "DESeq2":
        stat, p, adj_p = run_deseq2(data1, control)
        return[perturbation, stat, p, adj_p, stat]
    else:
         print("Test not known. Defaulting to MWU.")
         _, p = mannwhitneyu_cpu_axis(data1, control, axis=0, alternative=alternative)
    p_values_adjusted = multipletests(p, method='fdr_bh',alpha=0.3)[1]
    avg_z = []
    if expr_type != "z-normalized":
        avg_z = np.mean(data1, axis=0) - np.mean(control, axis=0)
    else:
        avg_z = np.mean(data1, axis=0)
    return [perturbation, stat, p, p_values_adjusted, avg_z]


def extract_perturbation_DEs_vs_control(adata, test, alternative="two-sided", expr_type="z-normalized", subsample_control = False, control_treatment=None):
    perturbations = list(adata.obs.gene_target.unique())
    control_matrix = None
    if control_treatment is None:
        perturbations.remove("NTC")
        control_matrix = adata[adata.obs.perturbed == "False", :].X.toarray()
    else:
        perturbations.remove(control_treatment)
        control_matrix = adata[adata.obs.gene_target == control_treatment, :].X.toarray()
    with tqdm_joblib(desc="Processing Perturbations", total=len(perturbations)) as progress_bar:
        results = Parallel(n_jobs=-1)(delayed(process_perturbation)(pert, control_matrix, adata, alternative, test, expr_type, subsample_control) for pert in perturbations)
    print("Done with DEG analysis")
    stat_df = pd.DataFrame({perturbation: stat for perturbation, stat, _, _, _ in results}, index=list(adata.var.index))
    p_values_df = pd.DataFrame({perturbation: pvalues for perturbation, _, pvalues, _, _ in results}, index=list(adata.var.index))
    adj_p_values_df = pd.DataFrame({perturbation: pvalues for perturbation, _, _, pvalues, _ in results}, index=list(adata.var.index))
    avg_z_df = pd.DataFrame({perturbation: avg_z for perturbation, _, _, _, avg_z in results}, index=list(adata.var.index))
    return stat_df, p_values_df, adj_p_values_df, avg_z_df


def plot_differentially_expressed_genes(p_values_df, avg_z_df, p_value_threshold=0.05, avg_z_threshold=0, deg_cutoff=25):
    # Initialize dictionaries to hold the counts of upregulated and downregulated genes
    significant_counts = {}
    
    # Iterate through the perturbations
    for perturbation in p_values_df.columns:
        # Get the p-values and average z-scores for the current perturbation
        p_values = p_values_df[perturbation]
        avg_z = avg_z_df[perturbation]
        
        # Determine which genes are significantly upregulated or downregulated
        upregulated = (p_values < p_value_threshold) & (avg_z > avg_z_threshold)
        downregulated = (p_values < p_value_threshold) & (avg_z <= -avg_z_threshold)
        
        significant_counts[perturbation] = (upregulated.sum(), downregulated.sum())
    
    # Calculate the maximum count for setting y-axis limits
    max_count = max(max(abs(count) for count in counts) for counts in significant_counts.values())
    
    # Order genes by total number of significantly regulated genes
    ordered_genes = sorted(significant_counts.keys(), key=lambda x: abs(sum(significant_counts[x])), reverse=True)
    
    # Plot the results
    plt.figure(figsize=(20, 12))
    
    for gene in ordered_genes:
        total_deg = significant_counts[gene][0] + significant_counts[gene][1]
        color_up = '#008000' if total_deg > deg_cutoff else '#BDE7BD'  # Darker green for upregulated
        color_down = '#800000' if total_deg > deg_cutoff else '#FFB6B3'  # Darker red for downregulated
        plt.bar(gene, significant_counts[gene][0], color=color_up)
        plt.bar(gene, -significant_counts[gene][1], color=color_down)  # Invert downregulated count
    
    plt.xlabel('Perturbation')
    plt.ylabel('Count of Significant Genes (p-value < 0.05)')
    plt.title('Count of Significantly Regulated Genes per Perturbation')
    plt.xticks(rotation=90)
    plt.ylim(-max_count, max_count)  # Set y-axis limits
    
    # Set y-axis to show positive values for both directions
    num_ticks = 10
    step = max_count / num_ticks
    y_ticks = np.arange(-max_count, max_count + step, step)
    y_labels = [str(abs(int(tick))) for tick in y_ticks]
    plt.yticks(y_ticks, y_labels)
    plt.legend(['Upregulated', 'Downregulated'])
    plt.axhline(0,color='black',linewidth=0.5)
    plt.grid(True, which='both', linestyle='-', color='gray', alpha=0.25)
    plt.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.25)
    plt.show()

def plot_differentially_expressed_genes_per_treatment(p_values_df, avg_z_df, treatments, p_value_threshold=0.05, avg_z_threshold=0, deg_cutoff=0):
    # Initialize dictionaries to hold the counts of upregulated and downregulated genes
    for treatment in treatments:
        perturbations = [pert for pert in p_values_df.columns if pert.split('_')[0] == treatment[0]] #This will currently only work if all start with different first letter
        significant_counts = {}
        
        # Iterate through the perturbations
        for perturbation in perturbations:
            # Get the p-values and average z-scores for the current perturbation
            p_values = p_values_df[perturbation]
            avg_z = avg_z_df[perturbation]
            
            # Determine which genes are significantly upregulated or downregulated
            upregulated = (p_values < p_value_threshold) & (avg_z > avg_z_threshold)
            downregulated = (p_values < p_value_threshold) & (avg_z <= -avg_z_threshold)
            
            significant_counts[perturbation] = (upregulated.sum(), downregulated.sum())
        
        # Calculate the maximum count for setting y-axis limits
        max_count = max(max(abs(count) for count in counts) for counts in significant_counts.values())
        
        # Order genes by total number of significantly regulated genes
        ordered_genes = sorted(significant_counts.keys(), key=lambda x: abs(sum(significant_counts[x])), reverse=True)
        
        # Plot the results
        plt.figure(figsize=(20, 12))
        
        for gene in ordered_genes:
            total_deg = significant_counts[gene][0] + significant_counts[gene][1]
            color_up = '#008000' if total_deg > deg_cutoff else '#BDE7BD'  # Darker green for upregulated
            color_down = '#800000' if total_deg > deg_cutoff else '#FFB6B3'  # Darker red for downregulated
            plt.bar(gene, significant_counts[gene][0], color=color_up)
            plt.bar(gene, -significant_counts[gene][1], color=color_down)  # Invert downregulated count
        
        plt.xlabel('Perturbation')
        plt.ylabel('Count of Significant Genes (p-value < 0.05)')
        plt.title(f'Count of Significantly Regulated Genes per Perturbation for treatment {treatment}')
        plt.xticks(rotation=90)
        plt.ylim(-max_count, max_count)  # Set y-axis limits
        
        # Set y-axis to show positive values for both directions
        num_ticks = 10
        step = max_count / num_ticks
        y_ticks = np.arange(-max_count, max_count + step, step)
        y_labels = [str(abs(int(tick))) for tick in y_ticks]
        plt.yticks(y_ticks, y_labels)
        plt.legend(['Upregulated', 'Downregulated'])
        plt.axhline(0,color='black',linewidth=0.5)
        plt.grid(True, which='both', linestyle='-', color='gray', alpha=0.25)
        plt.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.25)
        plt.show()

def plot_top_n_DEGs(p_values_df, adata, perturbation, n=10, log_transform=False, plot_group='both'):
    # Ensure the perturbation exists in the p_values_df columns
    if perturbation not in p_values_df.columns:
        print(f"Perturbation {perturbation} not found in p_values_df.")
        return
    
    # Get the top n DEGs based on p-values
    top_n_genes = p_values_df[perturbation].nsmallest(n).index
    
    # Extract expression data for the specified perturbation and control
    perturbed_data = adata[adata.obs.gene_target == perturbation][:, top_n_genes].X
    control_data = adata[adata.obs.perturbed == "False"][:, top_n_genes].X
    
    # Optionally log-transform the data to handle extreme values
    if log_transform:
        perturbed_data = np.log1p(np.nan_to_num(perturbed_data))
        control_data = np.log1p(np.nan_to_num(control_data))
    
    # Create a DataFrame for plotting
    expression_data = []
    if plot_group in ['perturbed', 'both']:
        for i, gene in enumerate(top_n_genes):
            for value in perturbed_data[:, i]:
                expression_data.append((gene, value, 'Perturbed'))
    if plot_group in ['control', 'both']:
        for i, gene in enumerate(top_n_genes):
            for value in control_data[:, i]:
                expression_data.append((gene, value, 'Control'))
    
    expression_df = pd.DataFrame(expression_data, columns=['Gene', 'Expression', 'Group'])
    
    # Create violin plots
    plt.figure(figsize=(12, 8))
    if plot_group == 'both':
        sns.violinplot(x='Gene', y='Expression', hue='Group', data=expression_df, split=True)
    else:
        sns.violinplot(x='Gene', y='Expression', data=expression_df, hue='Group', split=False)
    
    # Add mean values
    means = expression_df.groupby(['Gene', 'Group'])['Expression'].mean().reset_index()
    if plot_group == 'both':
        sns.pointplot(x='Gene', y='Expression', hue='Group', data=means, dodge=0.3, palette=['black', 'black'], 
                      markers='d', errorbar=None, linestyle='none', markersize=10)
    else:
        sns.pointplot(x='Gene', y='Expression', data=means, color='black', 
                      markers='d', errorbar=None, linestyle='none', markersize=10)
    
    plt.title(f'Top {n} DEGs for {perturbation}, {plot_group} cells')
    plt.xticks(rotation=90)
    plt.ylabel("Expression (Z-Normalized)")
    plt.tight_layout()
    if plot_group == 'both':
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    else:
        plt.legend().set_visible(False)
    plt.show()


def plot_random_hvgs(adata, gene_target, num_genes=20, seed=0, plot_type="jitter", ylabel="Expression in (scaled) UMIs"):
    """
    Plots the distribution of gene expression for random highly variable genes labeled with a certain gene target and "NTC".

    Parameters:
    - adata: AnnData object containing the single-cell data.
    - gene_target: The gene target of interest to filter the cells.
    - num_genes: Number of random highly variable genes to plot (default is 20).
    - seed: Seed for reproducibility (default is 0).
    - plot_type: Type of plot to generate ('jitter' or 'violin').
    """
    # Ensure reproducibility
    np.random.seed(seed)
    
    # Identify highly variable genes
    highly_variable_genes = adata.var[adata.var['highly_variable']].index

    # Randomly select num_genes highly variable genes
    selected_genes = np.random.choice(highly_variable_genes, size=num_genes, replace=False)

    # Subset the data to include only the selected genes
    adata_subset = adata[:, selected_genes]

    # Create a DataFrame for the expression data of the selected genes
    expression_data = pd.DataFrame(adata_subset.X.toarray(), columns=selected_genes, index=adata_subset.obs.index)
    expression_data['gene_target'] = adata_subset.obs['gene_target'].astype(str)

    # Filter the data to include only the specified gene target and "NTC"
    expression_data = expression_data[expression_data['gene_target'].isin([gene_target, 'NTC'])]

    # Combine gene and condition information
    expression_data_melted = expression_data.melt(id_vars='gene_target', var_name='Gene', value_name='Expression')
    expression_data_melted['Gene-Condition'] = expression_data_melted['gene_target'] + '-' + expression_data_melted['Gene']

    # Sort the Gene-Condition to ensure perturbation is plotted first
    expression_data_melted['Gene-Condition'] = pd.Categorical(expression_data_melted['Gene-Condition'], 
                                                              categories=sorted(expression_data_melted['Gene-Condition'].unique(), 
                                                                                key=lambda x: (x.split('-')[1], x.split('-')[0])),
                                                              ordered=True)
    
    # Plot the jitter or violin plots
    plt.figure(figsize=(20, 10))
    if plot_type == "jitter":
        sns.stripplot(x='Gene-Condition', y='Expression', hue='Gene-Condition', data=expression_data_melted, jitter=True, alpha=0.7, palette='tab20', legend=False)
    elif plot_type == "violin":
        sns.violinplot(x='Gene-Condition', y='Expression', hue='Gene-Condition', data=expression_data_melted, palette='tab20', legend=False)

    # Calculate mean and median
    means = expression_data_melted.groupby('Gene-Condition', observed=True)['Expression'].mean()
    medians = expression_data_melted.groupby('Gene-Condition', observed=True)['Expression'].median()
    
    # Add horizontal bars for mean and median
    for i, gene_condition in enumerate(means.index):
        plt.plot([i-0.4, i+0.4], [means[gene_condition], means[gene_condition]], color='black', linewidth=2)
        plt.plot([i-0.4, i+0.4], [medians[gene_condition], medians[gene_condition]], color='blue', linewidth=2)
    plt.text(0.99, 0.90, 'Black line: Mean\nBlue line: Median', 
             verticalalignment='bottom', horizontalalignment='right', 
             transform=plt.gca().transAxes,
             color='black', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    plt.title(f'Gene Expression for {num_genes} random highly variable genes for {gene_target} vs NTC')
    plt.xticks(rotation=90)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to fit title better
    plt.ylabel(ylabel)
    plt.xlabel("Perturbation-Gene")
    plt.show()


def plot_top_n_DEGs(p_values_df, adata, perturbation, n=10, log_transform=False, plot_group='both'):
    # Ensure the perturbation exists in the p_values_df columns
    if perturbation not in p_values_df.columns:
        print(f"Perturbation {perturbation} not found in p_values_df.")
        return
    
    # Get the top n DEGs based on p-values
    top_n_genes = p_values_df[perturbation].nsmallest(n).index
    
    # Extract expression data for the specified perturbation and control
    perturbed_data = adata[adata.obs.gene_target == perturbation][:, top_n_genes].X
    control_data = adata[adata.obs.perturbed == "False"][:, top_n_genes].X
    
    # Optionally log-transform the data to handle extreme values
    if log_transform:
        perturbed_data = np.log1p(np.nan_to_num(perturbed_data))
        control_data = np.log1p(np.nan_to_num(control_data))
    
    # Create a DataFrame for plotting
    expression_data = []
    if plot_group in ['perturbed', 'both']:
        for i, gene in enumerate(top_n_genes):
            for value in perturbed_data[:, i]:
                expression_data.append((gene, value, 'Perturbed'))
    if plot_group in ['control', 'both']:
        for i, gene in enumerate(top_n_genes):
            for value in control_data[:, i]:
                expression_data.append((gene, value, 'Control'))
    
    expression_df = pd.DataFrame(expression_data, columns=['Gene', 'Expression', 'Group'])
    
    # Create violin plots
    plt.figure(figsize=(12, 8))
    if plot_group == 'both':
        sns.violinplot(x='Gene', y='Expression', hue='Group', data=expression_df, split=True)
    else:
        sns.violinplot(x='Gene', y='Expression', data=expression_df, hue='Group', split=False)
    
    # Add mean values
    means = expression_df.groupby(['Gene', 'Group'])['Expression'].mean().reset_index()
    if plot_group == 'both':
        sns.pointplot(x='Gene', y='Expression', hue='Group', data=means, dodge=0.3, palette=['black', 'black'], 
                      markers='d', errorbar=None, linestyle='none', markersize=10)
    else:
        sns.pointplot(x='Gene', y='Expression', data=means, color='black', 
                      markers='d', errorbar=None, linestyle='none', markersize=10)
    
    plt.title(f'Top {n} DEGs for {perturbation}, {plot_group} cells')
    plt.xticks(rotation=90)
    plt.ylabel("Expression (Z-Normalized)")
    plt.tight_layout()
    if plot_group == 'both':
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    else:
        plt.legend().set_visible(False)
    plt.show()


def save_DEG_df(p_values_df, directionality_df, threshold, save=True, filepath=None, num_show=10):
    # Create a dictionary to hold the sorted and filtered index names for each column
    sorted_filtered_dict = {}
    p_values_dict = {}
    upregulated_dict = {}
    downregulated_dict = {}
    upregulated_p_values_dict = {}
    downregulated_p_values_dict = {}

    for column in p_values_df.columns:
        # Sort the column by values
        sorted_column = p_values_df[column].sort_values()
        
        # Filter the values based on the single threshold
        filtered_column = sorted_column[sorted_column < threshold]
        
        # Replace the values with the corresponding index names
        sorted_filtered_dict[column] = filtered_column.index.tolist()
        p_values_dict[column] = filtered_column.values.tolist()
        
        # Filter by directionality
        upregulated = filtered_column[directionality_df.loc[filtered_column.index, column] == 1]
        downregulated = filtered_column[directionality_df.loc[filtered_column.index, column] == -1]
        
        upregulated_dict[column] = upregulated.index.tolist()
        downregulated_dict[column] = downregulated.index.tolist()
        upregulated_p_values_dict[column] = upregulated.values.tolist()
        downregulated_p_values_dict[column] = downregulated.values.tolist()

    # Determine the maximum length of the lists in the dictionary
    max_length = max(len(lst) for lst in sorted_filtered_dict.values())

    # Create new DataFrames with the sorted and filtered index names
    sorted_filtered_df = pd.DataFrame({col: sorted_filtered_dict[col] + [None]*(max_length - len(sorted_filtered_dict[col])) for col in sorted_filtered_dict})
    p_values_df = pd.DataFrame({col: p_values_dict[col] + [None]*(max_length - len(p_values_dict[col])) for col in p_values_dict})
    
    max_length_up = max(len(lst) for lst in upregulated_dict.values())
    max_length_down = max(len(lst) for lst in downregulated_dict.values())
    
    upregulated_df = pd.DataFrame({col: upregulated_dict[col] + [None]*(max_length_up - len(upregulated_dict[col])) for col in upregulated_dict})
    upregulated_p_values_df = pd.DataFrame({col: upregulated_p_values_dict[col] + [None]*(max_length_up - len(upregulated_p_values_dict[col])) for col in upregulated_p_values_dict})
    
    downregulated_df = pd.DataFrame({col: downregulated_dict[col] + [None]*(max_length_down - len(downregulated_dict[col])) for col in downregulated_dict})
    downregulated_p_values_df = pd.DataFrame({col: downregulated_p_values_dict[col] + [None]*(max_length_down - len(downregulated_p_values_dict[col])) for col in downregulated_p_values_dict})
    
    # Sort the columns by the number of non-NaN elements
    sorted_filtered_df = sorted_filtered_df.loc[:, sorted_filtered_df.notna().sum().sort_values(ascending=False).index]
    p_values_df = p_values_df.loc[:, p_values_df.notna().sum().sort_values(ascending=False).index]
    upregulated_df = upregulated_df.loc[:, upregulated_df.notna().sum().sort_values(ascending=False).index]
    upregulated_p_values_df = upregulated_p_values_df.loc[:, upregulated_p_values_df.notna().sum().sort_values(ascending=False).index]
    downregulated_df = downregulated_df.loc[:, downregulated_df.notna().sum().sort_values(ascending=False).index]
    downregulated_p_values_df = downregulated_p_values_df.loc[:, downregulated_p_values_df.notna().sum().sort_values(ascending=False).index]

    if save and filepath:
        with pd.ExcelWriter(filepath) as writer:
            sorted_filtered_df.to_excel(writer, sheet_name='All Genes')
            p_values_df.to_excel(writer, sheet_name='P-Values (All Genes)')
            upregulated_df.to_excel(writer, sheet_name='Upregulated Genes')
            upregulated_p_values_df.to_excel(writer, sheet_name='P-Values (Upregulated)')
            downregulated_df.to_excel(writer, sheet_name='Downregulated Genes')
            downregulated_p_values_df.to_excel(writer, sheet_name='P-Values (Downregulated)')

    display(sorted_filtered_df.head(num_show))
    return sorted_filtered_df


def filter_perturbations_by_deg_cutoff(adata, p_values_df, number_degs, alpha=0.05):
    valid_genes = list(p_values_df.columns[((p_values_df < alpha).sum())>=number_degs])
    valid_genes.append("NTC")
    return adata[adata.obs.gene_target.isin(valid_genes),:].copy()



def create_HVG_set(adata, DEG_df, top_n_DEGs=10):
    """
    Create a set of highly variable genes (HVG) from the given AnnData object and a DataFrame of differentially expressed genes (DEGs).

    Parameters:
    adata (AnnData): AnnData object containing single-cell RNA-seq data.
    DEG_df (DataFrame): DataFrame containing differentially expressed genes.
    top_n_DEGs (int): Number of top DEGs to consider for each perturbation.

    Returns:
    AnnData: A subset of the input AnnData object containing only the HVGs.
    """
    
    # Get unique perturbations from adata.obs
    perturbations = list(adata.obs['gene_target'].unique())
    
    # Drop columns from DEG_df that are not in perturbations
    columns_to_keep = [col for col in DEG_df.columns if col in perturbations]
    DEG_df = DEG_df[columns_to_keep]
    
    # Select top_n_DEGs for each perturbation, or the maximum possible if fewer than top_n_DEGs
    top_DEGs = DEG_df.apply(lambda x: x.dropna().head(min(top_n_DEGs, len(x.dropna()))), axis=0)
    
    # Flatten the DataFrame to get a set of unique differentially expressed genes
    deg_values = top_DEGs.values.ravel()
    
    # Filter out any None or NaN values that may have been introduced
    deg_values = [gene for gene in deg_values if pd.notnull(gene)]
    unique_degs = set(deg_values)
    
    # Get the set of highly variable genes from adata
    unique_hvg = set(adata[:, adata.var['highly_variable']].var.index)
    
    # Create the union of the two sets
    HVG_set = list(unique_degs.union(unique_hvg))
    
    # Print the size of the resulting HVG set
    print(f"Constructing a set of {len(HVG_set)} unique HVGs from {len(unique_degs)} unique DEGs and {len(unique_hvg)} unique HVGs")

    # Return the subset of adata containing only the HVGs
    return adata[:, adata.var.index.isin(HVG_set)].copy()



def compute_mean_profile(adata, group_indices):
    # Compute the mean profile and ensure it's a 1D array
    mean_profile = adata[group_indices, :].X.mean(axis=0)
    return mean_profile.A1 if hasattr(mean_profile, 'A1') else mean_profile

def plot_gene_target_clustermap(adata, metric='correlation', n_jobs=-1, vmax=None, figsize=(20, 20)):
    # Group by gene_target and get indices for each group
    gene_target_groups = adata.obs.groupby('gene_target').indices

    # Initialize tqdm progress bar
    with tqdm_joblib(desc="Computing mean profiles", total=len(gene_target_groups)) as progress_bar:
        # Compute the mean expression profile for each group in parallel
        mean_profiles = Parallel(n_jobs=n_jobs)(
            delayed(compute_mean_profile)(adata, indices) 
            for indices in gene_target_groups.values()
        )
    
    # Convert mean profiles to a 2D array and reshape
    mean_profiles = np.vstack(mean_profiles)

    # Convert mean profiles to a DataFrame
    mean_profiles_df = pd.DataFrame(
        mean_profiles, 
        index=gene_target_groups.keys(),
        columns=adata.var_names
    )

    # Compute the pairwise distance matrix (using correlation or other metrics)
    if metric == 'correlation':
        # Compute the correlation matrix directly
        distance_matrix = mean_profiles_df.T.corr()
    else:
        # Compute pairwise distances using specified metric
        distance_matrix = pd.DataFrame(
            squareform(pdist(mean_profiles_df, metric=metric)),
            index=mean_profiles_df.index,
            columns=mean_profiles_df.index
        )

    # Plot the clustermap with clipping maximum values and set figure size
    g = sns.clustermap(distance_matrix, cmap='magma', vmax=vmax, figsize=figsize, cbar_kws={'shrink': 0.25})

    # Extract the reordered labels from the clustermap object
    reordered_labels_x = distance_matrix.columns[g.dendrogram_row.reordered_ind]
    reordered_labels_y = distance_matrix.columns[g.dendrogram_col.reordered_ind]

    # Set the ticks and labels to ensure all are shown
    g.ax_heatmap.set_xticks(np.arange(len(reordered_labels_x)))
    g.ax_heatmap.set_xticklabels(reordered_labels_x, rotation=90, fontsize=15)
    g.ax_heatmap.set_yticks(np.arange(len(reordered_labels_y)))
    g.ax_heatmap.set_yticklabels(reordered_labels_y, rotation=0, fontsize=15)


    plt.title('Pairwise correlations')
    plt.show()

    return distance_matrix


# Define the function to compute the mean normalized profile
def compute_mean_normalized_profile(adata, group_indices):
    mean_vector = adata[group_indices, :].X.mean(axis=0)
    normalized_mean_vector = (mean_vector - np.mean(mean_vector)) / np.std(mean_vector)
    return normalized_mean_vector.A1 if hasattr(normalized_mean_vector, 'A1') else normalized_mean_vector

# Define the function to compute the MDE with a 20-dimensional spectral embedding
def compute_MDE(adata, save=True, save_dir_stem=None, leiden_neighbors=5, preserve='neighbors'):
    # Group by gene_target and get indices for each group
    gene_target_groups = adata.obs.groupby('gene_target').indices

    # Initialize tqdm progress bar
    with tqdm_joblib(desc="Computing mean profiles", total=len(gene_target_groups)) as progress_bar:
        # Compute the mean expression profile for each group in parallel
        with parallel_backend('threading'):
            mean_profiles = Parallel(n_jobs=-1)(
                delayed(compute_mean_normalized_profile)(adata, indices) 
                for indices in gene_target_groups.values()
            )
    
    # Convert the list of mean profiles to a numpy array
    mean_profiles_array = np.array(mean_profiles)

    # Compute MDE
    mde = None
    embedding = None
    if preserve == "neighbors":
        # spectral_embedding = SpectralEmbedding(
        #     n_components=20,  # Ensure this matches the pymde embedding dimension
        #     affinity='nearest_neighbors',
        #     n_neighbors=7,
        #     eigen_solver='arpack'
        # )
        # initial_embedding = spectral_embedding.fit_transform(mean_profiles_array)
        mde = pymde.preserve_neighbors(mean_profiles_array, repulsive_fraction=5.0, n_neighbors = 5)
        embedding = mde.embed(max_iter=2000,print_every=100,verbose=True,eps=1e-8)
    else:
        mde = pymde.preserve_distances(mean_profiles_array)
        embedding = mde.embed(max_iter=1000,print_every=100,verbose=True)
    
    
    # Run Leiden clustering on the embeddings
    knn_graph = kneighbors_graph(embedding, n_neighbors=leiden_neighbors, include_self=False)
    sources, targets = knn_graph.nonzero()
    g = ig.Graph(directed=False)
    g.add_vertices(embedding.shape[0])
    g.add_edges(zip(sources, targets))
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
    
    # Get cluster labels and make pairwise correlation plot
    clusters = np.array(partition.membership)
    gene_targets = list(gene_target_groups.keys())
    embedding_df = pd.DataFrame(embedding, columns=['x', 'y'])
    embedding_df['gene_target'] = gene_targets
    embedding_df['cluster'] = clusters
    embedding_df['cluster'] = embedding_df['cluster'].astype(str)
    cluster_groups = embedding_df.groupby('cluster')
    centroids = cluster_groups[['x', 'y']].mean()
    pairwise_distances = pd.DataFrame(
        squareform(pdist(centroids, metric='euclidean')),
        index=centroids.index,
        columns=centroids.index
    )
    sns.clustermap(pairwise_distances, cmap='viridis')
    plt.title('Pairwise Distance Matrix of Cluster Centroids')
    plt.show()


    # Plot using Plotly with Pastel colormap for discrete clusters
    fig = px.scatter(embedding_df, x='x', y='y', text='gene_target', color='cluster',
                     hover_data={'x': True, 'y': True, 'gene_target': True},
                     title='MDE Embedding of Mean Normalized Profiles',
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(marker=dict(size=7), textposition='middle center',textfont=dict(size=4))
    
    fig.update_layout(
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        width=1200,  # Increase plot width
        height=1000,  # Increase plot height
        legend_title_text='Cluster',
        coloraxis_showscale=False
    )
    if save:
        pio.write_html(fig, file=save_dir_stem+"_MDE.html")
        embedding_df.to_excel(save_dir_stem+"_clusters.xlsx", index=False)
   
    fig.show()
    
    return embedding, clusters, mean_profiles_array


# Define the function to compute the MDE with a 20-dimensional spectral embedding
def compute_MDE_small(adata, save=True, save_dir_stem=None, leiden_neighbors=5, preserve='neighbors'):
    # Group by gene_target and get indices for each group
    gene_target_groups = adata.obs.groupby('gene_target').indices

    # Initialize tqdm progress bar
    with tqdm_joblib(desc="Computing mean profiles", total=len(gene_target_groups)) as progress_bar:
        # Compute the mean expression profile for each group in parallel
        with parallel_backend('threading'):
            mean_profiles = Parallel(n_jobs=-1)(
                delayed(compute_mean_normalized_profile)(adata, indices) 
                for indices in gene_target_groups.values()
            )
    
    # Convert the list of mean profiles to a numpy array
    mean_profiles_array = np.array(mean_profiles)

    # Compute MDE
    mde = None
    embedding = None
    if preserve == "neighbors":
        # spectral_embedding = SpectralEmbedding(
        #     n_components=20,  # Ensure this matches the pymde embedding dimension
        #     affinity='nearest_neighbors',
        #     n_neighbors=7,
        #     eigen_solver='arpack'
        # )
        # initial_embedding = spectral_embedding.fit_transform(mean_profiles_array)
        mde = pymde.preserve_neighbors(mean_profiles_array, repulsive_fraction=5.0, n_neighbors = 5)
        embedding = mde.embed(max_iter=1000,print_every=100,verbose=True,eps=1e-8)
    else:
        mde = pymde.preserve_distances(mean_profiles_array)
        embedding = mde.embed(max_iter=1000,print_every=100,verbose=True)
    
    
    # Run Leiden clustering on the embeddings
    knn_graph = kneighbors_graph(embedding, n_neighbors=leiden_neighbors, include_self=False)
    sources, targets = knn_graph.nonzero()
    g = ig.Graph(directed=False)
    g.add_vertices(embedding.shape[0])
    g.add_edges(zip(sources, targets))
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
    
    # Get cluster labels and make pairwise correlation plot
    clusters = np.array(partition.membership)
    gene_targets = list(gene_target_groups.keys())
    embedding_df = pd.DataFrame(embedding, columns=['x', 'y'])
    embedding_df['gene_target'] = gene_targets
    embedding_df['cluster'] = clusters
    embedding_df['cluster'] = embedding_df['cluster'].astype(str)
    cluster_groups = embedding_df.groupby('cluster')
    centroids = cluster_groups[['x', 'y']].mean()
    pairwise_distances = pd.DataFrame(
        squareform(pdist(centroids, metric='euclidean')),
        index=centroids.index,
        columns=centroids.index
    )
    sns.clustermap(pairwise_distances, cmap='viridis')
    plt.title('Pairwise Distance Matrix of Cluster Centroids')
    plt.show()


    # Plot using Plotly with Pastel colormap for discrete clusters
    fig = px.scatter(embedding_df, x='x', y='y', text='gene_target', color='cluster',
                     hover_data={'x': True, 'y': True, 'gene_target': True},
                     title='MDE of Mean Normalized Profiles',
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(marker=dict(size=8), textposition='middle center',textfont=dict(size=4))
    
    fig.update_layout(
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        width=600,  # Increase plot width
        height=600,  # Increase plot height
        legend_title_text='Cluster',
        coloraxis_showscale=False
    )
    if save:
        pio.write_html(fig, file=save_dir_stem+"_MDE.html")
        embedding_df.to_excel(save_dir_stem+"_clusters.xlsx", index=False)
   
    fig.show()
    
    return embedding, clusters, mean_profiles_array