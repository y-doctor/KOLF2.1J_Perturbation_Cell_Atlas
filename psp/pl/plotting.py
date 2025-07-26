import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import anndata as ad
import pandas as pd
from plotnine import (
    ggplot, aes, geom_bar, ggtitle, xlab, ylab,
    scale_fill_manual, geom_histogram, labs, theme,
    element_text, scale_y_continuous
)
import psp.utils as utils
from scipy.stats import gamma
from psp.utils import validate_anndata
import scanpy as sc
import matplotlib.colors as mcolors
plt.rcParams['font.family'] = 'Arial'
from matplotlib.colors import LogNorm, Normalize
from typing import Tuple, Dict, List
from scipy.spatial.distance import squareform
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import psp.da

def plot_cells_per_perturbation(adata: ad.AnnData, perturbation_key: str = 'gene_target', perturbed_key: str = 'perturbed', highlight_threshold: int = 100, y_max: int = 600, xlabel: str = "Number of Perturbations", ylabel: str = "Cells per Perturbation", tick_spacing: int = 100) -> plt.Figure:
    """
    Plots the number of cells per perturbation for perturbed cells in the AnnData object.

    This function filters the AnnData object for perturbed cells, counts the number of cells
    per perturbation, and creates a plot showing these counts. It highlights perturbations
    with more than `highlight_threshold` cells and adjusts the y-axis to display up to `y_max` cells, labeling the
    top as "`y_max`+".

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing single-cell data.
    - perturbation_key (str): The key in `adata.obs` to identify perturbations. Default is 'gene_target'.
    - perturbed_key (str): The key in `adata.obs` to identify perturbed cells. Default is 'perturbed'.
    - highlight_threshold (int): The threshold for highlighting perturbations. Print the number of perturbations with more than highlight_threshold cells. Default is 100.
    - y_max (int): The maximum value for the y-axis. Default is 600.

    Returns:
    - matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Filter for perturbed cells
    adata_perturbed = adata[adata.obs[perturbed_key] == "True"]
    
    # Count cells per perturbation
    counts = sorted(adata_perturbed.obs[perturbation_key].value_counts().values, reverse=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(range(1, len(counts) + 1), counts, color='#367CB7', linewidth=2)
    
    # Fill the area under the curve
    ax.fill_between(range(1, len(counts) + 1), counts, 0, color='#EDF6FF')
    
    # Set x and y labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Style the axes
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Draw the horizontal line at y=highlight_threshold
    ax.axhline(y=highlight_threshold, color='#5CB39D', linestyle='--', linewidth=1.5)
    
    # Find the number of perturbations with more than highlight_threshold cells
    num_perturbations_gt_threshold = sum(count > highlight_threshold for count in counts)
    print(f"{num_perturbations_gt_threshold}/{len(counts)} ({100*num_perturbations_gt_threshold/len(counts):.2f}%) have > {highlight_threshold} cells")
    
    # Adjust y-axis limits and label the top as "y_max+"
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, len(counts) + 1)
    y_ticks = list(range(0, y_max + 1, tick_spacing))
    y_labels = [str(y) for y in y_ticks[:-1]] + [f"{y_max}+"]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Remove the grid
    ax.grid(False)
    
    plt.show()
    return fig


def plot_umis_per_cell(adata: ad.AnnData, umi_key: str = 'n_UMI_counts') -> plt.Figure:
    """
    Plot a violin plot of UMIs per cell.

    This function generates a violin plot to visualize the distribution of Unique Molecular Identifiers (UMIs)
    per cell in the given AnnData object.

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing single-cell data.
    - umi_key (str): The key in adata.obs that contains UMI counts for each cell. Default is 'n_UMI_counts'.

    Returns:
    - matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Extract UMIs per cell
    umis_per_cell = adata.obs[umi_key]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot the violin
    sns.violinplot(data=umis_per_cell, ax=ax, inner='box', color='#EDF6FF', linewidth=2, saturation=1,linecolor="#367CB7")

    # Set x and y labels
    ax.set_xlabel('Single Cells', fontsize=12)
    ax.set_ylabel('UMIs per Cell', fontsize=12)
    
    # Style the axes
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Remove the grid
    ax.grid(False)
    
    plt.show()
    return fig


def plot_scatter(x: np.ndarray, y: np.ndarray, title: str = "Scatter Plot", xlabel: str = "X-axis", ylabel: str = "Y-axis") -> None:
    """
    Plots a scatter plot of the given x and y data.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, edgecolor='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.show()


def plot_histogram(data: np.ndarray, bins: int = 10, title: str = "Histogram", xlabel: str = "Value", ylabel: str = "Frequency") -> None:
    """
    Plots a histogram of the given data.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.show()


def plot_gRNA_distribution(adata: ad.AnnData) -> None:
    """
    Plots the distribution of the number of gRNAs per cell and calculates statistics 
    related to sgRNA calls and multiplet rates.
    """
    df = pd.DataFrame(adata.obs['n_gRNA'], columns=['n_gRNA'])
    df['n_gRNA'] = df['n_gRNA'].apply(lambda x: str(x) if x <= 6 else '>6')
    categories = df['n_gRNA'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = dict(zip(categories, colors[:len(categories)]))
    plot = (ggplot(df, aes(x='n_gRNA', fill='n_gRNA')) +
            geom_bar() +
            scale_fill_manual(values=color_map) +
            ggtitle('Number of gRNA Assigned') +
            xlab('n_gRNA') +
            ylab('Count'))
    plot.show()
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
    """
    adata_1_gRNA = adata[adata.obs['n_gRNA'] == 1, :]
    df = pd.DataFrame(adata_1_gRNA.obs['n_gRNA_UMIs'].astype(int), columns=['n_gRNA_UMIs'])
    plot = (ggplot(df, aes(x='n_gRNA_UMIs')) +
            geom_histogram(bins=50, fill='#AEC6CF') +
            labs(title='gRNA UMI Counts For Cells with 1 sgRNA Assigned', x='gRNA UMI Counts', y='Counts'))
    plot.show()


def plot_cells_per_guide_distribution(adata: ad.AnnData) -> None:
    """
    Plots the distribution of cells per guide RNA (gRNA) for cells with exactly one gRNA assigned.
    """
    adata_1_gRNA = adata[adata.obs['n_gRNA'] == 1, :]
    gRNA_counts = adata_1_gRNA.obs['gene_target'].value_counts()
    gRNA_counts_df = gRNA_counts.reset_index()
    gRNA_counts_df.columns = ['gene_target', 'count']
    ntc_count = gRNA_counts_df.loc[gRNA_counts_df['gene_target'] == 'NTC', 'count'].values[0]
    gRNA_counts_df = gRNA_counts_df[gRNA_counts_df['gene_target'] != 'NTC']
    plot = (ggplot(gRNA_counts_df, aes(x='reorder(gene_target, -count)', y='count')) +
            geom_bar(stat='identity', fill='#FFD1DC') +
            theme(axis_text_x=element_text(rotation=90, hjust=1), figure_size=(20, 6)) +
            scale_y_continuous(breaks=range(0, max(gRNA_counts_df['count']) + 50, 50)) +
            ggtitle(f'Number of Cells per Single Perturbation Target; {ntc_count} NTC Cells') +
            xlab('Perturbation Target') +
            ylab('Number of Cells'))
    plot.show()
    total_perts = len(gRNA_counts_df)
    over_50 = len(gRNA_counts_df[gRNA_counts_df['count'] >= 50])
    print(f"Number of perturbations with >= 50 cells with single guide assigned: {over_50}/{total_perts} ({100 * (over_50 / total_perts):.2f}%)")


def doublet_detection_sanity_check(adata: ad.AnnData) -> None:
    """
    Perform a sanity check for doublet detection by plotting histograms of key metrics.
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


def plot_sorted_bars(
    data: pd.Series,
    ylabel: str,
    title: str,
    repression_threshold: float = None,
    cells_threshold: int = None,
    label_interval: int = 100,
    invert_y: bool = False,
    vmin: float = 0,
    vmax: float = None,
) -> plt.Figure:
    """
    Plots sorted bar charts for knockdown efficiency or cell counts per sgRNA.
    """
    sorted_items = data.sort_values(ascending=False)
    keys = sorted_items.index.tolist()
    values = sorted_items.values.tolist()

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(keys, values, color='skyblue')
    
    if repression_threshold is not None:
        ax.axhline(repression_threshold, color='skyblue', ls=':')
    if cells_threshold is not None:
        ax.axhline(cells_threshold, color='skyblue', ls=':')
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=12)
    
    if invert_y:
        # Set the limits explicitly for an inverted plot
        if vmax is None:
            vmax = ax.get_ylim()[1]
        if vmin is None:
            vmin = ax.get_ylim()[0]
        ax.set_ylim(vmax, vmin)
    else:
        if vmin is not None and vmax is not None:
            ax.set_ylim(vmin, vmax)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', visible=False)
    
    plt.xticks(ticks=range(0, len(keys), label_interval), 
               labels=[keys[i] for i in range(0, len(keys), label_interval)])
    
    plt.tight_layout()
    return fig


def plot_percentage_perturbations_by_repression(
    adata: ad.AnnData,
    perturbation_col: str = "gene_target",
    xlabel: str = "Number of Perturbations",
    knockdown_col: str = "target_knockdown",
    knockdown_threshold: float = 0.3,
    figsize: tuple[float, float] = (5, 5),
    line_color: str = "#367CB7",
    fill_color: str = "#EDF6FF",
    threshold_color: str = "#5CB39D",
) -> plt.Figure:
    """
    Visualize the cumulative percentage of perturbations achieving specific knockdown levels.
    
    Parameters:
    - adata: AnnData object containing perturbation data
    - perturbation_col: Column in adata.obs containing perturbation identifiers
    - xlabel: Label for the x-axis
    - knockdown_col: Column in adata.obs containing knockdown efficiency values (0-1 scale)
    - knockdown_threshold: Threshold for meaningful repression (default 0.3 = 30%)
    - figsize: Dimensions of the output figure
    - line_color: Color for the main trend line
    - fill_color: Color for the area under the curve
    - threshold_color: Color for the threshold reference line
    
    Returns:
    - matplotlib Figure object containing the visualization
    
    Raises:
    - ValueError: If required columns are missing from adata.obs
    """
    # Validate input columns
    utils.validate_anndata(adata, required_obs=[perturbation_col, knockdown_col])
    perturbations = utils.get_perturbed_view(adata)
    
    # Calculate mean knockdown per perturbation
    total_knockdown = perturbations.obs.groupby(perturbation_col)[knockdown_col].mean()
    sorted_knockdown = np.sort(total_knockdown)[::-1]  # Descending sort
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main elements
    ax.fill_between(
        range(1, len(sorted_knockdown) + 1),
        0,
        sorted_knockdown * 100,
        color=fill_color
    )
    ax.plot(
        range(1, len(sorted_knockdown) + 1),
        sorted_knockdown * 100,
        color=line_color,
        linewidth=2
    )
    
    # Add threshold reference line
    ax.axhline(
        y=knockdown_threshold * 100,
        color=threshold_color,
        linestyle='--',
        linewidth=1.5
    )
    
    # Configure axis labels and ticks
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('% Repression', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(sorted_knockdown) + 1)
    ax.invert_yaxis()
    
    # Configure ticks and labels
    y_ticks = np.arange(0, 101, 20)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y}%" for y in y_ticks])
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Style elements
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Turn off grid
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    count = (sorted_knockdown >= knockdown_threshold).sum()
    print(f"Percentage of perturbations achieving {knockdown_threshold*100}% repression: {count/len(sorted_knockdown)*100:.2f}% ({count}/{len(sorted_knockdown)})")
    return fig

def plot_energy_distance_threshold(null_distances, experimental_group_distances, threshold=0.75):    
    """
    Plot energy distance distribution with gamma fit and threshold line.
    
    Parameters:
        null_distances: Array of null distribution energy distances
        experimental_group_distances: Dictionary of {experimental_group: energy_distance} for experimental groups
        threshold: Probability threshold for gamma distribution cutoff
    """
    # Fit gamma distribution
    alpha_mle, loc_mle, beta_mle = gamma.fit(null_distances, loc=0.3)
    
    # Prepare data
    perturbed_edist = np.array(list(experimental_group_distances.values()))
    x = np.linspace(np.min(null_distances), np.max(null_distances), 1000)
    pdf_fitted = gamma.pdf(x, alpha_mle, loc=loc_mle, scale=beta_mle)
    thresh_val = gamma.ppf(threshold, alpha_mle, loc=loc_mle, scale=beta_mle)

    # Create plot with consistent styling
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Histogram styling matching gRNA distribution plot
    ax.hist(null_distances, bins=300, density=True, alpha=0.6, 
            color='#AEC6CF', label="Control sgRNA")
    ax.hist(np.clip(perturbed_edist, 0, np.max(null_distances)*3), bins=300,
            density=True, alpha=0.6, color='#FFB3BA', label="Perturbing sgRNA")
    
    # Line styling consistent with other plots
    ax.plot(x, pdf_fitted, color='#367CB7', lw=2.5, label="Fitted Gamma Distribution")
    ax.axvline(x=thresh_val, color='#5CB39D', linestyle='--', lw=2, 
               label=f"{threshold*100}% Threshold")

    # Axis styling
    ax.set_xlabel("Energy Distance", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Energy Distance Distribution", fontsize=14, pad=12)
    
    # Grid and spine configuration
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('#444444')

    # Legend and ticks
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)

    plt.tight_layout()
    plt.show()
    
    return thresh_val


def plot_filtered_genes_inverted(
    results_dict, 
    p_value_threshold=0.05, 
    l2fc_threshold=0, 
    deg_cutoff=25, 
    min_total_deg=10,
    ytick_step = 20
):
    """
    Plot the number of upregulated and downregulated genes per perturbation as horizontal bars.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of DESeq2 results where keys are perturbation names and values are DataFrames
    p_value_threshold : float, optional
        P-value threshold for significance, by default 0.05
    l2fc_threshold : float, optional
        Log2 fold change threshold, by default 0
    deg_cutoff : int, optional
        Threshold for highlighting perturbations with high DEG counts, by default 25
    min_total_deg : int, optional
        Minimum total DEGs required to include a perturbation, by default 10
        
    Returns
    -------
    tuple
        (matplotlib.figure.Figure, tuple) containing the figure and a tuple of (ordered_perturbations, perturbation_counts)
    """
    significant_counts = {}

    total_perturbations = len(results_dict)

    # Process each perturbation's results
    for perturbation, df in results_dict.items():
        if df is None:
            continue
            
        p_values = df['padj']
        l2fc = df['log2FoldChange']
        upregulated = (p_values < p_value_threshold) & (l2fc > l2fc_threshold) & pd.notna(p_values)
        downregulated = (p_values < p_value_threshold) & (l2fc < -l2fc_threshold) & pd.notna(p_values)
        total_deg = upregulated.sum() + downregulated.sum()
        
        # Filter to show only perturbations with more than min_total_deg DEGs
        if total_deg >= min_total_deg:
            significant_counts[perturbation] = (upregulated.sum(), downregulated.sum())

    if not significant_counts:
        print("No perturbations with significant DEGs above the specified threshold.")
        return None, ([], [])

    # Set maximum count for x-axis
    max_count = 500
    
    # Sort perturbations by total DEG count in descending order
    ordered_perturbations = sorted(
        significant_counts.keys(),
        key=lambda x: abs(sum(significant_counts[x])),
        reverse=True
    )

    # Create figure with horizontal bars
    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, perturbation in enumerate(ordered_perturbations):
        total_deg = significant_counts[perturbation][0] + significant_counts[perturbation][1]
        
        # Use consistent colors for upregulated and downregulated DEGs
        color_up = '#BDE7BD'  # Light green for upregulated
        color_down = '#FFB6B3'  # Light red for downregulated
        
        # Plot horizontal bars
        ax.barh(idx, significant_counts[perturbation][0], color=color_up)  # Upregulated
        ax.barh(idx, -significant_counts[perturbation][1], color=color_down)  # Downregulated
    
    # Set axis labels
    ax.set_ylabel('Perturbation Number (>10 DEGs)')
    ax.set_xlabel(f'Count of Significant Genes (p-value < {p_value_threshold})')

    # Configure y-axis ticks
    ax.set_yticks(range(0, len(ordered_perturbations), ytick_step))

    # Configure x-axis limits and ticks
    ax.set_xlim(-max_count, max_count)
    ax.set_xticks(np.arange(-max_count, max_count + 1, step=100))
    ax.set_xticklabels(
        [str(abs(int(tick))) if abs(tick) != max_count else ">500" 
         for tick in np.arange(-max_count, max_count + 1, step=100)],
        fontsize=12
    )
    
    # Add legend and center line
    ax.legend(['Upregulated', 'Downregulated'], loc='upper right')
    ax.axvline(0, color='black', linewidth=0.5)
    
    # Remove grid and adjust layout
    ax.grid(False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Print metrics about DEG counts
    count_10 = sum(1 for pert in ordered_perturbations 
                   if significant_counts[pert][0] + significant_counts[pert][1] >= 10)
    print(f"Total perturbations with >= 10 DEGs: {count_10} ({100*count_10/total_perturbations:.2f}%)")
    
    count_25 = sum(1 for pert in ordered_perturbations 
                   if significant_counts[pert][0] + significant_counts[pert][1] >= 25)
    print(f"Total perturbations with >= 25 DEGs: {count_25} ({100*count_25/total_perturbations:.2f}%)")
    
    count_50 = sum(1 for pert in ordered_perturbations 
                   if significant_counts[pert][0] + significant_counts[pert][1] >= 50)
    print(f"Total perturbations with >= 50 DEGs: {count_50} ({100*count_50/total_perturbations:.2f}%)")
    
    # Return the figure and the ordered perturbations with their counts
    perturbation_counts = [significant_counts[pert] for pert in ordered_perturbations]
    return fig, (ordered_perturbations, perturbation_counts)


def plot_number_of_DEGs(adata: ad.AnnData, min_total_deg: int = 10, max_count: int = 500, p_value_threshold: float = 0.05) -> plt.Figure:
    """
    Plot the number of differentially expressed genes (DEGs) for each perturbation.

    This function generates a horizontal bar plot showing the count of upregulated and downregulated DEGs
    for each perturbation that has a total number of DEGs above a specified threshold.

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing the DEG summary in `adata.uns`.
    - min_total_deg (int, optional): Minimum number of total DEGs required to include a perturbation in the plot. Default is 10.
    - max_count (int, optional): Maximum count for the x-axis limit. Default is 500.
    - p_value_threshold (float, optional): Adjusted p-value threshold for significance. Default is 0.05.

    Returns:
    - matplotlib.figure.Figure: The matplotlib figure object containing the plot.
    """

    # Define colors for upregulated and downregulated DEGs
    color_up = '#BDE7BD'
    color_down = '#FFB6B3'

    # Filter and sort the summary DataFrame based on the total number of DEGs
    summary_df = adata.uns['Number_of_DEGs_per_perturbation']
    summary_df = summary_df[summary_df['Total_DEGs'] >= min_total_deg]
    summary_df = summary_df.sort_values(by='Total_DEGs', ascending=False)

    # Extract upregulated and downregulated DEG counts
    upregulated = summary_df['Total_Upregulated_DEGs']
    downregulated = summary_df['Total_Downregulated_DEGs']

    # Get the list of perturbations ordered by total DEGs
    ordered_perturbations = summary_df.index.tolist()

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot horizontal bars for each perturbation
    for idx, perturbation in enumerate(ordered_perturbations):
        up_count = upregulated[perturbation]
        down_count = downregulated[perturbation]
        ax.barh(idx, up_count, color=color_up)
        ax.barh(idx, -down_count, color=color_down)

    # Set axis labels
    ax.set_ylabel(f'Perturbation Number (> {min_total_deg} DEGs)')
    ax.set_xlabel(f'Count of Significant Genes (adj. p < {p_value_threshold})')

    # Configure y-ticks and x-ticks
    ax.set_yticks(range(0, len(ordered_perturbations), 100))
    ax.set_xlim(-max_count, max_count)
    ax.set_xticks(np.arange(-max_count, max_count + 1, step=100))
    ax.set_xticklabels([str(abs(int(tick))) if abs(tick) != max_count else ">500" for tick in np.arange(-max_count, max_count + 1, step=100)], fontsize=12)

    # Add legend and vertical line at x=0
    ax.legend(['Upregulated', 'Downregulated'], loc='upper right')
    ax.axvline(0, color='black', linewidth=0.5)

    # Disable grid and adjust subplot parameters
    ax.grid(False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # Display the plot
    plt.show()

    return fig


def plot_degs_dotplot(
    adata: ad.AnnData,
    degs_key: str = "Number_of_DEGs_per_perturbation",
    min_degs: int = 10,
    top_n: int = 20
) -> plt.Figure:
    """
    Plot dot plot of DEG counts from AnnData, labeling top perturbations with arrows.

    Parameters:
    - adata: AnnData object containing DEG results in .uns
    - degs_key: Key in adata.uns containing DEG summary DataFrame
    - min_degs: Minimum number of DEGs required to include a perturbation
    - top_n: Number of top perturbations to label with arrows

    Returns:
    - matplotlib Figure object
    """
    # Extract and sort DEG data
    degs_df = adata.uns[degs_key]
    degs_df = degs_df[degs_df['Total_DEGs'] >= min_degs]
    degs_df = degs_df.sort_values('Total_DEGs', ascending=False)
    
    # Prepare data for plotting
    perturbations = degs_df.index.tolist()
    deg_counts = degs_df['Total_DEGs'].tolist()
    top_perturbations = perturbations[:top_n]

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    x_values = range(len(perturbations))
    
    # Plot all points
    ax.plot(x_values, deg_counts, '.', color='#417dc1')
    
    # Annotation positioning
    annotation_x = 500
    y_offset = 80
    y_positions = [max(deg_counts) - i * y_offset for i in range(top_n)]

    # Add labels for top perturbations
    for idx, pert in enumerate(perturbations):
        if pert in top_perturbations:
            ax.annotate(
                pert,
                xy=(idx, deg_counts[idx]),
                xytext=(annotation_x, y_positions[top_perturbations.index(pert)]),
                textcoords='data',
                ha='left',
                fontsize=14,
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.2)
            )

    # Style plot
    ax.set_ylabel("Number of DEGs", fontsize=12)
    ax.set_xlabel(f"Perturbations Inducing >= {min_degs} DEGs", fontsize=12)
    ax.set_xticks([])
    ax.set_ylim(bottom=min_degs)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid(False)
    
    # Axis styling
    ax.set_facecolor('white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.5)

    plt.tight_layout()
    plt.show()
    return fig


def plot_perturbation_correlation(
    adata: ad.AnnData,
    key: str = "perturbation_correlation",
    use_clustered: bool = True,
    cmap: str = "RdYlBu",
    center: float = 0,
    vmin: float = -.5,
    vmax: float = 0.5,
    figsize: tuple[float, float] = (10, 8)
) -> plt.Figure:
    """
    Visualize correlation matrix using pre-computed clustering if available.
    
    Parameters:
        use_clustered: Whether to use pre-computed clustered data
    """
    corr_data = adata.uns[key]
    
    if use_clustered and 'clustered_dataframe' in corr_data:
        plot_df = corr_data['clustered_dataframe']
        title_suffix = " (Clustered)"
    else:
        plot_df = corr_data['dataframe']
        title_suffix = ""
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_df,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )
    ax.set_title(f"Perturbation Correlation Matrix{title_suffix}")
    return fig

def plot_perturbation_correlation_kde(
    adata: ad.AnnData,
    key: str = "perturbation_correlation",
    figsize: tuple[float, float] = (10, 8)
) -> plt.Figure:
    """
    Visualize the Kernel Density Estimate (KDE) of the perturbation correlation coefficients.

    Parameters:
        adata (ad.AnnData): The AnnData object containing the correlation data.
        key (str): The key in adata.uns where the correlation matrix is stored. Default is "perturbation_correlation".
        figsize (tuple[float, float]): The size of the figure to be created. Default is (10, 8).

    Returns:
        plt.Figure: The figure object containing the KDE plot.
    """
    # Extract the raw correlation matrix from the AnnData object
    corr_data = adata.uns[key]['raw_matrix']
    
    # Create a mask to exclude the diagonal elements (self-correlations)
    mask = np.where(~np.eye(corr_data.shape[0], dtype=bool))
    
    # Flatten the masked correlation data
    corr_data = corr_data[mask].flatten()
    
    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the KDE of the correlation coefficients
    sns.kdeplot(corr_data, ax=ax, cmap="RdYlBu", shade=True, shade_lowest=False)
    
    # Set the title and labels for the axes
    ax.set_title("KDE of Perturbation Correlation")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Density")
    
    # Adjust layout and display options
    plt.tight_layout()
    plt.grid(False)
    plt.show()
    
    return fig


def plot_perturbation_embedding_density(adata: ad.AnnData, groupby: str = "gene_target", perturbations: list[str] = None, title: str = None, figsize: tuple[float, float] = (10, 8), color_map: str = "Reds") -> plt.Figure:
    """
    Visualize the embedding density of perturbations in the UMAP space.

    Parameters:
        adata (ad.AnnData): The AnnData object containing the UMAP data.
        groupby (str): The key in adata.obs that indicates the perturbation group. Default is "gene_target".
        perturbations (list[str]): A list of perturbations to include in the plot. If None, all perturbations will be included.
        title (str): The title of the plot. If None, a default title will be used.
        figsize (tuple[float, float]): The size of the figure to be created. Default is (10, 8).

    Returns:
        plt.Figure: The figure object containing the embedding density plot.
    """
    # Validate input
    validate_anndata(adata, required_obs=[groupby])
    
    # Ensure all perturbations are present in adata.obs[groupby]
    if perturbations is not None:
        if not all(pert in adata.obs[groupby] for pert in perturbations):
            raise ValueError(f"All perturbations must be present in adata.obs[{groupby}], missing: {set(perturbations) - set(adata.obs[groupby])}")
    
    sc.tl.embedding_density(adata, groupby=groupby)
    fig = sc.pl.embedding_density(adata, groupby=groupby, group=perturbations, return_fig=True, color_map=color_map, title=title, figsize=figsize, fg_dotsize = 100)
    plt.show()
    return fig

def plot_etest_results(
    etest_df: pd.DataFrame,
    title: str = "E-test Results",
    point_size: int = 20,
    figsize: tuple = (9, 7),
    palette: dict = None,
    show_legend: bool = True,
    highlight_perturbations: list = None,
    significance_column: str = 'cross_batch_significant',
    pvalue_column: str = 'cross_batch_pvalue_adj',
) -> plt.Figure:
    """
    Visualize the results of an energy-based statistical test (E-test) with enhanced aesthetics.
    
    This function creates a scatter plot of the energy distance vs. statistical significance,
    with points colored by their significance status.
    
    Parameters:
    - etest_df (pd.DataFrame): DataFrame containing E-test results with columns:
        - 'edist': Energy distance values
        - 'pvalue_adj': Adjusted p-values
        - 'significant_adj': Boolean indicator of significance (or 'NTC' for controls)
    - title (str): Plot title. Default is "E-test Results".
    - point_size (int): Size of the scatter points. Default is 20.
    - figsize (tuple): Size of the figure as (width, height). Default is (9, 7).
    - palette (dict): Color mapping for significant, non-significant, and NTC points.
        Default is {True: '#5CB39D', False: '#E67C73', 'NTC': '#367CB7'}.
    - show_legend (bool): Whether to display the legend. Default is True.
    - highlight_perturbations (list): List of perturbation names to highlight with labels.
        Default is None (no highlights).
        
    Returns:
    - matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Ensure the required columns exist
    required_columns = ['edist', 'pvalue_adj', 'significant_adj']
    if not all(col in etest_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in etest_df.columns]
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    
    # To avoid issues with duplicate index labels during plotting, reset the index
    # and store the original index (assumed to be perturbation names) in a new column.
    etest_df = etest_df.copy().reset_index().rename(columns={'index': 'perturbation'})
    
    # Create log-transformed columns if they don't exist
    if 'log_edist' not in etest_df.columns:
        etest_df['log_edist'] = np.log10(etest_df['edist'])
    
    if f'neglog10_{pvalue_column}' not in etest_df.columns:
        etest_df[f'neglog10_{pvalue_column}'] = -np.log10(etest_df[pvalue_column])
    
    #Remove NTC from the dataframe
    etest_df = etest_df[etest_df[significance_column] != 'NTC']

    # Set default color palette if none provided
    if palette is None:
        palette = {
            "True": '#5CB39D',  # Teal for significant
            "False": '#E67C73', # Coral for non-significant
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with enhanced aesthetics
    scatter = sns.scatterplot(
        data=etest_df,
        x='log_edist',
        y=f'neglog10_{pvalue_column}',
        hue=significance_column,
        palette=palette,
        s=point_size,
        alpha=0.8,
        edgecolor='white',
        linewidth=0.5,
        ax=ax
    )
    
    # Remove grid styling
    ax.grid(False)
    
    # Style the axes
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Set labels and title
    ax.set_xlabel('Log10 Energy Distance', fontsize=16)
    ax.set_ylabel('-Log10 Adjusted p-value', fontsize=16)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis='both', labelsize=24)
    
    # Add count of significant perturbations
    sig_count = (etest_df[significance_column] == "True").sum()
    not_sig_count = (etest_df[significance_column] == "False").sum()
    
    # Configure legend to include the significant perturbation count
    if show_legend:
        # Define custom legend labels with the significant count included
        custom_labels = {
            True: f'Significant (n={sig_count})',
            False: f'Non-significant (n={not_sig_count})',
            'NTC': 'Control (NTC)'
        }
        
        handles, labels = ax.get_legend_handles_labels()
        # Update labels ensuring that boolean labels are correctly matched
        updated_labels = []
        for label in labels:
            if label == 'True' or label is True:
                updated_labels.append(custom_labels[True])
            elif label == 'False' or label is False:
                updated_labels.append(custom_labels[False])
            else:
                updated_labels.append(custom_labels.get(label, label))
        
        ax.legend(
            handles, 
            updated_labels,
            frameon=False,
            fontsize=14,
            markerscale=1.2,
            loc='best'
        )
    else:
        ax.get_legend().remove()
    
    # Highlight specific perturbations if requested
    if highlight_perturbations:
        # Find the positions of highlighted perturbations using the 'perturbation' column
        for pert in highlight_perturbations:
            if pert in etest_df['perturbation'].values:
                row = etest_df.loc[etest_df['perturbation'] == pert].iloc[0]
                ax.annotate(
                    pert,
                    xy=(row['log_edist'], row[f'neglog10_{pvalue_column}']),
                    xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=9,
                    color='black',
                    arrowprops=dict(arrowstyle='->', color='black', linewidth=0.8)
                )
    
    plt.tight_layout()
    return fig

def plot_energy_distance_vs_knockdown(
    adata: ad.AnnData,
    n_genes: int = 30,
    perturbation_key: str = "perturbation",
    gene_target_key: str = "gene_target",
    knockdown_key: str = "target_knockdown",
    edist_key: str = "perturbation_edist",
    figsize: tuple[float, float] = (12, 8),
    color_palette: dict = None,
    jitter_width: float = 0.5,
    show_legend: bool = True,
    box_linewidth: float = 1.2,
    box_facecolor: str = "#D0EFF4",
    box_edgecolor: str = "#7ABDE5",
    grid_alpha: float = 0,
    log_scale: bool = False,
    show_concordance_stats: bool = True,
    debug: bool = False
) -> plt.Figure:
    """
    Create box plots of energy distances for top perturbed genes, colored by knockdown level.
    
    Parameters:
    - adata: AnnData object containing perturbation data
    - n_genes: Number of top genes to display (ranked by mean energy distance)
    - perturbation_key: Column in adata.obs containing perturbation identifiers
    - gene_target_key: Column in adata.obs containing gene target names
    - knockdown_key: Column in adata.obs containing target knockdown values (0-1 scale)
    - edist_key: Column in adata.obs containing energy distance values
    - figsize: Dimensions of the output figure
    - color_palette: Dictionary mapping knockdown categories to colors; if None, uses default
    - jitter_width: Width of the jitter for scatter points
    - show_legend: Whether to display the color legend
    - box_linewidth: Line width for the box plots
    - box_facecolor: Fill color for the box plots
    - box_edgecolor: Color for the box plot outlines
    - grid_alpha: Alpha value for grid lines (0 means no grid)
    - log_scale: Whether to use logarithmic scale for y-axis
    - show_concordance_stats: Whether to print statistics about repression-edistance concordance
    - debug: Whether to print detailed debug information
    
    Returns:
    - matplotlib Figure object containing the visualization
    """
    # Validate input columns
    utils.validate_anndata(adata, required_obs=[perturbation_key, gene_target_key, knockdown_key, edist_key])
    
    # Exclude NTC controls if present
    adata_filtered = adata[adata.obs[gene_target_key] != "NTC"].copy()
    
    # Get total number of unique genes for verification
    total_unique_genes = adata_filtered.obs[gene_target_key].nunique()
    if debug:
        print(f"Total unique genes (excluding NTC): {total_unique_genes}")
    
    # Aggregate the data per gene target - replicating the provided example
    # First create a DataFrame from the relevant columns
    df = pd.DataFrame({
        'gene_target': adata_filtered.obs[gene_target_key],
        'perturbation': adata_filtered.obs[perturbation_key],
        'knockdown': adata_filtered.obs[knockdown_key],
        'edist': adata_filtered.obs[edist_key]
    })
    
    # Drop any rows with NaN values to prevent warnings
    df_clean = df.dropna(subset=['knockdown', 'edist'])
    if debug:
        dropped_rows = len(df) - len(df_clean)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NaN values")
    
    # Get all unique gene targets
    all_genes = df_clean[gene_target_key].unique()
    if debug:
        print(f"Unique genes after cleaning: {len(all_genes)}")
    
    # Calculate concordance between repression and energy distance
    if show_concordance_stats:
        # Dictionary to store per-perturbation stats for each gene
        gene_perturbation_stats = {}
        
        # Process each gene separately
        for gene in all_genes:
            gene_data = df_clean[df_clean[gene_target_key] == gene]
            
            # Group by perturbation and calculate mean values
            # Explicitly setting observed=True to avoid pandas warning
            pert_data = gene_data.groupby('perturbation', observed=True).agg({
                'knockdown': 'mean',
                'edist': 'mean'
            }).reset_index()
            
            # Store only if we have enough perturbations
            if len(pert_data) >= 2:
                gene_perturbation_stats[gene] = pert_data
        
        # Now process the concordance statistics
        concordant_genes = 0
        total_genes_analyzed = len(gene_perturbation_stats)
        gene_correlations = {}
        
        # Debug tracking
        genes_with_insufficient_perturbations = []
        genes_with_correlation_errors = []
        genes_with_nan_correlations = []
        
        for gene, pert_data in gene_perturbation_stats.items():
            # Get perturbation with max knockdown and max edist
            try:
                max_kd_idx = pert_data['knockdown'].idxmax()
                max_ed_idx = pert_data['edist'].idxmax()
                
                # Check if they are the same perturbation
                if pert_data.loc[max_kd_idx, 'perturbation'] == pert_data.loc[max_ed_idx, 'perturbation']:
                    concordant_genes += 1
                
                # Calculate correlation if we have at least 3 perturbations
                if len(pert_data) >= 3:
                    # Use scipy's pearsonr which is more robust than np.corrcoef
                    from scipy.stats import pearsonr
                    try:
                        r, p = pearsonr(pert_data['knockdown'], pert_data['edist'])
                        if not np.isnan(r):
                            gene_correlations[gene] = r
                        else:
                            genes_with_nan_correlations.append(gene)
                            if debug:
                                print(f"NaN correlation for gene {gene}")
                    except Exception as e:
                        genes_with_correlation_errors.append(gene)
                        if debug:
                            print(f"Error calculating correlation for gene {gene}: {str(e)}")
                else:
                    genes_with_insufficient_perturbations.append(gene)
            except Exception as e:
                if debug:
                    print(f"Error processing gene {gene}: {str(e)}")
        
        # Print detailed statistics
        if debug:
            print(f"\nDiagnostic Information:")
            print(f"- Total unique genes: {total_unique_genes}")
            print(f"- Genes with sufficient perturbations (>=2): {total_genes_analyzed}")
            print(f"- Genes with insufficient perturbations for correlation (<3): {len(genes_with_insufficient_perturbations)}")
            print(f"- Genes with correlation calculation errors: {len(genes_with_correlation_errors)}")
            print(f"- Genes with NaN correlations: {len(genes_with_nan_correlations)}")
        
        # Print results
        if total_genes_analyzed > 0:
            concordance_rate = (concordant_genes / total_genes_analyzed) * 100
            print(f"\nConcordance between max repression and max energy distance:")
            print(f"  • {concordant_genes}/{total_genes_analyzed} genes ({concordance_rate:.1f}%) have the same sgRNA showing maximum repression and maximum energy distance")
            
            # Calculate average correlation
            if gene_correlations:
                correlation_values = list(gene_correlations.values())
                avg_correlation = np.mean(correlation_values)
                median_correlation = np.median(correlation_values)
                print(f"  • Average Pearson correlation between repression and energy distance: {avg_correlation:.3f}")
                print(f"  • Median Pearson correlation: {median_correlation:.3f}")
                print(f"  • Found valid correlations for {len(gene_correlations)}/{total_genes_analyzed} genes")
    
    # Calculate mean knockdown and energy distance per gene target for display
    gene_stats = df_clean.groupby(gene_target_key, observed=True).agg({
        "knockdown": "mean",
        "edist": "mean"
    }).reset_index()
    
    # Sort by energy distance and select top n genes for display
    gene_stats = gene_stats.sort_values('edist', ascending=False)
    top_genes = gene_stats[gene_target_key].iloc[:n_genes].tolist()
    
    # Filter data to include only cells targeting top genes
    mask = df_clean[gene_target_key].isin(top_genes)
    df_top = df_clean[mask].copy()
    
    # Create knockdown categories for coloring (assuming data is pre-filtered for >30%)
    def categorize_knockdown(kd):
        if kd >= 0.9:
            return '>90%'
        elif kd >= 0.7:
            return '>70%'
        elif kd >= 0.5:
            return '>50%'
        else:
            return '>30%'  # Lowest category since data is pre-filtered
    
    # Calculate mean knockdown per perturbation for the top genes to color dots
    pert_stats = df_top.groupby([gene_target_key, 'perturbation'], observed=True).agg({
        'knockdown': 'mean',
        'edist': 'mean'
    }).reset_index()
    
    # Apply knockdown categorization
    pert_stats['knockdown_category'] = pert_stats['knockdown'].apply(categorize_knockdown)
    
    # Set up color palette
    if color_palette is None:
        color_palette = {
            '>90%': '#EF81AC',  # Pink
            '>70%': '#C294D4',  # Purple
            '>50%': '#C3E2E6',  # Light blue
            '>30%': '#FFDBCC',  # Light orange
        }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a custom boxplot for each gene
    boxprops = dict(linewidth=box_linewidth, facecolor=box_facecolor, edgecolor=box_edgecolor)
    whiskerprops = dict(linewidth=box_linewidth, color=box_edgecolor)
    medianprops = dict(linewidth=box_linewidth + 0.5, color='#444444')
    capprops = dict(linewidth=box_linewidth, color=box_edgecolor)
    
    # Order genes by mean energy distance
    gene_order = gene_stats[gene_target_key].iloc[:n_genes].tolist()
    positions = range(1, len(gene_order) + 1)
    
    # Dictionary to store per-gene data for boxplots
    gene_data = {}
    
    # Collect data points for each gene
    for gene in gene_order:
        gene_perts = pert_stats[pert_stats[gene_target_key] == gene]
        gene_data[gene] = gene_perts['edist'].values
    
    # Create boxplots by gene
    for i, gene in enumerate(gene_order):
        if len(gene_data[gene]) > 0:
            bp = ax.boxplot(
                gene_data[gene],
                positions=[positions[i]],
                widths=0.6,
                patch_artist=True,
                boxprops=boxprops,
                whiskerprops=whiskerprops,
                medianprops=medianprops,
                capprops=capprops,
                showcaps=True,
                showfliers=False
            )
    
    # Make sure all categories are represented in the legend
    handles, labels = [], []
    
    # Add dummy points for each category in the legend
    category_order = ['>90%', '>70%', '>50%', '>30%']
    for category in category_order:
        if category in color_palette:
            color = color_palette[category]
            # Create a dummy point for the legend
            dummy_handle = ax.scatter([], [], color=color, s=50, alpha=0.8, edgecolor='white', linewidth=0.5, label=category)
            handles.append(dummy_handle)
            labels.append(category)
    
    # Add the actual data points
    for i, gene in enumerate(gene_order):
        gene_perts = pert_stats[pert_stats[gene_target_key] == gene]
        
        # Add jitter to x position
        jitter = np.random.uniform(-jitter_width, jitter_width, size=len(gene_perts))
        x_pos = np.full(len(gene_perts), positions[i]) + jitter
        
        # Plot points colored by knockdown category
        for category in category_order:
            if category in color_palette:
                color = color_palette[category]
                category_mask = gene_perts['knockdown_category'] == category
                if category_mask.any():
                    ax.scatter(
                        x_pos[category_mask],
                        gene_perts.loc[category_mask, 'edist'],
                        color=color,
                        s=50,
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=0.5
                    )
    
    # Set y-axis to log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Set labels and ticks
    ax.set_ylabel('Energy Distance', fontsize=14)
    ax.set_xlabel('Gene Target', fontsize=14)
    ax.set_title(f'Energy Distance for Top {n_genes} Genes by Perturbation Effect', fontsize=16, pad=20)
    
    # Set x-ticks to gene names
    ax.set_xticks(positions)
    ax.set_xticklabels(gene_order, rotation=45, ha='right', fontsize=10)
    
    # Set grid styling based on alpha
    if grid_alpha > 0:
        ax.grid(axis='y', linestyle='--', alpha=grid_alpha)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    
    # Style axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # Add legend
    if show_legend and len(handles) > 0:
        ax.legend(
            handles,
            labels,
            title='Knockdown Level',
            fontsize=10,
            title_fontsize=11,
            loc='upper right',
            framealpha=0.9,
            frameon=True,
            edgecolor='#CCCCCC'
        )
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_repression_edist_correlation(
    adata: ad.AnnData,
    perturbation_key: str = "perturbation",
    gene_target_key: str = "gene_target",
    knockdown_key: str = "target_knockdown",
    edist_key: str = "perturbation_edist",
    min_perturbations: int = 3,
    figsize: tuple[float, float] = (10, 6),
    color: str = "#367CB7",
    highlight_color: str = "#E67C73",
    title: str = "Distribution of Repression-Energy Distance Correlations",
    kde: bool = True,
    rug: bool = True,
    include_gene_labels: bool = True,
    label_cutoff: float = 0.7,
    debug: bool = False
) -> plt.Figure:
    """
    Plot the distribution of Pearson correlations between repression levels and energy distances within genes.
    
    Parameters:
    - adata: AnnData object containing perturbation data
    - perturbation_key: Column in adata.obs containing perturbation identifiers
    - gene_target_key: Column in adata.obs containing gene target names
    - knockdown_key: Column in adata.obs containing target knockdown values (0-1 scale)
    - edist_key: Column in adata.obs containing energy distance values
    - min_perturbations: Minimum number of perturbations per gene to calculate correlation
    - figsize: Dimensions of the output figure
    - color: Color for the histogram/KDE
    - highlight_color: Color for genes with high correlation
    - title: Plot title
    - kde: Whether to show kernel density estimate
    - rug: Whether to show rug plot
    - include_gene_labels: Whether to label highly correlated genes
    - label_cutoff: Correlation threshold above which to label genes
    - debug: Whether to print detailed debug information
    
    Returns:
    - matplotlib Figure object containing the visualization
    """
    # Validate input columns
    utils.validate_anndata(adata, required_obs=[perturbation_key, gene_target_key, knockdown_key, edist_key])
    
    # Exclude NTC controls if present
    adata_filtered = adata[adata.obs[gene_target_key] != "NTC"].copy()
    
    # Get total number of unique genes for verification
    total_unique_genes = adata_filtered.obs[gene_target_key].nunique()
    if debug:
        print(f"Total unique genes (excluding NTC): {total_unique_genes}")
    
    # Create DataFrame from the relevant columns
    df = pd.DataFrame({
        'gene_target': adata_filtered.obs[gene_target_key],
        'perturbation': adata_filtered.obs[perturbation_key],
        'knockdown': adata_filtered.obs[knockdown_key],
        'edist': adata_filtered.obs[edist_key]
    })
    
    # Drop any rows with NaN values to avoid warnings
    df_clean = df.dropna(subset=['knockdown', 'edist'])
    if debug:
        dropped_rows = len(df) - len(df_clean)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NaN values")
    
    # Get all unique gene targets
    all_genes = df_clean['gene_target'].unique()
    if debug:
        print(f"Unique genes after cleaning: {len(all_genes)}")
    
    # Dictionary to store per-perturbation stats for each gene
    gene_perturbation_stats = {}
    
    # Process each gene separately
    for gene in all_genes:
        gene_data = df_clean[df_clean['gene_target'] == gene]
        
        # Group by perturbation and calculate mean values
        pert_data = gene_data.groupby('perturbation', observed=True).agg({
            'knockdown': 'mean',
            'edist': 'mean'
        }).reset_index()
        
        # Store only if we have enough perturbations
        if len(pert_data) >= min_perturbations:
            gene_perturbation_stats[gene] = pert_data
    
    # Calculate correlations for each gene with sufficient perturbations
    gene_correlations = {}
    failed_genes = []
    
    # Use scipy's pearsonr for more robust correlation calculation
    from scipy.stats import pearsonr
    
    for gene, pert_data in gene_perturbation_stats.items():
        try:
            # Double-check that we have enough valid perturbations
            if len(pert_data) >= min_perturbations:
                # Check for NaN or constant values
                kd_values = pert_data['knockdown'].values
                ed_values = pert_data['edist'].values
                
                if np.all(np.isfinite(kd_values)) and np.all(np.isfinite(ed_values)) and \
                   np.std(kd_values) > 0 and np.std(ed_values) > 0:
                    # Calculate correlation
                    r, p = pearsonr(kd_values, ed_values)
                    
                    if not np.isnan(r):
                        gene_correlations[gene] = r
                    else:
                        failed_genes.append((gene, "NaN correlation"))
                else:
                    failed_genes.append((gene, "Constant or non-finite values"))
            else:
                failed_genes.append((gene, f"Insufficient perturbations: {len(pert_data)} < {min_perturbations}"))
        except Exception as e:
            failed_genes.append((gene, str(e)))
            if debug:
                print(f"Error calculating correlation for gene {gene}: {str(e)}")
    
    # Diagnostic information
    if debug:
        print(f"\nDiagnostic Information:")
        print(f"- Total unique genes: {total_unique_genes}")
        print(f"- Genes with ≥{min_perturbations} perturbations: {len(gene_perturbation_stats)}")
        print(f"- Genes with valid correlations: {len(gene_correlations)}")
        print(f"- Genes that failed correlation calculation: {len(failed_genes)}")
        
        if failed_genes:
            print("\nSample of failed genes:")
            for i, (gene, reason) in enumerate(failed_genes[:5]):
                print(f"  {i+1}. {gene}: {reason}")
            if len(failed_genes) > 5:
                print(f"  ... and {len(failed_genes) - 5} more")
    
    if not gene_correlations:
        print(f"No genes have {min_perturbations} or more perturbations with valid data for correlation analysis.")
        return None
    
    # Extract correlation values for plotting
    correlations = list(gene_correlations.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram with KDE
    hist_kwargs = {'bins': 20, 'alpha': 0.7, 'edgecolor': 'black', 'color': color}
    kde_kwargs = {'color': color, 'linewidth': 2}
    rug_kwargs = {'color': color, 'alpha': 0.5}
    
    # Plot distribution
    sns.histplot(correlations, kde=kde, ax=ax, **hist_kwargs)
    
    # Add rug plot if requested
    if rug:
        sns.rugplot(correlations, ax=ax, **rug_kwargs)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add mean and median lines
    mean_corr = np.mean(correlations)
    median_corr = np.median(correlations)
    ax.axvline(x=mean_corr, color=highlight_color, linestyle='-', linewidth=2, 
              label=f'Mean = {mean_corr:.3f}')
    ax.axvline(x=median_corr, color='#5CB39D', linestyle='--', linewidth=2,
              label=f'Median = {median_corr:.3f}')
    
    # Add gene labels for high correlations if requested
    if include_gene_labels:
        # Identify genes with high correlation
        high_corr_genes = [(gene, corr) for gene, corr in gene_correlations.items() if corr >= label_cutoff]
        # Sort by correlation value (highest first)
        high_corr_genes.sort(key=lambda x: x[1], reverse=True)
        
        # Add text labels with slight offsets to avoid overlap
        for i, (gene, corr) in enumerate(high_corr_genes):
            # Stagger labels vertically to avoid overlap
            y_pos = ax.get_ylim()[1] * (0.95 - (i % 3) * 0.1)
            ax.text(corr, y_pos, gene, fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=highlight_color, alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('Pearson Correlation between Repression and Energy Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    
    # Improve styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.grid(False)
    
    # Add legend and stats
    ax.legend(frameon=False)
    
    # Add stats annotation
    positive_corr = sum(1 for c in correlations if c > 0)
    total_corr = len(correlations)
    percent_positive = (positive_corr / total_corr) * 100
    
    stats_text = (
        f"Total genes analyzed: {total_corr}/{total_unique_genes}\n"
        f"Positive correlations: {positive_corr} ({percent_positive:.1f}%)\n"
        f"Mean correlation: {mean_corr:.3f}\n"
        f"Median correlation: {median_corr:.3f}"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_edist_vs_degs(
    adata: ad.AnnData,
    edist_key: str = "perturbation_edist",
    deg_key: str = "n_DEGs_perturbation",
    edist_sig_key: str = "perturbation_cross_batch_significant",
    fdr_sig_key: str = "exceeds_ntc_fdr",
    perturbation_key: str = "perturbation",
    figsize: tuple = (10, 8),
    point_size: int = 70,
    highlight_genes: list = None,
    title: str = None,
    show_totals: bool = True,
    include_ntc: bool = False,
    save_path: str = None
) -> plt.Figure:
    """
    Create a publication-quality scatter plot comparing energy distance vs number of DEGs.
    
    Points are colored according to their significance status:
    - Both edist significant and exceeds NTC FDR threshold
    - Only edist significant
    - Only exceeds NTC FDR threshold
    - Neither significant
    
    Parameters:
    ----------
    adata : ad.AnnData
        AnnData object containing perturbation data
    edist_key : str
        Column in adata.obs containing energy distance values
    deg_key : str
        Column in adata.obs containing number of DEGs
    edist_sig_key : str
        Column in adata.obs indicating energy distance significance
    fdr_sig_key : str
        Column in adata.obs indicating NTC FDR significance
    perturbation_key : str
        Column in adata.obs containing perturbation identifiers
    figsize : tuple
        Size of the figure as (width, height)
    point_size : int
        Size of scatter points
    highlight_genes : list
        List of gene names to highlight with labels
    title : str
        Plot title (default: None - no title)
    show_totals : bool
        Whether to show category totals in the legend
    include_ntc : bool
        Whether to include NTC controls in the plot
    save_path : str
        If provided, save the figure to this path
        
    Returns:
    -------
    fig : plt.Figure
        The figure object containing the plot
    """
    # Set plotting style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.edgecolor'] = '#333333'
    
    # Import seaborn for regression with CI
    import seaborn as sns
    
    # Check required columns
    required_cols = [edist_key, deg_key, edist_sig_key, fdr_sig_key, perturbation_key]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create a working copy of the dataframe
    data = adata.obs.copy()
    
    # Filter out NTC controls if not included
    if not include_ntc:
        # Try different methods to identify NTC controls
        if 'gene_target' in data.columns:
            ntc_mask = data['gene_target'] == 'NTC'
        elif perturbation_key in data.columns:
            ntc_mask = data[perturbation_key].str.contains('NTC|NT|non-targeting', case=False, na=False)
        else:
            ntc_mask = pd.Series(False, index=data.index)
            
        # Apply filter if any NTCs found
        if ntc_mask.any():
            data = data[~ntc_mask]
    
    # Aggregate data by perturbation
    # For each perturbation, we take the mean edist and deg values
    # For significance, a perturbation is considered significant if any of its cells are significant
    agg_data = data.groupby(perturbation_key).agg({
        edist_key: 'mean',
        deg_key: 'mean',
        edist_sig_key: 'any',
        fdr_sig_key: 'any'
    }).reset_index()
    
    # Replace any negative values with 0 before log1p to avoid warnings
    agg_data[edist_key] = agg_data[edist_key].clip(lower=0)
    agg_data[deg_key] = agg_data[deg_key].clip(lower=0)
    
    # Apply log1p transformation
    agg_data['log1p_edist'] = np.log1p(agg_data[edist_key])
    agg_data['log1p_degs'] = np.log1p(agg_data[deg_key])
    
    # Define significance categories
    agg_data['category'] = 'neither'
    agg_data.loc[(agg_data[edist_sig_key]) & (~agg_data[fdr_sig_key]), 'category'] = 'edist_only'
    agg_data.loc[(~agg_data[edist_sig_key]) & (agg_data[fdr_sig_key]), 'category'] = 'fdr_only'
    agg_data.loc[(agg_data[edist_sig_key]) & (agg_data[fdr_sig_key]), 'category'] = 'both'
    
    # Define default color scheme if not provided
    color_scheme = {
        'both': '#66c5cc',     # Teal for both significant
        'edist_only': '#f6cf71', # Yellow for edist significant only
        'fdr_only': '#f89c74',  # Orange for FDR significant only
        'neither': '#b3b3b3'    # Gray for neither significant
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor('white')
    
    # Calculate category counts for legend
    category_counts = agg_data['category'].value_counts()
    all_categories = ['neither', 'fdr_only','edist_only', 'both']
    
    # Create more intuitive category labels
    if show_totals:
        category_labels = {
            'both': f"Both significant (n={category_counts.get('both', 0)})",
            'edist_only': f"E-Test Only (n={category_counts.get('edist_only', 0)})",
            'fdr_only': f"# DEGs count only (n={category_counts.get('fdr_only', 0)})",
            'neither': f"Neither (n={category_counts.get('neither', 0)})"
        }
    else:
        category_labels = {
            'both': "Both",
            'edist_only': "E-Test Only",
            'fdr_only': "# DEGs Only",
            'neither': "Neither"
        }
    
    # Print statistics to console instead of on the plot
    total_perturbations = len(agg_data)
    total_sig_edist = sum(category_counts.get(cat, 0) for cat in ['both', 'edist_only'])
    total_sig_fdr = sum(category_counts.get(cat, 0) for cat in ['both', 'fdr_only'])
    total_sig_both = category_counts.get('both', 0)
    
    percentage_any_sig = 100 * (total_sig_edist + total_sig_fdr - total_sig_both) / total_perturbations if total_perturbations > 0 else 0
    percentage_both_sig = 100 * total_sig_both / total_perturbations if total_perturbations > 0 else 0
    
    print(f"Plot statistics:")
    print(f"  • Total perturbations: {total_perturbations}")
    print(f"  • Perturbations with any significance: {percentage_any_sig:.1f}%")
    print(f"  • Perturbations significant in both metrics: {percentage_both_sig:.1f}%")
    
    # Create scatter plot for each category
    for category in all_categories:
        if category in category_counts:
            cat_data = agg_data[agg_data['category'] == category]
            ax.scatter(
                cat_data['log1p_degs'],  # x-axis is now DEGs 
                cat_data['log1p_edist'], # y-axis is now edist
                s=point_size,
                c=color_scheme[category],
                alpha=0.7,
                edgecolors='white',
                linewidths=0.5,
                label=category_labels[category]
            )
    
    # Handle NaN values before calculating correlation
    mask = ~(np.isnan(agg_data['log1p_edist']) | np.isnan(agg_data['log1p_degs']) | 
             np.isinf(agg_data['log1p_edist']) | np.isinf(agg_data['log1p_degs']))
    
    # Calculate correlation for legend
    pearson_r = None
    p_value = None
    
    # Only calculate correlation if we have valid data points
    if mask.sum() >= 2:
        try:
            # Filter out NaN/Inf values for correlation calculation
            x_valid = agg_data.loc[mask, 'log1p_degs']
            y_valid = agg_data.loc[mask, 'log1p_edist']
            
            from scipy.stats import pearsonr
            r, p = pearsonr(x_valid, y_valid)
            pearson_r = r
            p_value = p
            
            # Add regression line with confidence interval using seaborn
            sns.regplot(
                x='log1p_degs',
                y='log1p_edist',
                data=agg_data[mask],
                scatter=False,
                line_kws={'color': '#CCCCFF', 'linewidth': 1.0, 'linestyle': '--'},
                ci=95,
                ax=ax
            )
            
        except Exception as e:
            # Print error but don't crash the function
            print(f"Warning: Could not calculate correlation: {str(e)}")
    else:
        print("Warning: Not enough valid data points to calculate correlation")
    
    # Highlight specific genes if requested
    if highlight_genes and 'gene_target' in adata.obs.columns:
        # Create a mapping from perturbation to gene target
        pert_to_gene = dict(zip(adata.obs[perturbation_key], adata.obs['gene_target']))
        
        # Find perturbations for highlight genes
        for gene in highlight_genes:
            # Find all perturbations targeting this gene
            for pert, target in pert_to_gene.items():
                if target == gene and pert in agg_data[perturbation_key].values:
                    gene_data = agg_data[agg_data[perturbation_key] == pert]
                    if not gene_data.empty:
                        # Swap x and y for annotation coordinates
                        x = gene_data['log1p_degs'].values[0]
                        y = gene_data['log1p_edist'].values[0]
                        ax.annotate(
                            gene,
                            xy=(x, y),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=12,  # Increased font size
                            fontweight='bold',
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="white",
                                ec=color_scheme.get(gene_data['category'].values[0], '#333333'),
                                alpha=0.8
                            )
                        )
    
    # Set axis labels using standard notation instead of subscripts
    ax.set_xlabel('log1p(Number of DEGs)', fontsize=16, labelpad=10)  # Increased font size
    ax.set_ylabel('log1p(Energy Distance)', fontsize=16, labelpad=10)  # Increased font size
    
    # Add title only if specified
    if title:
        ax.set_title(title, fontsize=18, pad=15)  # Increased font size
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=24, width=1.5, length=5, pad=5)  # Increased tick label size
    ax.grid(False)
    
    # Create legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Add Pearson correlation to legend if available
    if pearson_r is not None:
        # Create a blank handle for the correlation text
        from matplotlib.lines import Line2D
        corr_handle = Line2D([], [], color='none')
        handles.append(corr_handle)
        
        # Format p-value string
        p_value_str = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.3f}"
        corr_label = f"Pearson r = {pearson_r:.3f}\np-value = {p_value_str}"
        labels.append(corr_label)
    
    # Add legend with proper positioning - now in upper left
    legend = ax.legend(
        handles=handles,
        labels=labels,
        frameon=False,
        fontsize=13,  # Increased font size
        title="Significance Categories",
        title_fontsize=14,  # Increased font size
        loc='upper left',
        borderaxespad=0.5,
        edgecolor='#CCCCCC'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_pairwise_matrix_clustermap(
    adata: ad.AnnData = None,
    matrix_df: pd.DataFrame = None,
    key: str = None,
    log_transform: bool = False,
    cmap: str = 'magma',
    figsize: tuple = (10, 10),
    method: str = 'average',
    metric: str = 'euclidean',
    title: str = None,
    vmin: float = None,
    vmax: float = None,
    clip_values: Tuple[float, float] = None,
    cbar_vmin: float = None,
    cbar_vmax: float = None,
    show_labels: bool = False,
    fontsize_row: int = 12,
    fontsize_col: int = 12,
    label_rotation_col: int = 90,
    log_scale_colorbar: bool = False,
    dendrogram_ratio: float = 0.1,
    random_state: int = 42,
    cbar_shrink: float = 0.3,
    cbar_pos: Tuple[float, float, float, float] = None
) -> plt.Figure:
    """
    Plot any pairwise matrix from AnnData.uns as a clustermap.
    
    This is a simplified, general-purpose function to plot any pairwise matrix stored in AnnData.uns.
    The matrix can be stored in different formats, and this function will handle them appropriately.
    
    Args:
        adata: AnnData object containing the pairwise matrix.
        matrix_df: DataFrame containing the pairwise matrix (if not provided, will be extracted from adata.uns[key]).
        key: Key in adata.uns containing the pairwise matrix.
        log_transform: Whether to log-transform the matrix values.
        cmap: Colormap to use.
        figsize: Figure size.
        method: Linkage method for clustering.
        metric: Distance metric for clustering.
        title: Title for the plot.
        vmin: Minimum value for colorbar.
        vmax: Maximum value for colorbar.
        clip_values: Tuple of minimum and maximum values to clip the data to before plotting (useful for log scales).
        cbar_vmin: Minimum value for colorbar.
        cbar_vmax: Maximum value for colorbar.
        show_labels: Whether to show labels.
        fontsize_row: Font size for row labels.
        fontsize_col: Font size for column labels.
        label_rotation_col: Rotation for column labels.
        log_scale_colorbar: Whether to use log scale for colorbar.
        dendrogram_ratio: Ratio of dendrogram height to main heatmap.
        random_state: Random seed for ensuring deterministic clustering.
        cbar_shrink: Factor to shrink the colorbar by (only used if cbar_pos is None).
        cbar_location: Location of the colorbar ('right', 'top', 'top-right', 'bottom') (only used if cbar_pos is None).
        cbar_pos: Tuple of (left, bottom, width, height) for precise colorbar positioning. Overrides cbar_location and cbar_shrink if provided.
        
    Returns:
        fig: Figure object.
    """
    import seaborn as sns
    from matplotlib.colors import LogNorm, Normalize
    import matplotlib.colors as mcolors
    from scipy.cluster.hierarchy import linkage
    import numpy as np
    
    # Set random seed for numpy to ensure deterministic results
    np.random.seed(random_state)
    
    # Get pairwise matrix
    if adata is not None and matrix_df is None:
        if key is None:
            raise ValueError("Either matrix_df or both adata and key must be provided")
        if key not in adata.uns:
            raise ValueError(f"Key '{key}' not found in adata.uns")
        
        # Check available keys and provide helpful error message
        if isinstance(adata.uns[key], dict):
            available_keys = list(adata.uns[key].keys())
            print(f"Available keys in adata.uns['{key}']: {available_keys}")
            
            # If dataframe is directly available, use it
            if 'dataframe' in adata.uns[key]:
                print(f"Using 'dataframe' from adata.uns['{key}']")
                matrix_df = adata.uns[key]['dataframe'].copy()
            # Otherwise, try to construct it from matrix and labels
            elif 'matrix' in adata.uns[key]:
                print(f"Using 'matrix' from adata.uns['{key}']")
                matrix = adata.uns[key]['matrix']
                
                # Get labels
                if 'perturbations' in adata.uns[key]:
                    labels = adata.uns[key]['perturbations']
                    if isinstance(labels, (list, np.ndarray)) and len(labels) == matrix.shape[0]:
                        print(f"Using 'perturbations' as labels for the matrix")
                        matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
                    else:
                        # No valid labels found, just use the matrix
                        print(f"No valid labels found, using matrix with default indices")
                        matrix_df = pd.DataFrame(matrix)
                else:
                    # No labels found, just use the matrix with default indices
                    print(f"No labels found, using matrix with default indices")
                    matrix_df = pd.DataFrame(matrix)
            else:
                # No matrix or dataframe found
                raise ValueError(
                    f"Could not find matrix data in adata.uns['{key}']. "
                    f"Expected keys 'dataframe' or 'matrix' but found: {available_keys}. "
                    f"Please provide the correct key or provide the matrix directly via matrix_df."
                )
        else:
            # The key itself might be the matrix
            print(f"Using adata.uns['{key}'] directly as matrix")
            if isinstance(adata.uns[key], np.ndarray):
                matrix_df = pd.DataFrame(adata.uns[key])
            else:
                matrix_df = adata.uns[key]
    
    if matrix_df is None:
        raise ValueError("Either adata and key, or matrix_df must be provided")
    
    # Make a copy of the matrix
    plot_df = matrix_df.copy()
    
    # Ensure numeric values
    plot_df = plot_df.astype(float)
    
    # Fill NaN values with 0
    plot_df = plot_df.fillna(0)
    
    # Clip values if requested
    if clip_values is not None:
        print(f"Clipping values below {clip_values[0]} to {clip_values[0]} and above {clip_values[1]} to {clip_values[1]}")
        plot_df.clip(lower=clip_values[0], upper=clip_values[1], inplace=True)
    
    # Log transform if requested
    if log_transform:
        print(f"Log-transforming values with np.log1p, clipping values below 0 to 0.")
        plot_df.clip(lower=0, inplace=True)
        plot_df = np.log1p(plot_df)
         
    # Pre-compute linkages for clustering
    print(f"Pre-computing linkages for clustering with method='{method}', metric='{metric}'")
    row_linkage = linkage(squareform(plot_df.values.round(8)), method=method, metric=metric)
    col_linkage = row_linkage.copy() if plot_df.shape[0] == plot_df.shape[1] else linkage(squareform(plot_df.T.values.round(8)).T, method=method, metric=metric)
    
    # Set up normalization for the heatmap
    if vmin is None:
        vmin = plot_df.min().min()
    if vmax is None:
        vmax = plot_df.max().max()
    
    # Create normalization for the colorbar
    if log_scale_colorbar:
        min_value = max(vmin, 1e-10)
        max_value = max(vmax, min_value*10)
        print(f"Colorscale: Using LogNorm with vmin={min_value}, vmax={max_value}")
        norm = LogNorm(vmin=min_value, vmax=max_value)
    else:
        print(f"Colorscale: Using linear normalization with vmin={vmin}, vmax={vmax}")
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Configure clustermap parameters
    clustermap_kwargs = {
        'data': plot_df,
        'figsize': figsize,
        'cmap': cmap,
        'row_linkage': row_linkage,
        'col_linkage': col_linkage,
        'row_colors': None,
        'col_colors': None,
        'norm': norm,
        'xticklabels': show_labels,
        'yticklabels': show_labels,
        'dendrogram_ratio': dendrogram_ratio,
        'colors_ratio': 0,
        'tree_kws': {'linewidths': 0.2},
        'row_cluster': True,
        'col_cluster': True,
        'vmin': cbar_vmin,
        'vmax': cbar_vmax,
    }
    
    # Handle colorbar positioning
    if cbar_pos is not None:
        clustermap_kwargs['cbar_pos'] = cbar_pos
    else:
        clustermap_kwargs['cbar_pos'] = (0.98, 0.90, 0.02, 0.1)
        
    
    # Create the clustermap
    g = sns.clustermap(**clustermap_kwargs)
    
    # Configure aesthetics
    if show_labels:
        # Adjust font size and rotation
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=label_rotation_col, fontsize=fontsize_col)
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=fontsize_row)
    
    # Add title if provided
    if title:
        full_title = f"{title} (clustered)"
        g.fig.suptitle(full_title, fontsize=16)
        g.fig.subplots_adjust(top=0.9)
    
    return g.fig


def plot_corrrelation_matrices(
    deg_df: pd.DataFrame = None,
    matrices: Dict[str, pd.DataFrame] = None,
    edistance_df: pd.DataFrame = None,
    adata: ad.AnnData = None,
    figsize: Tuple[float, float] = (20, 12),
    cmaps: List[str] = None,
    titles: List[str] = None,
    remove_diagonal: bool = True,
    show_axis: bool = True,
    tick_label_font_size: int = 8,
    perturbations: List[str] = None,
    save_path: str = None,
    ntc_delimeter: str = 'NTC',
    dpi: int = 300
) -> plt.Figure:
    """
    Plot multiple Jaccard matrices from DEG data in a grid layout.
    
    This function visualizes multiple Jaccard matrices computed with compute_deg_jaccard_matrix
    in a grid of clustered heatmaps. Each matrix is clustered independently.
    
    Parameters
    ----------
    deg_df : pd.DataFrame, optional
        DataFrame containing DEG data. If provided and matrices is None, 
        all five Jaccard matrices will be computed (all, upregulated, 
        downregulated, mismatch, and similarity_score)
    matrices : Dict[str, pd.DataFrame], optional
        Dictionary of pre-computed Jaccard matrices with their labels as keys
    edistance_df : pd.DataFrame, optional
        DataFrame containing energy distance data
    adata : ad.AnnData, optional
        AnnData object to store the matrices in, by default None
    figsize : Tuple[float, float], optional
        Figure size, by default (20, 12)
    cmaps : List[str], optional
        List of colormaps to use for each matrix, by default None
        If None, defaults to ['viridis', 'Blues', 'Reds', 'Purples', 'YlGnBu']
    titles : List[str], optional
        List of titles for each matrix, by default None
        If None, defaults to ['All DEGs', 'Upregulated DEGs', 'Downregulated DEGs', 
                             'Mismatched DEGs', 'Similarity Score']
    remove_diagonal : bool, optional
        Whether to remove the diagonal from the matrices, by default True
    show_axis : bool, optional
        Whether to show axis ticks and labels for the heatmaps
    tick_label_font_size : int, optional
        Font size for the tick labels, by default 8
    perturbations : List[str], optional
        List of perturbations to include in the matrices
    save_path : str, optional
        Path to save the figure, by default None
        If provided, the figure will be saved as both SVG and PNG
    dpi : int, optional
        DPI for the saved PNG figure, by default 300
    
    Returns
    -------
    plt.Figure
        The figure containing the grid of clustered heatmaps
    
    Examples
    --------
    >>> # Using DEG dataframe to compute all matrices
    >>> fig = plot_deg_jaccard_matrices(deg_df=deg_dataframe)
    >>> 
    >>> # Using pre-computed matrices
    >>> matrices = {
    >>>     'All DEGs': jaccard_all,
    >>>     'Upregulated DEGs': jaccard_up,
    >>>     'Downregulated DEGs': jaccard_down
    >>> }
    >>> fig = plot_deg_jaccard_matrices(matrices=matrices)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Define default colormaps and titles
    default_cmaps = ['Greens', 'Blues', 'Reds', 'Purples', 'YlGnBu']
    default_titles = ['All DEGs', 'Upregulated DEGs', 'Downregulated DEGs', 
                      'Mismatched DEGs', 'Similarity Score']
    
    # Compute matrices if not provided
    if matrices is None and deg_df is not None:
        matrices = {}
        matrices['Jaccard Index (All DEGs)'] = psp.da.compute_deg_jaccard_matrix(deg_df, adata=adata, comparison_type='all')
        matrices['Jaccard Index (Upregulated DEGs)'] = psp.da.compute_deg_jaccard_matrix(deg_df, adata=adata, comparison_type='upregulated')
        matrices['Jaccard Index (Downregulated DEGs)'] = psp.da.compute_deg_jaccard_matrix(deg_df, adata=adata, comparison_type='downregulated')
        matrices['Jaccard Index (Mismatched DEGs)'] = psp.da.compute_deg_jaccard_matrix(deg_df, adata=adata, comparison_type='mismatch')
        matrices['Similarity Score'] = psp.da.compute_deg_jaccard_matrix(deg_df, adata=adata, comparison_type='similarity_score')
    elif matrices is None:
        raise ValueError("Either deg_df or matrices must be provided")
    
    # Subset to requested perturbations if provided
    if perturbations is not None:
        perturbations = [p for p in perturbations if p != ntc_delimeter]
        matrices = {k: df.loc[perturbations, perturbations] for k, df in matrices.items()}
        if edistance_df is not None:
            perturbations.append(ntc_delimeter)
            edistance_df = edistance_df.loc[perturbations, perturbations]
    
    # Set up figure and grid
    fig = plt.figure(figsize=figsize)
    num_matrices = len(matrices)
    
    # Use fixed layout: 2 rows and 3 columns
    grid_rows, grid_cols = 2, 3
    gs = GridSpec(grid_rows, grid_cols, figure=fig)
    
    # Use provided titles and cmaps or default ones
    if titles is None:
        titles = list(matrices.keys())
    if cmaps is None:
        cmaps = default_cmaps[:num_matrices]
        # If we have more matrices than default cmaps, cycle through them
        if num_matrices > len(default_cmaps):
            cmaps = [default_cmaps[i % len(default_cmaps)] for i in range(num_matrices)]
    
    # Plot each matrix
    for i, (title, cmap) in enumerate(zip(titles, cmaps)):
        if title not in matrices:
            continue
            
        matrix = matrices[title].copy()
        
        # Remove diagonal if requested
        if remove_diagonal:
            if title == 'Similarity Score':
                # For similarity score, use higher value to clearly separate diagonal
                matrix = matrix - 2 * np.eye(len(matrix))
            elif title != 'Jaccard Index (Mismatched DEGs)':  # Mismatch already has zeros on diagonal
                matrix = matrix - np.eye(len(matrix))
        
        # Create subplot at fixed grid position
        row = i // grid_cols
        col = i % grid_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Create a clustermap without grid lines
        g = sns.clustermap(matrix, cmap=cmap, figsize=(8, 8), linewidths=0)
        plt.close()  # Close the clustermap figure
        
        # Plot the clustered data in our subplot
        im = ax.imshow(g.data2d, cmap=cmap, aspect='equal', interpolation='nearest')
        ax.set_title(title, fontsize=16)
        # Always disable any grid lines
        ax.grid(False)
        # Show or hide ticks/labels
        if show_axis:
            ax.set_xticks(range(len(g.data2d.columns)))
            ax.set_yticks(range(len(g.data2d.index)))
            ax.set_xticklabels(g.data2d.columns, rotation=90, fontsize=tick_label_font_size)
            ax.set_yticklabels(g.data2d.index, fontsize=tick_label_font_size)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Plot energy distance if provided (will occupy the 6th slot of the 2x3 grid)
    if edistance_df is not None:
        # Place in the last cell (6th) of the 2x3 grid
        idx = len(matrices)
        row = idx // grid_cols
        col = idx % grid_cols
        ax = fig.add_subplot(gs[row, col])
        # Prepare energy distance matrix: clip and log1p
        ed = edistance_df.copy()
        ed = ed.clip(lower=0)
        ed = np.log1p(ed)
        # Cluster the energy-distance matrix without grid lines
        g_ed = sns.clustermap(ed, cmap='flare', figsize=(8, 8), linewidths=0)
        plt.close(g_ed.fig)
        # Plot clustered data on our axis with equal aspect
        im_ed = ax.imshow(g_ed.data2d, cmap='flare', aspect='equal', interpolation='nearest')
        ax.set_title('Log1p Energy Distance', fontsize=16)
        # Always disable grid lines
        ax.grid(False)
        # Show or hide ticks/labels
        if show_axis:
            ax.set_xticks(range(g_ed.data2d.shape[1]))
            ax.set_xticklabels(g_ed.data2d.columns, rotation=90, fontsize=8)
            ax.set_yticks(range(g_ed.data2d.shape[0]))
            ax.set_yticklabels(g_ed.data2d.index, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        # Colorbar
        divider_ed = make_axes_locatable(ax)
        cax_ed = divider_ed.append_axes('right', size='3%', pad=0.05)
        plt.colorbar(im_ed, cax=cax_ed)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Remove extension if present
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        fig.savefig(f"{base_path}.svg")
        fig.savefig(f"{base_path}.png", dpi=dpi)
    
    return fig