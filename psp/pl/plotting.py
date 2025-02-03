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

def plot_cells_per_perturbation(adata: ad.AnnData, perturbation_key: str = 'gene_target', perturbed_key: str = 'perturbed', highlight_threshold: int = 100, y_max: int = 600) -> plt.Figure:
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
    ax.set_xlabel('Number of Perturbations', fontsize=12)
    ax.set_ylabel('Cells per Perturbation', fontsize=12)
    
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
    print(f"{num_perturbations_gt_threshold}/{len(counts)} ({100*num_perturbations_gt_threshold/len(counts):.2f}%) Perturbations have > {highlight_threshold} cells")
    
    # Adjust y-axis limits and label the top as "y_max+"
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, len(counts) + 1)
    y_ticks = list(range(0, y_max + 1, 100))
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
    sns.violinplot(data=umis_per_cell, ax=ax, inner='box', color='#EDF6FF', linewidth=2, saturation=1)

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
    invert_y: bool = False
) -> plt.Figure:
    """
    Plots sorted bar charts for knockdown efficiency or cell counts per sgRNA.
    
    Parameters:
    - data: Pandas Series with sgRNA names as index and values to plot
    - ylabel: Label for Y-axis
    - title: Plot title
    - repression_threshold: Optional threshold line for repression percentage
    - cells_threshold: Optional threshold line for cell counts
    - label_interval: Interval for x-axis labels
    - invert_y: Whether to invert Y-axis
    
    Returns:
    - matplotlib Figure object
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
        ax.invert_yaxis()
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='x', visible=False)
    
    plt.xticks(ticks=range(0, len(keys), label_interval), 
               labels=[keys[i] for i in range(0, len(keys), label_interval)])
    
    plt.tight_layout()
    return fig


def plot_percentage_perturbations_by_repression(
    adata: ad.AnnData,
    perturbation_col: str = "gene_target",
    knockdown_col: str = "total_knockdown",
    knockdown_threshold: float = 0.3,
    figsize: tuple[float, float] = (5, 5),
    line_color: str = "#367CB7",
    fill_color: str = "#EDF6FF",
    threshold_color: str = "#5CB39D"
) -> plt.Figure:
    """
    Visualize the cumulative percentage of perturbations achieving specific knockdown levels.
    
    Parameters:
    - adata: AnnData object containing perturbation data
    - perturbation_col: Column in adata.obs containing perturbation identifiers
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
    
    # Calculate mean knockdown per perturbation
    total_knockdown = adata.obs.groupby(perturbation_col)[knockdown_col].mean()
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
    ax.set_xlabel('Number of Perturbations', fontsize=12)
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
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
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
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
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


def plot_filtered_genes_inverted(results_dict, p_value_threshold=0.05, l2fc_threshold=0, deg_cutoff=25, min_total_deg=10):
    significant_counts = {}

    for perturbation, df in results_dict.items():
        p_values = df['padj']
        l2fc = df['log2FoldChange']
        upregulated = (p_values < p_value_threshold) & (l2fc > l2fc_threshold)
        downregulated = (p_values < p_value_threshold) & (l2fc < -l2fc_threshold)
        total_deg = upregulated.sum() + downregulated.sum()
        if total_deg >= min_total_deg:  # Filter to show only perturbations with more than min_total_deg DEGs
            significant_counts[perturbation] = (upregulated.sum(), downregulated.sum())

    if not significant_counts:
        print("No perturbations with significant DEGs above the specified threshold.")
        return

    max_count = 500
    # Sort in ascending order
    ordered_perturbations = sorted(significant_counts.keys(), key=lambda x: abs(sum(significant_counts[x])),reverse=True)

    fig, ax = plt.subplots(figsize=(6, 6))  # Inverted dimensions
    for idx, perturbation in enumerate(ordered_perturbations):
        total_deg = significant_counts[perturbation][0] + significant_counts[perturbation][1]
        color_up = '#BDE7BD' if total_deg > deg_cutoff else '#BDE7BD'
        color_down = '#FFB6B3' if total_deg > deg_cutoff else '#FFB6B3'
        ax.barh(idx, significant_counts[perturbation][0], color=color_up)  # Horizontal bar for upregulated
        ax.barh(idx, -significant_counts[perturbation][1], color=color_down)  # Horizontal bar for downregulated
    
    ax.set_ylabel('Perturbation Number (>10 DEGs)')
    ax.set_xlabel(f'Count of Significant Genes (p-value < {p_value_threshold})')

    # Place y-ticks every 100 perturbations and flip them
    ax.set_yticks(range(0, len(ordered_perturbations), 100))

    ax.set_xlim(-max_count, max_count)
    ax.set_xticks(np.arange(-max_count, max_count + 1, step=100))  # Set x-ticks every 100
    ax.set_xticklabels([str(abs(int(tick))) if abs(tick) != max_count else ">500" for tick in np.arange(-max_count, max_count + 1, step=100)], fontsize=12)
    ax.legend(['Upregulated', 'Downregulated'], loc='upper right')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.show()
    return fig, (ordered_perturbations, [significant_counts[pert] for pert in ordered_perturbations])


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