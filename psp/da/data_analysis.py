import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map
from functools import partial
import anndata as ad
import scanpy as sc
from scipy.cluster import hierarchy
from psp.utils import get_ntc_view, validate_anndata
from typing import Tuple, Dict, List, Optional
import pymde 
import plotly.io as pio 
import igraph as ig 
import plotly.express as px 
import leidenalg 
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from collections import defaultdict
import scperturb as scp
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm.auto import tqdm
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import issparse
import io, contextlib
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scperturb as scp
from scipy.spatial.distance import squareform, pdist
from tqdm_joblib import tqdm_joblib
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from joblib import parallel_backend
from tqdm.auto import tqdm
from scipy import sparse
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scperturb as scp
from scipy.spatial.distance import squareform, pdist
from tqdm_joblib import tqdm_joblib
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from joblib import parallel_backend
from tqdm.auto import tqdm
from scipy import sparse
import itertools
import random
import json

def _preprocess_for_etest(adata):
    """
    Preprocess AnnData object for energy distance calculations.
    
    This function prepares single-cell data for energy distance analysis by:
    1. Creating a copy of the original data
    2. Setting counts as the main matrix
    3. Filtering low-abundance genes
    4. Identifying highly variable genes
    5. Normalizing and scaling data
    6. Performing dimensionality reduction (PCA)
    7. Computing a neighborhood graph
    
    Parameters:
        adata (AnnData): Input AnnData object with raw counts in .layers['counts']
        
    Returns:
        AnnData: Preprocessed copy of the input data with PCA and neighbors computed
    """
    adata_edist = adata.copy()
    adata_edist.X = adata_edist.layers['counts'].copy()
    sc.pp.filter_genes(adata_edist, min_cells=3)
    n_var_max = 2000
    sc.pp.highly_variable_genes(adata_edist, n_top_genes=n_var_max, flavor='seurat_v3', subset=False, layer='counts')
    sc.pp.normalize_total(adata_edist, inplace=True)
    sc.pp.log1p(adata_edist)
    sc.pp.scale(adata_edist)
    sc.pp.pca(adata_edist, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata_edist)
    return adata_edist

def _subsample_adata_NTCs(adata, num_ntc_cells, random_seed=42):
    """
    Subsample non-targeting control (NTC) cells to a specific number.
    
    Randomly selects a subset of NTC cells while keeping all perturbed cells.
    This helps balance the dataset for downstream statistical analysis.
    
    Parameters:
        adata (AnnData): Input AnnData object
        num_ntc_cells (int): Number of NTC cells to retain
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        AnnData: Subsampled data with reduced number of NTC cells
    """
    np.random.seed(random_seed)
    ntc_cells = adata[adata.obs['perturbed'] == "False"].obs.index
    subsampled_ntc_cells = np.random.choice(ntc_cells, size=(len(ntc_cells) - num_ntc_cells), replace=False)
    adata_subsampled = adata[~adata.obs.index.isin(subsampled_ntc_cells)]
    return adata_subsampled

def _etest(subsampled_adata, seed=42, control_label='NTC', obsm_key='X_pca', dist='sqeuclidean', n_jobs=-1, verbose=True, runs=10000, correction_method='fdr_bh'):
    """
    Perform an energy-based statistical test to identify significant perturbations.
    
    This function:
    1. Subsamples NTC cells to a fixed number (400)
    2. Runs the energy test comparing each perturbation to controls
    3. Adjusts p-values for multiple testing
    4. Adds negative log10 p-values for visualization
    
    Parameters:
        subsampled_adata (AnnData): Preprocessed AnnData object
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        control_label (str, optional): Label of control perturbation. Defaults to 'NTC'.
        obsm_key (str, optional): Key in .obsm for coordinates. Defaults to 'X_pca'.
        dist (str, optional): Distance metric for energy test. Defaults to 'sqeuclidean'.
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to -1 (all cores).
        verbose (bool, optional): Whether to show progress. Defaults to True.
        runs (int, optional): Number of permutations for the test. Defaults to 10000.
        correction_method (str, optional): Method for multiple testing correction. Defaults to 'fdr_bh'.
        
    Returns:
        DataFrame: Results containing energy test statistics and adjusted p-values for each perturbation
    """
    adata_subsampled = _subsample_adata_NTCs(subsampled_adata, num_ntc_cells=400, random_seed=seed)
    etest = scp.etest(adata_subsampled, obs_key='perturbation', control=control_label, obsm_key=obsm_key, dist=dist, n_jobs=n_jobs, verbose=verbose, runs=runs, correction_method=correction_method)
    etest.loc[etest.index==control_label, 'significant_adj'] = control_label
    etest['neglog10_pvalue_adj'] = -np.log10(etest['pvalue_adj'])
    return etest

def _apply_cross_batch_correction(
    etest_df: pd.DataFrame,
    pvalue_col: str = "pvalue",
    correction_method: str = "fdr_bh",
    alpha: float = 0.05,
    control_label: str = "NTC"
) -> pd.DataFrame:
    """
    Apply multiple testing correction across all batches in combined etest results.
    
    This function takes the concatenated etest results from multiple batches and
    applies a consistent multiple testing correction across all perturbations,
    regardless of which batch they come from. This ensures a uniform false discovery
    rate control across the entire experiment.
    
    Parameters:
    - etest_df (pd.DataFrame): Combined DataFrame with etest results from multiple batches
    - pvalue_col (str): Column name containing the uncorrected p-values. Default is "pvalue"
    - correction_method (str): Method for multiple testing correction. Default is "fdr_bh"
        Options include: "bonferroni", "sidak", "holm", "fdr_bh" (Benjamini-Hochberg), etc.
    - alpha (float): Significance threshold. Default is 0.05
    - control_label (str): Label for control entries to exclude from correction. Default is "NTC"
    
    Returns:
    - pd.DataFrame: Updated DataFrame with cross-batch corrected p-values and significance
    
    Notes:
    - The function creates new columns:
      * 'cross_batch_pvalue_adj': The cross-batch corrected p-values
      * 'cross_batch_significant': Boolean indicator of significance based on corrected p-values
      * 'cross_batch_neglog10_pvalue_adj': Negative log10 of corrected p-values for visualization
    - Original columns are preserved
    - Control entries (e.g., NTC) are excluded from p-value adjustment and marked separately
    - When batched, there may be multiple control entries, one per batch
    """
    # Make a copy to avoid modifying the original
    result_df = etest_df.copy()
    
    # Ensure the pvalue column exists
    if pvalue_col not in result_df.columns:
        raise ValueError(f"Column '{pvalue_col}' not found in the DataFrame")
    
    # Identify control entries - handle both single and multi-batch scenarios
    # Controls can appear once per batch, but always with the exact control_label as index
    is_control = result_df.index == control_label
    
    # Also look for batch-specific control entries which might have been modified with a suffix
    # or appear with edist = 0 or other special identifiers
    if 'edist' in result_df.columns:
        # Controls often have energy distance = 0 since they're compared to themselves
        is_control = is_control | (result_df['edist'] == 0)
    
    # If significant_adj column exists, check for controls marked there
    if 'significant_adj' in result_df.columns:
        # In the _etest function, controls are marked with control_label in significant_adj
        is_control = is_control | (result_df['significant_adj'] == control_label)
    
    # Create mask for entries to correct (exclude controls)
    correction_mask = ~is_control
    
    # For compatibility, initialize columns with consistent types
    result_df['cross_batch_pvalue_adj'] = 1.0  # Default to 1.0 (not significant)
    result_df['cross_batch_significant'] = False  # Default to False (not significant)
    
    if correction_mask.sum() > 0:  # Only proceed if we have non-control entries
        # Extract p-values excluding controls
        pvalues_to_correct = result_df.loc[correction_mask, pvalue_col].values
        
        # Apply multiple testing correction
        reject, pvals_corrected, _, _ = multipletests(
            pvalues_to_correct, 
            alpha=alpha, 
            method=correction_method
        )
        
        # Add the corrected values back to the DataFrame (only for non-control entries)
        result_df.loc[correction_mask, 'cross_batch_pvalue_adj'] = pvals_corrected
        result_df.loc[correction_mask, 'cross_batch_significant'] = reject
    
    # Add negative log10 values for visualization
    result_df['cross_batch_neglog10_pvalue_adj'] = -np.log10(
        result_df['cross_batch_pvalue_adj'].fillna(1.0)
    )
    
    # Handle control entries - use the same pattern as in the original code
    # This matches what's done in the function _etest where:
    # etest.loc[etest.index==control_label, 'significant_adj'] = control_label
    if is_control.sum() > 0:
        # Keep pvalue_adj as 1.0 for controls
        # Set significant to the control label (this is what the original code did)
        result_df.loc[is_control, 'cross_batch_significant'] = control_label
    
    # Count significant perturbations (excluding controls)
    sig_count = ((result_df['cross_batch_significant'] == True) & correction_mask).sum()
    total_tests = correction_mask.sum()
    
    print(f"Cross-batch correction ({correction_method}):")
    print(f"  • Significant perturbations: {sig_count}/{total_tests} ({sig_count/total_tests*100:.2f}%)")
    print(f"  • Alpha threshold: {alpha}")
    print(f"  • Control entries excluded from correction: {is_control.sum()}")
    print(f"  • Total entries examined: {len(result_df)}")
    
    return result_df


# This was taken from scPerturb (Peidli et al. 2024) though I modified it to ensure deterministic results over the same random seed
def _etest_deterministic(adata, seed=42, obs_key='perturbation', control='NTC', 
                         obsm_key='X_pca', dist='sqeuclidean', n_jobs=-1, 
                         verbose=True, runs=10000, correction_method='fdr_bh', alpha=0.05):
    """
    A deterministic version of energy test that ensures the same set of permutations
    is used each time the function is called with the same seed.
    
    Parameters:
        adata: AnnData object
        seed: Random seed for reproducibility
        obs_key: Key in adata.obs for perturbation labels
        control: Label for control cells
        obsm_key: Key in adata.obsm for coordinates
        dist: Distance metric to use
        n_jobs: Number of jobs for parallelization
        verbose: Whether to show progress
        runs: Number of permutations
        correction_method: Method for multiple testing correction
        alpha: Significance threshold
        
    Returns:
        DataFrame with energy test results
    """
    groups = pd.unique(adata.obs[obs_key])
    control = [control] if isinstance(control, str) else control
    
    # Precompute pairwise distances
    pwds = {}
    for group in groups:
        x = adata[adata.obs[obs_key].isin([group] + control)].obsm[obsm_key].copy()
        pwd = pairwise_distances(x, x, metric=dist)
        pwds[group] = pwd
    
    # Set up deterministic permutation seeds
    master_rng = np.random.RandomState(seed)
    
    # Pre-generate all permutation seeds - this is the key to ensuring
    # the same set of permutations is used each time
    permutation_seeds = [master_rng.randint(0, 2**32) for _ in range(runs)]
    
    # Function to run a permutation with a specific seed
    def one_permutation(perm_seed):
        # Create a new RNG with this specific seed
        perm_rng = np.random.RandomState(perm_seed)
        
        df = pd.DataFrame(index=groups, columns=['edist'], dtype=float)
        M = np.sum(adata.obs[obs_key].isin(control))  # number of cells in control group
        
        for group in groups:
            if group in control:
                # Nothing to test here
                df.loc[group] = 0
                continue
                
            N = np.sum(adata.obs[obs_key]==group)
            
            # Get labels and shuffle them with this permutation's RNG
            labels = adata.obs[obs_key].values[adata.obs[obs_key].isin([group] + control)]
            shuffled_labels = perm_rng.permutation(labels)
            
            # Calculate energy distance using precomputed pairwise distances
            sc_pwd = pwds[group]
            idx = shuffled_labels==group
            
            # Calculate energy distance components
            factor = N / (N-1)  # sample_correct=True
            factor_c = M / (M-1)
            delta = np.sum(sc_pwd[idx, :][:, ~idx]) / (N * M)
            sigma = np.sum(sc_pwd[idx, :][:, idx]) / (N * N) * factor
            sigma_c = np.sum(sc_pwd[~idx, :][:, ~idx]) / (M * M) * factor_c
            
            edistance = 2 * delta - sigma - sigma_c
            df.loc[group] = edistance
            
        return df
    
    # Run the permutations in parallel, each with its pre-determined seed
    progress_func = tqdm if verbose else lambda x: x
    permutation_results = Parallel(n_jobs=n_jobs)(
        delayed(one_permutation)(seed) for seed in progress_func(permutation_seeds)
    )
    
    # Calculate the original (non-permuted) energy distances
    original_df = pd.DataFrame(index=groups, columns=['edist'], dtype=float)
    M = np.sum(adata.obs[obs_key].isin(control))
    
    for group in groups:
        if group in control:
            original_df.loc[group] = 0
            continue
            
        N = np.sum(adata.obs[obs_key]==group)
        
        # Get the original (non-shuffled) labels
        labels = adata.obs[obs_key].values[adata.obs[obs_key].isin([group] + control)]
        
        # Calculate energy distance
        sc_pwd = pwds[group]
        idx = labels==group
        
        factor = N / (N-1)
        factor_c = M / (M-1)
        delta = np.sum(sc_pwd[idx, :][:, ~idx]) / (N * M)
        sigma = np.sum(sc_pwd[idx, :][:, idx]) / (N * N) * factor
        sigma_c = np.sum(sc_pwd[~idx, :][:, ~idx]) / (M * M) * factor_c
        
        edistance = 2 * delta - sigma - sigma_c
        original_df.loc[group] = edistance
    
    # Count how many permutations had greater or equal energy distance
    # (indicates the permutation looks more extreme than the actual data)
    count_greater = np.zeros(len(groups), dtype=int)
    
    for i, group in enumerate(groups):
        for perm_result in permutation_results:
            if perm_result.loc[group, 'edist'] >= original_df.loc[group, 'edist']:
                count_greater[i] += 1
    
    # Calculate p-values
    pvalues = pd.Series(np.clip(count_greater, 1, np.inf) / runs, index=groups)
    
    # Apply multiple testing correction
    significant_adj, pvalue_adj, _, _ = multipletests(
        pvalues.values, alpha=alpha, method=correction_method
    )
    
    # Compile results into a DataFrame
    tab = pd.DataFrame({
        'edist': original_df['edist'],
        'pvalue': pvalues,
        'significant': pvalues < alpha,
        'pvalue_adj': pvalue_adj,
        'significant_adj': significant_adj
    }, index=groups)
    
    return tab

def _store_etest_results_in_anndata(adata: ad.AnnData, etest_results: pd.DataFrame, 
                                   key_added: str, perturbation_key: str) -> None:
    """
    Store E-test results in AnnData object for easy access.
    
    Parameters:
        adata: AnnData object to modify
        etest_results: DataFrame containing E-test results
        key_added: Key under which to store results in adata.uns
        perturbation_key: Key in adata.obs that identifies perturbations
    """
    # Create a copy to avoid modifying the original
    results_df = etest_results.copy()
    
    # Before converting, capture the original boolean values
    significance_map = {}
    if 'significant_adj' in results_df.columns:
        significance_map['batch'] = {idx: (val is True) for idx, val in results_df['significant_adj'].items() 
                                   if isinstance(val, bool)}
    
    if 'cross_batch_significant' in results_df.columns:
        significance_map['cross_batch'] = {idx: (val is True) for idx, val in results_df['cross_batch_significant'].items() 
                                        if isinstance(val, bool)}
    
    # Ensure all data types are h5ad-compatible
    for col in results_df.columns:
        # Check if column contains boolean values
        if results_df[col].dtype == bool:
            results_df[col] = results_df[col].astype(int)
        # Handle mixed types in significant columns
        elif col == 'significant_adj' or col == 'cross_batch_significant':
            results_df[col] = results_df[col].astype(str)
    
    # Store preprocessed results in adata.uns
    adata.uns[key_added] = results_df
    
    # Determine if cross-batch correction was applied
    has_cross_batch = 'cross_batch_pvalue_adj' in results_df.columns
    
    # Count significant perturbations using the original boolean values
    if has_cross_batch and 'cross_batch' in significance_map:
        n_significant = sum(significance_map['cross_batch'].values())
    elif 'batch' in significance_map:
        n_significant = sum(significance_map['batch'].values())
    else:
        n_significant = 0
    
    # Store parameters
    adata.uns[key_added + '_params'] = {
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'perturbation_key': perturbation_key,
        'n_perturbations': len(results_df),
        'n_significant': n_significant,
        'has_cross_batch_correction': has_cross_batch
    }
    
    # Map results to all cells
    perturbations = adata.obs[perturbation_key].unique()
    
    # Initialize columns in obs
    adata.obs['perturbation_edist'] = np.nan
    adata.obs['perturbation_pvalue'] = np.nan
    adata.obs['perturbation_significant'] = False
    
    # Add cross-batch columns if available
    if has_cross_batch:
        adata.obs['perturbation_cross_batch_pvalue'] = np.nan
        adata.obs['perturbation_cross_batch_significant'] = False
    
    # Helper function to get scalar value safely
    def get_scalar_value(value):
        """Extract scalar value from potential arrays, Series, or scalar values"""
        if hasattr(value, 'iloc') and len(value) > 0:  # For Series
            return value.iloc[0]
        elif hasattr(value, 'item'):  # For numpy arrays and some pandas objects
            return value.item()
        elif isinstance(value, (list, np.ndarray)) and len(value) > 0:  # For lists/arrays
            return value[0]
        else:
            return value  # Already scalar
    
    # Fill in values for each perturbation - using direct indexing to avoid reindexing issues
    for pert in perturbations:
        if pert in results_df.index:
            # Get indices of cells with this perturbation
            mask = adata.obs[perturbation_key] == pert
            indices = np.where(mask)[0]
            
            # Get results for this perturbation, ensuring we have scalar values
            edist = get_scalar_value(results_df.loc[pert, 'edist'])
            pvalue = get_scalar_value(results_df.loc[pert, 'pvalue_adj'])
            is_significant = significance_map.get('batch', {}).get(pert, False)
            
            # Assign values to cells using iloc for positional indexing (avoids index issues)
            adata.obs.iloc[indices, adata.obs.columns.get_loc('perturbation_edist')] = edist
            adata.obs.iloc[indices, adata.obs.columns.get_loc('perturbation_pvalue')] = pvalue
            adata.obs.iloc[indices, adata.obs.columns.get_loc('perturbation_significant')] = is_significant
            
            # Add cross-batch results if available
            if has_cross_batch:
                cross_batch_pvalue = get_scalar_value(results_df.loc[pert, 'cross_batch_pvalue_adj'])
                cross_batch_significant = significance_map.get('cross_batch', {}).get(pert, False)
                
                adata.obs.iloc[indices, adata.obs.columns.get_loc('perturbation_cross_batch_pvalue')] = cross_batch_pvalue
                adata.obs.iloc[indices, adata.obs.columns.get_loc('perturbation_cross_batch_significant')] = cross_batch_significant
    
    # Print diagnostic information
    if has_cross_batch:
        print(f"Cross-batch correction (fdr_bh):")
        print(f"  • Significant perturbations: {n_significant}/{len(results_df)} ({n_significant/len(results_df)*100:.2f}%)")
        print(f"  • Alpha threshold: 0.05")
    
    print(f"E-test results stored in AnnData:")
    print(f"  • adata.uns['{key_added}'] - Complete results DataFrame")
    print(f"  • adata.obs['perturbation_edist'] - Energy distance for each cell's perturbation")
    print(f"  • adata.obs['perturbation_pvalue'] - P-value for each cell's perturbation (batch-specific)")
    print(f"  • adata.obs['perturbation_significant'] - Whether cell's perturbation is significant (batch-specific)")
    
    if has_cross_batch:
        print(f"  • adata.obs['perturbation_cross_batch_pvalue'] - Cross-batch corrected p-value")
        print(f"  • adata.obs['perturbation_cross_batch_significant'] - Cross-batch significance")

def compute_etest(adata: ad.AnnData, n_jobs: int = -1, progress: bool = True, 
                  batch_key: str = "batch", perturbation_key: str = "perturbation",
                  control_label: str = "NTC", obsm_key: str = "X_pca", 
                  dist: str = "sqeuclidean", runs: int = 10000, 
                  correction_method: str = "fdr_bh", seed: int = 42,
                  num_ntc_cells: int = 400, max_perturbations_per_batch: int = 500,
                  key_added: str = "etest_results") -> pd.DataFrame:
    """
    Compute the energy test results for each batch in the dataset and combine them.
    
    Parameters:
        adata: AnnData object containing single-cell data
        n_jobs: Number of parallel jobs (-1 = all available cores)
        progress: Whether to show progress bars
        batch_key: Key in adata.obs for batch information
        perturbation_key: Key in adata.obs for perturbation information
        control_label: Label for control cells (typically "NTC")
        obsm_key: Key in adata.obsm for dimensionality reduction coordinates
        dist: Distance metric to use (e.g., "sqeuclidean")
        runs: Number of permutations for the energy test
        correction_method: Method for multiple testing correction
        seed: Random seed for reproducibility
        num_ntc_cells: Number of NTC cells to subsample to
        max_perturbations_per_batch: Maximum number of perturbations to process at once
        key_added: Key under which to store etest results in adata.uns
        
    Returns:
        Combined DataFrame with energy test results from all batches
    """
    # Set master seed
    np.random.seed(seed)
    
    # Check if the data has batches
    has_batches = batch_key in adata.obs.columns
    
    if has_batches:
        # Get unique batches
        batches = adata.obs[batch_key].unique()
        all_results = []
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            # Get batch-specific seed
            batch_seed = seed + batch_idx
            
            # Set seed for this batch
            np.random.seed(batch_seed)
            
            # Get batch data
            batch_adata = adata[adata.obs[batch_key] == batch].copy()
            
            if progress:
                print(f"Processing batch: {batch} - {batch_adata.n_obs} cells")
            
            # Check if batch has control cells
            if control_label not in batch_adata.obs[perturbation_key].values:
                if progress:
                    print(f"Skipping batch {batch}: No control cells found")
                continue
                
            # Check if we have enough control cells
            ntc_count = np.sum(batch_adata.obs[perturbation_key] == control_label)
            if ntc_count < num_ntc_cells:
                if progress:
                    print(f"Skipping batch {batch}: Not enough control cells ({ntc_count} < {num_ntc_cells})")
                continue
            
            # Preprocess batch data
            try:
                # Use preprocessing steps from _preprocess_for_etest
                batch_adata = _preprocess_for_etest(batch_adata)
                
                # Use batch-specific seed for subsampling
                np.random.seed(batch_seed)
                batch_adata_subsampled = _subsample_adata_NTCs(
                    batch_adata, 
                    num_ntc_cells=num_ntc_cells
                )
                
                # Get unique perturbations in this batch (excluding control)
                perturbations = np.unique(batch_adata_subsampled.obs[perturbation_key])
                perturbations = perturbations[perturbations != control_label]
                
                # Process perturbations in smaller batches
                batch_results = []
                
                for i in range(0, len(perturbations), max_perturbations_per_batch):
                    sub_batch_seed = batch_seed + i
                    
                    if progress:
                        print(f"  Processing perturbations {i+1}-{min(i+max_perturbations_per_batch, len(perturbations))} of {len(perturbations)}")
                    
                    # Get current subset of perturbations
                    current_perturbations = perturbations[i:i+max_perturbations_per_batch]
                    
                    # Create a subset AnnData with only these perturbations and the control
                    mask = np.isin(batch_adata_subsampled.obs[perturbation_key], 
                                  np.append(current_perturbations, control_label))
                    subset_adata = batch_adata_subsampled[mask]
                    
                    # Run our deterministic etest with multiprocessing
                    subset_result = _etest_deterministic(
                        subset_adata,
                        seed=sub_batch_seed,
                        obs_key=perturbation_key, 
                        control=control_label, 
                        obsm_key=obsm_key, 
                        dist=dist, 
                        n_jobs=n_jobs,  # Use requested parallelism
                        verbose=progress, 
                        runs=runs, 
                        correction_method=correction_method,
                        alpha=0.05,
                    )
                    
                    # Add to batch results
                    batch_results.append(subset_result)
                
                # Combine results from all sub-batches
                if batch_results:
                    etest_result = pd.concat(batch_results)
                    
                    # Mark control as itself for consistency
                    etest_result.loc[etest_result.index==control_label, 'significant_adj'] = control_label
                    
                    # Add negative log10 p-values for visualization
                    etest_result['neglog10_pvalue_adj'] = -np.log10(etest_result['pvalue_adj'])
                    
                    # Add batch information
                    etest_result['batch'] = batch
                    
                    # Append to results list
                    all_results.append(etest_result)
                
            except Exception as e:
                if progress:
                    print(f"Error processing batch {batch}: {str(e)}")
                continue
        
        # Combine all batch results
        if len(all_results) > 0:
            combined_results = pd.concat(all_results, axis=0)
            # Apply cross-batch correction with fixed seed
            np.random.seed(seed)
            combined_results = _apply_cross_batch_correction(
                combined_results, 
                pvalue_col='pvalue', 
                correction_method=correction_method,
                alpha=0.05
            )
            
            # Store results in AnnData object
            _store_etest_results_in_anndata(adata, combined_results, key_added, perturbation_key)
            
            return combined_results
        else:
            return pd.DataFrame()
    
    else:
        # Process the entire dataset as a single batch
        if progress:
            print(f"Processing entire dataset as a single batch - {adata.n_obs} cells")
        
        # Check if we have enough control cells
        ntc_count = np.sum(adata.obs[perturbation_key] == control_label)
        if ntc_count < num_ntc_cells:
            if progress:
                print(f"Not enough control cells ({ntc_count} < {num_ntc_cells})")
            return pd.DataFrame()
        
        # Preprocess the data
        try:
            # Use preprocessing steps from _preprocess_for_etest
            adata_processed = _preprocess_for_etest(adata)
            
            # Subsample NTC cells
            np.random.seed(seed)
            adata_subsampled = _subsample_adata_NTCs(
                adata_processed, 
                num_ntc_cells=num_ntc_cells
            )
            
            # Get unique perturbations (excluding control)
            perturbations = np.unique(adata_subsampled.obs[perturbation_key])
            perturbations = perturbations[perturbations != control_label]
            
            # Process perturbations in smaller batches
            all_results = []
            
            for i in range(0, len(perturbations), max_perturbations_per_batch):
                sub_batch_seed = seed + i
                
                if progress:
                    print(f"  Processing perturbations {i+1}-{min(i+max_perturbations_per_batch, len(perturbations))} of {len(perturbations)}")
                
                # Get current subset of perturbations
                current_perturbations = perturbations[i:i+max_perturbations_per_batch]
                
                # Create a subset AnnData with only these perturbations and the control
                mask = np.isin(adata_subsampled.obs[perturbation_key], 
                              np.append(current_perturbations, control_label))
                subset_adata = adata_subsampled[mask]
                
                # Run our deterministic etest with multiprocessing
                subset_result = _etest_deterministic(
                    subset_adata,
                    seed=sub_batch_seed,
                    obs_key=perturbation_key, 
                    control=control_label, 
                    obsm_key=obsm_key, 
                    dist=dist, 
                    n_jobs=n_jobs,  # Use requested parallelism
                    verbose=progress, 
                    runs=runs, 
                    correction_method=correction_method,
                    alpha=0.05,
                )
                
                # Add to results
                all_results.append(subset_result)
            
            # Combine results from all sub-batches
            if all_results:
                etest_result = pd.concat(all_results)
                
                # Mark control as itself for consistency
                etest_result.loc[etest_result.index==control_label, 'significant_adj'] = control_label
                
                # Add negative log10 p-values for visualization
                etest_result['neglog10_pvalue_adj'] = -np.log10(etest_result['pvalue_adj'])
                
                # Add a "batch" column with a single value for consistency with the batched version
                etest_result['batch'] = 'all'
                
                # Store results in AnnData object
                _store_etest_results_in_anndata(adata, etest_result, key_added, perturbation_key)
                
                return etest_result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            if progress:
                print(f"Error processing data: {str(e)}")
            return pd.DataFrame()


def analyze_perturbation_coherence(
    adata,
    deg_file_path,
    gene_target_key='gene_target',
    perturbation_key='perturbation',
    fdr_percentile=95,
    random_seed=42,
    save_path=None,
    figsize=(12, 8),
    coherence_column='coherent_target',
    batch_mode = False
):
    """
    Analyzes the coherence of perturbations targeting the same gene using Jaccard index.
    
    This function:
    1. Loads DEG data from an Excel file
    2. Filters to include only perturbations present in the AnnData object
    3. Calculates Jaccard indices between perturbations targeting the same gene
    4. Establishes an FDR threshold from random perturbation pairs
    5. Determines which gene targets have sufficient coherence
    6. Creates visualizations of the results
    7. Adds a column to adata.obs indicating whether each gene_target passes the coherence cutoff
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing perturbation information
    deg_file_path : str
        Path to Excel file containing differential expression results
    gene_target_key : str, optional (default: 'gene_target')
        Key in adata.obs containing gene target information
    perturbation_key : str, optional (default: 'perturbation')
        Key in adata.obs containing perturbation IDs
    fdr_percentile : int, optional (default: 95)
        Percentile of random Jaccard indices to use as FDR threshold
    random_seed : int, optional (default: 42)
        Seed for random number generation
    save_path : str, optional (default: None)
        Path to save the output figure. If None, figure is not saved
    figsize : tuple, optional (default: (12, 8))
        Figure size (width, height) in inches
    coherence_column : str, optional (default: 'coherent_target')
        Column name to use in adata.obs for coherence status
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'jaccard_df': DataFrame with Jaccard indices for same-gene perturbation pairs
        - 'gene_avg_jaccard': DataFrame with average Jaccard index per gene
        - 'coherent_genes': List of genes with sufficient coherence
        - 'fdr_threshold': FDR threshold value
        - 'random_jaccard_df': DataFrame with Jaccard indices for random perturbation pairs
        - 'figure': Matplotlib Figure object
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    from collections import defaultdict
    
    # Load the DEG data
    data = pd.read_csv(deg_file_path)
    
    # Get perturbations present in the AnnData object
    perturbations_in_adata = adata.obs[perturbation_key].unique().tolist()
    
    # Extract perturbation names from column headers
    all_perturbations = []
    pert_batch = {}
    for col in data.columns:
        if col.endswith('_DEGs'):
            pert = col.replace('_DEGs', '')
            if batch_mode:
                pre_split = pert + '_DEGs'
                pert ='_'.join(pert.split('_')[:-1])
                pert_batch[pert] = pre_split
            # Only include perturbations present in the AnnData object
            if pert in perturbations_in_adata:
                all_perturbations.append(pert)
    
    # Create a dictionary to store DEGs for each perturbation
    perturbation_degs = {}
    for pert in all_perturbations:
        # Get the DEGs column for this perturbation
        if batch_mode:
            degs_col = pert_batch[pert]
        else:
            degs_col = f"{pert}_DEGs"
        # Filter out NaN values and store the gene names
        genes = data[degs_col].dropna().tolist()
        perturbation_degs[pert] = set(genes)
    
    # Map perturbations to gene targets
    gene_to_perturbations = defaultdict(list)
    for pert in all_perturbations:
        # Get gene target from AnnData
        # Find all cells with this perturbation
        pert_cells = adata.obs[adata.obs[perturbation_key] == pert]
        if len(pert_cells) > 0:
            # Get the gene target (should be the same for all cells with this perturbation)
            gene = pert_cells[gene_target_key].iloc[0]
            gene_to_perturbations[gene].append(pert)
    
    # Calculate Jaccard index for perturbations targeting the same gene
    jaccard_results = []
    for gene, perts in gene_to_perturbations.items():
        if len(perts) > 1:  # Only consider genes with multiple perturbations
            for i in range(len(perts)):
                for j in range(i+1, len(perts)):
                    pert1, pert2 = perts[i], perts[j]
                    set1 = perturbation_degs[pert1]
                    set2 = perturbation_degs[pert2]
                    
                    # Calculate Jaccard index
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    jaccard = intersection / union if union > 0 else 0
                    
                    jaccard_results.append({
                        'Gene': gene,
                        'Perturbation1': pert1,
                        'Perturbation2': pert2,
                        'DEGs_Pert1': len(set1),
                        'DEGs_Pert2': len(set2),
                        'Intersection': intersection,
                        'Union': union,
                        'Jaccard_Index': jaccard
                    })
    
    # Convert results to DataFrame and sort
    jaccard_df = pd.DataFrame(jaccard_results)
    if len(jaccard_df) > 0:
        jaccard_df = jaccard_df.sort_values(['Gene', 'Jaccard_Index'], ascending=[True, False])
    
        # Calculate average Jaccard index per gene
        gene_avg_jaccard = jaccard_df.groupby('Gene')['Jaccard_Index'].mean().reset_index()
        gene_avg_jaccard = gene_avg_jaccard.sort_values('Jaccard_Index', ascending=False)
        
        # Calculate Jaccard index for random pairs of perturbations to establish FDR
        random_jaccard_results = []
        # Set a seed for reproducibility
        random.seed(random_seed)
        # Number of random pairs to sample (same as the number of same-gene pairs for fair comparison)
        n_random_pairs = len(jaccard_results)
        # Get all possible perturbation pairs
        sampled_pairs = set()
        
        # Try to get enough random pairs
        max_attempts = min(10000, n_random_pairs * 10)  # Avoid infinite loops
        attempts = 0
        
        while len(random_jaccard_results) < n_random_pairs and attempts < max_attempts:
            attempts += 1
            # Sample two different perturbations
            if len(all_perturbations) >= 2:
                pert1, pert2 = random.sample(all_perturbations, 2)
                # Find gene targets for these perturbations
                pert1_cells = adata.obs[adata.obs[perturbation_key] == pert1]
                pert2_cells = adata.obs[adata.obs[perturbation_key] == pert2]
                
                if len(pert1_cells) > 0 and len(pert2_cells) > 0:
                    gene1 = pert1_cells[gene_target_key].iloc[0]
                    gene2 = pert2_cells[gene_target_key].iloc[0]
                    pair_key = tuple(sorted([pert1, pert2]))
                    
                    # Only include pairs targeting different genes that haven't been sampled before
                    if gene1 != gene2 and pair_key not in sampled_pairs:
                        sampled_pairs.add(pair_key)
                        set1 = perturbation_degs[pert1]
                        set2 = perturbation_degs[pert2]
                        
                        # Calculate Jaccard index
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard = intersection / union if union > 0 else 0
                        
                        random_jaccard_results.append({
                            'Perturbation1': pert1,
                            'Perturbation2': pert2,
                            'DEGs_Pert1': len(set1),
                            'DEGs_Pert2': len(set2),
                            'Intersection': intersection,
                            'Union': union,
                            'Jaccard_Index': jaccard
                        })
        
        # Convert random results to DataFrame
        random_jaccard_df = pd.DataFrame(random_jaccard_results)
        
        # Calculate FDR threshold
        if len(random_jaccard_df) > 0:
            fdr_threshold = np.percentile(random_jaccard_df['Jaccard_Index'], fdr_percentile)
        else:
            fdr_threshold = 0
        
        # Add a column indicating whether the Jaccard index exceeds the FDR threshold
        jaccard_df['Exceeds_FDR'] = jaccard_df['Jaccard_Index'] > fdr_threshold
        
        # Determine which genes have sufficient coherence
        # A gene is coherent if the average Jaccard index of its perturbations exceeds the FDR threshold
        gene_avg_jaccard['Coherent'] = gene_avg_jaccard['Jaccard_Index'] > fdr_threshold
        coherent_genes = gene_avg_jaccard[gene_avg_jaccard['Coherent']]['Gene'].tolist()
        
        # Add coherence status to adata.obs
        coherence_dict = {gene: gene in coherent_genes for gene in gene_avg_jaccard['Gene']}
        # For genes that weren't evaluated (only one perturbation), set to False instead of NaN
        evaluated_genes = set(gene_avg_jaccard['Gene'])
        all_genes = set(adata.obs[gene_target_key].unique())
        for gene in all_genes - evaluated_genes:
            coherence_dict[gene] = False
            
        # Create a column in adata.obs to indicate if the gene target passes the coherence cutoff
        adata.obs[coherence_column] = adata.obs[gene_target_key].map(coherence_dict)
        # Fill any remaining NaN values with False (for NTCs or other cases)
        adata.obs[coherence_column] = adata.obs[coherence_column].fillna(False)
        
        # Print statistics about coherence status
        coherent_count = adata.obs[coherence_column].sum()
        total_cells = len(adata.obs)
        print(f"Added '{coherence_column}' column to adata.obs")
        print(f"- Cells with coherent targets: {coherent_count}/{total_cells} cells ({coherent_count/total_cells*100:.1f}%)")
        
        # Visualize the results
        fig = plt.figure(figsize=figsize)
        
        # Plot 1: Top genes by average Jaccard index
        plt.subplot(1, 2, 1)
        top_genes = gene_avg_jaccard.head(20)  # Top 20 genes
        ax = sns.barplot(x='Jaccard_Index', y='Gene', data=top_genes, palette='viridis')
        
        # Color bars based on coherence
        for i, coherent in enumerate(top_genes['Coherent']):
            bar_color = 'mediumseagreen' if coherent else 'lightgray'
            ax.patches[i].set_facecolor(bar_color)
            
        plt.title('Top 20 Genes by Average Jaccard Index')
        plt.xlabel('Average Jaccard Index')
        plt.ylabel('Gene')
        plt.axvline(fdr_threshold, color='red', linestyle='--', label=f'FDR Threshold: {fdr_threshold:.4f}')
        plt.legend()
        plt.grid(False)
        
        # Plot 2: Distribution of Jaccard indices with FDR threshold
        plt.subplot(1, 2, 2)
        if len(jaccard_df) > 0:
            sns.histplot(jaccard_df['Jaccard_Index'], bins=20, kde=True, label='Same Gene Pairs', alpha=0.7)
        if len(random_jaccard_df) > 0:
            sns.histplot(random_jaccard_df['Jaccard_Index'], bins=20, kde=True, label='Random Pairs', alpha=0.7)
        plt.title('Distribution of Jaccard Indices')
        plt.xlabel('Jaccard Index')
        plt.ylabel('Frequency')
        
        if len(jaccard_df) > 0:
            plt.axvline(jaccard_df['Jaccard_Index'].mean(), color='blue', linestyle='--', 
                        label=f'Same Gene Mean: {jaccard_df["Jaccard_Index"].mean():.3f}')
        plt.axvline(fdr_threshold, color='red', linestyle='--', 
                    label=f'FDR Threshold: {fdr_threshold:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.grid(False)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
        
        # Print summary statistics
        print(f"Perturbation coherence analysis:")
        print(f"- Total perturbations analyzed: {len(all_perturbations)}")
        print(f"- Total genes with multiple perturbations: {len(gene_to_perturbations)}")
        print(f"- FDR threshold (at {fdr_percentile}th percentile): {fdr_threshold:.4f}")
        print(f"- Perturbation pairs exceeding FDR: {jaccard_df['Exceeds_FDR'].sum()}/{len(jaccard_df)} ({jaccard_df['Exceeds_FDR'].mean()*100:.1f}%)")
        print(f"- Genes with coherent perturbations: {len(coherent_genes)}/{len(gene_avg_jaccard)} ({len(coherent_genes)/len(gene_avg_jaccard)*100:.1f}%)")
        
        return {
            'jaccard_df': jaccard_df,
            'gene_avg_jaccard': gene_avg_jaccard,
            'coherent_genes': coherent_genes,
            'fdr_threshold': fdr_threshold,
            'random_jaccard_df': random_jaccard_df,
            'figure': fig
        }
    else:
        print("No perturbation pairs found targeting the same gene.")
        # Still create the coherence column but set all values to False instead of NaN
        adata.obs[coherence_column] = False
        print(f"Added '{coherence_column}' column to adata.obs (all values are False)")
        
        return {
            'jaccard_df': pd.DataFrame(),
            'gene_avg_jaccard': pd.DataFrame(),
            'coherent_genes': [],
            'fdr_threshold': None,
            'random_jaccard_df': pd.DataFrame(),
            'figure': None
        }






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
    
    # Convert to numpy array
    mean_profiles = np.vstack(mean_profiles)
    
    # Store mean profiles in uns instead of as a layer
    adata.uns['mean_gene_target_profiles'] = {
        'profiles': mean_profiles,
        'gene_targets': list(categories)
    }

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
    sc.pl.umap(get_ntc_view(adata), color=[perturbation_key, batch_key], title=["NTC Cells", "NTC Cells - Batch"], frameon=False)
    
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


def __equal_subsampling(adata, obs_key, batch_key = None, random_seed=42, N_min=None):
    """Subsample cells while retaining same class sizes.

    This function is a modification of the scPerturb 'equal_subsampling' function. Simply modified to allow for random seed and to handle batch-specific NTC cells.
    This function is used to subsample the NTC cells to a fixed number.
    Classes are given by obs_key pointing to categorical in adata.obs.
    If N_min is given, downsamples to at least this number instead of the number
    of cells in the smallest class and throws out classes with less than N_min cells.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    obs_key: `str` in adata.obs.keys() (default: `perturbation`)
        Key in adata.obs specifying the groups to consider.
    N_min: `int` or `None` (default: `None`)
        If N_min is given, downsamples to at least this number instead of the number
        of cells in the smallest class and throws out classes with less than N_min cells.

    Returns
    -------
    subdata: :class:`~anndata.AnnData`
        Subsampled version of the original annotated data matrix.
    """
    # Determine grouping for subsampling: combine obs_key and batch_key if provided
    if batch_key is not None:
        group_series = adata.obs[obs_key].astype(str) + "_" + adata.obs[batch_key].astype(str)
    else:
        group_series = adata.obs[obs_key].astype(str)
    counts = group_series.value_counts()
    if N_min is not None:
        groups = counts.index[counts>=N_min]  # ignore groups with less than N_min cells to begin with
    else:
        groups=counts.index
    # We select downsampling target counts by min-max, i.e.
    # the largest N such that every group has at least N cells before downsampling.
    N = counts.min()
    N = N if N_min==None else np.max([N_min, N])
    print(f"Equal subsampling cells to {N} per perturbation")
    # subsample indices per group
    np.random.seed(random_seed)
    indices = []
    for group in groups:
        # get cell names belonging to the group
        cells = group_series[group_series == group].index.values
        indices.append(np.random.choice(cells, size=N, replace=False))
    selection = np.concatenate(indices)
    return adata[selection].copy()    


def _relative_z_normalization(adata):
    """
    Normalize data relative to the NTC cells from the same batch.
    """
    ntc_mean = adata[adata.obs['perturbation'] == 'NTC'].X.mean()
    adata.X = (adata.X - ntc_mean) / ntc_mean
    return adata


def _preprocess_for_pairwise_edist(adata, obs_key, batch_key=None, subsample_equal=True, random_seed=42):
    """
    Preprocess AnnData object for energy distance calculations.
    
    This function prepares single-cell data for energy distance analysis by:
    1. Creating a copy of the original data
    2. Setting counts as the main matrix
    3. Filtering low-abundance genes
    4. Identifying highly variable genes
    5. Normalizing and scaling data
    6. Performing dimensionality reduction (PCA)
    7. Computing a neighborhood graph
    
    Parameters:
        adata (AnnData): Input AnnData object with raw counts in .layers['counts']
        
    Returns:
        AnnData: Preprocessed copy of the input data with PCA and neighbors computed
    """
    if subsample_equal:
        adata_edist = __equal_subsampling(adata,)
    else:
        adata_edist = adata.copy()
    adata_edist.X = adata_edist.layers['counts'].copy()
    sc.pp.filter_genes(adata_edist, min_cells=3)
    n_var_max = 2000
    sc.pp.highly_variable_genes(adata_edist, n_top_genes=n_var_max, flavor='seurat_v3', subset=False, layer='counts', batch_key=batch_key)
    sc.pp.normalize_total(adata_edist, inplace=True)
    sc.pp.log1p(adata_edist)
    sc.pp.scale(adata_edist)
    sc.pp.pca(adata_edist, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata_edist)
    return adata_edist


def compute_pairwise_edist(
    adata: ad.AnnData,
    obsm_key: str = 'X_pca',
    perturbation_key: str = 'perturbation',
    dist: str = 'sqeuclidean',
    num_ntc_cells: int = 400,
    subsample_equal: bool = True,
    random_seed: int = 42,
    progress: bool = True,
    store_results: bool = True,
    key_added: str = 'pairwise_edist',
    batch_key: str = None
) -> pd.DataFrame:
    """
    Compute pairwise energy distances between all perturbations using scperturb's edist function.
    
    This function:
    1. Subsamples NTC cells to a fixed number (default 400)
    2. Computes pairwise energy distances between all perturbations
    3. Optionally stores results in adata.uns
    
    Parameters:
        adata: AnnData object containing single-cell data
        obsm_key: Key in adata.obsm for coordinates
        perturbation_key: Key in adata.obs for perturbation labels
        control_label: Label for control cells (non-targeting controls)
        dist: Distance metric for energy calculations
        num_ntc_cells: Number of NTC cells to retain in subsampling
        random_seed: Random seed for reproducibility
        progress: Whether to show progress information
        store_results: Whether to store results in adata.uns
        key_added: Key under which to store results in adata.uns
        
    Returns:
        DataFrame with pairwise energy distances
    """

    # Subsample NTC cells
    if not subsample_equal:
        print(f"Subsampling NTC cells to {num_ntc_cells}")
        adata_subsampled = _subsample_adata_NTCs(adata, num_ntc_cells=num_ntc_cells, random_seed=random_seed)
    else:
        adata_subsampled = __equal_subsampling(adata, obs_key=perturbation_key, batch_key=batch_key, random_seed=random_seed)
    
    
    # Ensure we have a preprocessed AnnData object
    if obsm_key not in adata_subsampled.obsm:
        # Preprocess data for energy distance calculation
        if progress:
            print("Preprocessing data for energy distance calculation")
        adata_edist = _preprocess_for_etest(adata_subsampled)
    else:
        adata_edist = adata_subsampled
    
    
    # Compute pairwise energy distances using scperturb's edist function
    if progress:
        print("Computing pairwise energy distances between all perturbations")
    
    # Call scp.edist function to compute energy distances
    edist_df = scp.edist(
        adata_edist, 
        obs_key=perturbation_key,
        obsm_key=obsm_key,
        dist=dist
    )
    
    # Store results in AnnData object if requested
    if store_results:
        # Store the energy distance matrix
        adata.uns[key_added] = {
            'matrix': edist_df.values,
            'perturbations': edist_df.columns.tolist(),
            'dataframe': edist_df,
            'params': {
                'obsm_key': obsm_key,
                'perturbation_key': perturbation_key,
                'dist': dist,
                'num_ntc_cells': num_ntc_cells,
                'random_seed': random_seed,
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
    return edist_df

def compute_deg_jaccard_matrix(
    deg_df: pd.DataFrame,
    adata: Optional[ad.AnnData] = None,
    key_added: str = 'deg_jaccard_matrix',
    l2fc_cutoff: float = 0.0,
    comparison_type: str = 'all'
) -> pd.DataFrame:
    """
    Compute pairwise Jaccard indices between perturbations based on their DEGs.
    
    Parameters
    ----------
    deg_df : pd.DataFrame
        DataFrame output from _save_DEG_df containing DEG information
    adata : Optional[ad.AnnData], optional
        AnnData object to store the matrix in, by default None
    key_added : str, optional
        Key to store the matrix in adata.uns if adata is provided, by default 'deg_jaccard_matrix'
    l2fc_cutoff : float, optional
        Only consider genes with absolute Log2FoldChange above this value, by default 0.0
    comparison_type : str, optional
        Type of comparison to perform, by default 'all'
        - 'all': Compare all DEGs regardless of direction
        - 'upregulated': Only compare upregulated genes (L2FC > 0)
        - 'downregulated': Only compare downregulated genes (L2FC < 0)
        - 'mismatch': Compare genes with opposite regulation (up in one, down in another)
        - 'similarity_score': Compute (Jaccard UP + Jaccard DOWN) / (1 + Jaccard MISMATCH)
        
    Returns
    -------
    pd.DataFrame
        Square matrix of Jaccard indices between perturbations
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Validate comparison_type
    valid_types = ['all', 'upregulated', 'downregulated', 'mismatch', 'similarity_score']
    if comparison_type not in valid_types:
        raise ValueError(f"comparison_type must be one of {valid_types}")
    
    # Extract perturbation names from column names (they end with _DEGs)
    perturbations = [col[:-5] for col in deg_df.columns if col.endswith('_DEGs')]
    n_perturbations = len(perturbations)
    
    # If using similarity_score, compute the individual matrices and combine them
    if comparison_type == 'similarity_score':
        # Compute matrices for up and down regulated genes
        up_matrix = compute_deg_jaccard_matrix(
            deg_df, adata=None, l2fc_cutoff=l2fc_cutoff, comparison_type='upregulated'
        ).values
        
        down_matrix = compute_deg_jaccard_matrix(
            deg_df, adata=None, l2fc_cutoff=l2fc_cutoff, comparison_type='downregulated'
        ).values
        
        mismatch_matrix = compute_deg_jaccard_matrix(
            deg_df, adata=None, l2fc_cutoff=l2fc_cutoff, comparison_type='mismatch'
        ).values
        
        # Calculate the similarity score: (Jaccard UP + Jaccard DOWN) / (1 + Jaccard MISMATCH)
        similarity_matrix = (up_matrix + down_matrix) / (1 + mismatch_matrix)
        
        # Convert to DataFrame
        jaccard_df = pd.DataFrame(
            similarity_matrix,
            index=perturbations,
            columns=perturbations
        )
        
    else:
        # Create lists to store all genes and their fold changes for each perturbation
        all_genes = set()
        gene_l2fc_dict = {}
        
        # First collect all genes and their fold changes
        for pert in perturbations:
            genes = deg_df[f"{pert}_DEGs"].dropna().values
            l2fcs = deg_df[f"{pert}_L2FC"].dropna().values
            
            # Filter by l2fc_cutoff if needed
            if l2fc_cutoff > 0:
                valid_indices = np.abs(l2fcs) >= l2fc_cutoff
                genes = genes[valid_indices]
                l2fcs = l2fcs[valid_indices]
            
            # Store gene-l2fc mapping for this perturbation
            gene_l2fc_dict[pert] = dict(zip(genes, l2fcs))
            all_genes.update(genes)
        
        # Convert to sorted list for consistent ordering
        all_genes = sorted(list(all_genes))
        
        if len(all_genes) == 0:
            # If no genes pass the filter, return identity matrix
            jaccard_matrix = np.eye(n_perturbations)
        else:
            # Create binary matrix based on comparison type
            if comparison_type == 'all':
                # Binary matrix: 1 if gene is a DEG for perturbation, 0 otherwise
                binary_matrix = np.zeros((n_perturbations, len(all_genes)), dtype=bool)
                for i, pert in enumerate(perturbations):
                    binary_matrix[i] = np.array([g in gene_l2fc_dict[pert] for g in all_genes])
                
                # Use pdist with jaccard metric (computes distance, which is 1 - similarity)
                jaccard_dist = pdist(binary_matrix, metric='jaccard')
                jaccard_matrix = 1 - squareform(jaccard_dist)  # Convert to similarity
                
            elif comparison_type == 'upregulated':
                # Binary matrix: 1 if gene is upregulated for perturbation, 0 otherwise
                binary_matrix = np.zeros((n_perturbations, len(all_genes)), dtype=bool)
                for i, pert in enumerate(perturbations):
                    binary_matrix[i] = np.array([
                        g in gene_l2fc_dict[pert] and gene_l2fc_dict[pert][g] > 0 
                        for g in all_genes
                    ])
                
                # Use pdist with jaccard metric
                jaccard_dist = pdist(binary_matrix, metric='jaccard')
                jaccard_matrix = 1 - squareform(jaccard_dist)
                
            elif comparison_type == 'downregulated':
                # Binary matrix: 1 if gene is downregulated for perturbation, 0 otherwise
                binary_matrix = np.zeros((n_perturbations, len(all_genes)), dtype=bool)
                for i, pert in enumerate(perturbations):
                    binary_matrix[i] = np.array([
                        g in gene_l2fc_dict[pert] and gene_l2fc_dict[pert][g] < 0 
                        for g in all_genes
                    ])
                
                # Use pdist with jaccard metric
                jaccard_dist = pdist(binary_matrix, metric='jaccard')
                jaccard_matrix = 1 - squareform(jaccard_dist)
                
            elif comparison_type == 'mismatch':
                # For mismatch, we need a custom pdist function
                # First create a matrix of gene regulation directions (-1, 0, 1)
                # where 1 = upregulated, -1 = downregulated, 0 = not a DEG
                sign_matrix = np.zeros((n_perturbations, len(all_genes)))
                for i, pert in enumerate(perturbations):
                    for j, gene in enumerate(all_genes):
                        if gene in gene_l2fc_dict[pert]:
                            l2fc = gene_l2fc_dict[pert][gene]
                            sign_matrix[i, j] = np.sign(l2fc)
                
                # Custom function to calculate mismatch Jaccard index for pdist
                def mismatch_jaccard(u, v):
                    """
                    Calculate Jaccard index for genes regulated in opposite directions.
                    Parameters:
                        u, v: regulation direction vectors (-1, 0, 1)
                    Returns:
                        Jaccard index for mismatched genes
                    """
                    # Find indices where either vector is non-zero (i.e., gene is a DEG)
                    either_deg = np.logical_or(u != 0, v != 0)
                    # Count total DEGs in either perturbation
                    union_count = np.sum(either_deg)
                    
                    if union_count == 0:
                        return 0  # No DEGs in either perturbation
                    
                    # Find where the product is negative (opposite directions)
                    opposite_dir = np.logical_and(u * v < 0, either_deg)
                    mismatch_count = np.sum(opposite_dir)
                    
                    # Return Jaccard index for mismatches
                    return mismatch_count / union_count
                
                # Calculate mismatch Jaccard indices directly
                mismatch_sim = pdist(sign_matrix, metric=mismatch_jaccard)
                jaccard_matrix = squareform(mismatch_sim)
                
                # No self-mismatch possible
                np.fill_diagonal(jaccard_matrix, 0.0)
            
        # Convert to DataFrame
        jaccard_df = pd.DataFrame(
            jaccard_matrix,
            index=perturbations,
            columns=perturbations
        )
    
    # Store in adata if provided
    if adata is not None:
        mtype = comparison_type.replace('_', '-')
        if l2fc_cutoff > 0:
            key = f"{key_added}_{mtype}_l2fc{l2fc_cutoff}"
        else:
            key = f"{key_added}_{mtype}"
        adata.uns[key] = jaccard_df
    
    return jaccard_df


def standard_preprocessing(adata: ad.AnnData, 
                          batch_key: str = None, 
                          layer: str = 'counts',
                          n_top_genes: int = 2000,
                          ) -> ad.AnnData:
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, batch_key=batch_key, flavor='seurat_v3', n_top_genes=n_top_genes, layer=layer)
    if not issparse(adata.X):
        adata.X = adata.layers[layer].copy()
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    #sc.pp.scale(adata)
    return adata

def compute_pairwise_edist_replicates(
    adata: ad.AnnData,
    obsm_key: str = 'X_pca',
    perturbation_key: str = 'perturbation',
    dist: str = 'sqeuclidean',
    num_ntc_cells: int = 400,
    subsample_equal: bool = True,
    random_seed: int = 42,
    n_reps: int = 30,
    batch_key: str = None,
    key_added: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run compute_pairwise_edist n_reps times with varying seeds and return
    the mean and variance of the resulting distance matrices.

    Parameters:
        adata: Annotated data matrix
        obsm_key, perturbation_key, dist, num_ntc_cells, subsample_equal: same as compute_pairwise_edist
        random_seed: Base seed for reproducibility
        n_reps: Number of replicates to run
        progress: Whether to print progress messages for each replicate
        batch_key: Batch key passed to compute_pairwise_edist

    Returns:
        mean_df: DataFrame of mean distances across replicates
        var_df: DataFrame of variance of distances across replicates
    """
    # Collect all replicate results
    sc.settings.verbosity = 0
    replicates = []
    # Collect all replicate results with progress bar
    for i in tqdm(range(n_reps), desc="Pairwise edist replicates"):
        seed_i = random_seed + i
        df = compute_pairwise_edist_silent(
            adata,
            obsm_key=obsm_key,
            perturbation_key=perturbation_key,
            dist=dist,
            num_ntc_cells=num_ntc_cells,
            subsample_equal=subsample_equal,
            random_seed=seed_i,
            progress=False,
            store_results=False,
            key_added=None,
            batch_key=batch_key
        )
        replicates.append(df)
    # Stack into array: shape (n_reps, N, N)
    arr = np.stack([df.values for df in replicates], axis=0)
    mean_arr = arr.mean(axis=0)
    var_arr = arr.var(axis=0, ddof=0)
    # Build DataFrames
    idx = replicates[0].index
    cols = replicates[0].columns
    mean_df = pd.DataFrame(mean_arr, index=idx, columns=cols)
    var_df = pd.DataFrame(var_arr, index=idx, columns=cols)
    # Store replicate summary in adata.uns if key_added is provided
    if key_added is not None:
        adata.uns[key_added] = {'mean': mean_df, 'variance': var_df}
        print(f"Stored replicate results in adata.uns['{key_added}']")
    sc.settings.verbosity = 4
    return mean_df, var_df

def compute_pairwise_edist_silent(
    adata: ad.AnnData,
    obsm_key: str = 'X_pca',
    perturbation_key: str = 'perturbation',
    dist: str = 'sqeuclidean',
    num_ntc_cells: int = 400,
    subsample_equal: bool = True,
    random_seed: int = 42,
    progress: bool = False,
    store_results: bool = False,
    key_added: str = None,
    batch_key: str = None
) -> pd.DataFrame:
    """
    Silent version of compute_pairwise_edist: suppresses all console output.
    """
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return compute_pairwise_edist(
            adata,
            obsm_key=obsm_key,
            perturbation_key=perturbation_key,
            dist=dist,
            num_ntc_cells=num_ntc_cells,
            subsample_equal=subsample_equal,
            random_seed=random_seed,
            progress=progress,
            store_results=store_results,
            key_added=key_added,
            batch_key=batch_key
        )

def compute_mimosca(
    adata: ad.AnnData,
    covariate_keys: List[str],
    layer: Optional[str] = None,
    key_added: str = "mimosca"
) -> pd.DataFrame:
    """
    Compute MIMOSCA-like linear model coefficients by regressing log-transformed gene expression on covariates.

    Parameters:
        adata: AnnData object containing single-cell data
        covariate_keys: list of column names in adata.obs to include as features (e.g., sgRNA, batch)
        layer: optional layer name in adata.layers to use for expression; if None, uses adata.X
        key_added: prefix under which to store results in adata.uns

    Returns:
        DataFrame of coefficients (covariates x genes)
    """
    import pandas as pd
    # Extract expression matrix
    if layer and layer in adata.layers:
        mat = adata.layers[layer].copy()
    else:
        mat = adata.X.copy()
    # Convert to dense array if sparse
    expr = mat.toarray() if hasattr(mat, "toarray") else np.array(mat)
    # Log-transform
    expr = np.log1p(expr)
    # Build design matrix
    cov_df = adata.obs[covariate_keys].copy()
    design = pd.get_dummies(cov_df, drop_first=False)
    # Add intercept
    design.insert(0, 'Intercept', 1)
    X = design.values
    # Compute coefficients via pseudo-inverse
    beta = np.linalg.pinv(X) @ expr
    # Create DataFrame of coefficients
    beta_df = pd.DataFrame(beta, index=design.columns, columns=adata.var_names)
    # Store in adata.uns
    adata.uns[f"{key_added}_beta"] = beta_df
    return beta_df


def relative_robust_z_normalization_df(
    adata,
    batch_key: str = 'batch',
    ctrl_key: str = 'perturbed',
    ctrl_value: str = 'False',
    save_layer: str = 'pre_robust_z'
) -> None:
    """
    Batch‐aware *robust* z‐normalization using pandas DataFrames.
    Uses control‐cell median and MAD instead of mean/std.

    Side effects
    ------------
    - Saves original adata.X into adata.layers[save_layer]
    - Overwrites adata.X with the batch‐normalized values (dense numpy array)
    """
    # 1) Save original
    if save_layer in adata.layers:
        raise ValueError(f"Layer '{save_layer}' already exists")
    adata.layers[save_layer] = adata.X.copy()

    # 2) Densify and build DataFrame
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    data = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    # 3) Prepare output DataFrame
    output_data = pd.DataFrame(
        np.zeros((adata.n_obs, adata.n_vars)),
        index=adata.obs_names,
        columns=adata.var_names
    )

    # 4) Map obs_names to integer positions once
    pos = data.index.get_indexer(adata.obs_names)
    arr = data.values  # raw ndarray

    # 5) Loop over batches
    for batch in adata.obs[batch_key].unique():
        mask = (adata.obs[batch_key] == batch).values
        all_pos  = pos[mask]
        ctrl_pos = pos[mask & (adata.obs[ctrl_key] == ctrl_value).values]

        # compute control median & MAD
        med = np.median(arr[ctrl_pos], axis=0)
        mad = np.median(np.abs(arr[ctrl_pos] - med), axis=0)

        # scale MAD to be comparable to σ (optional; 1.4826 for normal data)
        mad = mad * 1.4826

        # avoid division by zero
        mad[mad == 0] = 1.0

        # assign robust‐z values
        output_data.values[all_pos] = (arr[all_pos] - med) / mad

    # 6) Overwrite adata.X
    adata.X = output_data.values

import random
def relative_z_normalization_df(
    adata,
    batch_key: str = 'batch',
    ctrl_key: str = 'perturbed',
    ctrl_value: str = 'False',
    save_layer: str = 'pre_z_normalization'
) -> None:
    """
    Batch‐aware z‐normalization using pandas DataFrames.

    Parameters
    ----------
    adata
        AnnData object with .obs containing batch_key and ctrl_key columns.
    batch_key
        Column in adata.obs defining batch membership.
    ctrl_key
        Column in adata.obs defining control vs perturbed cells.
    ctrl_value
        Value in ctrl_key that denotes controls.
    save_layer
        Name under which to save the original adata.X.

    Side effects
    ------------
    - Saves original adata.X into adata.layers[save_layer]
    - Overwrites adata.X with the batch‐normalized values (dense numpy array)
    """
 # 1) save original
    if save_layer in adata.layers:
        raise ValueError(f"Layer '{save_layer}' already exists")
    adata.layers[save_layer] = adata.X.copy()

    # 2) pull out dense X as a numpy array
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()     # now X is (n_obs, n_vars) numpy
    else:
        X = X.copy()        # be safe if adata.X is a view

    n_obs, n_vars = X.shape
    out = np.zeros_like(X)  # final z-normalized pseudobulk

    # 3) for each batch, compute mu/std on controls and fill out
    obs = adata.obs
    for batch in obs[batch_key].unique():
        mask_batch = (obs[batch_key] == batch).values
        mask_ctrl  = mask_batch & (obs[ctrl_key] == ctrl_value).values

        # get control pseudocounts and compute mean/std per gene
        ctrl_mat = X[mask_ctrl]           # shape (#ctrl_cells, n_vars)
        mu  = ctrl_mat.mean(axis=0)       # (n_vars,)
        std = ctrl_mat.std(axis=0)
        std[std == 0] = 1.0

        # z-normalize all cells in this batch
        # broadcasting: (n_cells_in_batch, n_vars) - (n_vars,) / (n_vars,)
        out[mask_batch] = (X[mask_batch] - mu) / std

    # 4) clip in-place
    np.clip(out, -10, 10, out=out)

    # 5) write back
    adata.X = out


def pseudobulk_total(adata, layer='counts', batch_key='batch', n_samples=100, sample_size=200,
                     random_state=42, cpm_threshold=1):
    """
    Compute pseudobulk per gene_target and return:
      1) z-normalized pseudobulk DataFrame (including an 'NTC' zeros column),
      3) DataFrame of NTC mean CPM per batch,
      4) DataFrame of NTC variance CPM per batch.
    """
    import numpy as np
    import pandas as pd

   
    cpm_bulks = {}     # store raw CPM per target
    z_bulks = {}       # store z‐normalized per target
    ntc_means = {}     # store per‐batch NTC means
    ntc_stds = {}     # store per‐batch NTC variances

    # 2) Loop over batches
    for batch in adata.obs[batch_key].unique():
       adata_batch = adata[adata.obs[batch_key] == batch]
       X = adata_batch.layers['counts']
       if sparse.issparse(X):
           X = X.toarray()

       ctrl_idx = adata_batch[adata_batch.obs['gene_target'] == 'NTC'].obs.index

        # 2a) sample NTC pseudobulks and CPM‐normalize
       ntc_samps = []
       rng = np.random.default_rng(random_state)
       for _ in range(n_samples):
            # sample barcodes and convert to integer row indices for X
            sampled_barcodes = rng.choice(ctrl_idx, size=sample_size)
            sel = adata_batch.obs.index.get_indexer(sampled_barcodes)            
            vec_ntc = X[sel, :].sum(axis=0).astype(float)
            total_ntc = vec_ntc.sum()
            vec_ntc = vec_ntc / total_ntc * 1e6
            vec_ntc[vec_ntc < cpm_threshold] = 0
            ntc_samps.append(vec_ntc)
       ntc_samps = np.array(ntc_samps)
       ntc_mean = ntc_samps.mean(axis=0)
       ntc_std = ntc_samps.std(axis=0)
       ntc_std[ntc_std == 0] = 1.0
       ntc_means[batch] = ntc_mean
       ntc_stds[batch] = ntc_std

        # 2b) build CPM and z‐normalized vectors for each non‐NTC target
       for gene_target in adata_batch.obs['gene_target'].unique():
            if gene_target == 'NTC':
                continue
            cells = adata_batch[adata_batch.obs['gene_target'] == gene_target].obs.index
            tgt_idx = adata_batch.obs.index.get_indexer(cells)
            if tgt_idx.size == 0:
                continue
            vec_t = X[tgt_idx, :].sum(axis=0).astype(float)
            total_t = vec_t.sum()
            vec_t = vec_t / total_t * 1e6
            vec_t[vec_t < cpm_threshold] = 0
            cpm_bulks[gene_target] = vec_t
            z_bulks[gene_target] = (vec_t - ntc_mean) / ntc_std

    # 3) Assemble DataFrames
    genes = adata.var_names
    df_cpm = pd.DataFrame(cpm_bulks, index=genes)
    df_z = pd.DataFrame(z_bulks, index=genes)
    # add an 'NTC' column of zeros to the z‐normalized DF
    df_z['NTC'] = 0.0

    ntc_mean_df = pd.DataFrame(ntc_means, index=genes)
    ntc_std_df = pd.DataFrame(ntc_stds, index=genes)

    return df_z, df_cpm, ntc_mean_df, ntc_std_df

def pseudobulk_total_robust(
    adata,
    layer='counts',
    batch_key='batch',
    n_samples=100,
    sample_size=200,
    random_state=42,
    cpm_threshold=1
):
    """
    Compute pseudobulk per gene_target using a robust z–score (median & MAD) 
    instead of mean & SD. Returns:
      1) df_z:  genes × perturbations of robust z–scores (with an 'NTC' zeros column)
      2) df_cpm: genes × perturbations of raw CPM pseudobulks
      3) ntc_median_df: genes × batches of NTC medians
      4) ntc_mad_df: genes × batches of scaled MADs
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    cpm_bulks = {}
    z_bulks   = {}
    ntc_medians = {}
    ntc_mads    = {}

    # RNG can be re‑used per batch for reproducibility
    for batch in adata.obs[batch_key].unique():
        adata_batch = adata[adata.obs[batch_key] == batch]
        X = adata_batch.layers.get(layer, adata_batch.X)
        if sparse.issparse(X):
            X = X.toarray()

        ctrl_idx = adata_batch[adata_batch.obs['gene_target'] == 'NTC'].obs.index
        rng = np.random.default_rng(random_state)

        # 1) sample NTC pseudobulks and CPM‐normalize
        ntc_samps = []
        for _ in range(n_samples):
            sampled = rng.choice(ctrl_idx, size=sample_size, replace=True)
            sel = adata_batch.obs.index.get_indexer(sampled)
            vec = X[sel, :].sum(axis=0).astype(float)
            total = vec.sum()
            vec = vec / total * 1e6
            vec[vec < cpm_threshold] = 0
            ntc_samps.append(vec)
        ntc_samps = np.stack(ntc_samps, axis=0)

        # 2) compute robust stats
        ntc_median = np.median(ntc_samps, axis=0)
        mad = np.median(np.abs(ntc_samps - ntc_median), axis=0)
        scaled_mad = mad * 1.4826
        scaled_mad[scaled_mad == 0] = 1.0

        ntc_medians[batch] = ntc_median
        ntc_mads[batch]    = scaled_mad

        # 3) build CPM & robust‐z for each non‐NTC perturbation
        for tgt in adata_batch.obs['gene_target'].unique():
            if tgt == 'NTC':
                continue
            cells = adata_batch[adata_batch.obs['gene_target'] == tgt].obs.index
            idx = adata_batch.obs.index.get_indexer(cells)
            if idx.size == 0:
                continue

            vec_t = X[idx, :].sum(axis=0).astype(float)
            total_t = vec_t.sum()
            vec_t = vec_t / total_t * 1e6
            vec_t[vec_t < cpm_threshold] = 0

            cpm_bulks[tgt] = vec_t
            z_bulks[tgt]   = (vec_t - ntc_median) / scaled_mad

    # 4) assemble DataFrames
    genes = adata.var_names
    df_cpm = pd.DataFrame(cpm_bulks, index=genes)
    df_z   = pd.DataFrame(z_bulks, index=genes)
    df_z['NTC'] = 0.0  # add a zero‐column for NTC

    ntc_median_df = pd.DataFrame(ntc_medians, index=genes)
    ntc_mad_df = pd.DataFrame(ntc_mads,index=genes)

    return df_z, df_cpm, ntc_median_df, ntc_mad_df


def bulk_by_gene_target_mean(adata):
    """
    Bulk the AnnData object by 'gene_target' via mean.
    Returns a DataFrame with genes as rows and gene_target groups as columns.
    """
    # Extract expression data
    X = adata.X
    # Convert sparse matrix to dense if necessary
    if hasattr(X, 'toarray'):
        data = X.toarray()
    else:
        data = X
    # Build DataFrame (cells x genes)
    expr_df = pd.DataFrame(data, index=adata.obs.index, columns=adata.var_names)
    # Add gene_target labels
    expr_df['gene_target'] = adata.obs['gene_target']
    # Group by gene_target and compute mean, then transpose to get genes x groups
    bulk_df = expr_df.groupby('gene_target').mean().T
    return bulk_df


def bulk_by_sgRNA_mean(adata):
    """
    Bulk the AnnData object by 'gene_target' via mean.
    Returns a DataFrame with genes as rows and gene_target groups as columns.
    """
    # Extract expression data
    X = adata.X
    # Convert sparse matrix to dense if necessary
    if hasattr(X, 'toarray'):
        data = X.toarray()
    else:
        data = X
    # Build DataFrame (cells x genes)
    expr_df = pd.DataFrame(data, index=adata.obs.index, columns=adata.var_names)
    # Add gene_target labels
    expr_df['perturbation'] = adata.obs['perturbation']
    # Group by gene_target and compute mean, then transpose to get genes x groups
    bulk_df = expr_df.groupby('perturbation').mean().T
    return bulk_df

# Define the function to compute the mean normalized profile
def compute_mean_profile(adata, group_indices):
    mean_vector = adata[group_indices, :].X.mean(axis=0)
    return mean_vector.A1 if hasattr(mean_vector, 'A1') else mean_vector

def create_corrmatrix(adata, title='', mode='perturb', vmax=None, vmin=None):
    gene_target_groups = adata.obs.groupby('gene_target').indices
        # Initialize tqdm progress bar
    with tqdm_joblib(desc="Computing mean profiles", total=len(gene_target_groups)) as progress_bar:
        # Compute the mean expression profile for each group in parallel
        with parallel_backend('threading'):
            mean_profiles = Parallel(n_jobs=-1)(
                delayed(compute_mean_profile)(adata, indices) 
                for indices in gene_target_groups.values()
            )
    mean_profiles = pd.DataFrame(np.array(mean_profiles).T,index=list(adata.var.index),columns=list(gene_target_groups.keys()))
    if mode == 'perturb':
        corr_matrix = np.corrcoef(mean_profiles.T)
        distance_df = pd.DataFrame(corr_matrix, index=mean_profiles.columns, columns=mean_profiles.columns)
    elif mode == 'gene':
        corr_matrix = np.corrcoef(mean_profiles)
        distance_df = pd.DataFrame(corr_matrix, index=mean_profiles.index, columns=mean_profiles.index)
    lim = np.percentile(abs(corr_matrix.flatten()), 98)
    if vmax is None:
        vmax = lim
    if vmin is None:
        vmin = -lim
    fig = sns.clustermap(distance_df,cmap="RdBu",xticklabels=False, yticklabels=False,center=0, vmax=vmax, vmin=vmin)
    fig.ax_heatmap.set_xlabel('')
    fig.ax_heatmap.set_ylabel('');
    fig.ax_heatmap.set_title(title)
    fig.ax_row_dendrogram.set_visible(False)
    fig.ax_col_dendrogram.set_visible(False)
    #distance_df.to_csv("/tscc/projects/ps-malilab/ydoctor/iPSC_Pan_Genome/Pan_Genome/output_files/Aggregate_correlation_distances.csv")
    #fig.savefig("/tscc/projects/ps-malilab/ydoctor/iPSC_Pan_Genome/Pan_Genome/output_files/Aggregate_clustermap.svg")
    return(fig,distance_df,mean_profiles)



def get_corum_perturbation_pairs_from_list(
    perturbations, corum_complexes, min_fraction=0.66, random_state=42, sample = False
):
    """
    Identify CORUM-based perturbation links and sample an equal number
    of non-CORUM pairs.

    Only complexes with at least `min_fraction` of their members
    represented in `perturbations` are considered.

    Parameters
    ----------
    perturbations : list of str
        Perturbation identifiers (e.g., gene symbols).
    corum_complexes : dict
        Mapping from complex ID (or name) to list of member gene symbols.
    min_fraction : float, optional (default=0.66)
        Minimum fraction of a complex’s members that must appear in
        `perturbations` to include that complex.
    random_state : int, optional
        Seed for reproducible negative sampling.

    Returns
    -------
    same_complex_pairs : list of tuple
        Sorted list of unique (g1, g2) pairs drawn from filtered CORUM complexes.
    negative_pairs_sampled : list of tuple
        Randomly sampled list of pairs not present in any filtered complex,
        of the same length as `same_complex_pairs`.
    """
    import itertools
    import random

    # ensure unique perturbations
    perturbs = sorted(set(perturbations))

    # filter complexes by representation threshold
    filtered = {}
    num_present = 0
    for cid, members in corum_complexes.items():
        if len(set(members)) == 1:  # removes homodimers and complexes with identical members
            continue
        present = [g for g in members if g in perturbs]
        if members and (len(present) / len(members)) >= min_fraction:
            filtered[cid] = present
            num_present += 1
    
    print(f"{num_present} CORUM complexes present")

    # build positive CORUM pairs
    same_complex_pairs = set()
    for members in filtered.values():
        for g1, g2 in itertools.combinations(sorted(members), 2):
            if g1 == g2: #removes genes listed twice in the same complex
                continue
            same_complex_pairs.add((g1, g2))
    same_complex_pairs = sorted(same_complex_pairs)

    # all possible perturbation pairs
    all_pairs = set(itertools.combinations(perturbs, 2))

    # negatives are the pairs not in CORUM
    negative_candidates = list(all_pairs - set(same_complex_pairs))
    n_pos = len(same_complex_pairs)
    if n_pos > len(negative_candidates):
        raise ValueError(
            f"Need at least {n_pos} non-complex pairs, "
            f"but only {len(negative_candidates)} available."
        )

    # sample negatives
    if sample: 
        rng = random.Random(random_state)
        negative_pairs_sampled = rng.sample(negative_candidates, n_pos)

        return same_complex_pairs, negative_pairs_sampled
    else:
        return same_complex_pairs, negative_candidates


def benchmark_corum_vs_noncorum_pairs(data, corum_pairs, noncorum_pairs, group_key='gene_target'):
    """
    Compare distributions of pairwise values for CORUM vs non-CORUM pairs
    and perform a Mann-Whitney U test.

    Parameters
    ----------
    data : AnnData or pd.DataFrame
        If AnnData, compute a correlation matrix using create_corrmatrix and group_key.
        If DataFrame, use it directly as a precomputed pairwise matrix (rows/columns indexed by gene IDs).
    corum_pairs : list of tuple
        List of (g1, g2) pairs in the same CORUM complex.
    noncorum_pairs : list of tuple
        List of (g1, g2) pairs not in any CORUM complex.
    group_key : str, optional
        Key in adata.obs to group cells when computing correlations (ignored if data is DataFrame).

    Returns
    -------
    u_stat : float
        Mann-Whitney U statistic.
    p_value : float
        Two-sided p-value of the test.
    """
    import pandas as pd
    from anndata import AnnData

    # Decide whether to compute or reuse a pairwise matrix
    if isinstance(data, AnnData):
        corr_mat, _ = create_corrmatrix(data, '')
    elif isinstance(data, pd.DataFrame):
        corr_mat = data
    else:
        raise ValueError("`data` must be an AnnData object or a pandas DataFrame")

    # Extract the pairwise values
    corum_vals = [
        corr_mat.loc[g1, g2] if (g1 in corr_mat.index and g2 in corr_mat.columns) else 0
        for g1, g2 in corum_pairs
    ]
    noncorum_vals = [
        corr_mat.loc[g1, g2] if (g1 in corr_mat.index and g2 in corr_mat.columns) else 0
        for g1, g2 in noncorum_pairs
    ]

    # Plot the two distributions using KDE
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.set_context("paper", font_scale=1.2)
    
    # Plot KDEs
    sns.kdeplot(data=noncorum_vals, label='Non-CORUM pairs', color='#e74c3c', fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(data=corum_vals, label='CORUM pairs', color='#2ecc71', fill=True, alpha=0.3, ax=ax)
    
    # Customize plot
    ax.set_xlabel('Correlation', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    
    # Add legend with custom formatting
    ax.legend(frameon=False, fontsize=14)
    
    # Adjust layout and remove grid
    fig.tight_layout()
    ax.grid(False)
    
    # Make x-ticks larger
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.25, 0.75)
    
    # Remove top and right spines
    sns.despine(ax=ax)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.7)
    
    # Do not call plt.show() so the figure can be returned

    # Perform Mann-Whitney U test
    from scipy.stats import mannwhitneyu
    u_stat, p_value = mannwhitneyu(corum_vals, noncorum_vals, alternative='two-sided')
    print(f"Mann-Whitney U statistic: {u_stat:}, p-value: {p_value}")

    return fig, u_stat, p_value

import numpy as np
from sklearn.cluster import HDBSCAN
import pandas as pd

def compute_HDBSCAN_clusters(correlations, clusterer, pallete_clustered = None, vmin = None, vmax = None):
    labels = clusterer.fit_predict(1-correlations)
    labels_series = pd.Series(labels, index=correlations.index)
    # Map each cluster label to a distinct color
    unique_labels = sorted(labels_series.unique())
    palette = sns.color_palette("tab20", n_colors=len(unique_labels))
    lut = dict(zip(unique_labels, palette))
    row_colors = labels_series.map(lut)
    if vmin is None:
        vmin = -np.percentile(abs(correlations.values.flatten()), 98)
    if vmax is None:
        vmax = np.percentile(abs(correlations.values.flatten()), 98)
    # Plot a clustermap of the similarity matrix, annotating clusters with colored bars
    g = sns.clustermap(
        correlations,
        cmap="RdBu",
        vmin=vmin,
        vmax=vmax,
        center=0,
        row_colors=row_colors,
        col_colors=row_colors,
        xticklabels=False,
        yticklabels=False
    )
    # Build a legend for the cluster colors
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    plt.show()
    print(f"HDBSCAN Found {labels_series.max()+1} Clusters")
    clustered_genes = labels_series[labels_series != -1].index
    sim_clustered = correlations.loc[clustered_genes, clustered_genes]
    print(f"Correlation matrix on only genes assigned a HDBScan Cluster: {len(sim_clustered)} total genes")
    labels_clustered = labels_series.loc[clustered_genes]
    unique_labels_clustered = sorted(labels_clustered.unique())
    if pallete_clustered is None:
        palette_clustered = sns.color_palette("Spectral", n_colors=len(unique_labels_clustered))
    else:
        palette_clustered = pallete_clustered
    lut_clustered = dict(zip(unique_labels_clustered, palette_clustered))
    row_colors_clustered = labels_clustered.map(lut_clustered)
    # Plot a clustermap on the clustered subset
    lim = np.percentile(abs(sim_clustered.values.flatten()), 98)
    g2 = sns.clustermap(
        sim_clustered,
        cmap="RdBu",
        vmin=-.5, vmax=.5,
        center=0,
        row_colors=row_colors_clustered,
        col_colors=row_colors_clustered,
        linewidths=0,      # <-- no lines between cells
        linecolor=None,    # <-- explicit "no line" color
        xticklabels=False,
        yticklabels=False
    )
    # Hide dendrograms for cleaner presentation
    g2.ax_row_dendrogram.set_visible(False)
    g2.ax_col_dendrogram.set_visible(False)
    # Plot the LUT for clustered genes only (excluding noise)
    fig_lut2, ax_lut2 = plt.subplots(figsize=(max(6, len(unique_labels_clustered) * 0.5), 1))
    for idx, label in enumerate(unique_labels_clustered):
        ax_lut2.bar(idx, 1, color=lut_clustered[label], edgecolor='none')
        ax_lut2.text(idx, 1.05, str(label), ha='center', va='bottom', fontsize=10, rotation=90)
    ax_lut2.set_xticks([])
    ax_lut2.set_yticks([])
    # Add some padding to the x-axis limits for better visualization
    padding = 0.5
    ax_lut2.set_xlim(-0.5 - padding, len(unique_labels_clustered) - 0.5 + padding)
    ax_lut2.set_title("Cluster Color LUT")
    plt.tight_layout()
    plt.show()

    return g, g2, labels_series


def compute_HDBSCAN_clusters_20d_embedding(bulks, correlations, clusterer, axis = 1, pallete_clustered = None):
    dat = standardize(bulks, axis = axis)
    import pymde
    pymde.seed(42)
    mde = pymde.preserve_neighbors(dat.values.T, embedding_dim=20, n_neighbors=7,repulsive_fraction=5)
    Y = mde.embed(verbose=True, eps=1e-9, max_iter=1000)
    labels = clusterer.fit_predict(Y)
    labels_series = pd.Series(labels, index=bulks.index)
    # Map each cluster label to a distinct color
    unique_labels = sorted(labels_series.unique())
    palette = sns.color_palette("tab20", n_colors=len(unique_labels))
    lut = dict(zip(unique_labels, palette))
    row_colors = labels_series.map(lut)
    lim = np.percentile(abs(correlations.values.flatten()), 98)
    # Plot a clustermap of the similarity matrix, annotating clusters with colored bars
    g = sns.clustermap(
        correlations,
        cmap="RdBu",
        vmin=-lim,
        vmax=lim,
        center=0,
        row_colors=row_colors,
        col_colors=row_colors,
        xticklabels=False,
        yticklabels=False
    )
    # Build a legend for the cluster colors
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    plt.show()
    print(f"HDBSCAN Found {labels_series.max()+1} Clusters")
    clustered_genes = labels_series[labels_series != -1].index
    sim_clustered = correlations.loc[clustered_genes, clustered_genes]
    print(f"Correlation matrix on only genes assigned a HDBScan Cluster: {len(sim_clustered)} total genes")
    labels_clustered = labels_series.loc[clustered_genes]
    unique_labels_clustered = sorted(labels_clustered.unique())
    palette_clustered = sns.color_palette("tab20", n_colors=len(unique_labels_clustered))
    if pallete_clustered is None:
        palette_clustered = sns.color_palette("hls", n_colors=len(unique_labels_clustered))
    else:
        palette_clustered = pallete_clustered
    lut_clustered = dict(zip(unique_labels_clustered, palette_clustered))
    row_colors_clustered = labels_clustered.map(lut_clustered)
    # Plot a clustermap on the clustered subset
    lim = np.percentile(abs(sim_clustered.values.flatten()), 98)
    g2 = sns.clustermap(
        sim_clustered,
        cmap="RdBu",
        vmin=-.5, vmax=.5,
        center=0,
        row_colors=row_colors_clustered,
        col_colors=row_colors_clustered,
        linewidths=0,      # <-- no lines between cells
        linecolor=None,    # <-- explicit "no line" color   
        xticklabels=False,
        yticklabels=False
    )
    # Hide dendrograms for cleaner presentation
    g2.ax_row_dendrogram.set_visible(False)
    g2.ax_col_dendrogram.set_visible(False)
     # Plot the LUT for clustered genes only (excluding noise)
    fig_lut2, ax_lut2 = plt.subplots(figsize=(max(6, len(unique_labels_clustered) * 0.5), 1))
    for idx, label in enumerate(unique_labels_clustered):
        ax_lut2.bar(idx, 1, color=lut_clustered[label], edgecolor='none')
        ax_lut2.text(idx, 1.05, str(label), ha='center', va='bottom', fontsize=10, rotation=90)
    ax_lut2.set_xticks([])
    ax_lut2.set_yticks([])
    # Add some padding to the x-axis limits for better visualization
    padding = 0.5
    ax_lut2.set_xlim(-0.5 - padding, len(unique_labels_clustered) - 0.5 + padding)
    ax_lut2.set_title("Cluster Color LUT")
    plt.tight_layout()
    plt.show()
    return g, g2, labels_series


def compute_HDBSCAN_clusters_20d_embedding_T(bulks, correlations, clusterer, axis = 0, pallete_clustered = None):
    dat = standardize(bulks, axis = axis)
    import pymde
    pymde.seed(42)
    mde = pymde.preserve_neighbors(dat.values, embedding_dim=20, n_neighbors=7,repulsive_fraction=5)
    Y = mde.embed(verbose=True, eps=1e-9, max_iter=1000)
    labels = clusterer.fit_predict(Y)
    labels_series = pd.Series(labels, index=bulks.T.index)
    # Map each cluster label to a distinct color
    unique_labels = sorted(labels_series.unique())
    palette = sns.color_palette("tab20", n_colors=len(unique_labels))
    lut = dict(zip(unique_labels, palette))
    row_colors = labels_series.map(lut)
    lim = np.percentile(abs(correlations.values.flatten()), 98)
    # Plot a clustermap of the similarity matrix, annotating clusters with colored bars
    g = sns.clustermap(
        correlations,
        cmap="RdBu",
        vmin=-lim,
        vmax=lim,
        center=0,
        row_colors=row_colors,
        col_colors=row_colors,
        xticklabels=False,
        yticklabels=False
    )
    # Build a legend for the cluster colors
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    plt.show()
    print(f"HDBSCAN Found {labels_series.max()+1} Clusters")
    clustered_genes = labels_series[labels_series != -1].index
    sim_clustered = correlations.loc[clustered_genes, clustered_genes]
    print(f"Correlation matrix on only genes assigned a HDBScan Cluster: {len(sim_clustered)} total genes")
    labels_clustered = labels_series.loc[clustered_genes]
    unique_labels_clustered = sorted(labels_clustered.unique())
    palette_clustered = sns.color_palette("tab20", n_colors=len(unique_labels_clustered))
    if pallete_clustered is None:
        palette_clustered = sns.color_palette("hls", n_colors=len(unique_labels_clustered))
    else:
        palette_clustered = pallete_clustered
    lut_clustered = dict(zip(unique_labels_clustered, palette_clustered))
    row_colors_clustered = labels_clustered.map(lut_clustered)
    # Plot a clustermap on the clustered subset
    lim = np.percentile(abs(sim_clustered.values.flatten()), 98)
    g2 = sns.clustermap(
        sim_clustered,
        cmap="RdBu",
        vmin=-.5, vmax=.5,
        center=0,
        row_colors=row_colors_clustered,
        col_colors=row_colors_clustered,
        xticklabels=False,
        yticklabels=False
    )
    # Hide dendrograms for cleaner presentation
    g2.ax_row_dendrogram.set_visible(False)
    g2.ax_col_dendrogram.set_visible(False)
    # Plot the LUT for clustered genes only (excluding noise)
    fig_lut2, ax_lut2 = plt.subplots(figsize=(max(6, len(unique_labels_clustered) * 0.5), 1))
    for idx, label in enumerate(unique_labels_clustered):
        ax_lut2.bar(idx, 1, color=lut_clustered[label], edgecolor='none')
        ax_lut2.text(idx, 1.05, str(label), ha='center', va='bottom', fontsize=10, rotation=90)
    ax_lut2.set_xticks([])
    ax_lut2.set_yticks([])
    # Add some padding to the x-axis limits for better visualization
    padding = 0.5
    ax_lut2.set_xlim(-0.5 - padding, len(unique_labels_clustered) - 0.5 + padding)
    ax_lut2.set_title("Cluster Color LUT")
    plt.tight_layout()
    plt.show()
    return g, g2, labels_series

# # Create a DataFrame for the embedding
# embedding_df = pd.DataFrame(Y, columns=['x', 'y'])
# embedding_df['gene_target'] = bulks.index
# embedding_df.index = embedding_df['gene_target']


def bulk_by_gene_target_mean(adata):
    """
    Bulk the AnnData object by 'gene_target' via mean.
    Returns a DataFrame with genes as rows and gene_target groups as columns.
    """
    # Extract expression data
    X = adata.X
    # Convert sparse matrix to dense if necessary
    if hasattr(X, 'toarray'):
        data = X.toarray()
    else:
        data = X
    # Build DataFrame (cells x genes)
    expr_df = pd.DataFrame(data, index=adata.obs.index, columns=adata.var_names)
    # Add gene_target labels
    expr_df['gene_target'] = adata.obs['gene_target']
    # Group by gene_target and compute mean, then transpose to get genes x groups
    bulk_df = expr_df.groupby('gene_target').mean().T
    return bulk_df

import random
def relative_z_normalization_df(
    adata,
    batch_key: str = 'batch',
    ctrl_key: str = 'perturbed',
    ctrl_value: str = 'False',
    save_layer: str = 'pre_z_normalization'
) -> None:
    """
    Batch‐aware z‐normalization using pandas DataFrames.

    Parameters
    ----------
    adata
        AnnData object with .obs containing batch_key and ctrl_key columns.
    batch_key
        Column in adata.obs defining batch membership.
    ctrl_key
        Column in adata.obs defining control vs perturbed cells.
    ctrl_value
        Value in ctrl_key that denotes controls.
    save_layer
        Name under which to save the original adata.X.

    Side effects
    ------------
    - Saves original adata.X into adata.layers[save_layer]
    - Overwrites adata.X with the batch‐normalized values (dense numpy array)
    """
 # 1) save original
    if save_layer in adata.layers:
        raise ValueError(f"Layer '{save_layer}' already exists")
    adata.layers[save_layer] = adata.X.copy()

    # 2) pull out dense X as a numpy array
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()     # now X is (n_obs, n_vars) numpy
    else:
        X = X.copy()        # be safe if adata.X is a view

    n_obs, n_vars = X.shape
    out = np.zeros_like(X)  # final z-normalized pseudobulk

    # 3) for each batch, compute mu/std on controls and fill out
    obs = adata.obs
    for batch in obs[batch_key].unique():
        mask_batch = (obs[batch_key] == batch).values
        mask_ctrl  = mask_batch & (obs[ctrl_key] == ctrl_value).values

        # get control pseudocounts and compute mean/std per gene
        ctrl_mat = X[mask_ctrl]           # shape (#ctrl_cells, n_vars)
        mu  = ctrl_mat.mean(axis=0)       # (n_vars,)
        std = ctrl_mat.std(axis=0)
        std[std == 0] = 1.0

        # z-normalize all cells in this batch
        # broadcasting: (n_cells_in_batch, n_vars) - (n_vars,) / (n_vars,)
        out[mask_batch] = (X[mask_batch] - mu) / std

    # 4) clip in-place
    np.clip(out, -10, 10, out=out)

    # 5) write back
    adata.X = out

### Here, we are selecting the top 10 DEGs per perturbation and getting the union of these features
### Here, we are selecting the top 10 DEGs per perturbation and getting the union of these features
def extract_top_up_down_degs(combined_df, gene_targets, up_degs, down_degs,l2fc_threshold = 0):
    """
    Extract the top x upregulated (L2FC > 0) and top y downregulated (L2FC < 0) DEGs (genes) for each perturbation.
    
    Parameters:
    - combined_df (pd.DataFrame): The combined DataFrame with DEGs, L2FC, and Adj_P columns for each perturbation.
    - x (int): The number of top upregulated DEGs to extract.
    - y (int): The number of top downregulated DEGs to extract.
    
    Returns:
    - up_down_degs (dict): A dictionary where keys are perturbation names and values are lists of the top upregulated and downregulated DEGs (genes).
    """
    up_down_degs = set()
    perturbations = [col.split('_DEGs')[0] for col in combined_df.columns if '_DEGs' in col]
    perturbations = [p for p in perturbations if p.split('_')[0] in gene_targets]
    deg_l2fc_dict = {}
    for perturbation in perturbations:
        # Select the DEGs and L2FC columns for the perturbation
        degs_col = f'{perturbation}_DEGs'
        l2fc_col = f'{perturbation}_L2FC'
        
        # Create a DataFrame for the perturbation
        perturbation_df = combined_df[[degs_col, l2fc_col]].dropna()
        
        # Extract top x upregulated DEGs with their L2FCs
        upregulated_df = perturbation_df[perturbation_df[l2fc_col] > l2fc_threshold].sort_values(by=l2fc_col, ascending=False).head(up_degs)
        top_upregulated = upregulated_df[degs_col].tolist()
        top_upregulated_l2fc = upregulated_df[l2fc_col].tolist()
        
        # Extract top y downregulated DEGs with their L2FCs
        downregulated_df = perturbation_df[perturbation_df[l2fc_col] < -l2fc_threshold].sort_values(by=l2fc_col, ascending=True).head(down_degs)
        top_downregulated = downregulated_df[degs_col].tolist()
        top_downregulated_l2fc = downregulated_df[l2fc_col].tolist()
        
        # Combine the results
        up_down_degs.update(top_upregulated + top_downregulated)   
    return up_down_degs


def extract_HVGs(adata, umi_thresh=0.25, top_pct = 30):
    gene_means = np.asarray(adata.layers['pre_z_normalization'].mean(axis=0)).ravel()    # mean across cells for each gene
    genes_to_keep_by_count = gene_means > umi_thresh
    print(f"Genes with more than mean 0.25 UMIs: {sum(genes_to_keep_by_count)}")
    gene_variances = np.asarray(adata.X.var(axis=0)).ravel()
    genes_to_keep_by_var = gene_variances > np.percentile(gene_variances, 100-top_pct)
    print(f"Genes in the top 30th percent of variance: {sum(genes_to_keep_by_var)}")
    total_genes_to_keep = genes_to_keep_by_count & genes_to_keep_by_var
    print(f"Total HVGs: {sum(total_genes_to_keep)}")
    return adata.var.index[total_genes_to_keep]



def replogle_pipeline(adata, DEGs, batch_key = 'batch', n_up_degs = 5, n_down_degs = 5, umi_thresh = 0.25, pct_hvgs=30, l2fc_threshold = 0):
    # Normalization to median UMIs based on the NTCs with the least number of DEGs.
    median_NTC_umi_counts = {}
    for batch_name in adata.obs[batch_key].unique():
        median_UMIs_batch = get_ntc_view(adata[adata.obs[batch_key]==batch_name]).obs.n_UMI_counts.median()
        median_NTC_umi_counts[batch_name] = median_UMIs_batch

    min_median_umi_value = min(median_NTC_umi_counts.values())
    min_median_umi_key = min(median_NTC_umi_counts.items(), key=lambda x: x[1])[0]
    print(f"Minimum median NTC UMI count: {min_median_umi_value} from batch {min_median_umi_key}\n Normalizing to {min_median_umi_value} UMIs per cell")
    sc.pp.normalize_total(adata, target_sum = min_median_umi_value)

    #Applying relative Z-normalization
    print(f"Applying relative z-normalization per batch")
    relative_z_normalization_df(adata)

    #Extracting DEG based features
    gene_targets = adata.obs['gene_target'].unique()
    degs = extract_top_up_down_degs(DEGs, gene_targets, n_up_degs, n_down_degs, l2fc_threshold)
    print(f"Identified {len(degs)} unique genes that are the union of the top {n_up_degs} and bottom {n_down_degs} DEGs per perturbation")

    #Extracting HVG based features
    print("Identifying HVGs")
    hvgs = extract_HVGs(adata, umi_thresh=umi_thresh, top_pct=pct_hvgs)

    #Getting the union of HVGs
    total_hvgs = set(degs).union(set(hvgs))
    print(f"Building a feature set of {len(total_hvgs)} genes, comprising of the union of {len(degs)} DEGs and {len(hvgs)} HVGs.")
    adata_subset = adata[:,adata.var.index.isin(total_hvgs)].copy()

    return adata_subset


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple

def DEG_cosine_similarity(
    DEGs_df: pd.DataFrame,
    deg_suffix: str = '_DEGs',
    l2fc_suffix: str = '_L2FC',
    min_valid: int = 2,
    simplify_names: bool = True,
    cmap: str = 'RdBu',
    vmin: float = -.5,
    vmax: float = .5,
    center: float = 0,
    figsize: Tuple[float, float] = (10, 10),
    cbar_label: str = 'Cosine Similarity'
) -> Tuple[pd.DataFrame, sns.matrix.ClusterGrid]:
    """
    1) Load a table of DEGs per perturbation (expects columns like 'X_DEGs' and 'X_L2FC').
    2) Filters out perturbations with <= min_valid valid DEG/L2FC pairs.
    3) Builds a perturbation × gene matrix of L2FC values.
    4) Computes cosine similarity between perturbations.
    5) Plots & returns the clustermap.

    Returns:
        sim_df: DataFrame of pairwise cosine similarities.
        grid:   Seaborn ClusterGrid object for the clustermap.
    """
    # 1) Load
    degs = DEGs_df

    # 2) Identify perturbations
    perturbs = sorted({
        col[:-len(deg_suffix)]
        for col in degs.columns
        if col.endswith(deg_suffix)
    })

    # 3) Remove any with too few valid rows
    bad = [
        p for p in perturbs
        if degs.dropna(subset=[f"{p}{deg_suffix}", f"{p}{l2fc_suffix}"]).shape[0] < min_valid
    ]
    if bad:
        perturbs = [p for p in perturbs if p not in bad]
        print(f"Number of perturbations with one or less valid DEGs/L2FC: {len(bad)}")
        print(f"Removing perturbations: {bad}")

    # 4) Union of all DEG genes
    union_genes = sorted({
        gene
        for p in perturbs
        for gene in degs[f"{p}{deg_suffix}"].dropna().astype(str)
    })

    # 5) Build L2FC matrix
    mat = pd.DataFrame(0.0, index=perturbs, columns=union_genes)
    for p in perturbs:
        sub = degs.dropna(subset=[f"{p}{deg_suffix}", f"{p}{l2fc_suffix}"])
        genes = sub[f"{p}{deg_suffix}"].astype(str).tolist()
        l2fcs = sub[f"{p}{l2fc_suffix}"].tolist()
        mat.loc[p, genes] = l2fcs
    
    # 5) Remove rows where 0 or 1 entries are zero
    mask = (mat != 0).sum(axis=0) >= 2
    mat = mat.loc[:, mask]
    print(f"Removed {len(union_genes) - mat.shape[1]} genes with less than 2 entries that are nonzero.")

    # 6) Cosine similarity
    pert_pert_sim = pd.DataFrame(
        cosine_similarity(mat.values),
        index=mat.index,
        columns=mat.index
    )
    gene_gene_sim = pd.DataFrame(
        cosine_similarity(mat.values.T),
        index=mat.columns,
        columns=mat.columns
    )

    # 7) Simplify names if requested
    if simplify_names:
        pert_pert_sim = pert_pert_sim.rename(
            index=lambda x: x.split('_')[0],
            columns=lambda x: x.split('_')[0]
        )
        gene_gene_sim = gene_gene_sim.rename(
            index=lambda x: x.split('_')[0],
            columns=lambda x: x.split('_')[0]
        )
        mat = mat.rename(
            index=lambda x: x.split('_')[0],
            columns=lambda x: x.split('_')[0]
        )

    # 8) Plot
    grid1 = sns.clustermap(
        pert_pert_sim,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': cbar_label}
    )
    grid1.ax_row_dendrogram.set_visible(False)
    grid1.ax_col_dendrogram.set_visible(False)
    grid1.ax_heatmap.set_title('Perturbation-Perturbation Cosine Similarity')
    grid1.fig.set_size_inches(*figsize)
    plt.show()

    grid2 = sns.clustermap(
        gene_gene_sim,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': cbar_label}
    )
    grid2.ax_row_dendrogram.set_visible(False)
    grid2.ax_col_dendrogram.set_visible(False)
    grid2.ax_heatmap.set_title('Gene-Gene Cosine Similarity')
    grid2.fig.set_size_inches(*figsize)
    plt.show()
    return mat.T, pert_pert_sim, gene_gene_sim, grid1, grid2

def plot_pert_gene_summary_heatmap(gene_HDBSCAN_labels, perturbation_HDBSCAN_labels, mp, vmin = -0.4, vmax = 0.4):
    # Create a matrix to store mean z-scores between gene and perturbation clusters
    gene_clusters = gene_HDBSCAN_labels
    perturb_clusters = perturbation_HDBSCAN_labels

    # Get unique cluster labels (excluding noise points labeled as -1)
    unique_gene_clusters = sorted(set(gene_clusters) - {-1})
    unique_perturb_clusters = sorted(set(perturb_clusters) - {-1})

    # Initialize the matrix
    cluster_matrix = np.zeros((len(unique_gene_clusters), len(unique_perturb_clusters)))

    # For each gene cluster and perturbation cluster pair
    for i, gene_cluster in enumerate(unique_gene_clusters):
        for j, perturb_cluster in enumerate(unique_perturb_clusters):
            # Get genes in this gene cluster
            genes_in_cluster = np.where(gene_clusters == gene_cluster)[0]
            # Get perturbations in this perturbation cluster
            perts_in_cluster = np.where(perturb_clusters == perturb_cluster)[0]
            
            # Calculate mean z-score for this cluster pair
            cluster_matrix[i, j] = np.mean(mp.iloc[genes_in_cluster, perts_in_cluster].values)

    # Create a DataFrame for better visualization
    cluster_df = pd.DataFrame(
        cluster_matrix,
        index=[f'Gene Cluster {c}' for c in unique_gene_clusters],
        columns=[f'Perturb Cluster {c}' for c in unique_perturb_clusters]
    )

    # Create row and column colors based on cluster identity
    row_colors = sns.color_palette('hls', len(unique_gene_clusters))
    col_colors = sns.color_palette('Spectral', len(unique_perturb_clusters))

    # Create a dictionary mapping cluster numbers to colors
    row_color_dict = dict(zip(unique_gene_clusters, row_colors))
    col_color_dict = dict(zip(unique_perturb_clusters, col_colors))

    # Create color vectors for rows and columns
    row_colors = [row_color_dict[c] for c in unique_gene_clusters]
    col_colors = [col_color_dict[c] for c in unique_perturb_clusters]

    # Plot the heatmap with colored rows and columns
    g = sns.clustermap(cluster_df, 
                cmap='RdBu', 
                center=0,
                vmax=vmax,
                vmin=vmin,
                row_colors=row_colors,
                col_colors=col_colors,
                xticklabels=False,  # Turn off x ticks
                yticklabels=False,
                square=False)  # Turn off y ticks

    g.fig.set_size_inches(12, 12)

    # 1) Annotate the row‐color bar with the true gene‐cluster IDs in the clustered order:
    for new_pos, orig_idx in enumerate(g.dendrogram_row.reordered_ind):
        gene_cluster = unique_gene_clusters[orig_idx]
        # place the text at x = -0.5 (just left of the color bar), y = new_pos + 0.5
        g.ax_row_colors.text(-0.5, new_pos + 0.5, str(gene_cluster),
                            ha='center', va='center', color='black')

    # 2) Annotate the column‐color bar similarly:
    for new_pos, orig_idx in enumerate(g.dendrogram_col.reordered_ind):
        perturb_cluster = unique_perturb_clusters[orig_idx]
        # place the text at x = new_pos + 0.5, y = -0.5 (just above the color bar)
        g.ax_col_colors.text(new_pos + 0.5, -0.5, str(perturb_cluster),
                            ha='center', va='center', color='black')
    g.ax_heatmap.set_xlabel('Perturbation Clusters')
    g.ax_heatmap.set_ylabel('Gene Clusters')
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    plt.show()


def standardize(df, axis = 0):
    col_means = df.values.mean(axis=axis, keepdims=True)  # shape: (1, n_columns)
    col_stds  = df.values.std(axis=axis, keepdims=True)
    col_stds[col_stds == 0] = 1

    mp_standardized = (df.values - col_means) / col_stds
    mp_standardized = pd.DataFrame(mp_standardized, index=df.index, columns=df.columns).T
    return mp_standardized


def embed_spectral_mde(mp_standardized, n_components = 20, n_neighbors = 7, random_state = 42, repulsive_fraction = 5):
    embedder = SpectralEmbedding(
        n_components=n_components,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        eigen_solver='arpack',
        random_state=random_state
    )
    X = embedder.fit_transform(mp_standardized)
    pymde.seed(random_state)
    mde = pymde.preserve_neighbors(X, embedding_dim=2, n_neighbors=n_neighbors,repulsive_fraction=repulsive_fraction)
    Y = mde.embed(verbose=True, eps=1e-9, max_iter=6000)
    return Y

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
import igraph as ig
import leidenalg
from sklearn.neighbors import kneighbors_graph

def plot_cluster_agreement(Y, perturbation_HDBSCAN_labels):
    # Make everything deterministic
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    sns.set(style="whitegrid", font_scale=1.2)

    # Define neighbor sizes to test
    neighbor_sizes = list(range(2, 20))

    # Restrict to points assigned by HDBSCAN
    mask = perturbation_HDBSCAN_labels >= 0
    hdb_labels = perturbation_HDBSCAN_labels[mask].values

    # Evaluate clustering agreement for each neighbor size
    ari_scores = []
    ami_scores = []
    for k in neighbor_sizes:
        knn_graph = kneighbors_graph(Y, n_neighbors=k, include_self=False)
        sources, targets = knn_graph.nonzero()
        g = ig.Graph(directed=False)
        g.add_vertices(Y.shape[0])
        g.add_edges(list(zip(sources, targets)))
        # Leiden clustering with fixed seed
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            seed=seed
        )
        leiden_labels = np.array(partition.membership)
        # Compute metrics
        ari_scores.append(adjusted_rand_score(hdb_labels, leiden_labels[mask]))
        ami_scores.append(adjusted_mutual_info_score(hdb_labels, leiden_labels[mask]))

    # Plot results
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(neighbor_sizes, ari_scores, marker='o', linestyle='-', color='C0', label='ARI')
    plt.plot(neighbor_sizes, ami_scores, marker='s', linestyle='--', color='C1', label='AMI')
    plt.xlabel('Number of Leiden Neighbors (k)')
    plt.ylabel('Adjusted score')
    plt.title('Clustering agreement vs. k (Leiden vs. HDBSCAN)')
    plt.xticks(neighbor_sizes)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()



import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
from sklearn.neighbors import kneighbors_graph

def plot_mde(Y, perturbation_HDBSCAN_labels, bulk_df, n_neighbors = 6, corum_complexes = None, seed = 12, save_dir_stem = None, marker_size = 7, marker_opacity = 0.7):
    corum, non_corum = get_corum_perturbation_pairs_from_list(bulk_df.columns.unique(), corum_complexes)

    # --- helper to convert "rgb(r, g, b)" strings to matplotlib tuples ---
    def _rgb_str_to_tuple(s: str):
        r, g, b = map(int, re.findall(r'\d+', s))
        return (r/255, g/255, b/255)

    # --- 1) Leiden clustering on the 2D coords ---
    knn_graph = kneighbors_graph(Y, n_neighbors=n_neighbors, include_self=False)
    sources, targets = knn_graph.nonzero()
    g = ig.Graph(directed=False)
    g.add_vertices(Y.shape[0])
    g.add_edges(list(zip(sources, targets)))
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        seed=seed
    )
    leiden_labels = np.array(partition.membership)

    # --- 2) Build embedding_df ---
    embedding_df = pd.DataFrame(Y, columns=['x', 'y'])
    embedding_df['gene_target']    = bulk_df.columns
    embedding_df['leiden cluster'] = leiden_labels.astype(str)
    hdbscan_map = dict(perturbation_HDBSCAN_labels)
    embedding_df['hdbscan cluster'] = embedding_df['gene_target'].map(hdbscan_map)
    embedding_df.index = embedding_df['gene_target']

    # order the Leiden labels numerically
    leiden_cats = sorted(embedding_df['leiden cluster'].unique(), key=lambda x: int(x))
    embedding_df['leiden cluster'] = pd.Categorical(
        embedding_df['leiden cluster'],
        categories=leiden_cats,
        ordered=True
    )

    # --- 3) Interactive Plotly scatter ---
    fig = px.scatter(
        embedding_df,
        x='x', y='y',
        text='gene_target',
        color='leiden cluster',
        category_orders={'leiden cluster': leiden_cats},
        hover_data=['x', 'y', 'gene_target', 'hdbscan cluster'],
        title='MDE Embedding of Mean Normalized Profiles',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(
        marker=dict(size=marker_size, opacity=marker_opacity),
        textposition='middle center',
        textfont=dict(size=1)
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(traceorder='normal'),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        width=1200,
        height=1000,
        legend_title_text='Leiden Cluster',
        coloraxis_showscale=False
    )

    import plotly.io as pio #type: ignore
    if save_dir_stem is not None:
        pio.write_html(fig, file=save_dir_stem+"_MDE.html", include_plotlyjs="inline",full_html=True)
        embedding_df.to_excel(save_dir_stem+"_MDE_clusters.xlsx", index=False)
    fig.show()

    fig,ax = plt.subplots(1,1,figsize=(10,10))
    # convert Plotly Bold colors to matplotlib-friendly RGB tuples
    bold_list     = px.colors.qualitative.Bold
    leiden_colors = [_rgb_str_to_tuple(c) for c in bold_list]
    leiden_colors = [leiden_colors[i % len(leiden_colors)] for i in range(len(leiden_cats))]
    for i, lc in enumerate(leiden_cats):
        sel = embedding_df['leiden cluster'] == lc
        ax.scatter(
            embedding_df.loc[sel, 'x'],
            embedding_df.loc[sel, 'y'],
            color=leiden_colors[i],
            s=marker_size,
            alpha=marker_opacity,
            edgecolor='white',
            linewidth=0.3,
            zorder=2
        )
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    # Save just this panel to an SVG file
    fig.savefig(save_dir_stem + "_MDE.svg", format='svg', bbox_inches='tight')


    # --- prepare CORUM membership and palettes for static plots ---
    corum_genes = {g for pair in corum for g in pair}
    is_in_corum  = embedding_df['gene_target'].isin(corum_genes)
    hdb          = embedding_df['hdbscan cluster']
    coords       = embedding_df.set_index('gene_target')[['x','y']]

    # convert Plotly Bold colors to matplotlib-friendly RGB tuples
    bold_list     = px.colors.qualitative.Bold
    leiden_colors = [_rgb_str_to_tuple(c) for c in bold_list]
    leiden_colors = [leiden_colors[i % len(leiden_colors)] for i in range(len(leiden_cats))]

    # HDBSCAN palette (exclude noise = -1)
    hdb_clusters = sorted(c for c in hdb.unique() if c != -1)
    hdb_palette  = sns.color_palette('hls', len(hdb_clusters))

    # seaborn clean style
    sns.set_style('white')
    sns.set_style({'axes.grid': False})

    # --- 4) Static 1×4 Matplotlib figure ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)

    # Panel 1: Leiden clusters
    ax = axes[0]
    for i, lc in enumerate(leiden_cats):
        sel = embedding_df['leiden cluster'] == lc
        ax.scatter(
            embedding_df.loc[sel, 'x'],
            embedding_df.loc[sel, 'y'],
            color=leiden_colors[i],
            s=10,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.3,
            zorder=2
        )
    ax.set_title('Leiden clusters')
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    # Panel 2: HDBSCAN clusters
    ax = axes[1]
    ax.scatter(
        embedding_df.loc[hdb == -1, 'x'],
        embedding_df.loc[hdb == -1, 'y'],
        color='lightgray', s=15, alpha=0.1, zorder=1
    )
    for i, c in enumerate(hdb_clusters):
        sel = (hdb == c)
        ax.scatter(
            embedding_df.loc[sel, 'x'],
            embedding_df.loc[sel, 'y'],
            color=hdb_palette[i],
            s=15,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.3,
            zorder=2
        )
    ax.set_title('HDBSCAN clusters')
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


    # Panel 3: CORUM membership (solid dark‑blue dots)
    ax = axes[2]
    # draw CORUM lines behind
    for g1, g2 in corum:
        if g1 in coords.index and g2 in coords.index:
            x1, y1 = coords.loc[g1]
            x2, y2 = coords.loc[g2]
            ax.plot(
                [x1, x2], [y1, y2],
                color='darkblue', linewidth=0.5, alpha=0.2, zorder=1,
            )
    # non‑CORUM points
    ax.scatter(
        embedding_df.loc[~is_in_corum, 'x'],
        embedding_df.loc[~is_in_corum, 'y'],
        color='lightgray', s=10, alpha=0.3, zorder=2,
        label='Non CORUM'
    )
    # CORUM points (solid dark blue)
    ax.scatter(
        embedding_df.loc[ is_in_corum, 'x'],
        embedding_df.loc[ is_in_corum, 'y'],
        color='darkblue', s=10, alpha=0.8, zorder=3,
        label='CORUM Pairs'
    )
    ax.set_title('CORUM membership')
    ax.legend(loc='lower left', frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    # Panel 4: Overlay of HDBSCAN + CORUM
    ax = axes[3]
    ax.scatter(
        embedding_df.loc[~is_in_corum, 'x'],
        embedding_df.loc[~is_in_corum, 'y'],
        color='lightgray', s=10, alpha=0.1, zorder=1
    )
    for g1, g2 in corum:
        if g1 in coords.index and g2 in coords.index:
            x1, y1 = coords.loc[g1]
            x2, y2 = coords.loc[g2]
            ax.plot(
                [x1, x2], [y1, y2],
                color='darkblue', linewidth=0.5, alpha=0.2, zorder=2
            )
    for i, c in enumerate(hdb_clusters):
        sel = (hdb == c)
        ax.scatter(
            embedding_df.loc[sel, 'x'],
            embedding_df.loc[sel, 'y'],
            color=hdb_palette[i],
            s=15,
            alpha=0.9,
            edgecolor='white',
            linewidth=0.4,
            zorder=3
        )
    # CORUM genes as open circles on top
    ax.scatter(
        embedding_df.loc[ is_in_corum, 'x'],
        embedding_df.loc[ is_in_corum, 'y'],
        facecolors='none', edgecolors='black',
        s=15, linewidth=.5, zorder=4,
        label='CORUM genes'
    )
    ax.set_title('Overlay: HDBSCAN + CORUM')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.show()
    if save_dir_stem is not None:
        fig.savefig(save_dir_stem + "_MDE_views.svg", format='svg', bbox_inches='tight')
    return embedding_df


def plot_corum_distances(embedding_df, corum_pairs, save_dir_stem = None):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial.distance import pdist, squareform

    # Prepare embedding coordinates and mapping from gene to row index
    coords = embedding_df[['x', 'y']].values
    genes = embedding_df['gene_target'].tolist()
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Compute full pairwise Euclidean distance matrix
    dist_mat = squareform(pdist(coords, metric='euclidean'))

    # Keep only CORUM pairs where both genes appear in the embedding
    valid_corum_pairs = [(g1, g2) for g1, g2 in corum_pairs if g1 in gene_to_idx and g2 in gene_to_idx]

    # Build a boolean mask for CORUM pairs in the distance matrix
    n = dist_mat.shape[0]
    mask_corum = np.zeros((n, n), dtype=bool)
    for g1, g2 in valid_corum_pairs:
        i, j = gene_to_idx[g1], gene_to_idx[g2]
        mask_corum[i, j] = True
        mask_corum[j, i] = True

    # Extract only the upper‐triangular entries (i < j) to avoid duplicates and self‐distances
    i_upper, j_upper = np.triu_indices(n, k=1)
    all_dists = dist_mat[i_upper, j_upper]
    is_corum = mask_corum[i_upper, j_upper]

    # Split distances into CORUM vs non‐CORUM
    corum_dists = all_dists[is_corum]
    noncorum_dists = all_dists[~is_corum]

    # Compute means and their ratio\
    all_distances = all_dists.mean()
    mean_corum = corum_dists.mean()/all_distances if corum_dists.size else np.nan
    mean_noncorum = noncorum_dists.mean()/all_distances if noncorum_dists.size else np.nan
    ratio = mean_noncorum / mean_corum if mean_corum else np.nan

    print(f"Normalized Mean CORUM pair distance    : {mean_corum:.4f}")
    print(f"Normalized Mean non-CORUM pair distance: {mean_noncorum:.4f}")
    print(f"Distance ratio (NON:CORUM)   : {ratio:.4f}")

    # Plot density distribution histograms
    plt.figure(figsize=(8, 6))
    sns.histplot(corum_dists, stat='density', bins=50, color='blue', alpha=0.5, label='CORUM pairs')
    sns.histplot(noncorum_dists, stat='density', bins=50, color='orange', alpha=0.5, label='Non-CORUM pairs')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.title('Density Distribution of Pairwise Distances')
    plt.legend()
    plt.tight_layout()
    plt.show()
    if save_dir_stem is not None:
        plt.savefig(save_dir_stem + "_corum_distances.svg", format='svg', bbox_inches='tight')



import pandas as pd
import gseapy as gp

def annotate_clusters(
    embedding_df: pd.DataFrame,
    output_path: str,
    cluster_cols: list[str] = ['leiden cluster', 'hdbscan cluster'],
    libraries: list[str] = [
        'CORUM_Links',
        'CORUM',
        'KEGG_2021_Human',
        'GO_Biological_Process_2025',
        'GO_Molecular_Function_2025',
        'GO_Cellular_Component_2025',
        'Reactome_Pathways_2024'
    ],
    corum_path: str = '/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/KOLF_Perturbation_Atlas_Analysis/analysis_scripts/investigations/corum_humanComplexes.txt',
    pval_cutoff: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Groups genes by each cluster_col,
    2) annotates with all given libraries,
    3) writes MDE + each annotated sheet to `output_path`,
    4) returns (leiden_df, hdbscan_df) for any downstream use.
    """
    # --- load CORUM once ---
    corum_table = pd.read_csv(corum_path, sep='\t')
    complexes = {
        name: subs.split(';')
        for name, subs in zip(corum_table['complex_name'],
                              corum_table['subunits_gene_name'])
    }
    all_genes = embedding_df['gene_target'].unique().tolist()

    def annotate_single(df: pd.DataFrame, lib: str) -> pd.DataFrame:
        anns = []
        for genes_str in df['gene_list']:
            genes = genes_str.split(',')
            if lib == 'CORUM_Links':
                links = [
                    c for c, members in complexes.items()
                    if len(set(members) & set(genes)) >= 0.66 * len(members)
                ]
                anns.append('; '.join(links) or 'No links')
            else:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=[lib],
                    organism='Human',
                    background=all_genes,
                    outdir=None
                )
                if enr.results is None or enr.results.empty:
                    anns.append("No significant enrichment")
                else:
                    sig = enr.results[enr.results['Adjusted P-value'] < pval_cutoff]
                    if sig.empty:
                        anns.append("No significant enrichment")
                    else:
                        terms = sig.sort_values('Adjusted P-value')['Term']
                        anns.append('; '.join(terms.tolist()))
        df[f'{lib}_Annotation'] = anns
        return df

    # --- build and annotate each cluster‐df ---
    annotated_dfs = []
    for col in cluster_cols:
        # Exclude noise cluster = -1
        filtered = embedding_df[embedding_df[col] != -1]

        # Group by the valid clusters
        df = (
            filtered
            .groupby(col)['gene_target']
            .agg(lambda x: ','.join(x))
            .reset_index()
            .rename(columns={'gene_target': 'gene_list'})
        )

        # Annotate each library in turn
        for lib in libraries:
            df = annotate_single(df, lib)

        annotated_dfs.append(df)

    # --- write everything to one workbook ---
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        embedding_df.to_excel(writer, sheet_name='MDE', index=False)
        for col, df in zip(cluster_cols, annotated_dfs):
            # sheet names cannot exceed 31 chars in Excel
            sheet = col if len(col) <= 31 else col[:28] + '...'
            df.to_excel(writer, sheet_name=sheet, index=False)

    return annotated_dfs

