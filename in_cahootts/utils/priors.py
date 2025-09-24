"""
Prior knowledge setup utilities for neural ODE models.
"""

import torch
import pandas as pd
import numpy as np
from inferelator.preprocessing import ManagePriors
from typing import Tuple, Optional


def setup_priors_no_holdouts(gold_standard_df: pd.DataFrame) -> torch.Tensor:
    """
    Setup prior knowledge without holdouts.
    
    Args:
        gold_standard_df: Gold standard DataFrame
        
    Returns:
        torch.Tensor: Prior matrix
    """
    prior = torch.tensor(gold_standard_df.values, dtype=torch.float32).T
    return prior


def setup_priors(gold_standard_df: pd.DataFrame, seed: int, gene_names: list,
                shuffled: bool = False, shuffled_seed: Optional[int] = None) -> torch.Tensor:
    """
    Setup prior knowledge with cross-validation holdouts.
    
    Args:
        gold_standard_df: Gold standard DataFrame
        seed: Random seed for cross-validation
        gene_names: List of gene names
        shuffled: Whether to shuffle priors
        shuffled_seed: Seed for shuffling (optional)
        
    Returns:
        torch.Tensor: Prior matrix
    """
    split_axis = 0
    split = 0.2
    
    # Cross-validate gold standard
    prior_df, gs_df = ManagePriors.cross_validate_gold_standard(
        gold_standard_df, gold_standard_df, split_axis, split, seed
    )
    
    # Shuffle if requested
    if shuffled and shuffled_seed is not None:
        prior_df = ManagePriors.shuffle_priors(prior_df, -1, shuffled_seed)
    
    # Reindex and fill missing values
    prior_df = prior_df.reindex(gene_names, axis=0).fillna(0).astype(int)
    
    # Convert to tensor
    prior = torch.tensor(prior_df.values, dtype=torch.float32).T
    
    return prior


def setup_priors_erv_calc(gold_standard_df: pd.DataFrame, seed: int, gene_names: list,
                         return_gs: bool = False) -> Tuple[torch.Tensor, ...]:
    """
    Setup prior knowledge for ERV calculation.
    
    Args:
        gold_standard_df: Gold standard DataFrame
        seed: Random seed for cross-validation
        gene_names: List of gene names
        return_gs: Whether to return gold standard
        
    Returns:
        torch.Tensor or tuple: Prior matrix (and optionally gold standard)
    """
    split_axis = 0
    split = 0.2
    
    # Cross-validate gold standard
    prior_df, gs_df = ManagePriors.cross_validate_gold_standard(
        gold_standard_df, gold_standard_df, split_axis, split, seed
    )
    
    # Reindex and fill missing values
    prior_df = prior_df.reindex(gene_names, axis=0).fillna(0).astype(int)
    
    if return_gs:
        gs_df = gs_df.reindex(gene_names, axis=0).fillna(0).astype(int)
        prior = torch.tensor(prior_df.values, dtype=torch.float32).T
        gs = torch.tensor(gs_df.values, dtype=torch.float32).T
        return prior, gs
    else:
        prior = torch.tensor(prior_df.values, dtype=torch.float32).T
        return prior


def create_prior_matrix(gold_standard_df: pd.DataFrame, gene_names: list,
                       use_holdouts: bool = True, seed: int = 257,
                       shuffled: bool = False, shuffled_seed: Optional[int] = None) -> torch.Tensor:
    """
    Create prior matrix with flexible options.
    
    Args:
        gold_standard_df: Gold standard DataFrame
        gene_names: List of gene names
        use_holdouts: Whether to use cross-validation holdouts
        seed: Random seed
        shuffled: Whether to shuffle priors
        shuffled_seed: Seed for shuffling
        
    Returns:
        torch.Tensor: Prior matrix
    """
    if use_holdouts:
        return setup_priors(gold_standard_df, seed, gene_names, shuffled, shuffled_seed)
    else:
        return setup_priors_no_holdouts(gold_standard_df)


def validate_prior_matrix(prior_matrix: torch.Tensor, gene_names: list) -> bool:
    """
    Validate prior matrix dimensions and values.
    
    Args:
        prior_matrix: Prior matrix to validate
        gene_names: List of gene names
        
    Returns:
        bool: True if valid
    """
    # Check dimensions
    if prior_matrix.shape[1] != len(gene_names):
        print(f"Warning: Prior matrix shape {prior_matrix.shape} doesn't match gene count {len(gene_names)}")
        return False
    
    # Check for valid values (0s and 1s)
    unique_values = torch.unique(prior_matrix)
    if not torch.all((unique_values == 0) | (unique_values == 1)):
        print(f"Warning: Prior matrix contains non-binary values: {unique_values}")
        return False
    
    return True


def get_prior_statistics(prior_matrix: torch.Tensor) -> dict:
    """
    Get statistics about the prior matrix.
    
    Args:
        prior_matrix: Prior matrix
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'shape': prior_matrix.shape,
        'sparsity': (prior_matrix == 0).float().mean().item(),
        'density': (prior_matrix == 1).float().mean().item(),
        'total_edges': prior_matrix.sum().item(),
        'max_connections': prior_matrix.sum(dim=0).max().item(),
        'min_connections': prior_matrix.sum(dim=0).min().item(),
        'mean_connections': prior_matrix.sum(dim=0).mean().item()
    }
    
    return stats