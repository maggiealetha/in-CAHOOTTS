"""
Experiment utilities for neural ODE models.
"""

import os
import torch
import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, List, Optional, Tuple


def create_output_directory(run_number: int, use_prior: bool = False, 
                          decay: bool = False, shuffled: bool = False) -> str:
    """
    Create output directory structure for experiments.
    
    Args:
        run_number: Run number for the experiment
        use_prior: Whether using prior knowledge
        decay: Whether using decay modeling
        shuffled: Whether using shuffled priors
        
    Returns:
        str: Path to created directory
    """
    if use_prior:
        if shuffled:
            output_path = os.path.join("yeast_with_prior", "shuffled")
        else:
            output_path = "yeast_with_prior"
    else:
        output_path = "yeast_no_prior"
    
    if decay:
        print(f'Adding decay to path: {output_path}')
        output_path = os.path.join(output_path, "decay")
        print(f'Decay path: {output_path}')
    
    dir_path = os.path.join(output_path, f"run{run_number}")
    print(f"Creating directory: {dir_path}")
    
    # Create the full directory path (including parent directories)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_cross_validation_results(cv_fold: int, final_epoch: int, model: torch.nn.Module,
                                train_losses: List[float], val_losses: List[float], 
                                r2_scores: List[float], output_dir: str,
                                mape_scores: Optional[List[float]] = None,
                                shuffled: bool = False, decay: bool = False,
                                rate: Optional[float] = None,
                                prior_weight: Optional[float] = None,
                                prior_losses: Optional[List[float]] = None,
                                weight_decay: Optional[float] = None,
                                learning_rate: Optional[float] = None):
    """
    Save cross-validation results and model.
    
    Args:
        cv_fold: Cross-validation fold number
        final_epoch: Final training epoch
        model: Trained model
        train_losses: Training losses
        val_losses: Validation losses
        r2_scores: R² scores
        output_dir: Output directory
        mape_scores: MAPE scores (optional)
        shuffled: Whether using shuffled priors
        decay: Whether using decay modeling
        rate: Learning rate (optional)
        prior_weight: Prior weight (optional)
        prior_losses: Prior losses (optional)
        weight_decay: Weight decay (optional)
        learning_rate: Learning rate (optional)
    """
    # Create filename suffix based on parameters
    if rate is not None:
        suffix = f"r{rate:.4f}"
    elif prior_weight is not None:
        suffix = f"pw{prior_weight:.4f}_wd{weight_decay:.6f}_lr{learning_rate:.6f}"
    else:
        suffix = ""
    
    # Save model
    if suffix:
        model_path = os.path.join(output_dir, f"model_cv{cv_fold}_epochs{final_epoch}_{suffix}.pt")
        train_loss_path = os.path.join(output_dir, f"tr_loss_cv{cv_fold}_epochs{final_epoch}_{suffix}.csv")
        val_loss_path = os.path.join(output_dir, f"vd_loss_cv{cv_fold}_epochs{final_epoch}_{suffix}.csv")
        r2_path = os.path.join(output_dir, f"r2_cv{cv_fold}_epochs{final_epoch}_{suffix}.csv")
    else:
        model_path = os.path.join(output_dir, f"model_cv{cv_fold}_epochs{final_epoch}.pt")
        train_loss_path = os.path.join(output_dir, f"tr_loss_cv{cv_fold}_epochs{final_epoch}.csv")
        val_loss_path = os.path.join(output_dir, f"vd_loss_cv{cv_fold}_epochs{final_epoch}.csv")
        r2_path = os.path.join(output_dir, f"r2_cv{cv_fold}_epochs{final_epoch}.csv")
    
    # Save model and metrics
    torch.save(model.state_dict(), model_path)
    np.savetxt(train_loss_path, train_losses, delimiter=',')
    np.savetxt(val_loss_path, val_losses, delimiter=',')
    np.savetxt(r2_path, r2_scores, delimiter=',')
    
    # Save additional metrics if provided
    if mape_scores is not None:
        mape_path = os.path.join(output_dir, f"mape_cv{cv_fold}_epochs{final_epoch}_{suffix}.csv")
        np.savetxt(mape_path, mape_scores, delimiter=',')
    
    if prior_losses is not None and prior_weight is not None:
        prior_loss_path = os.path.join(output_dir, f"prior_loss_cv{cv_fold}_epochs{final_epoch}_{suffix}.csv")
        np.savetxt(prior_loss_path, prior_losses, delimiter=',')


def save_experiment_metadata(output_dir: str, config: Dict[str, Any],
                           gs_seeds: List[int], data_seeds: List[int],
                           prior_seeds: Optional[List[int]] = None,
                           shuffled_seeds: Optional[List[int]] = None,
                           decay: bool = False):
    """
    Save experiment metadata including seeds and configuration.
    
    Args:
        output_dir: Output directory
        config: Configuration dictionary
        gs_seeds: Gold standard seeds
        data_seeds: Data seeds
        prior_seeds: Prior seeds (optional)
        shuffled_seeds: Shuffled seeds (optional)
        decay: Whether using decay modeling
    """
    # Prepare metadata
    if prior_seeds and shuffled_seeds:
        meta_data = np.vstack((gs_seeds, data_seeds, prior_seeds, shuffled_seeds)).T
        cols = ['gs_seed', 'data_seed', 'prior_seed', 'shuffled_seed']
    elif prior_seeds:
        meta_data = np.vstack((gs_seeds, data_seeds, prior_seeds)).T
        cols = ['gs_seed', 'data_seed', 'prior_seed']
    else:
        meta_data = np.vstack((gs_seeds, data_seeds)).T
        cols = ['gs_seed', 'data_seed']
    
    # Save metadata
    pd.DataFrame(meta_data, columns=cols).to_csv(
        os.path.join(output_dir, "meta_data.tsv"), sep='\t'
    )
    pd.DataFrame([config]).to_csv(
        os.path.join(output_dir, 'config_settings.csv'), index=False
    )


def load_experiment_metadata(directory: str, shuffled: bool = False) -> Tuple[List[int], ...]:
    """
    Load experiment metadata.
    
    Args:
        directory: Directory containing metadata
        shuffled: Whether shuffled seeds are present
        
    Returns:
        tuple: Seeds (gs_seeds, data_seeds, [shuffled_seeds])
    """
    seeds = pd.read_csv(os.path.join(directory, 'meta_data.tsv'), 
                       header=0, sep='\t', index_col=0)
    
    gs_seeds = seeds['gs_seed'].to_list()
    data_seeds = seeds['data_seed'].to_list()
    
    if shuffled and 'shuffled_seed' in seeds.columns:
        shuffled_seeds = seeds['shuffled_seed'].to_list()
        return gs_seeds, data_seeds, shuffled_seeds
    else:
        return gs_seeds, data_seeds


def generate_random_seed() -> int:
    """
    Generate a random seed.
    
    Returns:
        int: Random seed between 0 and 1000
    """
    return np.random.randint(0, 1000)


# Legacy function names for backward compatibility
def mkdirs(run_num, use_prior=False, decay=False, shuffled=False):
    """Legacy function name for create_output_directory."""
    return create_output_directory(run_num, use_prior, decay, shuffled)


def save_per_cv(cv, final_epoch, func, tr_mse, vd_mse, r2, output_dir, 
                mape=None, shuffled=False, decay=False, rate_=None,
                prior_weight=None, prior_loss=None, wd=None, lr=None):
    """Legacy function name for save_cross_validation_results."""
    return save_cross_validation_results(
        cv, final_epoch, func, tr_mse, vd_mse, r2, output_dir,
        mape_scores=mape, shuffled=shuffled, decay=decay, rate=rate_,
        prior_weight=prior_weight, prior_losses=prior_loss,
        weight_decay=wd, learning_rate=lr
    )


def save_meta_data(output_dir, config, gs_seeds, data_seeds, 
                  prior_seeds=None, shuffled_seeds=None, decay=False):
    """Legacy function name for save_experiment_metadata."""
    return save_experiment_metadata(
        output_dir, config, gs_seeds, data_seeds,
        prior_seeds, shuffled_seeds, decay
    )


def load_meta_data(directory, shuffled=False):
    """Legacy function name for load_experiment_metadata."""
    return load_experiment_metadata(directory, shuffled)


def gen_seed():
    """Legacy function name for generate_random_seed."""
    return generate_random_seed()