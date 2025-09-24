"""
in-CAHOOTTs: Neural ODE for Gene Regulatory Networks

A neural ODE framework for modeling gene regulatory networks with prior knowledge integration.
"""

__version__ = "1.0.0"
__author__ = "Maggie"

from .models import ODEFunc, SoftPriorODEFunc, BlockODEFunc
from .training import Trainer
from .data import (
    TimeDataset, preprocess_data_general, preprocess_data_biophysical, split_data,
    create_dataloaders, generate_batch_times, setup_multiple_dataloaders,
    select_sparse_random_subset, select_random_subset,
    # Legacy function names
    create_dls, gen_dls, gen_batch_t, setup_dls, _shuffle_time_data
)
from .utils import (
    # Metrics
    calculate_r2, calculate_pseudo_r2, training_metrics,
    calculate_rss, calculate_tss, calculate_r2_components,
    calculate_pseudo_r2_per_feature, calculate_r2_per_feature,
    dropout_hook, check_plateau,
    # Config
    load_config, save_config,
    # Experiment utilities
    create_output_directory, save_cross_validation_results, save_experiment_metadata,
    load_experiment_metadata, generate_random_seed,
    # Prior utilities
    setup_priors_no_holdouts, setup_priors, setup_priors_erv_calc,
    create_prior_matrix, validate_prior_matrix, get_prior_statistics,
    # Legacy function names
    mkdirs, save_per_cv, save_meta_data, load_meta_data, gen_seed
)

__all__ = [
    # Models
    "ODEFunc",
    "SoftPriorODEFunc", 
    "BlockODEFunc",
    # Training
    "Trainer",
    # Data handling
    "TimeDataset",
    "preprocess_data_general",
    "preprocess_data_biophysical",
    "split_data",
    "create_dataloaders",
    "generate_batch_times",
    "setup_multiple_dataloaders",
    "select_sparse_random_subset",
    "select_random_subset",
    # Metrics
    "calculate_r2",
    "calculate_pseudo_r2", 
    "training_metrics",
    "calculate_rss",
    "calculate_tss",
    "calculate_r2_components",
    "calculate_pseudo_r2_per_feature",
    "calculate_r2_per_feature",
    "dropout_hook",
    "check_plateau",
    # Config
    "load_config", 
    "save_config",
    # Experiment utilities
    "create_output_directory",
    "save_cross_validation_results", 
    "save_experiment_metadata",
    "load_experiment_metadata",
    "generate_random_seed",
    # Prior utilities
    "setup_priors_no_holdouts",
    "setup_priors",
    "setup_priors_erv_calc", 
    "create_prior_matrix",
    "validate_prior_matrix",
    "get_prior_statistics",
    # Legacy names
    "create_dls",
    "gen_dls", 
    "gen_batch_t",
    "setup_dls",
    "_shuffle_time_data",
    "mkdirs",
    "save_per_cv", 
    "save_meta_data", 
    "load_meta_data",
    "gen_seed",
    "calc_rss",
    "calc_tss", 
    "calc_r2",
    "calc_pseudo_r2_per_feature",
    "calc_r2_per_feature"
]