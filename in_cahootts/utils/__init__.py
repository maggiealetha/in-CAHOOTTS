"""
Utility functions for neural ODE models.
"""

from .metrics import (
    calculate_r2, calculate_pseudo_r2, training_metrics,
    calculate_rss, calculate_tss, calculate_r2_components,
    calculate_pseudo_r2_per_feature, calculate_r2_per_feature,
    dropout_hook, check_plateau,
    # Legacy function names
    calc_rss, calc_tss, calc_r2, calc_pseudo_r2_per_feature, calc_r2_per_feature
)
from .config import load_config, save_config
from .experiment import (
    create_output_directory, save_cross_validation_results, save_experiment_metadata,
    load_experiment_metadata, generate_random_seed,
    # Legacy function names
    mkdirs, save_per_cv, save_meta_data, load_meta_data, gen_seed
)
from .priors import (
    setup_priors_no_holdouts, setup_priors, setup_priors_erv_calc,
    create_prior_matrix, validate_prior_matrix, get_prior_statistics
)

__all__ = [
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
    # Legacy function names
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