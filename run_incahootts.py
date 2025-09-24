#!/usr/bin/env python3
"""
in-CAHOOTTs: Unified Experiment Runner

This script provides a unified interface for running all types of in-CAHOOTTs experiments:
- Pretraining (cell cycle, drug perturbation, sparse sampling)
- Prior-integrated training
- Block model validation

Usage:
    python run_incahootts.py <config_file> <output_dir> [--experiment-type <type>]

Examples:
    # Cell cycle pretraining
    python run_incahootts.py configs/cell_cycle.yaml results/cell_cycle --experiment-type cell_cycle
    
    # Drug perturbation with IVP subset
    python run_incahootts.py configs/drug_perturbation.yaml results/drug_pert --experiment-type drug_perturbation
    
    # Sparse sampling experiment
    python run_incahootts.py configs/sparse_sampling.yaml results/sparse --experiment-type sparse_sampling
    
    # Prior-integrated training
    python run_incahootts.py configs/prior_training.yaml results/prior --experiment-type prior
"""

import sys
import argparse
import torch
import pandas as pd
import scanpy as sc
import os
import numpy as np
from pathlib import Path

# Import our new in_cahootts package
from incahootts import (
    Trainer, ODEFunc, SoftPriorODEFunc, BlockODEFunc,
    preprocess_data_general, preprocess_data_biophysical, split_data,
    create_dataloaders, generate_batch_times, setup_multiple_dataloaders,
    load_config, save_config, create_output_directory, 
    save_cross_validation_results, save_experiment_metadata, generate_random_seed,
    setup_priors_no_holdouts, setup_priors
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run in-CAHOOTTs experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('config_file', help='Path to YAML configuration file')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--experiment-type', 
                       choices=['cell_cycle', 'drug_perturbation', 'sparse_sampling', 'prior', 'block_model', 'post_training'],
                       default='cell_cycle',
                       help='Type of experiment to run (default: cell_cycle)')
    parser.add_argument('--pretrained-model', 
                       help='Path to pretrained model for post-training (required for post_training experiment type)')
    parser.add_argument('--prior-weight', type=float, default=0.99,
                       help='Prior weight for post-training (default: 0.99)')
    parser.add_argument('--freeze-lambda', action='store_true', default=True,
                       help='Freeze lambda layers in post-training (default: True)')
    
    return parser.parse_args()

def load_and_preprocess_data(config):
    """Load and preprocess data based on configuration."""
    print(f"Loading data from: {config['data_file']}")
    
    # Load data
    yeast = sc.read_h5ad(config['data_file'])
    
    # Filter data based on experiment type
    if config.get('experiment_filter'):
        if config['experiment_filter'] == 'pools_12':
            adata = yeast[(yeast.obs['Pool']==2)|(yeast.obs['Pool']==1)]
        elif config['experiment_filter'] == 'experiment_2':
            adata = yeast[yeast.obs['Experiment']==2]
        else:
            adata = yeast
    else:
        adata = yeast
    
    del yeast
    
    # Preprocess data
    if config.get('decay', False):
        print("Using biophysical preprocessing (decay data)")
        data, time_vector = preprocess_data_biophysical(adata)
    else:
        print("Using general preprocessing")
        if config.get('time_axis') == 'cc':
            data, time_vector = preprocess_data_general(adata, time_axis='cc')
        else:
            data, time_vector = preprocess_data_general(adata)
    
    var_names = adata.var_names
    print(f"Data shape: {data.shape}, Time points: {len(time_vector)}")
    
    return data, time_vector, var_names

def load_gold_standard(config, var_names):
    """Load gold standard network."""
    if not config.get('use_prior', False):
        return None
    
    print(f"Loading gold standard from: {config['gold_standard_file']}")
    gold_standard = pd.read_csv(
        config['gold_standard_file'],
        sep="\t",
        index_col=0
    ).reindex(
        var_names,
        axis=0
    ).fillna(0).astype(int)
    
    return gold_standard

def create_model(config, n_genes, device, prior=None):
    """Create model based on configuration."""
    model_type = config.get('model_type', 'base')
    use_prior = config.get('use_prior', False)
    dropout = config.get('dropout', 0.2)
    
    if model_type == 'block_model':
        print("Creating BlockODEFunc model")
        model = BlockODEFunc(device, n_genes, use_prior=use_prior, prior_=prior, dropout=dropout)
    elif use_prior:
        print("Creating SoftPriorODEFunc model")
        model = SoftPriorODEFunc(device, n_genes, use_prior=use_prior, prior_=prior, dropout=dropout)
    else:
        print("Creating ODEFunc model")
        model = ODEFunc(device, n_genes, use_prior=use_prior, prior_=prior, dropout=dropout)
    
    return model

def setup_training_data(config, data, time_vector):
    """Setup training and validation data."""
    cvs = config.get('cvs', 5)
    decay = config.get('decay', False)
    
    # Split data for cross-validation
    train, train_t, test, test_t, n_genes = split_data(
        data, time_vector, 
        seed=generate_random_seed(), 
        decay=decay
    )
    
    # Create data loaders
    train_loaders, val_loaders, batch_times = setup_multiple_dataloaders(
        train, train_t, test, test_t,
        sequence_length=config.get('sl', 29),
        n_dataloaders=config.get('n_dls', 3),
        t_min=config.get('tmin', 0),
        t_max=config.get('tmax', 87),
        time_step=config.get('ts', 1),
        bigT_for_scaling=config.get('bigT_for_scaling', None)
    )
    
    return train_loaders, val_loaders, batch_times, n_genes

def load_pretrained_model(pretrained_path, device, n_genes, dropout=0.2):
    """Load pretrained model and prepare for post-training."""
    print(f"Loading pretrained model from: {pretrained_path}")
    
    # Create base model (without prior)
    model = ODEFunc(device, n_genes, use_prior=False, prior_=None, dropout=dropout)
    
    # Load pretrained weights
    if not Path(pretrained_path).exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    
    state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict)
    
    print("Pretrained model loaded successfully")
    return model

def prepare_model_for_post_training(model, prior_matrix, prior_weight, freeze_lambda=True):
    """Prepare model for post-training with prior knowledge."""
    print(f"Preparing model for post-training with prior weight: {prior_weight}")
    
    # Convert to SoftPriorODEFunc
    model_with_prior = SoftPriorODEFunc(
        device=model.device,
        ndim=model.ndim,
        use_prior=True,
        prior_=prior_matrix,
        prior_weight=prior_weight,
        dropout=model.dropout
    )
    
    # Copy pretrained weights
    model_with_prior.load_state_dict(model.state_dict(), strict=False)
    
    # Set prior
    model_with_prior.prior_ = prior_matrix.to(model.device)
    model_with_prior.use_prior = True
    model_with_prior.prior_weight = prior_weight
    
    # Initialize weights from prior
    model_with_prior._initialize_weights_from_prior()
    
    # Freeze lambda layers if requested
    if freeze_lambda:
        print("Freezing lambda layers...")
        for name, param in model_with_prior.named_parameters():
            if 'lambda_' in name:
                param.requires_grad = False
                print(f"Frozen: {name}")
    
    return model_with_prior

def run_experiment(config, experiment_type, output_dir, args=None):
    """Run the specified experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    data, time_vector, var_names = load_and_preprocess_data(config)
    
    # Load gold standard if needed
    gold_standard = load_gold_standard(config, var_names)
    
    # Create output directory
    output_path = create_output_directory(
        output_dir, 
        use_prior=config.get('use_prior', False),
        decay=config.get('decay', False),
        shuffled=config.get('shuffle_prior', False)
    )
    
    # Setup trainer
    trainer = Trainer(
        device=device,
        learning_rate=config.get('lr', 0.001),
        weight_decay=config.get('wd', 1e-5),
        patience=config.get('patience', 20)
    )
    
    # Run cross-validation
    cvs = config.get('cvs', 5)
    gs_seeds = []
    data_seeds = []
    shuffled_seeds = []
    
    for cv in range(cvs):
        print(f"\n=== Cross-validation fold {cv+1}/{cvs} ===")
        
        # Generate seeds
        gs_seeds.append(generate_random_seed())
        data_seeds.append(generate_random_seed())
        if config.get('shuffle_prior', False):
            shuffled_seeds.append(generate_random_seed())
        
        # Setup data
        train_loaders, val_loaders, batch_times, n_genes = setup_training_data(config, data, time_vector)
        
        # Setup prior if needed
        prior = None
        if config.get('use_prior', False) and gold_standard is not None:
            prior = setup_priors(
                gold_standard, gs_seeds[-1], var_names,
                shuffled=config.get('shuffle_prior', False),
                shuffled_seed=shuffled_seeds[-1] if shuffled_seeds else None
            )
        
        # Create model
        model = create_model(config, n_genes, device, prior)
        
        # Run training based on experiment type
        epochs = config.get('epochs', 500)
        val_frequency = config.get('val_frequency', 20)
        
        if experiment_type == 'cell_cycle':
            history = trainer.train_cell_cycle(
                model, train_loaders, val_loaders, batch_times,
                epochs=epochs,
                sparse_sampling=config.get('sparse_sampling', False),
                interval_minutes=config.get('interval_minutes', 30),
                val_frequency=val_frequency
            )
        elif experiment_type == 'drug_perturbation':
            history = trainer.train_drug_perturbation(
                model, train_loaders, val_loaders, batch_times,
                epochs=epochs,
                ivp_subset=config.get('ivp_subset', 10),
                time_subset=config.get('time_subset', 30),
                sparse_sampling=config.get('sparse_sampling', False),
                interval_minutes=config.get('interval_minutes', 30),
                val_frequency=val_frequency
            )
        elif experiment_type == 'sparse_sampling':
            history = trainer.train_sparse_sampling(
                model, train_loaders, val_loaders, batch_times,
                epochs=epochs,
                interval_minutes=config.get('interval_minutes', 30),
                val_frequency=val_frequency
            )
        elif experiment_type == 'prior':
            history = trainer.train_with_prior(
                model, train_loaders, val_loaders, batch_times,
                epochs=epochs,
                sparse_sampling=config.get('sparse_sampling', False),
                interval_minutes=config.get('interval_minutes', 30),
                val_frequency=val_frequency
            )
        elif experiment_type == 'post_training':
            # Post-training requires a pretrained model
            if not args.pretrained_model:
                raise ValueError("Post-training requires --pretrained-model argument")
            
            # Load pretrained model and convert to SoftPriorODEFunc
            pretrained_model = load_pretrained_model(args.pretrained_model, device, n_genes, config.get('dropout', 0.2))
            model = prepare_model_for_post_training(
                pretrained_model, prior, args.prior_weight, args.freeze_lambda
            )
            
            history = trainer.train_with_prior(
                model, train_loaders, val_loaders, batch_times,
                epochs=epochs,
                sparse_sampling=config.get('sparse_sampling', False),
                interval_minutes=config.get('interval_minutes', 30),
                val_frequency=val_frequency
            )
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Save results
        final_epoch = len(history['train_losses']) - 1
        save_cross_validation_results(
            cv, final_epoch, model, 
            history['train_losses'], history['val_losses'], history['r2_scores'],
            output_path,
            shuffled=config.get('shuffle_prior', False),
            decay=config.get('decay', False)
        )
        
        print(f"CV {cv+1} completed. Final train loss: {history['train_losses'][-1]:.6f}")
        if history['val_losses'][-1] is not None:
            print(f"Final validation loss: {history['val_losses'][-1]:.6f}")
        if history['r2_scores'][-1] is not None:
            print(f"Final R² score: {history['r2_scores'][-1]:.6f}")
    
    # Save metadata
    save_experiment_metadata(
        output_path, config, gs_seeds, data_seeds,
        prior_seeds=gs_seeds, shuffled_seeds=shuffled_seeds
    )
    
    print(f"\n=== Experiment completed ===")
    print(f"Results saved to: {output_path}")
    print("Done!")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Add experiment type to config
    config['experiment_type'] = args.experiment_type
    
    print(f"Running {args.experiment_type} experiment")
    print(f"Config: {args.config_file}")
    print(f"Output: {args.output_dir}")
    
    # Run experiment
    run_experiment(config, args.experiment_type, args.output_dir, args)

if __name__ == "__main__":
    main()