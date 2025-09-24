#!/usr/bin/env python3
"""
in-CAHOOTTs: Post-Training with Prior Knowledge

This script performs post-training (second stage) by loading a pretrained model
and fine-tuning it with prior knowledge integration.

Usage:
    python run_post_training.py <config_file> <pretrained_model_path> <output_dir>

Examples:
    # Post-train with prior knowledge
    python run_post_training.py configs/post_training.yaml pretrained_models/model_0.pth results/post_training
    
    # Post-train with specific prior weight
    python run_post_training.py configs/post_training.yaml pretrained_models/model_0.pth results/post_training --prior-weight 0.99
"""

import sys
import argparse
import torch
import pandas as pd
import scanpy as sc
import os
import numpy as np
import copy
from pathlib import Path

# Import our new in_cahootts package
from incahootts import (
    Trainer, ODEFunc, SoftPriorODEFunc,
    preprocess_data_general, preprocess_data_biophysical, split_data,
    create_dataloaders, generate_batch_times, setup_multiple_dataloaders,
    load_config, save_config, create_output_directory, 
    save_cross_validation_results, save_experiment_metadata, generate_random_seed,
    setup_priors_no_holdouts, setup_priors
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run in-CAHOOTTs post-training with prior knowledge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('config_file', help='Path to YAML configuration file')
    parser.add_argument('pretrained_model_path', help='Path to pretrained model weights')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--prior-weight', type=float, default=0.99,
                       help='Prior weight for regularization (default: 0.99)')
    parser.add_argument('--freeze-lambda', action='store_true', default=True,
                       help='Freeze lambda layers (default: True)')
    parser.add_argument('--cv-fold', type=int, default=0,
                       help='Cross-validation fold to use (default: 0)')
    
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
    
    # Print trainable parameters
    trainable_params = []
    frozen_params = []
    for name, param in model_with_prior.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Frozen parameters: {len(frozen_params)}")
    
    return model_with_prior

def setup_training_data(config, data, time_vector):
    """Setup training and validation data."""
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

def run_post_training(config, pretrained_path, output_dir, prior_weight, freeze_lambda, cv_fold):
    """Run post-training with prior knowledge."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    data, time_vector, var_names = load_and_preprocess_data(config)
    
    # Load gold standard
    gold_standard = load_gold_standard(config, var_names)
    
    # Create output directory
    output_path = create_output_directory(
        output_dir, 
        use_prior=True,  # Always true for post-training
        decay=config.get('decay', False),
        shuffled=config.get('shuffle_prior', False)
    )
    
    # Setup data
    train_loaders, val_loaders, batch_times, n_genes = setup_training_data(config, data, time_vector)
    
    # Load pretrained model
    pretrained_model = load_pretrained_model(pretrained_path, device, n_genes, config.get('dropout', 0.2))
    
    # Setup prior
    prior = setup_priors_no_holdouts(gold_standard)
    
    # Prepare model for post-training
    model = prepare_model_for_post_training(
        pretrained_model, prior, prior_weight, freeze_lambda
    )
    
    # Setup trainer
    trainer = Trainer(
        device=device,
        learning_rate=config.get('lr', 0.001),
        weight_decay=config.get('wd', 1e-5),
        patience=config.get('patience', 20)
    )
    
    # Run post-training
    print(f"\n=== Starting Post-Training ===")
    print(f"Prior weight: {prior_weight}")
    print(f"Freeze lambda: {freeze_lambda}")
    print(f"Epochs: {config.get('epochs', 500)}")
    
    epochs = config.get('epochs', 500)
    val_frequency = config.get('val_frequency', 20)
    
    history = trainer.train_with_prior(
        model, train_loaders, val_loaders, batch_times,
        epochs=epochs,
        sparse_sampling=config.get('sparse_sampling', False),
        interval_minutes=config.get('interval_minutes', 30),
        val_frequency=val_frequency
    )
    
    # Save results
    final_epoch = len(history['train_losses']) - 1
    save_cross_validation_results(
        cv_fold, final_epoch, model, 
        history['train_losses'], history['val_losses'], history['r2_scores'],
        output_path,
        shuffled=config.get('shuffle_prior', False),
        decay=config.get('decay', False)
    )
    
    # Save metadata
    save_experiment_metadata(
        output_path, config, [generate_random_seed()], [generate_random_seed()],
        prior_seeds=[generate_random_seed()], shuffled_seeds=[]
    )
    
    # Print final results
    print(f"\n=== Post-Training Completed ===")
    print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    if history['val_losses'][-1] is not None:
        print(f"Final validation loss: {history['val_losses'][-1]:.6f}")
    if history['r2_scores'][-1] is not None:
        print(f"Final R² score: {history['r2_scores'][-1]:.6f}")
    if history.get('prior_losses'):
        print(f"Final prior loss: {history['prior_losses'][-1]:.6f}")
    
    print(f"Results saved to: {output_path}")
    print("Done!")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config_file)
    
    print(f"Running post-training experiment")
    print(f"Config: {args.config_file}")
    print(f"Pretrained model: {args.pretrained_model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Prior weight: {args.prior_weight}")
    print(f"Freeze lambda: {args.freeze_lambda}")
    
    # Run post-training
    run_post_training(
        config, args.pretrained_model_path, args.output_dir, 
        args.prior_weight, args.freeze_lambda, args.cv_fold
    )

if __name__ == "__main__":
    main()