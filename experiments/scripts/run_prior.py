#!/usr/bin/env python3
"""
Prior integration experiment runner for neural ODE models.

This script runs experiments with prior knowledge integration.
"""

import sys
import torch
import pandas as pd
import scanpy as sc
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from incahootts import ODEFunc, SoftPriorODEFunc, Trainer
from incahootts.data import preprocess_data_general, split_data, TimeDataset
from incahootts.utils.config import load_config
from incahootts.utils.metrics import training_metrics


def run_prior_experiment(config_path: str, output_dir: str):
    """
    Run prior integration neural ODE experiment.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    yeast_data = sc.read_h5ad(config['data_files']['data_file'])
    adata = yeast_data[yeast_data.obs['Experiment'] == config['data_files']['experiment_filter']]
    del yeast_data
    
    # Preprocess data
    data, time_vector = preprocess_data_general(adata)
    
    # Load gold standard for prior
    gold_standard = pd.read_csv(
        config['data_files']['gold_standard_file'],
        sep="\t",
        index_col=0
    ).reindex(adata.var_names, axis=0).fillna(0).astype(int)
    
    # Convert to tensor for prior
    prior_matrix = torch.tensor(gold_standard.values, dtype=torch.float32)
    
    # Split data
    train_data, train_time, test_data, test_time = split_data(
        data, time_vector, 
        seed=config['experiment']['random_seed'],
        split=0.2)
    
    # Create data loaders
    train_dataset = TimeDataset(
        train_data, train_time,
        sequence_length=config['data']['sequence_length']
    )
    test_dataset = TimeDataset(
        test_data, test_time,
        sequence_length=config['data']['sequence_length']
    )
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Stage 1: Pretraining
    print("Stage 1: Pretraining...")
    base_model = ODEFunc(
        device=device,
        ndim=data.shape[1],
        neurons=config['model']['neurons'],
        dropout=config['model']['dropout'],
        use_prior=False
    )
    
    trainer = Trainer(
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience']
    )
    
    pretrain_history = trainer.train_pretraining(
        model=base_model,
        train_loaders=[train_loader],
        val_loaders=[test_loader],
        batch_times=[torch.linspace(0, 1, config['data']['sequence_length'])],
        epochs=config['training']['epochs'] // 2  # Use half epochs for pretraining
    )
    
    # Stage 2: Prior integration
    print("Stage 2: Prior integration...")
    prior_model = SoftPriorODEFunc(
        device=device,
        ndim=data.shape[1],
        neurons=config['model']['neurons'],
        dropout=config['model']['dropout'],
        use_prior=True,
        prior_=prior_matrix,
        prior_weight=config['model']['prior_weight']
    )
    
    # Copy pretrained weights
    prior_model.load_state_dict(base_model.state_dict())
    
    prior_history = trainer.train_with_prior(
        model=prior_model,
        train_loaders=[train_loader],
        val_loaders=[test_loader],
        batch_times=[torch.linspace(0, 1, config['data']['sequence_length'])],
        epochs=config['training']['epochs'] // 2  # Use half epochs for prior training
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    torch.save(base_model.state_dict(), os.path.join(output_dir, 'base_model.pth'))
    torch.save(prior_model.state_dict(), os.path.join(output_dir, 'prior_model.pth'))
    
    # Save training histories
    import json
    with open(os.path.join(output_dir, 'pretrain_history.json'), 'w') as f:
        json.dump(pretrain_history, f, indent=2)
    
    with open(os.path.join(output_dir, 'prior_history.json'), 'w') as f:
        json.dump(prior_history, f, indent=2)
    
    # Calculate final metrics
    pretrain_metrics = training_metrics(
        pretrain_history['train_losses'],
        pretrain_history['val_losses'],
        pretrain_history['r2_scores']
    )
    
    prior_metrics = training_metrics(
        prior_history['train_losses'],
        prior_history['val_losses'],
        prior_history['r2_scores'],
        prior_history.get('prior_losses')
    )
    
    with open(os.path.join(output_dir, 'pretrain_metrics.json'), 'w') as f:
        json.dump(pretrain_metrics, f, indent=2)
    
    with open(os.path.join(output_dir, 'prior_metrics.json'), 'w') as f:
        json.dump(prior_metrics, f, indent=2)
    
    print(f"Experiment completed. Results saved to {output_dir}")
    print(f"Pretraining R² score: {pretrain_metrics['final_r2']:.4f}")
    print(f"Prior integration R² score: {prior_metrics['final_r2']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_prior.py <config_path> <output_dir>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    run_prior_experiment(config_path, output_dir)