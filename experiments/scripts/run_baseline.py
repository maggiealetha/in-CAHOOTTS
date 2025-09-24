#!/usr/bin/env python3
"""
Baseline experiment runner for neural ODE models.

This script runs baseline experiments without prior knowledge integration.
"""

import sys
import torch
import pandas as pd
import scanpy as sc
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from incahootts import ODEFunc, Trainer
from incahootts.data import preprocess_data_general, split_data, TimeDataset
from incahootts.utils.config import load_config
from incahootts.utils.metrics import training_metrics


def run_baseline_experiment(config_path: str, output_dir: str):
    """
    Run baseline neural ODE experiment.
    
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
    
    # Load gold standard
    gold_standard = pd.read_csv(
        config['data_files']['gold_standard_file'],
        sep="\t",
        index_col=0
    ).reindex(adata.var_names, axis=0).fillna(0).astype(int)
    
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
    
    # Create model
    model = ODEFunc(
        device=device,
        ndim=data.shape[1],
        neurons=config['model']['neurons'],
        dropout=config['model']['dropout'],
        use_prior=config['model']['use_prior'],
        calc_erv_=config['model']['calc_erv']
    )
    
    # Create trainer
    trainer = Trainer(
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience']
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train_pretraining(
        model=model,
        train_loaders=[train_loader],
        val_loaders=[test_loader],
        batch_times=[torch.linspace(0, 1, config['data']['sequence_length'])],
        epochs=config['training']['epochs'],
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    
    # Save training history
    import json
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Calculate final metrics
    metrics = training_metrics(
        history['train_losses'],
        history['val_losses'],
        history['r2_scores']
    )
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Experiment completed. Results saved to {output_dir}")
    print(f"Final R² score: {metrics['final_r2']:.4f}")
    print(f"Best R² score: {metrics['best_r2']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_baseline.py <config_path> <output_dir>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    run_baseline_experiment(config_path, output_dir)