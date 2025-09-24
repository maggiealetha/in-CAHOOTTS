#!/usr/bin/env python3
"""
in-CAHOOTTs Command Line Interface

A simplified CLI for running common in-CAHOOTTs experiments.

Usage:
    python incahootts_cli.py <experiment_type> <config_file> <output_dir>

Examples:
    # Run cell cycle pretraining
    python incahootts_cli.py cell-cycle configs/cell_cycle.yaml results/cell_cycle
    
    # Run drug perturbation experiment
    python incahootts_cli.py drug-pert configs/drug_pert.yaml results/drug_pert
    
    # Run sparse sampling experiment
    python incahootts_cli.py sparse configs/sparse.yaml results/sparse
    
    # Run prior-integrated training
    python incahootts_cli.py prior configs/prior.yaml results/prior
    
    # Run block model validation
    python incahootts_cli.py block configs/block.yaml results/block
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main CLI function."""
    if len(sys.argv) != 4:
        print("Usage: python incahootts_cli.py <experiment_type> <config_file> <output_dir>")
        print("\nExperiment types:")
        print("  cell-cycle    - Cell cycle pretraining with dense sampling")
        print("  drug-pert     - Drug perturbation with IVP subset approach")
        print("  sparse        - Sparse sampling with interval-based selection")
        print("  prior         - Prior-integrated training")
        print("  block         - Block model validation")
        print("  post-train    - Post-training with prior knowledge (requires pretrained model)")
        print("\nExamples:")
        print("  python incahootts_cli.py cell-cycle configs/cell_cycle.yaml results/cell_cycle")
        print("  python incahootts_cli.py drug-pert configs/drug_pert.yaml results/drug_pert")
        sys.exit(1)
    
    experiment_type = sys.argv[1]
    config_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Map CLI experiment types to internal types
    type_mapping = {
        'cell-cycle': 'cell_cycle',
        'drug-pert': 'drug_perturbation',
        'sparse': 'sparse_sampling',
        'prior': 'prior',
        'block': 'block_model',
        'post-train': 'post_training'
    }
    
    if experiment_type not in type_mapping:
        print(f"Error: Unknown experiment type '{experiment_type}'")
        print("Valid types: cell-cycle, drug-pert, sparse, prior, block, post-train")
        sys.exit(1)
    
    internal_type = type_mapping[experiment_type]
    
    # Check if config file exists
    if not Path(config_file).exists():
        print(f"Error: Config file '{config_file}' not found")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the experiment
    cmd = [
        'python', 'run_in_cahootts.py',
        config_file, output_dir,
        '--experiment-type', internal_type
    ]
    
    print(f"Running {experiment_type} experiment...")
    print(f"Config: {config_file}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"experiment completed successfully!")
        print(f"Results saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"experiment failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"experiment interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
