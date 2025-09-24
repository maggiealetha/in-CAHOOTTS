# in-CAHOOTTs: Running Experiments

This guide explains how to run experiments using the unified in-CAHOOTTs framework.

## Quick Start

### 1. Simple CLI (Recommended for beginners)

```bash
# Cell cycle pretraining
python incahootts_cli.py cell-cycle experiments/configs/cell_cycle_example.yaml results/cell_cycle

# Drug perturbation experiment
python incahootts_cli.py drug-pert experiments/configs/drug_perturbation_example.yaml results/drug_pert

# Sparse sampling experiment
python incahootts_cli.py sparse experiments/configs/sparse_sampling_example.yaml results/sparse

# Prior-integrated training
python incahootts_cli.py prior experiments/configs/prior_training_example.yaml results/prior

# Block model validation
python incahootts_cli.py block experiments/configs/block_model_example.yaml results/block

# Post-training with prior knowledge
python incahootts_cli.py post-train experiments/configs/post_training_example.yaml results/post_training
```

### 2. Full CLI (More control)

```bash
# Cell cycle pretraining
python run_incahootts.py experiments/configs/cell_cycle_example.yaml results/cell_cycle --experiment-type cell_cycle

# Drug perturbation with IVP subset
python run_incahootts.py experiments/configs/drug_perturbation_example.yaml results/drug_pert --experiment-type drug_perturbation

# Sparse sampling
python run_incahootts.py experiments/configs/sparse_sampling_example.yaml results/sparse --experiment-type sparse_sampling

# Prior-integrated training
python run_incahootts.py experiments/configs/prior_training_example.yaml results/prior --experiment-type prior

# Block model validation
python run_incahootts.py experiments/configs/block_model_example.yaml results/block --experiment-type block_model

# Post-training with prior knowledge
python run_incahootts.py experiments/configs/post_training_example.yaml results/post_training --experiment-type post_training --pretrained-model pretrained_models/model_0.pth
```

## Experiment Types

### 1. Cell Cycle Pretraining (`cell_cycle`)
- **Purpose**: Pretrain model on cell cycle data
- **Sampling**: Dense time sampling with double shuffling
- **Use case**: Initial model training before prior integration
- **Data**: Typically uses pools 1 and 2 with cell cycle time axis

### 2. Drug Perturbation (`drug_perturbation`)
- **Purpose**: Train model for drug perturbation experiments
- **Sampling**: IVP subset + random time points
- **Use case**: Modeling cellular responses to drug treatments
- **Data**: Typically uses experiment 2 data

### 3. Sparse Sampling (`sparse_sampling`)
- **Purpose**: Train model with interval-based sparse time sampling
- **Sampling**: `select_sparse_random_subset` with configurable intervals
- **Use case**: When you want to sample time points at regular intervals
- **Data**: Any experiment type

### 4. Prior-Integrated Training (`prior`)
- **Purpose**: Train model with prior knowledge integration
- **Sampling**: Dense sampling (can be made sparse with `sparse_sampling: true`)
- **Use case**: Fine-tuning pretrained models with biological priors
- **Data**: Any experiment type with gold standard network

### 5. Block Model (`block_model`)
- **Purpose**: Validate against vanilla neural ODE
- **Sampling**: Dense time sampling with double shuffling
- **Use case**: Baseline comparison experiments
- **Data**: Typically uses pools 1 and 2 with cell cycle time axis

### 6. Post-Training (`post_training`)
- **Purpose**: Fine-tune pretrained models with prior knowledge
- **Sampling**: Dense sampling (can be made sparse with `sparse_sampling: true`)
- **Use case**: Second stage of training after pretraining
- **Data**: Any experiment type with gold standard network
- **Requirements**: Requires pretrained model weights

## Configuration Files

Configuration files are YAML files that specify all experiment parameters. Example configurations are provided in `experiments/configs/`.

### Key Configuration Parameters

#### Data Configuration
```yaml
data_file: yeast_data/2021_INFERELATOR_DATA.h5ad  # Path to data file
experiment_filter: pools_12  # Data filtering: pools_12, experiment_2, or none
time_axis: cc  # Time axis type: cc for cell cycle
decay: false  # Whether to use decay data format
```

#### Model Configuration
```yaml
model_type: base  # Model type: base, soft_prior, block_model
use_prior: false  # Whether to use prior knowledge
dropout: 0.2  # Dropout rate
```

#### Training Configuration
```yaml
epochs: 500  # Number of training epochs
lr: 0.001  # Learning rate
wd: 1.0e-05  # Weight decay
val_frequency: 20  # Validation frequency (every N epochs)
patience: 20  # Early stopping patience
```

#### Data Loading Configuration
```yaml
cvs: 5  # Number of cross-validation folds
sl: 29  # Sequence length
n_dls: 3  # Number of data loaders
tmin: 0  # Minimum time
tmax: 87  # Maximum time
ts: 1  # Time step
```

#### Experiment-Specific Configuration
```yaml
# For drug perturbation
ivp_subset: 10  # Number of initial time points
time_subset: 30  # Number of random time points

# For sparse sampling
interval_minutes: 30  # Time interval for sparse sampling

# For prior training
gold_standard_file: yeast_data/YEASTRACT_20230601_BOTH.tsv.gz
shuffle_prior: false  # Whether to shuffle prior network
```

## Output Structure

Results are saved in the specified output directory with the following structure:

```
results/
├── model_0.pth  # Model weights for CV fold 0
├── model_1.pth  # Model weights for CV fold 1
├── ...
├── train_loss_0.npy  # Training losses for CV fold 0
├── val_loss_0.npy    # Validation losses for CV fold 0
├── r2_0.npy          # R² scores for CV fold 0
├── ...
├── metadata.json     # Experiment metadata
└── config.yaml       # Configuration used
```

## Advanced Usage

### Custom Experiment Types

You can create custom experiment types by modifying the configuration and using the appropriate training method:

```python
# In your custom script
from incahootts import Trainer

trainer = Trainer(device=device)

# Custom training with specific parameters
history = trainer.train_pretraining(
    model=model,
    train_loaders=train_loaders,
    val_loaders=val_loaders,
    batch_times=batch_times,
    epochs=500,
    experiment_type="cell_cycle",  # or "drug_perturbation"
    sparse_sampling=True,  # Enable sparse sampling
    interval_minutes=15,   # Custom interval
    val_frequency=10       # More frequent validation
)
```

### Combining Experiment Types

You can combine different sampling strategies:

```yaml
# Cell cycle with sparse sampling
experiment_type: cell_cycle
sparse_sampling: true
interval_minutes: 30

# Drug perturbation with sparse sampling (overrides IVP approach)
experiment_type: drug_perturbation
sparse_sampling: true
interval_minutes: 30

# Prior training with sparse sampling
experiment_type: prior
sparse_sampling: true
interval_minutes: 30
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Config file not found**: Check the path to your config file
3. **Data file not found**: Ensure the data file path in config is correct
4. **Import errors**: Make sure you've installed the package: `pip install -e .`

### Performance Tips

1. **Use validation frequency**: Set `val_frequency: 20` to validate every 20 epochs
2. **Adjust sequence length**: Smaller `sl` values use less memory
3. **Use appropriate device**: The script automatically detects CUDA availability

## Migration from Old Scripts

If you're migrating from the old run scripts:

1. **Old**: `run_nODE_cc-pools_12_no_prior_pretraining_block_model.py`
   **New**: `python incahootts_cli.py block experiments/configs/block_model_example.yaml results/block`

2. **Old**: `run_nODE-time_subset.py`
   **New**: `python incahootts_cli.py drug-pert experiments/configs/drug_perturbation_example.yaml results/drug_pert`

3. **Old**: `run_nODE-soft_constraint_prior-l2_only-full_prior.py`
   **New**: `python incahootts_cli.py prior experiments/configs/prior_training_example.yaml results/prior`

The new scripts provide the same functionality with a cleaner, more maintainable interface.