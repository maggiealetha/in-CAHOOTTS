"""
Evaluation metrics for neural ODE models.
"""

import torch
import numpy as np


def calculate_r2(y_true, y_pred):
    """
    Calculate R² score.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    if isinstance(y_true, torch.Tensor):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()
    else:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


def calculate_pseudo_r2(y_true, y_pred, exclude_low_variance=0.028, epsilon=1e-6):
    """
    Calculate pseudo R² score excluding low variance genes.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        exclude_low_variance: Variance threshold for exclusion
        epsilon: Small value to avoid division by zero
        
    Returns:
        Pseudo R² score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
    
    # Calculate variance for each gene
    gene_vars = np.var(y_true, axis=0)
    
    # Filter out low variance genes
    high_var_mask = gene_vars > exclude_low_variance
    
    if np.sum(high_var_mask) == 0:
        return 0.0
    
    # Calculate R² for high variance genes only
    y_true_filtered = y_true[:, high_var_mask]
    y_pred_filtered = y_pred[:, high_var_mask]
    
    ss_res = np.sum((y_true_filtered - y_pred_filtered) ** 2)
    ss_tot = np.sum((y_true_filtered - np.mean(y_true_filtered)) ** 2)
    
    if ss_tot < epsilon:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_r2_per_feature(y_true, y_pred, exclude_low_variance=0.028, epsilon=1e-6):
    """
    Calculate R² score per feature (gene).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        exclude_low_variance: Variance threshold for exclusion
        epsilon: Small value to avoid division by zero
        
    Returns:
        R² scores per feature
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
    
    # Calculate variance for each gene
    gene_vars = np.var(y_true, axis=0)
    
    # Filter out low variance genes
    high_var_mask = gene_vars > exclude_low_variance
    
    r2_scores = np.zeros(y_true.shape[1])
    
    for i in range(y_true.shape[1]):
        if high_var_mask[i]:
            ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
            
            if ss_tot > epsilon:
                r2_scores[i] = 1 - (ss_res / ss_tot)
    
    return r2_scores


def training_metrics(train_losses, val_losses, r2_scores, prior_losses=None):
    """
    Calculate training metrics summary.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        r2_scores: R² scores
        prior_losses: Prior regularization losses (optional)
        
    Returns:
        dict: Metrics summary
    """
    metrics = {
        'final_train_loss': train_losses[-1] if train_losses else 0.0,
        'final_val_loss': val_losses[-1] if val_losses else 0.0,
        'final_r2': r2_scores[-1] if r2_scores else 0.0,
        'best_r2': max(r2_scores) if r2_scores else 0.0,
        'best_epoch': np.argmax(r2_scores) if r2_scores else 0,
        'convergence_epoch': len(train_losses) - 1
    }
    
    if prior_losses:
        metrics['final_prior_loss'] = prior_losses[-1]
        metrics['avg_prior_loss'] = np.mean(prior_losses)
    
    return metrics


def check_plateau(values, patience=5, threshold=0.01):
    """
    Check if values have plateaued.
    
    Args:
        values: List of values to check
        patience: Number of epochs to wait for improvement
        threshold: Minimum improvement threshold
        
    Returns:
        bool: True if plateaued
    """
    if len(values) < patience:
        return False
    
    recent_values = values[-patience:]
    improvement = max(recent_values) - min(recent_values)
    
    return improvement < threshold


def calculate_rss(true_x, pred_x):
    """
    Calculate residual sum of squares.
    
    Args:
        true_x: True values
        pred_x: Predicted values
        
    Returns:
        torch.Tensor: RSS value
    """
    return torch.nn.MSELoss(reduction='none')(true_x, pred_x).sum()


def calculate_tss(true_x):
    """
    Calculate total sum of squares.
    
    Args:
        true_x: True values
        
    Returns:
        torch.Tensor: TSS value
    """
    return torch.sum((true_x - torch.mean(true_x, dim=(0, 1))) ** 2)


def calculate_r2_components(true_x, pred_x):
    """
    Calculate R² using RSS and TSS components.
    
    Args:
        true_x: True values
        pred_x: Predicted values
        
    Returns:
        torch.Tensor: R² value
    """
    rss = calculate_rss(true_x, pred_x)
    tss = calculate_tss(true_x)
    return 1 - (rss / tss)


def calculate_pseudo_r2_per_feature(true_x, pred_x, exclude_low_variance=0.028, 
                                   epsilon=1e-6, genes=None, return_genes=False):
    """
    Calculate pseudo R² per feature excluding low variance genes.
    
    Args:
        true_x: True values
        pred_x: Predicted values
        exclude_low_variance: Variance threshold for exclusion
        epsilon: Small value to avoid division by zero
        genes: Specific gene indices to return (optional)
        return_genes: Whether to return gene-specific values
        
    Returns:
        torch.Tensor or tuple: Mean R² or (mean R², gene R² values)
    """
    if isinstance(true_x, torch.Tensor):
        true_x = true_x.detach().numpy()
        pred_x = pred_x.detach().numpy()
    
    rss_per_feature = np.sum((true_x - pred_x) ** 2, axis=0)
    tss_per_feature = np.sum(true_x ** 2, axis=0)
    n = true_x.shape[0]
    variance_per_feature = tss_per_feature / (n - 1)
    
    # Create mask for low-variance features
    var_mask = variance_per_feature < exclude_low_variance
    
    # Set RSS and TSS to 0 for low-variance features
    rss_per_feature[var_mask] = 0
    tss_per_feature[var_mask] = 0
    
    r2_per_feature = 1 - (rss_per_feature / (tss_per_feature + epsilon))
    
    if genes is not None:
        print(f"R² for specified genes: {r2_per_feature[genes]}")
    
    if return_genes and genes is not None:
        return np.mean(r2_per_feature), r2_per_feature[genes]
    else:
        return np.mean(r2_per_feature)


def calculate_r2_per_feature(true_x, pred_x, exclude_low_variance=0.028, 
                            epsilon=1e-6, genes=None, return_genes=False):
    """
    Calculate R² per feature excluding low variance genes.
    
    Args:
        true_x: True values
        pred_x: Predicted values
        exclude_low_variance: Variance threshold for exclusion
        epsilon: Small value to avoid division by zero
        genes: Specific gene indices to return (optional)
        return_genes: Whether to return gene-specific values
        
    Returns:
        torch.Tensor or tuple: Mean R² or (mean R², gene R² values)
    """
    if isinstance(true_x, torch.Tensor):
        true_x = true_x.detach().numpy()
        pred_x = pred_x.detach().numpy()
    
    rss_per_feature = np.sum((true_x - pred_x) ** 2, axis=0)
    tss_per_feature = np.sum((true_x - np.mean(true_x, axis=0)) ** 2, axis=0)
    n = true_x.shape[0]
    variance_per_feature = tss_per_feature / (n - 1)
    
    # Create mask for low-variance features
    var_mask = variance_per_feature < exclude_low_variance
    
    # Set RSS and TSS to 0 for low-variance features
    rss_per_feature[var_mask] = 0
    tss_per_feature[var_mask] = 0
    
    r2_per_feature = 1 - (rss_per_feature / (tss_per_feature + epsilon))
    
    if genes is not None:
        print(f"R² for specified genes: {r2_per_feature[genes]}")
    
    if return_genes and genes is not None:
        return np.mean(r2_per_feature), r2_per_feature[genes]
    else:
        return np.mean(r2_per_feature)


def dropout_hook(module, input, output):
    """
    Hook to monitor dropout rate in a module.
    
    Args:
        module: PyTorch module
        input: Module input
        output: Module output
    """
    # Count the number of zero elements in the output
    zero_count = (output == 0).sum().item()
    total_count = output.numel()
    dropout_rate = zero_count / total_count
    print(f"Estimated Dropout Rate: {dropout_rate}")


# Legacy function names for backward compatibility
def calc_rss(true_x, pred_x):
    """Legacy function name for calculate_rss."""
    return calculate_rss(true_x, pred_x)


def calc_tss(true_x):
    """Legacy function name for calculate_tss."""
    return calculate_tss(true_x)


def calc_r2(true_x, pred_x):
    """Legacy function name for calculate_r2_components."""
    return calculate_r2_components(true_x, pred_x)


def calc_pseudo_r2_per_feature(true_x, pred_x, exclude_low_variance=0.028, 
                              epsilon=1e-6, genes=None, gpf=False):
    """Legacy function name for calculate_pseudo_r2_per_feature."""
    return calculate_pseudo_r2_per_feature(true_x, pred_x, exclude_low_variance, 
                                         epsilon, genes, gpf)


def calc_r2_per_feature(true_x, pred_x, exclude_low_variance=0.028, 
                       epsilon=1e-6, genes=None, gpf=False):
    """Legacy function name for calculate_r2_per_feature."""
    return calculate_r2_per_feature(true_x, pred_x, exclude_low_variance, 
                                  epsilon, genes, gpf)