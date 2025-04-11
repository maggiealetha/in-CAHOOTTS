import torch
import torch.nn as nn
import numpy as np

#monitor training
def training_metrics(losses, losses_d, vd_loss_e, r2_e):
    return {
        'trajectory_loss': np.mean(losses[-100:]),  # Recent average
        'decay_loss': np.mean(losses_d[-100:]),
        'validation_loss': vd_loss_e[-1],
        'r2': r2_e[-1],
        'loss_stability': np.std(losses[-100:])  # Check if losses are stable
    }

def calc_rss(true_x, pred_x):
    return torch.nn.MSELoss(reduction='none')(true_x,pred_x).sum()

def calc_tss(true_x):
    return torch.sum((true_x-torch.mean(true_x, dim=(0,1)))**2)

def calc_r2(true_x,pred_x):
    return 1-(calc_rss(true_x,pred_x)/calc_tss(true_x))

def calc_pseudo_r2_per_feature(true_x, pred_x, exclude_low_variance=.028, epsilon=1e-6, genes=[4972, 3367, 1663, 5257, 4541, 5574], gpf = False):
    
    rss_per_feature = torch.sum((true_x - pred_x) ** 2, dim=0)
    tss_per_feature = torch.sum((true_x) ** 2, dim=0)
    n = true_x.shape[0]
    variance_per_feature = tss_per_feature / (n - 1)
    
    # Create a mask for low-variance features
    var_mask = variance_per_feature < exclude_low_variance

    # Set RSS and TSS to 0 for low-variance features
    rss_per_feature[var_mask] = 0
    tss_per_feature[var_mask] = 0
    
    r2_per_feature = 1 - (rss_per_feature / (tss_per_feature + epsilon))
    print(r2_per_feature[genes])
    
    if gpf:       
        return torch.mean(r2_per_feature), r2_per_feature[genes]
    else:
        return torch.mean(r2_per_feature)
        
def calc_r2_per_feature(true_x, pred_x, exclude_low_variance=.028, epsilon=1e-6, genes=[4972, 3367, 1663, 5257, 4541, 5574], gpf = False):
    
    rss_per_feature = torch.sum((true_x - pred_x) ** 2, dim=0)
    tss_per_feature = torch.sum((true_x - torch.mean(true_x, dim=0)) ** 2, dim=0)
    n = true_x.shape[0]
    variance_per_feature = tss_per_feature / (n - 1)
    
    # Create a mask for low-variance features
    var_mask = variance_per_feature < exclude_low_variance

    # Set RSS and TSS to 0 for low-variance features
    rss_per_feature[var_mask] = 0
    tss_per_feature[var_mask] = 0
    
    r2_per_feature = 1 - (rss_per_feature / (tss_per_feature + epsilon))
    #print(r2_per_feature[genes])
    
    if gpf:       
        return torch.mean(r2_per_feature), r2_per_feature[genes]
    else:
        return torch.mean(r2_per_feature)


def dropout_hook(module, input, output):
    # Count the number of zero elements in the output
    zero_count = (output == 0).sum().item()
    total_count = output.numel()
    dropout_rate = zero_count / total_count
    print(f"Estimated Dropout Rate: {dropout_rate}")

def check_plateau(mse_values, patience=5, threshold=0.01):
    """
    Check if the model is plateauing based on R-squared values.

    Parameters:
    mse_values (list): List of MSE values over epochs.
    patience (int): Number of epochs to wait before declaring a plateau.
    threshold (float): Minimum change to consider as not plateauing.

    Returns:
    bool: True if plateauing, False otherwise.
    """
    # if len(mse_values) < patience or len(r2_values) < patience:
    #     return False

    mse_recent = mse_values[-patience:]

    mse_change = np.abs(np.diff(mse_recent))

    mse_plateau = np.all(mse_change < threshold)

    return mse_plateau
