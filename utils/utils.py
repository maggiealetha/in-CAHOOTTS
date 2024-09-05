import torch
import torch.nn as nn

def calc_rss(true_x, pred_x):
    return torch.nn.MSELoss(reduction='none')(true_x,pred_x).sum()

def calc_tss(true_x):
    return torch.sum((true_x-torch.mean(true_x, dim=(0,1)))**2)

def calc_r2(true_x,pred_x):
    return 1-(calc_rss(true_x,pred_x)/calc_tss(true_x))

def dropout_hook(module, input, output):
    # Count the number of zero elements in the output
    zero_count = (output == 0).sum().item()
    total_count = output.numel()
    dropout_rate = zero_count / total_count
    print(f"Estimated Dropout Rate: {dropout_rate}")
