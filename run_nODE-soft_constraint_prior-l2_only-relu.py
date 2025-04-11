import sys
import torch
import pandas as pd
import scanpy as sc
import os
import numpy as np
import copy

from mag_nODE_model.utils.setup_tools import load_config, mkdirs, gen_seed, setup_priors, _get_data_from_ad, preprocess_data_general, preprocess_data_biophysical, split_data, setup_dls, save_per_cv, save_meta_data
from mag_nODE_model.models.base_model_soft_prior_relu import ODEFunc
from mag_nODE_model.train.training_soft_prior import _training_soft_prior as _training_

def evaluate_prior_weights(device, n_genes, prior_matrix, dl_list, vdl_list, batch_t_list,
                         prior_weights=[0.001, 0.01, 0.1, 0.5], weight_decays = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7], pre_loaded_func = None):
    """Evaluate different prior weights and their impact on model performance"""
    results = {}

    for _wd in weight_decays:
        for weight in prior_weights:
            print(f"\nTesting prior_weight: {weight}")
    
            if pre_loaded_func is not None:
                func = copy.deepcopy(pre_loaded_func)
                func.to(device)
                func.prior_ = prior_matrix.to(device)
                func.use_prior = True
                func.prior_weight=weight
                func._initialize_weights_from_prior()
                func.train()
            
            else: # Initialize model
                func = ODEFunc(
                    device=device,
                    ndim=n_genes,
                    use_prior=True,
                    prior_=prior_matrix,
                    prior_weight=weight,
                    dropout=0.1
                )
                func.to(device)
    
            
            
            # Train model
            losses, vd_losses, r2, prior_losses, epochs = _training_(
                niters=600,  # Reduced epochs for quick testing
                func=func,
                device=device,
                dls=dl_list,
                vdls=vdl_list,
                batch_ts=batch_t_list,
                lr=1e-3,
                wd=_wd
            )
    
            final_niter = epochs
            
            save_per_cv(cv, final_niter, func, losses, vd_losses, r2, dir_path, shuffled = config['shuffle_prior'],  prior_weight = weight, prior_loss = prior_losses, wd = _wd)
    
            save_meta_data(dir_path, config, gs_seeds, data_seeds, prior_seeds = gs_seeds, shuffled_seeds = shuffled_seeds)
            
            
            # Calculate metrics
            final_loss = losses[-1]
            final_r2 = r2[-1]
            
            # Get weight adherence to prior
            input_layer = func.encoder[1]
            prior = prior_matrix.to(input_layer.weight.device)
            mask = (prior != 0)
            weight_deviation = torch.mean((input_layer.weight[mask] - prior[mask])**2).item()
            
            results[weight] = {
                'final_loss': final_loss,
                'final_r2': final_r2,
                'weight_deviation': weight_deviation,
                'training_losses': losses
            }
            
            print(f"Final loss: {final_loss:.4f}")
            print(f"Final R2: {final_r2:.4f}")
            print(f"Weight deviation from prior: {weight_deviation:.4f}")
    
        if pre_loaded_func is not None:
            del func
            torch.cuda.empty_cache()  # Clear GPU cache
    
    return results

# Find best weight based on combined metrics
def find_best_weight(results):
    best_weight = None
    best_score = float('inf')
    
    for weight, metrics in results.items():
        # Normalize each metric to 0-1 range
        loss_norm = metrics['final_loss'] / max(r['final_loss'] for r in results.values())
        r2_norm = 1 - (metrics['final_r2'] / max(r['final_r2'] for r in results.values()))
        dev_norm = metrics['weight_deviation'] / max(r['weight_deviation'] for r in results.values())
        
        # Combined score (lower is better)
        score = loss_norm + r2_norm + dev_norm
        
        if score < best_score:
            best_score = score
            best_weight = weight
    
    return best_weight
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_file = str(sys.argv[1])
config = load_config(config_file)

_cvs = config['cvs']
niters_ = config['epochs']
niters=niters_
dropout_ = config['dropout']

yeast = sc.read_h5ad(config['data_file'])
adata = yeast[yeast.obs['Experiment']==2]
del yeast

gs_file = config['gold_standard_file']

if config['decay']:
    data, time_vector = preprocess_data_biophysical(adata)

else:    
    data, time_vector = preprocess_data_general(adata)

var_names = adata.var_names

gold_standard = pd.read_csv(
    gs_file,
    sep="\t",
    index_col=0
).reindex(
    var_names,
    axis=0
).fillna(
    0
).astype(int)

dir_path = mkdirs(sys.argv[2], use_prior = config['use_prior'], decay = config['decay'], shuffled = config['shuffle_prior'])

gs_seeds = []
shuffled_seeds = []
data_seeds = []

for cv in range(0, _cvs):
    #generate random seeds for cross validation
    gs_seeds.append(gen_seed())
    data_seeds.append(gen_seed())
    
    if config['shuffle_prior']:
        shuffled_seeds.append(gen_seed())        
        
    train, train_t, test, test_t, n_genes = split_data(data, time_vector, seed = data_seeds[-1], decay = config['decay'])
    
    dl_list, vdl_list, batch_t_list = setup_dls(train, train_t, test, test_t, ts = config['ts'], sl = config['sl'], n_dls = config['n_dls'], tmin = config['tmin'], tmax = config['tmax'])
      
    func = ODEFunc(device, n_genes, use_prior=False, prior_ = None, dropout=dropout_)
    func.load_state_dict(torch.load('yeast_no_prior/decay/run4164293/model_cv1_epochs500.pt', map_location=device))

    for name, param in func.named_parameters():
        if 'lambda_' in name:
            param.requires_grad = False  # Freeze this layer
    
    lambda_weights = []
    for name, param in func.named_parameters():
        if not param.requires_grad:
            print(f"Name: {name} is frozen.")
            lambda_weights.append(param[0])

    for name, param in func.named_parameters():
        if param.requires_grad:
            print(f"Name: {name} is not frozen.")

    if config['use_prior']:
        prior = setup_priors(gold_standard, gs_seeds[-1], var_names, shuffled=config['shuffle_prior'], shuffled_seed = shuffled_seeds)

        
    #func.mask_input_weights(func.prior_, func.encoder[1], 'weight')

    prior_weights = [0.99]
    results = evaluate_prior_weights(device, n_genes, prior , dl_list, vdl_list, batch_t_list,
                                   prior_weights=prior_weights, pre_loaded_func = func)

    best_weight = find_best_weight(results)
    print(f"\nBest prior_weight: {best_weight}")

    # tr_loss, vd_loss, r2, niter = _training_(niters, func, device, dl_list, vdl_list, batch_t_list, lr = float(config['lr']), wd = float(config['wd']), decay = config['decay'], decay_weight = config['decay_weight'])
    # final_niter = niter

    # lambda_weights_after = []
    # for name, param in func.named_parameters():
    #     if not param.requires_grad:
    #         lambda_weights_after.append(param[0])

    # print("lambda in weights:", torch.unique(lambda_weights_after[0]==lambda_weights[0]))
    # print("lambda out weights:", torch.unique(lambda_weights_after[1]==lambda_weights[1]))
    
#     save_per_cv(cv, final_niter, func, tr_loss, vd_loss, r2, dir_path, shuffled = config['shuffle_prior'], decay = config['decay'])

# save_meta_data(dir_path, config, gs_seeds, data_seeds, prior_seeds = gs_seeds, shuffled_seeds = shuffled_seeds)
    
# print(tr_loss, vd_loss, r2, niter)
print('done.')
