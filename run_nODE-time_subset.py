import sys
import torch
import pandas as pd
import scanpy as sc
import os

from mag_nODE_model.utils.setup_tools import load_config, mkdirs, gen_seed, setup_priors, _get_data_from_ad, preprocess_data_general, preprocess_data_biophysical, split_data, setup_dls, save_per_cv, save_meta_data
from mag_nODE_model.models.base_model import ODEFunc
from mag_nODE_model.train.training_time_subset import _training_subset_time as _training_

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
    
    dl_list, vdl_list, batch_t_list = setup_dls(train, train_t, test, test_t, ts = config['ts'], sl = config['sl'], n_dls = config['n_dls'], tmin = config['tmin'], tmax = config['tmax'], bigT_for_scaling =  config['bigT_for_scaling'])
    
    if config['use_prior']:
        prior = setup_priors(gold_standard, gs_seeds[-1], var_names, shuffled=config['shuffle_prior'], shuffled_seed = shuffled_seeds)
        func = ODEFunc(device, n_genes, use_prior=config['use_prior'], prior_ = prior, dropout=dropout_)
    
    else:  
        func = ODEFunc(device, n_genes, use_prior=config['use_prior'], prior_ = None, dropout=dropout_)    
    
    tr_loss, vd_loss, r2, niter = _training_(niters, func, device, dl_list, vdl_list, batch_t_list, lr = float(config['lr']), wd = float(config['wd']), decay = config['decay'], ivp_subset = config['ivp_subset'], time_subset = config['time_subset'], subset_time = config['subset_time'])
    final_niter = niter
    
    save_per_cv(cv, final_niter, func, tr_loss, vd_loss, r2, dir_path, shuffled = config['shuffle_prior'], decay = config['decay'])

save_meta_data(dir_path, config, gs_seeds, data_seeds, prior_seeds = gs_seeds, shuffled_seeds = shuffled_seeds)
    
print(tr_loss, vd_loss, r2, niter)
print('done.')
