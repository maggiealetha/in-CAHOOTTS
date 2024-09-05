import sys
import torch
import pandas as pd
import scanpy as sc

from mag_nODE_model.setup_tools import setup_priors, preprocess_data_general, split_data, setup_dls
from mag_nODE_model.base_model import ODEFunc
from mag_nODE_model.training import _training_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cvs = 1
niters_ = 10 #sys.arg
niters=niters_
dropout_ = 0.2

yeast = sc.read_h5ad('yeast_data/2021_INFERELATOR_DATA.h5ad')
adata = yeast[yeast.obs['Experiment']==2]
del yeast

gs_file = 'yeast_data/YEASTRACT_20230601_BOTH.tsv.gz'

data, time_vector, count_scaler = preprocess_data_general(adata)
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

gs_seed = 0
prior = setup_priors(gold_standard, gs_seed, var_names)

data_seed = 0
train, train_t, test, test_t, n_genes = split_data(data, time_vector, seed = data_seed)

dl_list, vdl_list, batch_t_list = setup_dls(train, train_t, test, test_t)

func = ODEFunc(device, n_genes, use_prior=False, prior_ = None, dropout=dropout_)

#func = ODEFunc(device, n_genes, use_prior=True, prior_ = prior, dropout=dropout_)

tr_loss, vd_loss, r2, r2_per_t, niter = _training_(niters, func, device, dl_list, vdl_list, batch_t_list)

print(tr_loss, vd_loss, r2, r2_per_t, niter)
print('done.')