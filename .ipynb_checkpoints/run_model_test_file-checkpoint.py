import sys
import torch
import pandas as pd
import scanpy as sc

from .setup_tools import setup_priors, preprocess_data_general, split_data, create_dls
from .base_model import ODEFunc
from .training import _training_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ts = 1
sl = 30
tmin = 0
tmax= 60
bigT=60
niters_ = int(sys.argv[1])

_cvs = 1
niters=niters_
lr = float(sys.argv[2])
wd = float(sys.argv[3])
dropout_ = float(sys.argv[4])
file_name = str(sys.argv[5])

yeast = sc.read_h5ad('yeast_data/2021_INFERELATOR_DATA.h5ad')
adata = yeast[yeast.obs['Experiment']==2]
del yeast
var_names = adata.var_names

gs_file = 'yeast_data/YEASTRACT_20230601_BOTH.tsv.gz'
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

data, time_vector, count_scaler = preprocess_data_general(adata)

cvs = [c for c in range(0,_cvs)]
aupr_r = []
aupr_b = []
gs_seeds = []
data_seeds = []
tr_mse_cvs = []
vd_mse_cvs = []
r2_cvs = []

for cv in cvs:

    #generate random seeds for cross validation
    gs_seed = np.random.randint(0,1000)
    gs_seeds.append(gs_seed)
    data_seed = np.random.randint(0,1000)
    data_seeds.append(data_seed)

    prior = setup_priors(gold_standard, gs_seed, var_names)
    train, train_t, test, test_t, n_genes = split_data(data, time_vector, seed = data_seed)

    dl_1, vdl_1 = create_dls(train, train_t, test, test_t, ts, sl, tmin, tmin+sl)
    batch_t_1 = torch.linspace(tmin+4,(bigT/2),sl+1)/(tmax)
    batch_t_1 = batch_t_1[:sl]
    dl_2, vdl_2 = create_dls(train, train_t, test, test_t, ts, sl, tmin+sl, tmax)
    batch_t_2 = torch.linspace((bigT/2),bigT,sl+1)/(tmax)
    batch_t_2 = batch_t_2[:sl]

    tr_loss, vd_loss, r2, r2_per_t, niter = _training_(niters, func, device, dl_1, vdl_1, batch_t_1, dl_2, vdl_2, batch_t_2)

    