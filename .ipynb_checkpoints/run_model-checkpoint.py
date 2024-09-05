import sys
import torch
import pandas as pd
import scanpy as sc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


yeast = sc.read_h5ad('yeast_data/2021_INFERELATOR_DATA.h5ad')
adata = yeast[yeast.obs['Experiment']==2]
del yeast

gs_file = 'yeast_data/YEASTRACT_20230601_BOTH.tsv.gz'
gs_tsv = pd.read_csv(gs_file, sep = '\t', index_col = 0)

ts = 1
sl = 30
tmin = 0
tmax= 60
bigT=60
cv = 0
niters_ = int(sys.argv[1])

_cvs = 1
niters=niters_
lr = float(sys.argv[2])
wd = float(sys.argv[3])
dropout_ = float(sys.argv[4])
file_name = str(sys.argv[5])

data, time_vector, count_scaler, rm_idx = preprocess_data(adata)
_gs_ = gs_tsv.T[np.asarray(rm_idx)].T

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

    train, train_t, test, test_t = split_data(data, time_vector, seed = data_seed)

    #would be cool to automate this as well over the different times that you want to make it more user friendly
    dl_1, vdl_1, n_genes = create_dls(train, train_t, test, test_t, ts, sl, tmin, tmin+sl)
    batch_t_1 = torch.linspace(tmin+4,(bigT/2),sl+1)/(tmax)
    batch_t_1 = batch_t_1[:sl]
    dl_2, vdl_2, n_genes = create_dls(train, train_t, test, test_t, ts, sl, tmin+sl, tmax)
    batch_t_2 = torch.linspace((bigT/2),bigT,sl+1)/(tmax)
    batch_t_2 = batch_t_2[:sl]

    if cv != 0:
        del func
    func = ODEFunc(device, n_genes, use_prior=True, prior_ = prior, dropout=dropout_)
    optimizer = opt = optim.Adam(func.parameters(), lr=lr, weight_decay=wd)
    prior = prior.to(device)

    #training
    tr_loss, vd_loss, r2, r21,r22, niter = _training_(niters, dl_1, vdl_1, batch_t_1, dl_2, vdl_2, batch_t_2, func, optimizer, prior, device)
    niters_ = niter

    #save model
    torch.save(func.state_dict(), "yeast_no_prior_activation_tuning/gelu_60min_epochs%i_wd%.6f_lr%.4f_cv%i_sl%i_dropout%.2f_d_test.pt"%(niters_, wd, lr, cv, sl, dropout_))

    #plot_training(niters, tr_loss, vd_loss, r2)
    tr_mse_cvs = tr_loss
    vd_mse_cvs = vd_loss
    r2_cvs = r2

    np.savetxt('yeast_no_prior_activation_tuning/tr_gelu_60min_epochs%i_wd%.6f_lr%.4f_cv%i_sl%i_dropout%.2f_d_test.csv'%(niters_, wd, lr, cv, sl, dropout_), tr_mse_cvs, delimiter=',')
    np.savetxt('yeast_no_prior_activation_tuning/vd_gelu_60min_epochs%i_wd%.6f_lr%.4f_cv%i_sl%i_dropout%.2f_d_test.csv'%(niters_, wd, lr, cv, sl, dropout_), vd_mse_cvs, delimiter=',')
    np.savetxt('yeast_no_prior_activation_tuning/r2_gelu_60min_epochs%i_wd%.6f_lr%.4f_cv%i_sl%i_dropout%.2f_d_test.csv'%(niters_, wd, lr, cv, sl, dropout_), r2_cvs, delimiter=',')


# tr_mse_cvs=np.asarray(tr_mse_cvs)
# vd_mse_cvs=np.asarray(vd_mse_cvs)
# r2_mse_cvs=np.asarray(r2_mse_cvs)


meta_data = np.vstack((gs_seeds,data_seeds)).T
pd.DataFrame(meta_data, columns = ['gs_seed', 'data_seed']).to_csv('yeast_no_prior_activation_tuning/metadata_gelu_60min_epochs%i_wd%.6f_lr%.4f_cv%i_sl%i_dropout%.2f_d_test.tsv' %(niters_, wd, lr, cv, sl, dropout_), sep = '\t')

print('done.')
