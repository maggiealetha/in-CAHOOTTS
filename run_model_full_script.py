#run_nODE_yeast_minimal_mbiological_grn_simulator/l.py

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

from time_dataset import TimeDataset
from torch.utils.data import DataLoader

import pandas as pd

from scipy.sparse import isspmatrix

import time

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import scanpy as sc

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from _trunc_robust_scaler import TruncRobustScaler

import sys

from torch.nn.utils import prune
from inferelator.preprocessing import ManagePriors

#try softplus activation function?
#define ode
class ODEFunc(nn.Module):
    '''NODE class implementation '''


    def __init__(self, device, ndim, explicit_time=False, neurons=158, use_prior = False, prior_ = None, scaler=None, dropout=None):
        ''' Initialize a new ODEFunc '''
        super(ODEFunc, self).__init__()

        self.ndim = ndim
        self.scale = scaler
        self.dropout_ = dropout

        self.lambda_ = nn.Sequential()
        self.lambda_.add_module('linear_in', nn.Linear(ndim, neurons, bias = False))
        self.lambda_.add_module('activation_0', nn.Softplus())
        self.lambda_.add_module('linear_out', nn.Linear(neurons,ndim, bias = False))
        self.lambda_.add_module('activation_1', nn.Softplus())

        self.encoder = nn.Sequential()
        self.encoder.add_module('gene_dropout', nn.Dropout(p=dropout))
        self.encoder.add_module('linear_in', nn.Linear(ndim, neurons, bias = False))
        self.encoder.add_module('tf_dropout', nn.Dropout(p=0.1))
        self.encoder.add_module('activation_0', nn.GELU())
        self.encoder.add_module('meta_0', nn.Linear(neurons,neurons, bias = False))
        self.encoder.add_module('activation_2', nn.GELU())
        self.encoder.add_module('meta_1', nn.Linear(neurons,neurons, bias = False))
        self.encoder.add_module('activation_3', nn.GELU())
        self.encoder.add_module('linear_out', nn.Linear(neurons,ndim, bias = False))
        self.encoder.add_module('activation_1', nn.ReLU(inplace=True))

        if use_prior:

            self.mask_input_weights(prior, self.encoder[1], 'weight')


        self.lambda_.to(device)
        self.encoder.to(device)
        print("device in model init: ", device)

    def forward(self, t, y):
        lambda_ = self.lambda_(y)
        alpha = self.encoder(y)
        decay_mod = torch.mul(lambda_,y)*-1
        dxdt = decay_mod + alpha
        return(dxdt)

    def get_biophys(self, t, y_, c_scale=None, v_scale=None):

        #scaler = 1/c_scale
        y = y_#*scaler
        lambda_ = self.lambda_(y)
        alpha = self.alpha(y).detach().numpy()#*v_scale
        decay_mod = torch.mul(lambda_,y).detach().numpy()*-1#*v_scale
        dxdt = decay_mod + alpha
        return(dxdt, alpha, decay_mod)#, lambda_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_priors(gs_df, seed):

    split_axis = 0
    split = 0.2

    _prior, _gs = ManagePriors.cross_validate_gold_standard(gs_df,gs_df, split_axis, split,seed)

    held_out = np.array(_gs.index)
    prior = torch.from_numpy(_prior.values).float()
    gs = torch.from_numpy(_gs.values).float()

    return prior, gs, held_out

def preprocess_data(data_obj):

    indices = [np.where(adata.var.index == g)[0][0] for g in ['KANMX','NATMX','Q0285']]
    genes_to_keep = list(adata.var.index)
    genes_to_keep.remove('KANMX')
    genes_to_keep.remove('NATMX')
    genes_to_keep.remove('Q0285')

    counts_scaling = TruncRobustScaler(with_centering=False) #TruncRobustScaler(with_centering=False)#
    #velo_scaling = MinMaxScaler()

    time_vector = data_obj.obs['program_rapa_time'].values

    data_obj.X = data_obj.X.astype(np.float32)
    sc.pp.normalize_per_cell(data_obj, min_counts=0)

    #_log_data = np.log1p(data_obj.X.A)
    data = counts_scaling.fit_transform(data_obj.X.A)

    #remove genes not in the matrix
    data_ = np.delete(data, indices, axis=1)

    return data_, time_vector, counts_scaling.scale_, genes_to_keep

def split_data(data, time, seed = 257, split = 0.2):

    validation_size = split
    random_seed = seed

    train_idx, test_idx = train_test_split(
        np.arange(data.shape[0]),
        test_size=validation_size,
        random_state=random_seed
    )

    _train = data[train_idx, :]
    _train_t = time_vector[train_idx]
    _test = data[test_idx, :]
    _test_t = time_vector[test_idx]

    return _train, _train_t, _test, _test_t

def create_dls(_train, _tr_t, _test, _t_t, ts, sl, tmin, tmax):

    n_genes = _train.shape[-1]

    td = TimeDataset(
    _train,
    _tr_t,
    tmin,
    tmax,
    t_step=ts,
    sequence_length=sl
    )
    print(td.n)
    dl = DataLoader(
            td,
            batch_size=20,#td.n
            drop_last=True,
        )


    vd = TimeDataset(
    _test,
    _t_t,
    tmin,
    tmax,
    t_step=ts,
    sequence_length=sl
    )
    print(vd.n)
    vdl = DataLoader(
            vd,
            batch_size=20,#vd.n
            drop_last=True,
        )

    return dl, vdl, n_genes

def _shuffle_time_data(dl):
    try:
        dl.dataset.shuffle()
    except AttributeError:
        pass

def calc_rss(true_x, pred_x):
    return torch.nn.MSELoss(reduction='none')(true_x,pred_x).sum()

def calc_tss(true_x):
    return torch.sum((true_x-torch.mean(true_x, dim=(0,1)))**2)

def calc_r2(true_x,pred_x):
    return 1-(calc_rss(true_x,pred_x)/calc_tss(true_x))

def training_single_dl(niters, dl1, vdl1, batch_t1, func, optimizer):

    loss_f = torch.nn.MSELoss()
    losses = []
    vd_loss = []
    r2_e_1 = []
    r2_e_2 = []
    r2_e = []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
    factor=0.9, patience=30, threshold=1e-09,
    threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-09, verbose=True)
    for iter in tqdm(range(niters + 1)):
        optimizer.zero_grad()
        step_loss = []
        _tr_loss=0
        #random_numbers = np.random.choice(np.arange(0, 5723), 300, replace=False)
        for d in dl1:
            batch_x0 = torch.mean(d[:,0,:],dim=0).to(device)
            _batch_x = torch.mean(d,dim=0).to(device)
            _pred_x = odeint(func = func, y0=batch_x0, t=batch_t1, method='rk4').to(device)
            tr_loss = loss_f(_pred_x,_batch_x)
            #tr_loss.backward()
            _tr_loss+= tr_loss
            step_loss.append(tr_loss.item())

        _tr_loss.backward()
        optimizer.step()
        scheduler.step(np.mean(step_loss))
        losses.append(np.mean(step_loss))
        if np.isnan(losses[-1]):
            print('nan encountered, stopping at: ', iter)
            break
        _shuffle_time_data(dl1)

        if iter%20 == 0:
            print("epoch: ", iter)
            for vd in vdl1:
                batch_x0 = torch.mean(vd[:,0,:],dim=0).to(device)
                _batch_x = torch.mean(vd,dim=0).to(device)
                _pred_x = odeint(func = func, y0=batch_x0, t=batch_t1, method='rk4').to(device)
                vd_loss.append(loss_f(_pred_x,_batch_x).item())
                r2_e_1.append(calc_r2(_batch_x,_pred_x).item())
                print(r2_e_1[-1])

            r2_e_1_m = np.mean(r2_e_1)

            r2_e.append(np.mean([r2_e_1_m]))
            if (iter > 100) and (r2_e[-4] > r2_e[-1]):
                print('early stopping at epoch: ', iter)
                break
            _shuffle_time_data(vdl1)

    return losses, vd_loss, r2_e, r2_e_1, iter
    
def training__(niters, dl1, vdl1, batch_t1, dl2, vdl2, batch_t2, func, optimizer):

    loss_f = torch.nn.MSELoss()
    losses = []
    vd_loss = []
    r2_e_1 = []
    r2_e_2 = []
    r2_e = []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
    factor=0.9, patience=30, threshold=1e-09,
    threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-09, verbose=True)
    for iter in tqdm(range(niters + 1)):
        optimizer.zero_grad()
        step_loss = []
        _tr_loss=0
        #random_numbers = np.random.choice(np.arange(0, 5723), 300, replace=False)
        for d in dl1:
            batch_x0 = torch.mean(d[:,0,:],dim=0).to(device)
            _batch_x = torch.mean(d,dim=0).to(device)
            _pred_x = odeint(func = func, y0=batch_x0, t=batch_t1, method='rk4').to(device)
            tr_loss = loss_f(_pred_x,_batch_x)
            #tr_loss.backward()
            _tr_loss+= tr_loss
            step_loss.append(tr_loss.item())
        for d in dl2:
            batch_x0 = torch.mean(d[:,0,:],dim=0).to(device)
            _batch_x = torch.mean(d,dim=0).to(device)
            #print(_batch_x.shape)
            _pred_x = odeint(func = func, y0=batch_x0, t=batch_t2, method='rk4').to(device)
            #print(_pred_x.shape)
            tr_loss = loss_f(_pred_x,_batch_x)
            _tr_loss+= tr_loss
            #_tr_loss+= tr_loss
            step_loss.append(tr_loss.item())

        _tr_loss.backward()
        optimizer.step()
        scheduler.step(np.mean(step_loss))
        losses.append(np.mean(step_loss))
        if np.isnan(losses[-1]):
            print('nan encountered, stopping at: ', iter)
            break
        _shuffle_time_data(dl1)
        _shuffle_time_data(dl2)

        if iter%20 == 0:
            print("epoch: ", iter)
            for vd in vdl1:
                batch_x0 = torch.mean(vd[:,0,:],dim=0).to(device)
                _batch_x = torch.mean(vd,dim=0).to(device)
                _pred_x = odeint(func = func, y0=batch_x0, t=batch_t1, method='rk4').to(device)
                vd_loss.append(loss_f(_pred_x,_batch_x).item())
                r2_e_1.append(calc_r2(_batch_x,_pred_x).item())
                print(r2_e_1[-1])
            for vd in vdl2:
                batch_x0 = torch.mean(vd[:,0,:],dim=0).to(device)
                _batch_x = torch.mean(vd,dim=0).to(device)
                _pred_x = odeint(func = func, y0=batch_x0, t=batch_t1, method='rk4').to(device)
                vd_loss.append(loss_f(_pred_x,_batch_x).item())
                r2_e_2.append(calc_r2(_batch_x,_pred_x).item())
                print(r2_e_2[-1])

            r2_e_1_m = np.mean(r2_e_1)
            r2_e_2_m = np.mean(r2_e_2)

            r2_e.append(np.mean([r2_e_1_m, r2_e_2_m]))
            if (iter > 100) and (r2_e[-4] > r2_e[-1]):
                print('early stopping at epoch: ', iter)
                break
            _shuffle_time_data(vdl1)
            _shuffle_time_data(vdl2)

    return losses, vd_loss, r2_e, r2_e_1,r2_e_2, iter

def plot_training(niters, tr_loss, vd_loss, r2):
    #mse
    plt.figure(figsize = (6,6), dpi = 300)
    plt.plot(np.linspace(0, niters, niters+1),tr_loss, label = "training")
    plt.plot(np.linspace(0, niters, len(vd_loss)), vd_loss, label = "validation")#len(vd_loss)),[r.detach().numpy() for r in vd_loss], alpha = 0.7)
    plt.xlim(0,niters+30)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend(loc='best')


    #r2
    plt.figure(figsize = (6,6), dpi = 300)
    plt.plot(np.linspace(0, niters, len(r2)), r2)
    plt.ylim(0,0.5)
    plt.show()


from sklearn.metrics import precision_recall_curve, auc

def calculate_aupr(true_adjacency, predicted_adjacency):
    # Flatten matrices to 1D arrays
    y_true = true_adjacency.flatten()
    y_scores = predicted_adjacency.flatten()

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Calculate area under the precision-recall curve (AUPR)
    aupr = auc(recall, precision)

    return aupr

def get_aupr(model, _gs, idx):

    gs = (_gs.detach().cpu().numpy().astype(np.float64) !=0).astype(float)
    model_state_dict = model.state_dict()
    pred_adj = model_state_dict['alpha.linear_out.weight'][idx].detach().cpu().numpy().astype(np.float64)
    pa = (pred_adj !=0).astype(float)

    return calculate_aupr(gs,pred_adj),calculate_aupr(gs,pa)

def dropout_hook(module, input, output):
    # Count the number of zero elements in the output
    zero_count = (output == 0).sum().item()
    total_count = output.numel()
    dropout_rate = zero_count / total_count
    print(f"Estimated Dropout Rate: {dropout_rate}")

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

    #randomly select prior and data
    #prior, gs, held_out_connections = setup_priors(_gs_,gs_seed)
    train, train_t, test, test_t = split_data(data, time_vector, seed = data_seed)
    dl_1, vdl_1, n_genes = create_dls(train, train_t, test, test_t, ts, sl, tmin, sl)
    batch_t_1 = torch.linspace(0,bigT/2,sl+1)/(bigT)
    batch_t_1 = batch_t_1[:sl]
    # dl_2, vdl_2, n_genes = create_dls(train, train_t, test, test_t, ts, sl, sl, tmax)
    # batch_t_2 = torch.linspace(bigT/2,bigT,sl+1)/(bigT)
    # batch_t_2 = batch_t_2[:sl]

    if cv != 0:
        del func

    print('dropout: ', dropout_)
    func = ODEFunc(device, n_genes, use_prior=False, prior_ = None, dropout = dropout_)
    print('dropout post initialization: ', func.dropout_)

    for name, module in func.named_modules():
        if isinstance(module, nn.Dropout):
            module.register_forward_hook(dropout_hook)

    optimizer = opt = optim.Adam(func.parameters(), lr=lr, weight_decay=wd)

    #training
    tr_loss, vd_loss, r2, r21,r22, niter = training__(niters, dl_1, vdl_1, batch_t_1, dl_2, vdl_2, batch_t_2, func, optimizer)
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
