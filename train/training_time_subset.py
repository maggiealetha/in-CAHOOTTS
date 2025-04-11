#clean_runs_for_time_holdouts
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import prune

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

from  ..utils.setup_tools import _shuffle_time_data, select_random_subset
from ..utils.utils import calc_r2_per_feature as calc_r2
from ..utils.utils import check_plateau

from torchmetrics.regression import MeanAbsolutePercentageError

def _training_subset_time(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1, ivp_subset = 10, time_subset = 30, subset_time = False):
    loss_f = torch.nn.MSELoss()
    losses = [] 
    vd_loss_e = []
    r2_e = []

    losses_d = []
    vd_loss_d_e = []
    
    optimizer = opt = optim.Adam(func.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.9, patience=patience_, threshold=1e-09,
                                                     threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-09, verbose=True)
    if func.use_prior:
        print('sending prior to device')
        prior = func.prior_.to(device)
        
    for iter in tqdm(range(niters + 1)):
        optimizer.zero_grad()
        step_loss = []
        step_loss_d = []
        _tr_loss = 0

        for i in range(len(dls)):
            dl = dls[i]    # Training dataset
            vdl = vdls[i]  # Validation dataset
            batch_t_ = batch_ts[i]  # Batch time corresponding to the dataset

            if subset_time:
                batch_t = torch.cat(((batch_t_[:ivp_subset]),(select_random_subset(batch_t_, ivp_subset, time_subset))))
                
            else:
                batch_t = batch_t_
            #batch_t = torch.cat(((batch_t_[:ivp_subset]),(select_random_subset(batch_t_, time_subset))))
            batch_idx = (batch_t*(len(batch_t_)-1)).int()

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,batch_idx,:,0],dim=0).to(device)


                else:
                    #print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d[:, batch_idx, :], dim=0).to(device)

                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)
                tr_loss = loss_f(_pred_x, _batch_x)
                                
                tr_loss.backward()

                step_loss.append(tr_loss.item())

                if decay: # to avoid training model in parallel pass predicted expression into the model
                    
                    decay_velo = torch.mean(d[:, batch_idx,:,1],dim=0).to(device)
                    _pred_x_detached = _pred_x.detach()
                    pred_decay_velo = func.get_decay(_pred_x_detached).to(device) 

                    dv_loss = decay_weight*loss_f(pred_decay_velo, decay_velo)
                    dv_loss.backward()
                    step_loss_d.append(dv_loss.item())

        optimizer.step()

        if func.use_prior:
            #print('in pruning function')
            prune.remove(func.encoder[1], 'weight') #assumes dropout
            prune.custom_from_mask(func.encoder[1], name='weight', mask= prior != 0)

        scheduler.step(np.mean(step_loss))
        losses.append(np.mean(step_loss))
        losses_d.append(np.mean(step_loss_d))

        if np.isnan(losses[-1]):
            print('nan encountered, stopping at: ', iter)
            break

        for i in range(len(dls)):
            _shuffle_time_data(dls[i])    # Shuffle training data
            _shuffle_time_data(vdls[i])  # Shuffle validation data

        if iter % 20 == 0:
            r2_e_list = [[] for _ in range(len(vdls))]  # Separate r2_e for each pair of dataset
            vd_loss = [[] for _ in range(len(vdls))]
            
            for i in range(len(vdls)):
                vdl = vdls[i]
                # Validation loop
                for vd in vdl:
                    if decay:
                        batch_x0 = torch.mean(vd[:,0,:,0],dim=0).to(device)
                        _batch_x = torch.mean(vd[:,batch_idx,:,0],dim=0).to(device)
                    else:
                        batch_x0 = torch.mean(vd[:,0,:], dim=0).to(device)
                        _batch_x = torch.mean(vd[:, batch_idx, :], dim=0).to(device)
                        
                    _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)
                    vd_loss[i].append(loss_f(_pred_x, _batch_x).item())
                    r2_e_list[i].append(calc_r2(_batch_x, _pred_x).item())
                    print(f'R2 for dataset {i+1}: ', r2_e_list[i][-1])
            
            for r2_e_dataset in r2_e_list:
                print('r2_e_dataset', len(r2_e_dataset), 'ndls', len(dls))
            r2_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in r2_e_list]
            r2_e.append(np.mean(r2_e_m))

            vd_e_m = [np.mean(vd_e_dataset) for vd_e_dataset in vd_loss]
            vd_loss_e.append(np.mean(vd_e_m))

            # if (iter > 200) and (r2_e[-4] > r2_e[-1]):
            #     print('early stopping at epoch: ', iter)
            #     break

    return losses, vd_loss_e, r2_e, iter