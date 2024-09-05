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

from  .setup_tools import _shuffle_time_data
from .utils import calc_r2


def _training_(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False):
    loss_f = torch.nn.MSELoss()
    losses = []
    vd_loss_e = []
    r2_e = []
    optimizer = opt = optim.Adam(func.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.9, patience=30, threshold=1e-09,
                                                     threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-09, verbose=True)
    if func.use_prior:
        print('sending prior to device')
        prior = func.prior_.to(device)
        
    for iter in tqdm(range(niters + 1)):
        optimizer.zero_grad()
        step_loss = []
        _tr_loss = 0

        for i in range(len(dls)):
            dl = dls[i]    # Training dataset
            vdl = vdls[i]  # Validation dataset
            batch_t = batch_ts[i]  # Batch time corresponding to the dataset

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)

                    decay_velo = torch.mean(d[:,0,:,1],dim=0).to(device)
                    pred_decay_velo = func.get_decay(batch_x0)

                    dv_loss = loss_f(pred_decay_velo, decay_velo)
                    dv_loss.backward(retain_graph=True)


                else:
                    print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d, dim=0).to(device)

                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)
                tr_loss = loss_f(_pred_x, _batch_x)
                tr_loss.backward()

                step_loss.append(tr_loss.item())

        optimizer.step()

        if func.use_prior:
            print('in pruning function')
            prune.remove(func.encoder[1], 'weight') #assumes dropout
            prune.custom_from_mask(func.encoder[1], name='weight', mask= prior != 0)

        scheduler.step(np.mean(step_loss))
        losses.append(np.mean(step_loss))

        if np.isnan(losses[-1]):
            print('nan encountered, stopping at: ', iter)
            break

        for i in range(len(dls)):
            _shuffle_time_data(dls[i])    # Shuffle training data
            _shuffle_time_data(vdls[i])  # Shuffle validation data

        if iter % 2 == 0:
            r2_e_list = [[] for _ in range(len(vdls))]  # Separate r2_e for each pair of dataset
            vd_loss = [[] for _ in range(len(vdls))]
            
            for i in range(len(vdls)):
                vdl = vdls[i]
                # Validation loop
                for vd in vdl:
                    if decay:
                        batch_x0 = torch.mean(vd[:,0,:,0],dim=0).to(device)
                        _batch_x = torch.mean(vd[:,:,:,0],dim=0).to(device)
                    else:
                        batch_x0 = torch.mean(vd[:,0,:], dim=0).to(device)
                        _batch_x = torch.mean(vd, dim=0).to(device)
                        
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

            if (iter > 100) and (r2_e[-4] > r2_e[-1]):
                print('early stopping at epoch: ', iter)
                break

    return losses, vd_loss_e, r2_e, iter


def __training__(niters, func, device, *args,  lr = 1e-3, wd = 1e-5, decay = False):
    loss_f = torch.nn.MSELoss()
    losses = []
    vd_loss = []
    r2_e = []
    r2_e_list = [[] for _ in range(len(args)//3)]  # Separate r2_e for each pair of dataset

    optimizer = opt = optim.Adam(func.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.9, patience=30, threshold=1e-09,
                                                     threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-09, verbose=True)
    if func.use_prior:
        prior = func.prior_.to(device)
        
    for iter in tqdm(range(niters + 1)):
        optimizer.zero_grad()
        step_loss = []
        _tr_loss = 0

        for i in range(0, len(args)//3):
            dl = args[i*3]    # Training dataset
            vdl = args[i*3+1] # Validation dataset
            batch_t = args[i*3 + 2]  # Batch time corresponding to the dataset
            print(len(args), len(args)//3)

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)

                    decay_velo = torch.mean(d[:,0,:,1],dim=0).to(device)
                    pred_decay_velo = func.get_decay(batch_x0)

                    dv_loss = loss_f(pred_decay_velo, decay_velo)
                    dv_loss.backward(retain_graph=True)


                else:
                    print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d, dim=0).to(device)

                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)
                tr_loss = loss_f(_pred_x, _batch_x)
                tr_loss.backward()

                step_loss.append(tr_loss.item())

            # Validation loop
            if iter % 2 == 0:
                for vd in vdl:
                    if decay:
                        batch_x0 = torch.mean(vd[:,0,:,0],dim=0).to(device)
                        _batch_x = torch.mean(vd[:,:,:,0],dim=0).to(device)
                    else:
                        batch_x0 = torch.mean(vd[:,0,:], dim=0).to(device)
                        _batch_x = torch.mean(vd, dim=0).to(device)
                        
                    _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)
                    vd_loss.append(loss_f(_pred_x, _batch_x).item())
                    r2_e_list[i].append(calc_r2(_batch_x, _pred_x).item())
                    print(f'R2 for dataset {i+1}: ', r2_e_list[i][-1])

        optimizer.step()

        if func.use_prior:
            prune.remove(func.encoder[1], 'weight') #assumes dropout
            prune.custom_from_mask(func.encoder[1], name='weight', mask= prior != 0)

        scheduler.step(np.mean(step_loss))
        losses.append(np.mean(step_loss))

        if np.isnan(losses[-1]):
            print('nan encountered, stopping at: ', iter)
            break

        for i in range(0, len(args)//3):
            _shuffle_time_data(args[i*3])    # Shuffle training data
            _shuffle_time_data(args[i*3+1])  # Shuffle validation data

        if iter % 20 == 0:
            r2_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in r2_e_list]
            r2_e.append(np.mean(r2_e_m))

            if (iter > 100) and (r2_e[-4] > r2_e[-1]):
                print('early stopping at epoch: ', iter)
                break

    return losses, vd_loss, r2_e, r2_e_list, iter
