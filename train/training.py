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

def _training_annealing_weights(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1, rate_ = 3):
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
        #print('sending prior to device')
        prior = func.prior_.to(device)
        #print("prior device: ", prior.device)
        #print("linear in device: ", func.encoder[1].weight.device)
        
    for iter in tqdm(range(niters + 1)):
        optimizer.zero_grad()
        step_loss = []
        step_loss_d = []
        _tr_loss = 0

        # Gradually apply the mask constraints
        if func.use_prior:
            func.mask_input_weights_with_annealing(
                mask=prior,
                module=func.encoder[1],  # Assuming this is your target layer
                epoch=iter,
                max_epochs=niters,
                annealing_schedule='exponential',  # or 'linear'
                rate = rate_
            )

            #print("mask applied in training loop")


        for i in range(len(dls)):
            dl = dls[i]    # Training dataset
            batch_t = batch_ts[i]  # Batch time corresponding to the dataset

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)


                else:
                    #print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d, dim=0).to(device)
                
                #print("data sending into model")
                
                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)
                
                #print("pre loss function")
                
                tr_loss = loss_f(_pred_x, _batch_x)

                #print("pre weights backward pass")
                                
                tr_loss.backward()

                step_loss.append(tr_loss.item())

                if decay: # to avoid training model in parallel pass predicted expression into the model
                    
                    decay_velo = torch.mean(d[:,:,:,1],dim=0).to(device)
                    _pred_x_detached = _pred_x.detach()
                    pred_decay_velo = func.get_decay(_pred_x_detached).to(device) 

                    dv_loss = decay_weight*loss_f(pred_decay_velo, decay_velo)
                    dv_loss.backward()
                    step_loss_d.append(dv_loss.item())
            
            _shuffle_time_data(dls[i])    # Shuffle training data


        optimizer.step()

        # if func.use_prior:
        #     #print('in pruning function')
        #     prune.remove(func.encoder[1], 'weight') #assumes dropout
        #     prune.custom_from_mask(func.encoder[1], name='weight', mask= prior != 0)

        scheduler.step(np.mean(step_loss))
        losses.append(np.mean(step_loss))
        losses_d.append(np.mean(step_loss_d))

        if np.isnan(losses[-1]):
            print('nan encountered, stopping at: ', iter)
            break

        for i in range(len(dls)):
            _shuffle_time_data(dls[i])    # Shuffle training data

        if iter % 20 == 0:
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
    
                _shuffle_time_data(vdl)  # Shuffle validation data
            
            for r2_e_dataset in r2_e_list:
                print('r2_e_dataset', len(r2_e_dataset), 'ndls', len(dls))
            r2_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in r2_e_list]
            r2_e.append(np.mean(r2_e_m))

            vd_e_m = [np.mean(vd_e_dataset) for vd_e_dataset in vd_loss]
            vd_loss_e.append(np.mean(vd_e_m))


            #if (iter > 100) and check_plateau(r2_e):
            #    print('early stopping at epoch: ', iter)
            #    break

    return losses, vd_loss_e, r2_e, iter #losses_d,

def _training_subset_time(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1):
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
            batch_t = torch.cat(((batch_t_[:10]),(select_random_subset(batch_t_,10, 30))))
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

    return losses, losses_d, vd_loss_e, r2_e, iter

def _training_combined_loss(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1):
    loss_f = torch.nn.MSELoss()
    losses = [] 
    vd_loss_e = []
    r2_e = []

    losses_d = []
    vd_loss_d_e = []
    dv_loss = None
    
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
            batch_t = batch_ts[i]  # Batch time corresponding to the dataset

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)


                else:
                    #print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d, dim=0).to(device)

                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)
                tr_loss = loss_f(_pred_x, _batch_x)
                                

                step_loss.append(tr_loss.item())

                if decay: # to avoid training model in parallel pass predicted expression into the model
                    
                    decay_velo = torch.mean(d[:,:,:,1],dim=0).to(device)
                    _pred_x_detached = _pred_x.detach()
                    pred_decay_velo = func.get_decay(_pred_x_detached).to(device) 

                    dv_loss = decay_weight*loss_f(pred_decay_velo, decay_velo)
                    step_loss_d.append(dv_loss.item())

        total_loss = tr_loss + dv_loss
        total_loss.backward()

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

            # if (iter > 200) and (r2_e[-4] > r2_e[-1]):
            #     print('early stopping at epoch: ', iter)
            #     break

    return losses, losses_d, vd_loss_e, r2_e, iter
    
def _training_(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1):
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

        for sh_d in range(0,2):#shuffle data twice per epoch to increase data exposure

            for i in range(len(dls)):
                dl = dls[i]    # Training dataset
                batch_t = batch_ts[i]  # Batch time corresponding to the dataset
    
                # Training loop for dataset
                for d in dl:
    
                    if decay:
                        batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                        _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)
    
    
                    else:
                        #print('standard data setup')
                        batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                        _batch_x = torch.mean(d, dim=0).to(device)
    
                    _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)
                    tr_loss = loss_f(_pred_x, _batch_x)
                                    
                    tr_loss.backward()
    
                    step_loss.append(tr_loss.item())
    
                    if decay: # to avoid training model in parallel pass predicted expression into the model
                        
                        decay_velo = torch.mean(d[:,:,:,1],dim=0).to(device)
                        _pred_x_detached = _pred_x.detach()
                        pred_decay_velo = func.get_decay(_pred_x_detached).to(device) 
    
                        dv_loss = decay_weight*loss_f(pred_decay_velo, decay_velo)
                        dv_loss.backward()
                        step_loss_d.append(dv_loss.item())
                
                _shuffle_time_data(dls[i])    # Shuffle training data


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

        if iter % 20 == 0:
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
    
                _shuffle_time_data(vdl)  # Shuffle validation data
            
            for r2_e_dataset in r2_e_list:
                print('r2_e_dataset', len(r2_e_dataset), 'ndls', len(dls))
            r2_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in r2_e_list]
            r2_e.append(np.mean(r2_e_m))

            vd_e_m = [np.mean(vd_e_dataset) for vd_e_dataset in vd_loss]
            vd_loss_e.append(np.mean(vd_e_m))


            #if (iter > 100) and check_plateau(r2_e):
            #    print('early stopping at epoch: ', iter)
            #    break

    return losses, vd_loss_e, r2_e, iter #losses_d,


def _training_t_dist(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1, t_dist = 5):
    loss_f = torch.nn.MSELoss()
    losses = [] 
    vd_loss_e = []
    r2_e = []
    mape_e = []

    losses_d = []
    vd_loss_d_e = []
    
    optimizer = opt = optim.Adam(func.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.9, patience=patience_, threshold=1e-09,
                                                     threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-09, verbose=True)
    mape = MeanAbsolutePercentageError().to(device)

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
            batch_t = batch_ts[i]  # Batch time corresponding to the dataset

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)


                else:
                    #print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d, dim=0).to(device)

                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)

                tr_loss = loss_f(_pred_x, _batch_x)
                tr_loss.backward()
                step_loss.append(tr_loss.item())

                if decay: # to avoid training model in parallel pass predicted expression into the model
                    
                    t_dist_sampler = np.linspace(0, d.shape[1]-1, int(d.shape[1]/t_dist), dtype=int)
                    
                    decay_velo = torch.mean(d[:,t_dist_sampler,:,1],dim=0).to(device)
                    _pred_x_detached = _pred_x[t_dist_sampler, :].detach()
                    
                    pred_decay_velo = func.get_decay(_pred_x_detached).to(device) 
                     
                    dv_loss = loss_f(pred_decay_velo, decay_velo)

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
            with torch.no_grad():
                r2_e_list = [[] for _ in range(len(vdls))]  # Separate r2_e for each pair of dataset
                vd_loss = [[] for _ in range(len(vdls))]
                mape_e_list = [[] for _ in range(len(vdls))]
                
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
                        mape_e_list[i].append(mape(_pred_x, _batch_x).item())
                        print(f'R2 for dataset {i+1}: ', r2_e_list[i][-1])
            
            for r2_e_dataset in r2_e_list:
                print('r2_e_dataset', len(r2_e_dataset), 'ndls', len(dls))
            r2_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in r2_e_list]
            r2_e.append(np.mean(r2_e_m))
            
            mape_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in mape_e_list]
            mape_e.append(np.mean(mape_e_m))
            
            vd_e_m = [np.mean(vd_e_dataset) for vd_e_dataset in vd_loss]
            vd_loss_e.append(np.mean(vd_e_m))

            # if (iter > 200) and (r2_e[-4] > r2_e[-1]):
            #     print('early stopping at epoch: ', iter)
            #     break

    return losses, losses_d, vd_loss_e, r2_e, mape_e, iter
    
def _training_no_stack_mape(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1, annealing = False):
    loss_f = torch.nn.MSELoss()
    losses = [] 
    vd_loss_e = []
    r2_e = []
    mape_e = []

    losses_d = []
    vd_loss_d_e = []
    
    optimizer = opt = optim.Adam(func.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.9, patience=patience_, threshold=1e-09,
                                                     threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-09, verbose=True)
    mape = MeanAbsolutePercentageError().to(device)

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
            batch_t = batch_ts[i]  # Batch time corresponding to the dataset

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)


                else:
                    #print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d, dim=0).to(device)

                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)

                if decay: # to avoid training model in parallel pass predicted expression into the model
                    
                    decay_velo = torch.mean(d[:,:,:,1],dim=0).to(device)
                    _pred_x_detached = _pred_x.detach()
                    pred_decay_velo = func.get_decay(_pred_x_detached).to(device)
                     
                    #dv_loss_init = loss_f(pred_decay_velo[0], decay_velo[0])
                    
                    #lower_bound_violations = torch.clamp(torch.abs(decay_velo.min(dim=0).values) - torch.abs(pred_decay_velo), max=0)
                    #lower_bound_loss = torch.mean(lower_bound_violations**2)

                    dv_loss = loss_f(pred_decay_velo, decay_velo)#dv_loss_init + decay_weight*lower_bound_loss

                    decay_variance = torch.var(pred_decay_velo)
                    if decay_variance > 1.0:  # threshold for when to apply constraint
                        variance_penalty = 0.05 * decay_variance  # smaller weight (0.05) for gentler effect
                        dv_loss = dv_loss + variance_penalty
                    
                    dv_loss.backward(retain_graph=True)
                    step_loss_d.append(dv_loss.item())
                
                tr_loss = loss_f(_pred_x, _batch_x)
                tr_loss.backward()

                step_loss.append(tr_loss.item())

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
            with torch.no_grad():
                r2_e_list = [[] for _ in range(len(vdls))]  # Separate r2_e for each pair of dataset
                vd_loss = [[] for _ in range(len(vdls))]
                mape_e_list = [[] for _ in range(len(vdls))]
                
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
                        mape_e_list[i].append(mape(_pred_x, _batch_x).item())
                        print(f'R2 for dataset {i+1}: ', r2_e_list[i][-1])
            
            for r2_e_dataset in r2_e_list:
                print('r2_e_dataset', len(r2_e_dataset), 'ndls', len(dls))
            r2_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in r2_e_list]
            r2_e.append(np.mean(r2_e_m))
            
            mape_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in mape_e_list]
            mape_e.append(np.mean(mape_e_m))
            
            vd_e_m = [np.mean(vd_e_dataset) for vd_e_dataset in vd_loss]
            vd_loss_e.append(np.mean(vd_e_m))

            # if (iter > 200) and (r2_e[-4] > r2_e[-1]):
            #     print('early stopping at epoch: ', iter)
            #     break

    return losses, losses_d, vd_loss_e, r2_e, mape_e, iter
    
def _training_no_stack(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1, annealing = False):
    loss_f = torch.nn.MSELoss()
    losses = [] 
    vd_loss_e = []
    r2_e = []
    mape_e = []

    losses_d = []
    vd_loss_d_e = []
    
    optimizer = opt = optim.Adam(func.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.9, patience=patience_, threshold=1e-09,
                                                     threshold_mode='abs', cooldown=0, min_lr=1e-5, eps=1e-09, verbose=True)
    mape = MeanAbsolutePercentageError().to(device)

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
            batch_t = batch_ts[i]  # Batch time corresponding to the dataset

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)


                else:
                    #print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d, dim=0).to(device)

                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)

                if decay: # to avoid training model in parallel pass predicted expression into the model
                    #print(_pred_x.shape)
                    
                    #_pred_x0 = _pred_x[0,:].to(device) #torch.mean(_pred_x[:,0,:,0],dim=0).to(device)
                    
                    decay_velo = torch.mean(d[:,:,:,1],dim=0).to(device)
                    _pred_x_detached = _pred_x.detach()
                    pred_decay_velo = func.get_decay(_pred_x_detached).to(device) #torch.stack([func.get_decay(_pred_x_detached[_t_,:]) for _t_ in range(len(_pred_x))]).to(device)

                    if annealing:
                        decay_weight_ =  decay_weight*(iter / niters)
                    else:
                        decay_weight_ = decay_weight
                        
                    dv_loss = decay_weight_*loss_f(pred_decay_velo, decay_velo)
                    dv_loss.backward(retain_graph=True)
                    step_loss_d.append(dv_loss.item())
                
                tr_loss = loss_f(_pred_x, _batch_x)
                tr_loss.backward()

                step_loss.append(tr_loss.item())

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
            mape_e_list = [[] for _ in range(len(vdls))]
            
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
                    mape_e_list[i].append(mape(_pred_x, _batch_x).item())
                    print(f'R2 for dataset {i+1}: ', r2_e_list[i][-1])
            
            for r2_e_dataset in r2_e_list:
                print('r2_e_dataset', len(r2_e_dataset), 'ndls', len(dls))
            r2_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in r2_e_list]
            r2_e.append(np.mean(r2_e_m))
            
            mape_e_m = [np.mean(r2_e_dataset) for r2_e_dataset in mape_e_list]
            mape_e.append(np.mean(mape_e_m))
            
            vd_e_m = [np.mean(vd_e_dataset) for vd_e_dataset in vd_loss]
            vd_loss_e.append(np.mean(vd_e_m))

            # if (iter > 200) and (r2_e[-4] > r2_e[-1]):
            #     print('early stopping at epoch: ', iter)
            #     break

    return losses, losses_d, vd_loss_e, r2_e, iter

def _training_old(niters, func, device, dls, vdls, batch_ts,  lr = 1e-3, wd = 1e-5, decay = False, patience_ = 30, decay_weight = 1):
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
            batch_t = batch_ts[i]  # Batch time corresponding to the dataset

            # Training loop for dataset
            for d in dl:

                if decay:
                    batch_x0 = torch.mean(d[:,0,:,0],dim=0).to(device)
                    _batch_x = torch.mean(d[:,:,:,0],dim=0).to(device)


                else:
                    #print('standard data setup')
                    batch_x0 = torch.mean(d[:,0,:], dim=0).to(device)
                    _batch_x = torch.mean(d, dim=0).to(device)

                _pred_x = odeint(func=func, y0=batch_x0, t=batch_t, method='rk4').to(device)

                if decay: # to avoid training model in parallel pass predicted expression into the model
                    #print(_pred_x.shape)
                    
                    #_pred_x0 = _pred_x[0,:].to(device) #torch.mean(_pred_x[:,0,:,0],dim=0).to(device)
                    
                    decay_velo = torch.mean(d[:,:,:,1],dim=0).to(device)
                    _pred_x_detached = _pred_x.detach()
                    pred_decay_velo = torch.stack([func.get_decay(_pred_x_detached[_t_,:]) for _t_ in range(len(_pred_x))]).to(device)

                    dv_loss = decay_weight*loss_f(pred_decay_velo, decay_velo)
                    dv_loss.backward(retain_graph=True)
                    step_loss_d.append(dv_loss.item())
                
                tr_loss = loss_f(_pred_x, _batch_x)
                tr_loss.backward()

                step_loss.append(tr_loss.item())

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

            # if (iter > 200) and (r2_e[-4] > r2_e[-1]):
            #     print('early stopping at epoch: ', iter)
            #     break

    return losses, losses_d, vd_loss_e, r2_e, iter


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
