import pandas as pd
import scanpy as sc
import numpy as np
import torch
import os
import yaml

from inferelator.preprocessing import ManagePriors
from _trunc_robust_scaler import TruncRobustScaler
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split


from .time_dataset import TimeDataset
from torch.utils.data import DataLoader

def select_random_subset(batch_t, ivp_offset, subset_size):
    """
    Select a random subset of values from a 1-D tensor and return them in sorted order.

    :param batch_t: 1-D tensor with values (e.g., time vector).
    :param subset_size: Number of elements to select from the tensor.
    :return: A new tensor containing the selected subset of values in sorted order.
    """
    if subset_size > len(batch_t):
        raise ValueError("Subset size must be less than or equal to the length of batch_t.")

    # Randomly select indices without replacement
    indices = torch.randperm(len(batch_t)-ivp_offset)[:subset_size] + ivp_offset
    
    # Select the subset and sort it
    selected_subset = batch_t[indices]
    sorted_subset = torch.sort(selected_subset).values  # Sort the selected subset

    return sorted_subset
    
# Function to load yaml configuration file
def load_config(config_name):
    # folder to load config file
    CONFIG_PATH = "mag_nODE_model/"
    if config_name[0] == 'm':
        _config_name = config_name
        print(_config_name)
    else:
        _config_name = os.path.join(CONFIG_PATH, config_name)
        print(_config_name)
    with open(_config_name) as file: 
        config = yaml.safe_load(file)
        config['use_prior'] = bool(config['use_prior'])
        config['shuffle_prior'] = bool(config['shuffle_prior'])
        config['decay'] = bool(config['decay'])

    return config

def gen_seed():
    return np.random.randint(0,1000)
    
def setup_priors(gs_df, seed, gene_names, shuffled=False, shuffled_seed=None): #gold_standard, gs_seed, var_names
    
    split_axis = 0
    split = 0.2
    
    _prior, _gs = ManagePriors.cross_validate_gold_standard(gs_df,gs_df, split_axis, split,seed)

    if shuffled:
        _prior = ManagePriors.shuffle_priors(_prior, -1, shuffled_seed[-1])
    
    _prior = _prior.reindex(gene_names, axis=0).fillna(0).astype(int)
    
    prior = torch.tensor(_prior.values, dtype=torch.float32).T
    
    return prior
    
def setup_priors_erv_calc(gs_df, seed, gene_names, return_gs = False): #gold_standard, gs_seed, var_names

    split_axis = 0
    split = 0.2

    _prior, _gs = ManagePriors.cross_validate_gold_standard(gs_df,gs_df, split_axis, split,seed)

    _prior = _prior.reindex(gene_names, axis=0).fillna(0).astype(int)

    if return_gs:

            _gs= _gs.reindex(gene_names, axis=0).fillna(0).astype(int)

            return _prior, _gs

    else:

        return _prior

def _get_data_from_ad( 
    adata,
    layers,
    agg_func=np.add,
    densify=False,
    **kwargs
):

    """from Chris Jackson"""

    if isinstance(layers, (tuple, list)):
        _output = _get_data_from_ad(adata, layers[0], densify=densify).copy()
        for layer in layers[1:]:
            agg_func(
                _output,
                _get_data_from_ad(adata, layer, densify=densify),
                out=_output,
                **kwargs
            )

    elif layers == 'X':
        _output = adata.X

    else:
        _output = adata.layers[layers]

    if densify:
        try:
            _output = _output.toarray()
        except AttributeError:
            pass

    return _output
    
def preprocess_data_general(data_obj, time_axis = 'rapa', post_process = False):

    counts_scaling = TruncRobustScaler(with_centering=False)

    time_vector = None
    
    if time_axis == 'rapa':
        time_vector = data_obj.obs['program_rapa_time'].values
    else:
        time_vector = data_obj.obs['program_cc_time'].values

    data_obj.X = data_obj.X.astype(np.float32)
    sc.pp.normalize_per_cell(data_obj, min_counts=0)

    data = counts_scaling.fit_transform(data_obj.X.A)


    if post_process:
        return data, time_vector, counts_scaling.scale_

    else:
        return data, time_vector

def preprocess_data_biophysical(data_obj, post_process = False):

    time_vector = data_obj.obs['program_rapa_time'].values

    counts_layer='X'
    count_scaling = TruncRobustScaler(with_centering=False)

    count_data = _get_data_from_ad(
        data_obj,
        counts_layer
    )

    count_data = count_scaling.fit_transform(count_data)

    try:
        count_data = count_data.toarray()
    except AttributeError:
        pass

    data_ = [count_data]
    decay_velocity_layers=('decay_constants', 'denoised')
    velocity_data = _get_data_from_ad(
                data_obj,
                decay_velocity_layers,
                np.multiply,
                densify=True
            )
    velocity_data *= -1

    #velo_scaling = TruncRobustScaler(with_centering=False)

    # data_.append(
    #             velo_scaling.fit_transform(
    #                 velocity_data
    #             )
    #         )

    data_.append(
            velocity_data*(1/count_scaling.scale_)
            )

    data_ = np.stack(data_, axis=-1)

    if post_process:
        return data_, time_vector, count_scaling.scale_

    else:
        return data_, time_vector#, count_scaling.scale_, velo_scaling.scale_

def split_data(data, time, seed = 257, split = 0.2, decay = False):

    validation_size = split
    random_seed = seed

    train_idx, test_idx = train_test_split(
        np.arange(data.shape[0]),
        test_size=validation_size,
        random_state=random_seed
    )

    _train = data[train_idx, :]
    _train_t = time[train_idx]
    _test = data[test_idx, :]
    _test_t = time[test_idx]

    if decay:
        n_genes = _train.shape[-2]
    else:
        n_genes = _train.shape[-1]

    return _train, _train_t, _test, _test_t, n_genes

def create_dls(_train, _tr_t, _test, _t_t, ts, sl, tmin, tmax, bs = 20):
    print('in c_dls ts: ', ts, 'sl: ', sl, 'tmin: ', tmin, 'tmax: ', tmax)

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
            batch_size=bs,
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
            batch_size=bs,
            drop_last=True,
        )

    return dl, vdl
    
def gen_batch_t(tmin_seg, tmax_seg, sl, tmax, scaling_tmax=None):
    
    scaling_factor = scaling_tmax if scaling_tmax is not None else tmax
    
    batch_t = torch.linspace(tmin_seg, tmax_seg, sl) / scaling_factor

    return batch_t  

def gen_dls(train, train_t, test, test_t, ts = 1, sl = 30, tmin = 0, tmax=60):
    
    dl, vdl = create_dls(train, train_t, test, test_t, ts, sl, tmin, tmin+sl)

    return dl, vdl

def setup_dls(train, train_t, test, test_t, ts = 1, sl = 30, n_dls = 2, tmin = 0, tmax = 60, use_offset = True, bigT_for_scaling = None):

    dls = []
    vdls = [] 
    batch_ts = []
    
    bigT = sl*n_dls
    if use_offset:
        offset = tmin * -1
        print(offset)
    else:
        offset=0

    for i in range(n_dls):
        if i == 0:
            tmin_segment = tmin + offset
            tmax_segment = tmin_segment + sl
        else:
            tmin_segment = tmin + offset + i * sl
            tmax_segment = tmin_segment + sl
            
        # Ensure we do not exceed bigT
        tmax_segment = min(tmax_segment, bigT)

        # Simulate dataloader generation and time batching
        dl, vdl = create_dls(train, train_t, test, test_t, ts, sl, tmin_segment, tmax_segment)
        dls.append(dl)
        vdls.append(vdl)

        print(tmin_segment, tmax_segment, sl, tmax)
        batch_t = gen_batch_t(tmin_segment, tmax_segment, sl, tmax, scaling_tmax = bigT_for_scaling)

        batch_ts.append(batch_t)

    return dls, vdls, batch_ts

def _shuffle_time_data(dl):
    try:
        dl.dataset.shuffle()
    except AttributeError:
        pass

def mkdirs(run_num, use_prior = False, decay = False, shuffled = False):
    prior_bool=""
    output_path=""

    if use_prior:
        prior_bool = 'yeast_with_prior/'

        if shuffled:
            output_path += os.path.join(prior_bool,"shuffled")

    else:
        prior_bool = 'yeast_no_prior/'

    if decay:
        print('pre decay', output_path)
        output_path += os.path.join(prior_bool,"decay")
        print('in decay', output_path)
    
    else:
        output_path += os.path.join(prior_bool)
    
    dir_path = os.path.join(output_path, "run"+str(run_num))
    print(dir_path)
    #print(os.getcwd()) 
    
    os.mkdir(dir_path)
    return(dir_path)

def save_per_cv(cv, final_epoch, func, tr_mse, vd_mse, r2, output_dir, mape = None, shuffled = False, decay = False, rate_  = None, prior_weight = None, prior_loss = None, wd = None):

    if rate_ is not None:
        torch.save(func.state_dict(), os.path.join(output_dir, "model_cv%i_epochs%i_r%.4f.pt" %(cv, final_epoch, rate_)))
        np.savetxt(os.path.join(output_dir, "tr_loss_cv%i_epochs%i_r%.4f.csv" %(cv, final_epoch, rate_)), tr_mse, delimiter=',')
        np.savetxt(os.path.join(output_dir, "vd_loss_cv%i_epochs%i_r%.4f.csv" %(cv, final_epoch, rate_)), vd_mse, delimiter=',')
        np.savetxt(os.path.join(output_dir, "r2_cv%i_epochs%i_r%.4f.csv" %(cv, final_epoch, rate_)), r2, delimiter=',')
    if prior_weight is not None:
        torch.save(func.state_dict(), os.path.join(output_dir, "model_cv%i_epochs%i_pw%.4f_wd%.6f.pt" %(cv, final_epoch, prior_weight, wd)))
        np.savetxt(os.path.join(output_dir, "tr_loss_cv%i_epochs%i_pw%.4f_wd%.6f.csv" %(cv, final_epoch, prior_weight, wd)), tr_mse, delimiter=',')
        np.savetxt(os.path.join(output_dir, "vd_loss_cv%i_epochs%i_pw%.4f_wd%.6f.csv" %(cv, final_epoch, prior_weight, wd)), vd_mse, delimiter=',')
        np.savetxt(os.path.join(output_dir, "r2_cv%i_epochs%i_pw%.4f_wd%.6f.csv" %(cv, final_epoch, prior_weight, wd)), r2, delimiter=',')
        np.savetxt(os.path.join(output_dir, "prior_loss_cv%i_epochs%i_pw%.4f_wd%.6f.csv" %(cv, final_epoch, prior_weight, wd)), prior_loss, delimiter=',')
    else:
        torch.save(func.state_dict(), os.path.join(output_dir, "model_cv%i_epochs%i.pt" %(cv, final_epoch)))
        np.savetxt(os.path.join(output_dir, "tr_loss_cv%i_epochs%i.csv" %(cv, final_epoch)), tr_mse, delimiter=',')
        np.savetxt(os.path.join(output_dir, "vd_loss_cv%i_epochs%i.csv" %(cv, final_epoch)), vd_mse, delimiter=',')
        np.savetxt(os.path.join(output_dir, "r2_cv%i_epochs%i.csv" %(cv, final_epoch)), r2, delimiter=',')
        if mape is not None:
            np.savetxt(os.path.join(output_dir, "mape_cv%i_epochs%i.csv" %(cv, final_epoch)), mape, delimiter=',')

def save_meta_data(output_dir, config, gs_seeds, data_seeds, prior_seeds = None, shuffled_seeds = None, decay = False):
    

    if (len(prior_seeds) > 0) and (len(shuffled_seeds) > 0):
        meta_data = np.vstack((gs_seeds,data_seeds, prior_seeds, shuffled_seeds)).T
        cols = ['gs_seed', 'data_seed', 'prior_seed', 'shuffled_seed']
   
    elif (len(prior_seeds) > 0):
        meta_data = np.vstack((gs_seeds,data_seeds, prior_seeds)).T
        cols = ['gs_seed', 'data_seed', 'prior_seed']
    
    else:
        meta_data = np.vstack((gs_seeds,data_seeds)).T
        cols = ['gs_seed', 'data_seed']
            
    pd.DataFrame(meta_data, columns = cols).to_csv(os.path.join(output_dir, "meta_data.tsv"), sep = '\t')
    pd.DataFrame([config]).to_csv(os.path.join(output_dir, 'config_settings.csv'), index=False)

def load_meta_data(dir, shuffled = False):
    seeds = pd.read_csv(dir+'/meta_data.tsv', header =0, sep = '\t', index_col = 0)
    gs_seeds = seeds['gs_seed'].to_list()
    data_seeds = seeds['data_seed'].to_list()
    
    if shuffled:#len(seeds.columns) > 3:
        shuffled_seeds = seeds['shuffled_seed'].to_list()
        return gs_seeds, data_seeds, shuffled_seeds
    else:
        return gs_seeds, data_seeds