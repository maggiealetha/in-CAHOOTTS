"""
Data preprocessing utilities for neural ODE models.
"""

import numpy as np
import scanpy as sc
from sklearn.preprocessing import RobustScaler
from _trunc_robust_scaler import TruncRobustScaler


def preprocess_data_general(data_obj, time_axis='rapa', post_process=False):
    """
    Preprocess data for general neural ODE training.
    
    Args:
        data_obj: AnnData object containing gene expression data
        time_axis: Time axis to use ('rapa' or 'cc')
        post_process: Whether to return scaling information
        
    Returns:
        tuple: (processed_data, time_vector) or (processed_data, time_vector, scaler)
    """
    counts_scaling = TruncRobustScaler(with_centering=False)
    
    # Extract time vector
    if time_axis == 'rapa':
        time_vector = data_obj.obs['program_rapa_time'].values
    else:
        time_vector = data_obj.obs['program_cc_time'].values
    
    # Preprocess data
    data_obj.X = data_obj.X.astype(np.float32)
    sc.pp.normalize_per_cell(data_obj, min_counts=0)
    
    # Scale data
    data = counts_scaling.fit_transform(data_obj.X.A)
    
    if post_process:
        return data, time_vector, counts_scaling
    else:
        return data, time_vector


def preprocess_data_biophysical(data_obj, post_process=False):
    """
    Preprocess data for biophysical neural ODE training with decay modeling.
    
    Args:
        data_obj: AnnData object containing gene expression data
        post_process: Whether to return scaling information
        
    Returns:
        tuple: (processed_data, time_vector) or (processed_data, time_vector, scaler)
    """
    time_vector = data_obj.obs['program_rapa_time'].values
    
    # Process count data
    count_scaling = TruncRobustScaler(with_centering=False)
    count_data = _get_data_from_ad(data_obj, 'X')
    count_data = count_scaling.fit_transform(count_data)
    
    try:
        count_data = count_data.toarray()
    except AttributeError:
        pass
    
    data_ = [count_data]
    
    # Process velocity data
    decay_velocity_layers = ('decay_constants', 'denoised')
    velocity_data = _get_data_from_ad(
        data_obj,
        decay_velocity_layers,
        np.multiply,
        densify=True
    )
    velocity_data *= -1
    
    data_.append(velocity_data * (1 / count_scaling.scale_))
    data_ = np.stack(data_, axis=-1)
    
    if post_process:
        return data_, time_vector, count_scaling.scale_
    else:
        return data_, time_vector


def _get_data_from_ad(adata, layer, func=None, densify=False):
    """
    Extract data from AnnData object.
    
    Args:
        adata: AnnData object
        layer: Layer name or tuple of layer names
        func: Function to apply to multiple layers
        densify: Whether to densify sparse matrices
        
    Returns:
        Extracted data
    """
    if isinstance(layer, str):
        data = adata.layers[layer] if layer in adata.layers else adata.X
    else:
        # Multiple layers
        data = [adata.layers[l] if l in adata.layers else adata.X for l in layer]
        if func is not None:
            data = func(*data)
        else:
            data = np.stack(data, axis=-1)
    
    if densify and hasattr(data, 'toarray'):
        data = data.toarray()
    
    return data


def split_data(data, time_vector, seed=257, split=0.2, decay=False):
    """
    Split data into train and test sets.
    
    Args:
        data: Gene expression data
        time_vector: Time points
        seed: Random seed
        split: Test set fraction
        decay: Whether data includes decay information
        
    Returns:
        tuple: (train_data, train_time, test_data, test_time)
    """
    np.random.seed(seed)
    
    if decay:
        # For decay data, split along first axis
        n_samples = data.shape[0]
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * (1 - split))
        
        train_data = data[indices[:split_idx]]
        test_data = data[indices[split_idx:]]
        train_time = time_vector[indices[:split_idx]]
        test_time = time_vector[indices[split_idx:]]
    else:
        # For general data, use sklearn split
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_time, test_time = train_test_split(
            data, time_vector, test_size=split, random_state=seed
        )
    
    return train_data, train_time, test_data, test_time