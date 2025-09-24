"""
Data loader utilities for neural ODE models.
"""

import torch
from torch.utils.data import DataLoader
from .datasets import TimeDataset


def create_dataloaders(train_data, train_time, test_data, test_time, 
                      time_step=1, sequence_length=30, time_min=0, time_max=60, 
                      batch_size=20):
    """
    Create training and validation data loaders.
    
    Args:
        train_data: Training data
        train_time: Training time points
        test_data: Test data
        test_time: Test time points
        time_step: Time step for sampling
        sequence_length: Length of sequences
        time_min: Minimum time for dataset
        time_max: Maximum time for dataset
        batch_size: Batch size for data loaders
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    print(f'Creating dataloaders: time_step={time_step}, sequence_length={sequence_length}, '
          f'time_min={time_min}, time_max={time_max}')
    
    # Create training dataset
    train_dataset = TimeDataset(
        train_data,
        train_time,
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
        sequence_length=sequence_length
    )
    print(f'Training dataset size: {train_dataset.n}')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True
    )
    
    # Create validation dataset
    val_dataset = TimeDataset(
        test_data,
        test_time,
        time_min=time_min,
        time_max=time_max,
        time_step=time_step,
        sequence_length=sequence_length
    )
    print(f'Validation dataset size: {val_dataset.n}')
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False
    )
    
    return train_loader, val_loader


def generate_batch_times(time_min, time_max, sequence_length, scaling_max=None):
    """
    Generate batch time vectors.
    
    Args:
        time_min: Minimum time
        time_max: Maximum time
        sequence_length: Length of time sequence
        scaling_max: Maximum value for scaling (optional)
        
    Returns:
        torch.Tensor: Batch time vector
    """
    scaling_factor = scaling_max if scaling_max is not None else time_max
    batch_t = torch.linspace(time_min, time_max, sequence_length) / scaling_factor
    return batch_t


def setup_multiple_dataloaders(train_data, train_time, test_data, test_time,
                              time_step=1, sequence_length=30, n_dataloaders=2,
                              time_min=0, time_max=60, use_offset=True, 
                              scaling_max=None, batch_size=20):
    """
    Setup multiple data loaders for different time segments.
    
    Args:
        train_data: Training data
        train_time: Training time points
        test_data: Test data
        test_time: Test time points
        time_step: Time step for sampling
        sequence_length: Length of sequences
        n_dataloaders: Number of data loaders to create
        time_min: Minimum time
        time_max: Maximum time
        use_offset: Whether to use time offset
        scaling_max: Maximum value for scaling
        batch_size: Batch size for data loaders
        
    Returns:
        tuple: (train_loaders, val_loaders, batch_times)
    """
    train_loaders = []
    val_loaders = []
    batch_times = []
    
    total_time = time_max - time_min
    
    if use_offset:
        offset = time_min * -1
        print(f"Using offset: {offset}")
    else:
        offset = 0
    
    for i in range(n_dataloaders):
        if i == 0:
            # First segment covers full time range
            segment_time_min = time_min + offset
            segment_time_max = time_max + offset
            print(f"Segment {i}: time_max = {segment_time_max}")
        else:
            # Subsequent segments are sequential
            segment_time_min = time_min + offset + i * sequence_length
            segment_time_max = segment_time_min + sequence_length
        
        # Ensure we don't exceed total time
        segment_time_max = min(segment_time_max, total_time)
        
        # Create data loaders for this segment
        train_loader, val_loader = create_dataloaders(
            train_data, train_time, test_data, test_time,
            time_step=time_step, sequence_length=sequence_length,
            time_min=segment_time_min, time_max=segment_time_max,
            batch_size=batch_size
        )
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        
        print(f"Segment {i}: time_min={segment_time_min}, time_max={segment_time_max}, "
              f"sequence_length={sequence_length}, time_max={time_max}")
        
        # Generate batch times for this segment
        batch_t = generate_batch_times(
            segment_time_min, segment_time_max, sequence_length,
            scaling_max=scaling_max
        )
        batch_times.append(batch_t)
    
    return train_loaders, val_loaders, batch_times


def shuffle_time_data(data_loader):
    """
    Shuffle time data in the data loader.
    
    Args:
        data_loader: DataLoader to shuffle
    """
    try:
        data_loader.dataset.shuffle()
    except AttributeError:
        pass

def select_sparse_random_subset(batch_t, interval_minutes=20):
    """
    Select a sparse random subset of time points with jitter.
    
    Args:
        batch_t: Time vector tensor
        interval_minutes: Interval between time points in minutes
        
    Returns:
        torch.Tensor: Sparse subset of time points
    """
    max_time = len(batch_t) - 1
    base_times = torch.arange(0, max_time + interval_minutes, interval_minutes, dtype=torch.float32)
    
    jitter_range = interval_minutes / 4
    target_times = base_times.clone()
    
    if len(target_times) > 1:
        jitter = (torch.rand(len(target_times) - 1) - 0.5) * 2 * jitter_range
        target_times[1:] += jitter
        target_times = torch.clamp(target_times, 0, max_time)
    
    indices = []
    for target_time in target_times:
        closest_idx = torch.argmin(torch.abs(batch_t - target_time / max_time))
        indices.append(closest_idx.item())
    
    # Remove duplicates and sort
    indices = sorted(list(set(indices)))
    sparse_subset = batch_t[torch.tensor(indices)]
    
    return sparse_subset


def select_random_subset(batch_t, ivp_offset, subset_size):
    """
    Select a random subset of values from a 1-D tensor and return them in sorted order.
    
    Args:
        batch_t: 1-D tensor with values (e.g., time vector)
        ivp_offset: Offset for initial value problem
        subset_size: Number of elements to select from the tensor
        
    Returns:
        torch.Tensor: Sorted subset of selected values
        
    Raises:
        ValueError: If subset_size is larger than available elements
    """
    if subset_size > len(batch_t):
        raise ValueError("Subset size must be less than or equal to the length of batch_t.")
    
    # Randomly select indices without replacement
    indices = torch.randperm(len(batch_t) - ivp_offset)[:subset_size] + ivp_offset
    
    # Select the subset and sort it
    selected_subset = batch_t[indices]
    sorted_subset = torch.sort(selected_subset).values
    
    return sorted_subset
