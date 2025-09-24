"""
Dataset classes for neural ODE training.
"""

import torch
import numpy as np
import warnings
from scipy.sparse import isspmatrix


class TimeDataset(torch.utils.data.Dataset):
    """
    Dataset for time-series gene expression data.
    
    Handles time-stratified sampling and sequence length management
    for neural ODE training.
    """
    
    def __init__(self, data, time_vector, sequence_length=None, 
                 time_min=None, time_max=None, time_step=None):
        """
        Initialize TimeDataset.
        
        Args:
            data: Gene expression data
            time_vector: Time points corresponding to data
            sequence_length: Length of sequences to sample
            time_min: Minimum time for stratification
            time_max: Maximum time for stratification
            time_step: Time step for stratification
        """
        self.data = data
        self.time_vector = time_vector
        self._base_time_vector = time_vector.copy()
        
        # Time stratification properties
        self.time_min = time_min
        self.time_max = time_max
        self.time_step = time_step
        self.shuffle_time_limits = None
        
        # Sequence length properties
        self._sequence_length = sequence_length
        self._sequence_length_options = None
        
        # Sampling properties
        self.n = 0
        self.n_steps = None
        self.with_replacement = True
        self.rng = None
        
        # Stratification indices
        self.strat_idxes = None
        self.shuffle_idxes = None
        
        # Initialize stratification if time parameters provided
        if all(x is not None for x in [time_min, time_max, time_step]):
            self._generate_stratified_indices()
    
    @property
    def sequence_length(self):
        """Get current sequence length."""
        return self._sequence_length
    
    @sequence_length.setter
    def sequence_length(self, x):
        """Set sequence length."""
        if x is None:
            self._sequence_length = None
        elif isinstance(x, (tuple, list, np.ndarray)):
            self._sequence_length_options = x
            self.shuffle_sequence_length()
        else:
            self._sequence_length = int(x)
    
    def _generate_stratified_indices(self):
        """Generate stratified indices for time-based sampling."""
        if self.time_min is None or self.time_max is None or self.time_step is None:
            return
        
        # Create time bins
        time_bins = np.arange(self.time_min, self.time_max + self.time_step, self.time_step)
        
        # Assign each time point to a bin
        bin_assignments = np.digitize(self.time_vector, time_bins) - 1
        bin_assignments = np.clip(bin_assignments, 0, len(time_bins) - 2)
        
        # Create stratified indices
        self.strat_idxes = []
        for bin_idx in range(len(time_bins) - 1):
            bin_indices = np.where(bin_assignments == bin_idx)[0]
            if len(bin_indices) > 0:
                self.strat_idxes.append(bin_indices)
    
    def shuffle(self):
        """Shuffle the dataset."""
        if self.strat_idxes is not None:
            # Shuffle within each time stratum
            self.shuffle_idxes = []
            for stratum in self.strat_idxes:
                shuffled_stratum = stratum.copy()
                np.random.shuffle(shuffled_stratum)
                self.shuffle_idxes.append(shuffled_stratum)
        else:
            # Simple shuffle
            indices = np.arange(len(self.data))
            np.random.shuffle(indices)
            self.shuffle_idxes = [indices]
    
    def shuffle_sequence_length(self):
        """Randomly select a sequence length from options."""
        if self._sequence_length_options is not None:
            self._sequence_length = np.random.choice(self._sequence_length_options)
    
    def shuffle_time_vector(self):
        """Shuffle time vector within limits."""
        if self.shuffle_time_limits is not None:
            min_time, max_time = self.shuffle_time_limits
            shuffled_times = np.random.uniform(min_time, max_time, len(self.time_vector))
            self.time_vector = np.sort(shuffled_times)
    
    def __getitem__(self, idx):
        """Get item from dataset."""
        if self.shuffle_idxes is not None:
            # Use stratified sampling
            stratum_idx = idx % len(self.shuffle_idxes)
            item_idx = self.shuffle_idxes[stratum_idx][idx // len(self.shuffle_idxes)]
        else:
            item_idx = idx
        
        # Get data and time
        data_item = self.data[item_idx]
        time_item = self.time_vector[item_idx]
        
        # Convert to tensor if needed
        if not isinstance(data_item, torch.Tensor):
            data_item = torch.tensor(data_item, dtype=torch.float32)
        if not isinstance(time_item, torch.Tensor):
            time_item = torch.tensor(time_item, dtype=torch.float32)
        
        return data_item, time_item
    
    def __len__(self):
        """Get dataset length."""
        if self.shuffle_idxes is not None:
            return sum(len(stratum) for stratum in self.shuffle_idxes)
        return len(self.data)