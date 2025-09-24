"""
Base Neural ODE model for gene regulatory network modeling.

This module contains the core ODEFunc class and its variants for different training scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import prune


class ODEFunc(nn.Module):
    """
    Base Neural ODE function for modeling gene regulatory networks.
    
    This is the core model used for pretraining without prior knowledge.
    
    Args:
        device: PyTorch device for computation
        ndim: Number of genes/dimensions
        explicit_time: Whether to use explicit time input (default: False)
        neurons: Number of neurons in hidden layers (default: 158)
        use_prior: Whether to use prior knowledge (default: False)
        prior_: Prior knowledge matrix (default: None)
        scaler: Data scaler (default: None)
        dropout: Dropout rate (default: None)
        calc_erv_: Whether to calculate ERV (default: False)
    """
    
    def __init__(self, device, ndim, explicit_time=False, neurons=158, 
                 use_prior=False, prior_=None, scaler=None, dropout=None, calc_erv_=False):
        super().__init__()
        
        self.ndim = ndim
        self.neurons = neurons
        self._model_device = device
        self.scale = scaler
        self.dropout_ = dropout
        self.use_prior = use_prior
        self.prior_ = prior_
        self.calc_erv_ = calc_erv_
        
        # Lambda network for decay modeling
        self.lambda_ = nn.Sequential()
        self.lambda_.add_module('linear_in', nn.Linear(ndim, neurons, bias=False))
        self.lambda_.add_module('activation_0', nn.Softplus())
        self.lambda_.add_module('linear_out', nn.Linear(neurons, ndim, bias=False))
        self.lambda_.add_module('activation_1', nn.Softplus())
        
        # Encoder network for regulatory interactions
        self.encoder = nn.Sequential()
        self.encoder.add_module('gene_dropout', nn.Dropout(p=dropout))
        self.encoder.add_module('linear_in', nn.Linear(ndim, neurons, bias=False))
        self.encoder.add_module('tf_dropout', nn.Dropout(p=0.1))
        self.encoder.add_module('activation_0', nn.GELU())
        self.encoder.add_module('meta_0', nn.Linear(neurons, neurons, bias=False))
        self.encoder.add_module('activation_2', nn.GELU())
        self.encoder.add_module('meta_1', nn.Linear(neurons, neurons, bias=False))
        self.encoder.add_module('activation_3', nn.GELU())
        self.encoder.add_module('linear_out', nn.Linear(neurons, ndim, bias=False))
        self.encoder.add_module('activation_1', nn.ReLU())
        
        if calc_erv_:
            self._drop_tf = None  # Will be set externally
            self._mask = self._create_mask()
        
        self.lambda_.to(device)
        self.encoder.to(device)
        print(f"ODEFunc initialized on device: {device}")
    
    def forward(self, t, y):
        """
        Forward pass of the neural ODE.
        
        Args:
            t: Time points
            y: Gene expression values
            
        Returns:
            dxdt: Time derivatives of gene expression
        """
        lambda_ = self.lambda_(y)
        decay_mod = torch.mul(-lambda_, y)
        
        if self.calc_erv_:
            # Apply mask to zero out the relevant neurons for ERV calculation
            mask_diag = torch.diag(self._mask)
            a = self.encoder.linear_in(y)  
            tfa = self.encoder.activation_0(a)
            tfa = tfa @ mask_diag
            b = self.encoder.meta_0(tfa)
            c = self.encoder.activation_2(b)
            d = self.encoder.meta_1(c)
            e = self.encoder.activation_3(d)
            f = self.encoder.linear_out(e)
            alpha = self.encoder.activation_1(f)
        else:
            alpha = self.encoder(y)
        
        dxdt = decay_mod + alpha
        return dxdt
    
    def get_decay(self, y):
        """Get decay component of the model."""
        lambda_ = self.lambda_(y)
        decay_mod = torch.mul(lambda_, y) * -1
        return decay_mod
    
    def get_decay_rate(self, y):
        """Get decay rates (lambda values)."""
        return self.lambda_(y)
    
    def get_biophys(self, t, y_, c_scale=None, v_scale=None):
        """
        Get biophysical components for analysis.
        
        Returns:
            dxdt: Time derivatives
            alpha: Regulatory interactions
            decay_mod: Decay component
            lambda_: Decay rates
        """
        y = y_
        lambda_ = self.lambda_(y)
        alpha = self.encoder(y).detach().numpy()
        decay_mod = torch.mul(lambda_, y).detach().numpy() * -1
        dxdt = decay_mod + alpha
        return dxdt, alpha, decay_mod, lambda_
    
    def _create_mask(self):
        """Create a mask tensor based on indices in _drop_tf for ERV calculation."""
        mask = torch.ones(self.neurons, dtype=torch.float32).to(self._model_device)
        if hasattr(self, '_drop_tf') and self._drop_tf is not None:
            mask[self._drop_tf] = 0
        return mask


class SoftPriorODEFunc(ODEFunc):
    """
    Neural ODE with soft prior knowledge integration.
    
    This model extends the base ODEFunc to incorporate prior knowledge
    about gene regulatory networks through regularization.
    
    Args:
        device: PyTorch device for computation
        ndim: Number of genes/dimensions
        explicit_time: Whether to use explicit time input (default: False)
        neurons: Number of neurons in hidden layers (default: 158)
        use_prior: Whether to use prior knowledge (default: False)
        prior_: Prior knowledge matrix (default: None)
        scaler: Data scaler (default: None)
        dropout: Dropout rate (default: None)
        prior_weight: Weight for prior regularization (default: 0.1)
    """
    
    def __init__(self, device, ndim, explicit_time=False, neurons=158, 
                 use_prior=False, prior_=None, scaler=None, dropout=None, prior_weight=0.1):
        super().__init__(device, ndim, explicit_time, neurons, use_prior, prior_, scaler, dropout)
        
        self.prior_weight = prior_weight
        self.scaled_prior = None
        
        # Initialize weights based on prior if available
        if use_prior and prior_ is not None:
            self.prior_.to(device)
            self._initialize_weights_from_prior()
    
    def _initialize_weights_from_prior(self):
        """Initialize encoder weights based on prior knowledge."""
        if self.prior_ is not None:
            input_layer = self.encoder[1]  # linear_in layer
            with torch.no_grad():
                # Initialize weights to match prior where available
                input_layer.weight.data = self.prior_.clone()
    
    def get_prior_loss(self):
        """
        Calculate regularization loss based on deviation from prior.
        
        Uses L2 regularization for weights where prior is zero and
        deviation penalty for weights where prior is non-zero.
        
        Returns:
            prior_loss: Regularization loss term
        """
        if not self.use_prior or self.prior_ is None:
            return 0.0
        
        input_layer = self.encoder[1]  # linear_in layer
        prior = self.prior_.to(input_layer.weight.device)
        
        # Create masks for zero and non-zero values in prior
        nonzero_mask = (prior != 0)
        zero_mask = (prior == 0)
        
        # Calculate deviation from prior where prior exists (non-zero values)
        nonzero_loss = torch.sum((input_layer.weight[nonzero_mask] - prior[nonzero_mask]) ** 2)
        
        # Apply L2 regularization where prior is zero
        zero_loss = torch.sum(input_layer.weight[zero_mask] ** 2)
        
        # Weight the two loss components
        nonzero_weight = self.prior_weight
        zero_weight = self.prior_weight * 0.5  # Tune this ratio as needed
        
        total_loss = nonzero_weight * nonzero_loss + zero_weight * zero_loss
        
        return total_loss


class BlockODEFunc(nn.Module):
    """
    Block-structured Neural ODE for validation against vanilla neural ODE.
    
    This model uses a different architecture for comparison purposes.
    
    Args:
        device: PyTorch device for computation
        ndim: Number of genes/dimensions
        explicit_time: Whether to use explicit time input (default: False)
        neurons: Number of neurons in hidden layers (default: 158)
        use_prior: Whether to use prior knowledge (default: False)
        prior_: Prior knowledge matrix (default: None)
        scaler: Data scaler (default: None)
        dropout: Dropout rate (default: None)
        calc_erv_: Whether to calculate ERV (default: False)
    """
    
    def __init__(self, device, ndim, explicit_time=False, neurons=158, 
                 use_prior=False, prior_=None, scaler=None, dropout=None, calc_erv_=False):
        super().__init__()
        
        self.ndim = ndim
        self.neurons = neurons
        self._model_device = device
        self.scale = scaler
        self.dropout_ = dropout
        self.use_prior = use_prior
        self.prior_ = prior_
        self.calc_erv_ = calc_erv_
        
        # Block-structured encoder network
        self.encoder = nn.Sequential()
        self.encoder.add_module('gene_dropout', nn.Dropout(p=dropout))
        self.encoder.add_module('linear_in', nn.Linear(ndim, neurons, bias=False))
        self.encoder.add_module('tf_dropout', nn.Dropout(p=0.1))
        self.encoder.add_module('activation_0', nn.GELU())
        self.encoder.add_module('meta_0', nn.Linear(neurons, neurons, bias=False))
        self.encoder.add_module('activation_2', nn.GELU())
        self.encoder.add_module('meta_1', nn.Linear(neurons, neurons, bias=False))
        self.encoder.add_module('activation_3', nn.GELU())
        self.encoder.add_module('meta_2', nn.Linear(neurons, neurons, bias=False))
        self.encoder.add_module('activation_4', nn.Softplus())
        self.encoder.add_module('linear_out', nn.Linear(neurons, ndim, bias=False))
        self.encoder.add_module('activation_1', nn.ReLU())
        
        if calc_erv_:
            self._drop_tf = None  # Will be set externally
            self._mask = self._create_mask()
        
        self.encoder.to(device)
        print(f"BlockODEFunc initialized on device: {device}")
    
    def forward(self, t, y):
        """Forward pass of the block neural ODE."""
        alpha = self.encoder(y)
        return alpha
    
    def get_biophys(self, t, y_, c_scale=None, v_scale=None):
        """Get biophysical components for analysis."""
        y = y_
        alpha = self.encoder(y).detach().numpy()
        return alpha, alpha, None, None
    
    def _create_mask(self):
        """Create a mask tensor for ERV calculation."""
        mask = torch.ones(self.neurons, dtype=torch.float32).to(self._model_device)
        if hasattr(self, '_drop_tf') and self._drop_tf is not None:
            mask[self._drop_tf] = 0
        return mask