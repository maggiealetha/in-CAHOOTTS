"""
Unified trainer class for neural ODE models.

This module provides a clean interface for training different variants of neural ODE models
for gene regulatory network modeling.
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint


class Trainer:
    """
    Unified trainer for neural ODE models.
    
    Handles both pretraining (without prior) and fine-tuning (with prior) scenarios.
    """
    
    def __init__(self, device, learning_rate=1e-3, weight_decay=1e-5, patience=30):
        """
        Initialize trainer.
        
        Args:
            device: PyTorch device for computation
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            patience: Patience for learning rate scheduling
        """
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'r2_scores': [],
            'prior_losses': []
        }

    def train_pretraining(self, model, train_loaders, val_loaders, batch_times, 
                         epochs=700, experiment_type="cell_cycle", ivp_subset=10, time_subset=30,
                         sparse_sampling=False, interval_minutes=30, val_frequency=20):
        """
        Train model without prior knowledge (pretraining).
        
        Args:
            model: Neural ODE model to train
            train_loaders: List of training data loaders
            val_loaders: List of validation data loaders
            batch_times: List of time vectors for each loader
            epochs: Number of training epochs
            experiment_type: Type of experiment ("cell_cycle" or "drug_perturbation")
            ivp_subset: Number of initial time points for drug perturbation
            time_subset: Number of random time points for drug perturbation
            sparse_sampling: Whether to use sparse time sampling (overrides experiment_type logic)
            interval_minutes: Time interval in minutes for sparse sampling (default: 30)
            val_frequency: How often to run validation (default: 20 epochs)
            
        Returns:
            dict: Training history
        """
        print("Starting pretraining...")
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=self.patience, 
            threshold=1e-09, threshold_mode='abs', cooldown=0, 
            min_lr=1e-5, eps=1e-09, verbose=True
        )
        
        loss_fn = torch.nn.MSELoss()
        
        # Get training configuration
        config = self._get_training_config(
            experiment_type=experiment_type,
            sparse_sampling=sparse_sampling,
            interval_minutes=interval_minutes,
            ivp_subset=ivp_subset,
            time_subset=time_subset
        )
        
        for epoch in tqdm(range(epochs + 1)):
            optimizer.zero_grad()
            
            train_losses = []
            
            # Training loop using method dispatch
            for _ in range(config['shuffle_count']):
                for i, (train_loader, batch_t_) in enumerate(zip(train_loaders, batch_times)):
                    # Get batch times using configured strategy
                    batch_t, batch_idx = config['batch_time_func'](
                        batch_t_, 
                        interval_minutes=interval_minutes,
                        ivp_subset=ivp_subset,
                        time_subset=time_subset
                    )
                    
                    for batch in train_loader:
                        # Process batch using configured strategy
                        batch_x0, batch_x = config['batch_processing'](batch, batch_t, batch_idx)
                        
                        # Forward pass
                        pred_x = odeint(func=model, y0=batch_x0, t=batch_t, method='rk4').to(self.device)
                        loss = loss_fn(pred_x, batch_x)
                        
                        loss.backward()
                        train_losses.append(loss.item())
                    
                    # Shuffle training data
                    self._shuffle_time_data(train_loader)
            
            optimizer.step()
            scheduler.step(np.mean(train_losses))
            
            # Store training history
            self.training_history['train_losses'].append(np.mean(train_losses))
            
            # Validation (only every val_frequency epochs)
            if epoch % val_frequency == 0 or epoch == epochs:
                val_loss, r2_score = self._validate(model, val_loaders, batch_times)
                self.training_history['val_losses'].append(val_loss)
                self.training_history['r2_scores'].append(r2_score)
            
            # Check for NaN
            if np.isnan(self.training_history['train_losses'][-1]):
                print(f'NaN encountered, stopping at epoch: {epoch}')
                break
        
        return self.training_history
    
    def _get_training_config(self, experiment_type, sparse_sampling, **kwargs):
        """Get training configuration based on experiment type and sparse sampling."""
        if sparse_sampling:
            return {
                'batch_time_func': self._get_sparse_batch_times,
                'shuffle_count': 1,
                'batch_processing': self._process_batch_indexed
            }
        elif experiment_type == "drug_perturbation":
            return {
                'batch_time_func': self._get_ivp_batch_times,
                'shuffle_count': 1,
                'batch_processing': self._process_batch_indexed
            }
        else:  # cell_cycle
            return {
                'batch_time_func': self._get_dense_batch_times,
                'shuffle_count': 2,
                'batch_processing': self._process_batch_dense
            }
    
    def _get_sparse_batch_times(self, batch_t_, interval_minutes=30, **kwargs):
        """Get batch times using sparse sampling strategy."""
        from ..data import select_sparse_random_subset
        batch_t = select_sparse_random_subset(batch_t_, interval_minutes)
        batch_idx = (batch_t * (len(batch_t_) - 1)).int()
        return batch_t, batch_idx
    
    def _get_ivp_batch_times(self, batch_t_, ivp_subset=10, time_subset=30, **kwargs):
        """Get batch times using IVP subset strategy."""
        from ..data import select_random_subset
        batch_t = torch.cat((
            batch_t_[:ivp_subset], 
            select_random_subset(batch_t_, ivp_subset, time_subset)
        ))
        batch_idx = (batch_t * (len(batch_t_) - 1)).int()
        return batch_t, batch_idx
    
    def _get_dense_batch_times(self, batch_t_, **kwargs):
        """Get batch times using dense sampling strategy."""
        return batch_t_, None
    
    def _process_batch_dense(self, batch, batch_t, batch_idx=None):
        """Process batch using dense sampling (no indexing)."""
        # Handle decay data format (4D) vs regular data format (3D)
        if batch.dim() == 4:  # decay data: [batch, time, features, 2]
            batch_x0 = torch.mean(batch[:, 0, :, 0], dim=0).to(self.device)
            batch_x = torch.mean(batch[:, :, :, 0], dim=0).to(self.device)
        else:  # regular data: [batch, time, features]
            batch_x0 = torch.mean(batch[:, 0, :], dim=0).to(self.device)
            batch_x = torch.mean(batch, dim=0).to(self.device)
        return batch_x0, batch_x
    
    def _process_batch_indexed(self, batch, batch_t, batch_idx):
        """Process batch using indexed sampling (for IVP subset and sparse sampling)."""
        # Handle decay data format (4D) vs regular data format (3D)
        if batch.dim() == 4:  # decay data: [batch, time, features, 2]
            batch_x0 = torch.mean(batch[:, 0, :, 0], dim=0).to(self.device)
            batch_x = torch.mean(batch[:, batch_idx, :, 0], dim=0).to(self.device)
        else:  # regular data: [batch, time, features]
            batch_x0 = torch.mean(batch[:, 0, :], dim=0).to(self.device)
            batch_x = torch.mean(batch[:, batch_idx, :], dim=0).to(self.device)
        return batch_x0, batch_x
    
    def train_sparse_sampling(self, model, train_loaders, val_loaders, batch_times,
                             epochs=700, interval_minutes=30, val_frequency=20):
        """
        Train model with sparse time sampling using interval-based selection.
        
        Uses select_sparse_random_subset for interval-based sparse time sampling.
        Single pass through data per epoch (no double shuffling).
        
        Args:
            model: Neural ODE model to train
            train_loaders: List of training data loaders
            val_loaders: List of validation data loaders
            batch_times: List of time vectors for each loader
            epochs: Number of training epochs
            interval_minutes: Time interval in minutes for sparse sampling (default: 30)
            val_frequency: How often to run validation (default: 20 epochs)
            
        Returns:
            dict: Training history
        """
        return self.train_pretraining(
            model=model,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            batch_times=batch_times,
            epochs=epochs,
            sparse_sampling=True,
            interval_minutes=interval_minutes,
            val_frequency=val_frequency
        )
    
    def train_drug_perturbation(self, model, train_loaders, val_loaders, batch_times,
                               epochs=700, ivp_subset=10, time_subset=30, 
                               sparse_sampling=False, interval_minutes=30, val_frequency=20):
        """
        Train model for drug perturbation experiments with IVP subset approach.
        
        Uses sparse time sampling with initial value problem (IVP) subset + random time points.
        Single pass through data per epoch (no double shuffling).
        
        Args:
            model: Neural ODE model to train
            train_loaders: List of training data loaders
            val_loaders: List of validation data loaders
            batch_times: List of time vectors for each loader
            epochs: Number of training epochs
            ivp_subset: Number of initial time points (default: 10)
            time_subset: Number of random time points (default: 30)
            sparse_sampling: Whether to use sparse time sampling instead of IVP approach
            interval_minutes: Time interval in minutes for sparse sampling (default: 30)
            val_frequency: How often to run validation (default: 20 epochs)
            
        Returns:
            dict: Training history
        """
        return self.train_pretraining(
            model=model,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            batch_times=batch_times,
            epochs=epochs,
            experiment_type="drug_perturbation",
            ivp_subset=ivp_subset,
            time_subset=time_subset,
            sparse_sampling=sparse_sampling,
            interval_minutes=interval_minutes,
            val_frequency=val_frequency
        )
    
    def train_cell_cycle(self, model, train_loaders, val_loaders, batch_times,
                        epochs=700, sparse_sampling=False, interval_minutes=30, val_frequency=20):
        """
        Train model for cell cycle experiments with dense time sampling.
        
        Uses dense time sampling (all time points) with double shuffling per epoch
        to maximize data exposure and improve training efficiency.
        
        Args:
            model: Neural ODE model to train
            train_loaders: List of training data loaders
            val_loaders: List of validation data loaders
            batch_times: List of time vectors for each loader
            epochs: Number of training epochs
            sparse_sampling: Whether to use sparse time sampling instead of dense sampling
            interval_minutes: Time interval in minutes for sparse sampling (default: 30)
            val_frequency: How often to run validation (default: 20 epochs)
            
        Returns:
            dict: Training history
        """
        return self.train_pretraining(
            model=model,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            batch_times=batch_times,
            epochs=epochs,
            experiment_type="cell_cycle",
            sparse_sampling=sparse_sampling,
            interval_minutes=interval_minutes,
            val_frequency=val_frequency
        )
    
    def train_with_prior(self, model, train_loaders, val_loaders, batch_times,
                        epochs=500, sparse_sampling=False, interval_minutes=30, val_frequency=20):
        """
        Train model with prior knowledge integration.
        
        Args:
            model: Neural ODE model with prior integration
            train_loaders: List of training data loaders
            val_loaders: List of validation data loaders
            batch_times: List of time vectors for each loader
            epochs: Number of training epochs
            sparse_sampling: Whether to use sparse time sampling
            interval_minutes: Time interval in minutes for sparse sampling (default: 30)
            val_frequency: How often to run validation (default: 20 epochs)
            
        Returns:
            dict: Training history
        """
        print("Starting training with prior knowledge...")
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=self.patience, 
            threshold=1e-09, threshold_mode='abs', cooldown=0, 
            min_lr=1e-5, eps=1e-09, verbose=True
        )
        
        loss_fn = torch.nn.MSELoss()
        
        # Get training configuration for prior training (always dense unless sparse_sampling=True)
        config = self._get_training_config(
            experiment_type="cell_cycle",  # Default to cell cycle for prior training
            sparse_sampling=sparse_sampling,
            interval_minutes=interval_minutes
        )
        
        for epoch in tqdm(range(epochs + 1)):
            optimizer.zero_grad()
            
            train_losses = []
            prior_losses = []
            
            # Training loop using method dispatch
            for _ in range(config['shuffle_count']):
                for i, (train_loader, batch_t_) in enumerate(zip(train_loaders, batch_times)):
                    # Get batch times using configured strategy
                    batch_t, batch_idx = config['batch_time_func'](
                        batch_t_, 
                        interval_minutes=interval_minutes
                    )
                    
                    for batch in train_loader:
                        # Process batch using configured strategy
                        batch_x0, batch_x = config['batch_processing'](batch, batch_t, batch_idx)
                        
                        # Forward pass
                        pred_x = odeint(func=model, y0=batch_x0, t=batch_t, method='rk4').to(self.device)
                        reconstruction_loss = loss_fn(pred_x, batch_x)
                        
                        # Add prior regularization if enabled
                        if model.use_prior:
                            prior_loss = model.get_prior_loss()
                            total_loss = reconstruction_loss + prior_loss
                            prior_losses.append(prior_loss.item())
                        else:
                            total_loss = reconstruction_loss
                        
                        total_loss.backward()
                        train_losses.append(reconstruction_loss.item())
                    
                    # Shuffle training data
                    self._shuffle_time_data(train_loader)
            
            optimizer.step()
            scheduler.step(np.mean(train_losses))
            
            # Store training history
            self.training_history['train_losses'].append(np.mean(train_losses))
            
            # Validation (only every val_frequency epochs)
            if epoch % val_frequency == 0 or epoch == epochs:
                val_loss, r2_score = self._validate(model, val_loaders, batch_times)
                self.training_history['val_losses'].append(val_loss)
                self.training_history['r2_scores'].append(r2_score)

            
            if prior_losses:
                self.training_history['prior_losses'].append(np.mean(prior_losses))
            
            # Check for NaN
            if np.isnan(self.training_history['train_losses'][-1]):
                print(f'NaN encountered, stopping at epoch: {epoch}')
                break
        
        return self.training_history
    
    def _validate(self, model, val_loaders, batch_times):
        """Validate model on validation data."""
        model.eval()
        val_losses = []
        r2_scores = []
        
        with torch.no_grad():
            for val_loader, batch_t in zip(val_loaders, batch_times):
                for batch in val_loader:
                    batch_x0 = torch.mean(batch[:, 0, :], dim=0).to(self.device)
                    batch_x = torch.mean(batch, dim=0).to(self.device)
                    
                    pred_x = odeint(func=model, y0=batch_x0, t=batch_t, method='rk4').to(self.device)
                    loss = torch.nn.MSELoss()(pred_x, batch_x)
                    val_losses.append(loss.item())
                    
                    # Calculate R² score
                    r2 = self._calculate_r2(batch_x, pred_x)
                    r2_scores.append(r2)
        
        model.train()
        return np.mean(val_losses), np.mean(r2_scores)
    
    def _calculate_r2(self, y_true, y_pred):
        """Calculate R² score."""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()
    
    def _shuffle_time_data(self, data_loader):
        """Shuffle time data in the loader."""
        from ..data import shuffle_time_data
        shuffle_time_data(data_loader)