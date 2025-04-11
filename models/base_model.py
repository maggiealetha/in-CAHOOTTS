import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import prune

class ODEFunc(nn.Module):
    '''NODE class implementation '''


    def __init__(self, device, ndim, explicit_time=False, neurons=158, use_prior = False, prior_ = None, scaler=None, dropout=None, calc_erv_ = False):
        ''' Initialize a new ODEFunc '''
        super().__init__()

        self.ndim = ndim
        self.neurons = neurons
        self._model_device = device
        self.scale = scaler
        self.dropout_ = dropout
        self.use_prior = use_prior
        self.prior_ = prior_
        self.calc_erv_ = calc_erv_

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
        self.encoder.add_module('activation_1', nn.ReLU())#inplace=True

        # if use_prior:

        #     self.mask_input_weights(self.prior_, self.encoder[1], 'weight')

        if calc_erv_:
            self._drop_tf = drop_tf_ #tf to drop
            self._mask = self._create_mask()


        self.lambda_.to(device)
        self.encoder.to(device)
        print("device in model init: ", device)

    def forward(self, t, y):

        # if hasattr(self, 'frozen_decay') and hasattr(self, 'frozen_times'):
        #     # Find nearest timepoint for current t
        #     t_idx = torch.argmin(torch.abs(self.frozen_times - t))
        #     decay_mod = torch.mul(-self.frozen_decay[t_idx], y)
        # else:


        lambda_ = self.lambda_(y)
        decay_mod = torch.mul(-lambda_,y)

        if self.calc_erv_:
            # Apply mask to zero out the relevant neurons
            #print("in masking TFA, masking TF: ", self._drop_tf, "tf val in mask pre diag: ", self._mask[self._drop_tf])
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
        return(dxdt)
        
    def get_decay(self, y):
        
        lambda_ = self.lambda_(y)
        decay_mod = torch.mul(lambda_,y)*-1
        
        return decay_mod
        
    def get_decay_rate(self, y):
                
        return self.lambda_(y)

    def get_biophys(self, t, y_, c_scale=None, v_scale=None):

        #scaler = 1/c_scale
        y = y_#*scaler
        lambda_ = self.lambda_(y)
        alpha = self.encoder(y).detach().numpy()#*v_scale
        decay_mod = torch.mul(lambda_,y).detach().numpy()*-1#*v_scale
        dxdt = decay_mod + alpha
        return(dxdt, alpha, decay_mod, lambda_)
    
    def _create_mask(self):
        ''' Create a mask tensor based on indices in _drop_tf '''
        mask = torch.ones(self.neurons, dtype=torch.float32).to(self._model_device)
        mask[self._drop_tf] = 0
        #print("inside create mask, dropping tf: ", self._drop_tf)
        return mask
        
    def mask_input_weights(
        self,
        mask,
        module=None,
        use_mask_weights=False,
        layer_name='weight',
        weight_vstack=None
    ):
        """
        Apply a mask to layer weights
    
        :param mask: Mask tensor. Non-zero values will be retained,
            and zero values will be masked to zero in the layer weights
        :type mask: torch.Tensor
        :param encoder: Module to mask, use self.encoder if this is None,
            defaults to None
        :type encoder: torch.nn.Module, optional
        :param use_mask_weights: Set the weights equal to values in mask,
            defaults to False
        :type use_mask_weights: bool, optional
        :param layer_name: Module weight name,
            defaults to 'weight'
        :type layer_name: str, optional
        :param weight_vstack: Number of times to stack the mask, for cases
            where the layer weights are also stacked, defaults to None
        :type weight_vstack: _type_, optional
        :raises ValueError: Raise error if the mask and module weights are
            different sizes
        """
    
        if module is not None:
            encoder = module
        elif isinstance(self.encoder, torch.nn.Sequential):
            encoder = self.encoder[1]
        else:
            encoder = self.encoder
    
        if weight_vstack is not None and weight_vstack > 1:
            mask = torch.vstack([mask for _ in range(weight_vstack)])
    
        if mask.shape != getattr(encoder, layer_name).shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match weights {layer_name} "
                f"shape {getattr(encoder, layer_name).shape}"
            )
    
        # Replace initialized encoder weights with prior weights
        if use_mask_weights:
            setattr(
                encoder,
                layer_name,
                torch.nn.parameter.Parameter(
                    torch.clone(mask)
                )
            )
    
        # Mask to prior
        prune.custom_from_mask(
            encoder,
            name=layer_name,
            mask=mask != 0
        )

    def mask_input_weights_with_annealing(self, mask, module, epoch, max_epochs, 
                                        annealing_schedule='linear', rate = 2):
        """Apply mask gradually over training epochs"""
        if annealing_schedule == 'linear':
            # Linearly increase mask strength
            alpha = epoch / max_epochs
        elif annealing_schedule == 'exponential':
            # Exponentially increase mask strength
            alpha = 1 - np.exp(-rate * epoch / max_epochs)
        
        # Interpolate between original weights and masked weights
        original_weights = getattr(module, 'weight').clone()
        masked_weights = original_weights * (mask != 0)
        interpolated_weights = (1 - alpha) * original_weights + alpha * masked_weights
        
        #Update weights
        setattr(module, 'weight', 
                torch.nn.Parameter(interpolated_weights.to(module.weight.device)))

        #module.weight = torch.nn.Parameter(interpolated_weights.to(module.weight.device))

        print("successfully applied?")

