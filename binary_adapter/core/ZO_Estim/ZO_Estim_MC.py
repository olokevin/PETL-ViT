from typing import Callable

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scipy.stats import qmc
from .ZO_Estim_entry import split_model, split_named_model, SplitedLayer, SplitedParam
from .QMC_sampler import sphere_n, coord_basis, block_mask_generator, layer_mask_generator

class ZO_Estim_MC(nn.Module):
    """
    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
    """
    
    def __init__(
        self,
        model: nn.Module,
        obj_fn: Callable,
        # For param perturb ZO. A list of SplitedParam. Specifies what Tensors should be optimized.
        trainable_param_list: list = None,
        # For actv  perturb ZO. A list of SplitedLayer. Specifies what layers' activations should be perturbed.
        trainable_layer_list: list = None,

        sigma: float = 0.1,
        n_sample: int = 20,
        signSGD: bool = False,
        
        quantize_method: str = 'None',  # 'None', 'u_fp-grad_fp', 'u_fp-grad_int', 'u_int-grad_int', 'u_int-grad_fp'
        estimate_method: str = 'forward',
        sample_method: str = 'gaussian',
        ):
        super().__init__()

        self.model = model
        self.obj_fn = obj_fn

        if trainable_param_list == 'all':
            self.trainable_param_list = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    # self.trainable_param_list.append(name) 
                    self.trainable_param_list.append(param) 
        else:
            self.trainable_param_list = trainable_param_list
        
        splited_named_modules = split_named_model(model)
        self.splited_layer_list = []
        idx = 0
        for name, layer in splited_named_modules.items():
            self.splited_layer_list.append(SplitedLayer(idx, name, layer))
            # print(name, layer)
            idx += 1
        
        if trainable_layer_list == 'all':
            self.trainable_layer_list = self.splited_layer_list
        else:
            self.trainable_layer_list = trainable_layer_list

        self.sigma = sigma
        self.n_sample = n_sample
        self.signSGD = signSGD

        self.quantize_method = quantize_method
        
        self.estimate_method = estimate_method
        self.sample_method = sample_method

        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        
        self.forward_counter = 0
    
    def _init_sampler(self, dimension):
        if self.sample_method == 'sobol':
            sampler = qmc.Sobol(d=dimension, scramble=False)
        elif self.sample_method == 'halton':
            sampler = qmc.Halton(d=dimension, scramble=True)
        elif self.sample_method == 'sphere_n':
            sampler = sphere_n(n=dimension)
        elif self.sample_method == 'coord_basis':
            sampler = coord_basis(dimension=dimension)
        else:
            sampler = None
        return sampler
            
    ### Generate random vectors from a normal distribution
    def _sample_unit_sphere(self, dimension, device):
        
        if self.sample_method == 'uniform':
            sample = torch.randn(dimension, device=device)
            sample = torch.nn.functional.normalize(sample, p=2, dim=0)
        elif self.sample_method == 'gaussian':
            sample = torch.randn(dimension, device=device) / dimension
        elif self.sample_method == 'bernoulli':
            ### Rademacher
            if 'u_int' in self.quantize_method:
                sample = torch.ones(dimension, device=device) - 2*torch.bernoulli(0.5*torch.ones(dimension, device=device))
            else:
                sample = torch.ones(dimension, device=device) - 2*torch.bernoulli(0.5*torch.ones(dimension, device=device))
                sample = sample / torch.sqrt(torch.tensor(dimension, device=device))
        elif self.sample_method == 'coord_basis':
            sample = next(self.sampler)
            sample = sample.to(device)
        elif self.sample_method in ('sobol', 'halton'):
            if self.sampler == None:
                raise ValueError('Need sampler input')
            else:
                sample = torch.Tensor(self.sampler.random(1)).squeeze()
                sample = 2*sample-torch.ones_like(sample)
                sample = torch.nn.functional.normalize(sample, p=2, dim=0)
                sample = sample.to(device)
        elif self.sample_method == 'sphere_n':
            sample = next(self.sampler)
            sample = sample.to(device)
        else:
            return NotImplementedError('Unlnown sample method', self.sample_method)
        
        return sample
    
    def _sample_unit_sphere_quantized(self, shape, sample_method, device):
        if sample_method == 'bernoulli':
            sample = torch.ones(shape, device=device) - 2*torch.bernoulli(0.5*torch.ones(shape, device=device))
        else:
            return NotImplementedError('Unlnown sample method', self.sample_method)
        
        return sample
    
    def get_single_param_ZO_gradient(self, splited_param, block_in, old_loss, sigma, estimate_method, sample_method):
        idx = splited_param.odx
        param = splited_param.param

        param_dim = param.numel()
        param_shape = param.shape
        param_vec = param.view(-1)

        param_ZO_grad = torch.zeros_like(param_vec, device=self.device)

        if sample_method == 'coord_basis':
            for i in range(param_dim):
                old_param_vec = param_vec[i] * 1
                # pos
                param_vec[i] = param_vec[i] + sigma
                _, pos_loss = self.obj_fn(starting_idx=idx, input=block_in, return_loss_reduction='mean')
                param_vec[i] = param_vec[i] - sigma

                # neg
                if estimate_method == 'forward':
                    param_ZO_grad[i] = (pos_loss - old_loss) / sigma
                elif estimate_method == 'antithetic':
                    param_vec[i] = param_vec[i] - sigma
                    _, neg_loss = self.obj_fn(starting_idx=idx, input=block_in, return_loss_reduction='mean')
                    param_vec[i] = param_vec[i] + sigma

                    param_ZO_grad[i] = (pos_loss - neg_loss) / 2 / sigma
                else:
                    raise NotImplementedError('Unknown estimate method')
        elif sample_method == 'bernoulli':
            for i in range(self.n_sample):
                ### Generate random perturbation with the same shape as the parameter
                u = self._sample_unit_sphere_quantized(param_vec.shape, sample_method, self.device)
                old_param_vec = param_vec * 1

                ### Add perturbation to the parameter
                # pos
                param_vec.add_(u * sigma)
                _, pos_loss = self.obj_fn(starting_idx=idx, input=block_in, return_loss_reduction='mean')
                param_vec.sub_(u * sigma)

                # neg
                if estimate_method == 'forward':
                    param_ZO_grad += (pos_loss - old_loss) / sigma * u
                elif estimate_method == 'antithetic':
                    param_vec.sub_(u * sigma)
                    _, neg_loss = self.obj_fn(starting_idx=idx, input=block_in, return_loss_reduction='mean')
                    param_vec.add_(u * sigma)

                    param_ZO_grad += (pos_loss - neg_loss) / 2 / sigma * u
                
                ### Estimate gradient
            param_ZO_grad = param_ZO_grad / self.n_sample
            # param_ZO_grad = param_ZO_grad / 8
        else:
            return NotImplementedError('sample method not implemented yet')

        param_ZO_grad = param_ZO_grad.view(param_shape)
        return param_ZO_grad
    
    def get_param_ZO_gradient(self, old_loss):

        for splited_param in self.trainable_param_list:
            block_in = self.obj_fn(ending_idx=splited_param.idx, return_loss_reduction='no_loss')
            ### TODO: could further specify sigma, estimate_method, sample_method for different params
            param_ZO_grad = self.get_single_param_ZO_gradient(splited_param, block_in, old_loss, self.sigma, self.estimate_method, self.sample_method)

            splited_param.grad = param_ZO_grad
    
    def get_actv_ZO_gradient(self):
        _, old_loss = self.obj_fn(return_loss_reduction='none')

        ### Generate random perturbation with the same shape as the parameter

        ### Add perturbation to the parameter

        ### Estimate gradient

        for splited_layer in self.trainable_layer_list:
            assert hasattr(splited_layer.layer, 'perturb_forward_flag')

            block_in = self.obj_fn(ending_idx=splited_layer.idx, return_loss_reduction='no_loss')

            post_actv_shape = splited_layer.layer.get_output_shape()
            batch_sz = post_actv_shape[0]
            mask = torch.ones_like(post_actv_shape)
            
            for i in range(self.n_sample):
                
                ### Generate random perturbation with the same shape as the parameter
                if self.sample_method == 'coord_basis':
                    raise NotImplementedError
                else:
                    if hasattr(self, 'sync_batch_perturb') and self.sync_batch_perturb:
                        feature_shape = torch.prod(post_actv_shape[1:]).item()
                        u = mask * torch.tile(self._sample_unit_sphere_quantized(feature_shape, self.sample_method, self.device).unsqueeze(0), (batch_sz, 1))
                    else:
                        u = mask * self._sample_unit_sphere_quantized(post_actv_shape, self.sample_method, self.device)    

                ### Add perturbation to the parameter
                splited_layer.layer.en_perturb_forward(u)
                _, pos_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='none')
                self.forward_counter += 1

                ### Estimate gradient
                if self.estimate_method == 'forward':
                    ZO_grad += (pos_loss - old_loss).view(-1,1) / self.sigma * u

                elif self.estimate_method == 'antithetic':
                    splited_layer.layer.en_perturb_forward(-u)
                    _, neg_loss = self.obj_fn(starting_idx=splited_layer.idx, input=block_in, return_loss_reduction='none')
                    self.forward_counter += 1
                    
                    ZO_grad += (pos_loss - neg_loss).view(-1,1) / 2.0 / self.sigma * u

            splited_layer.layer.disable_perturb_forward()
            ZO_grad = (ZO_grad / self.n_sample / batch_sz).view(post_actv_shape)

            pre_activ = splited_layer.layer.pre_actv

            if splited_layer.type == nn.Linear:
                splited_layer.layer.weight.grad = torch.matmul(ZO_grad.T, pre_activ)
                splited_layer.layer.bias.grad = torch.mean(ZO_grad, dim=0)
            else:
                splited_layer.layer.local_backward(ZO_grad)            

    def update_obj_fn(self, obj_fn):
        self.obj_fn = obj_fn
    
    def get_forward_cnt(self):
        return self.forward_counter
    
    def estimate_grad(self):
        
        # self.model.zero_grad()
        outputs, old_loss = self.obj_fn()
        if self.trainable_layer_list is not None:
            self.get_actv_ZO_gradient()
        
        if self.trainable_param_list is not None:
            self.get_param_ZO_gradient(old_loss)
        
        return outputs, old_loss, self.estim_grads