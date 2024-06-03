import torch
import torch.nn as nn

from .ZO_utils import SplitedLayer, SplitedParam, split_model
from .ZO_Estim_MC import ZO_Estim_MC

def create_opt_layer_list(layer_list, opt_able_layers_dict):
    if isinstance(layer_list, str):
        return opt_able_layers_dict[layer_list]
    elif isinstance(layer_list, list):
        opt_layers = []
        for layer_str in layer_list:
            opt_layers.append(opt_able_layers_dict[layer_str])
        return tuple(opt_layers)
    else:
        raise (ValueError("opt_layers_strs should either be a string of a list of strings"))

def build_ZO_Estim(config, model, opt_able_layers_dict):
    if config.name == 'ZO_Estim_MC':
        ### Splited model
        split_modules_list = split_model(model)
        splited_param_list = None
        splited_layer_list = None

        ### Param perturb 
        if config.param_perturb_param_list is not None:
            param_perturb_param_list = config.param_perturb_param_list
            if config.param_perturb_block_idx_list == 'all':
                param_perturb_block_idx_list = list(range(len(split_modules_list)))
            else:
                param_perturb_block_idx_list = config.param_perturb_block_idx_list
            
            splited_param_list = []
            for name, param in model.named_parameters():
                if any(keyword in name for keyword in param_perturb_param_list):
                    block_idx = int(name.split('.')[1])
                    if block_idx in param_perturb_block_idx_list:
                        splited_param_list.append(SplitedParam(idx=block_idx, name=name, param=param))
        
        ### Actv perturb 
        if config.actv_perturb_layer_list is not None:
            actv_perturb_layer_list = create_opt_layer_list(config.actv_perturb_layer_list, opt_able_layers_dict)
            if config.actv_perturb_block_idx_list == 'all':
                actv_perturb_block_idx_list = list(range(len(split_modules_list)))
            else:
                actv_perturb_block_idx_list = config.actv_perturb_block_idx_list

            splited_layer_list = []
            for name, layer in model.named_modules():
                if type(layer) in actv_perturb_layer_list:
                  block_idx = int(name.split('.')[1])
                  if block_idx in actv_perturb_block_idx_list:
                      splited_layer_list.append(SplitedLayer(idx=block_idx, name=name, layer=layer))

        ZO_Estim = ZO_Estim_MC(
            model = model, 
            obj_fn_type = config.obj_fn_type,
            splited_param_list = splited_param_list,
            splited_layer_list = splited_layer_list,

            sigma = config.sigma,
            n_sample  = config.n_sample,
            signSGD = config.signSGD,
            
            quantized = config.quantized,
            estimate_method = config.estimate_method,
            sample_method = config.sample_method,
            normalize_perturbation = config.normalize_perturbation
        )
        return ZO_Estim
    else:
        return NotImplementedError

def build_obj_fn(obj_fn_type, **kwargs):
    if obj_fn_type == 'classifier':
        obj_fn = build_obj_fn_classifier(**kwargs)
    elif obj_fn_type == 'classifier_layerwise':
        obj_fn = build_obj_fn_classifier_layerwise(**kwargs)
    elif obj_fn_type == 'classifier_acc':
        obj_fn = build_obj_fn_classifier_acc(**kwargs)
    else:
        return NotImplementedError
    return obj_fn

def build_obj_fn_classifier(data, target, model, criterion):
    def _obj_fn():
        y = model(data)
        return y, criterion(y, target)
    
    return _obj_fn

def build_obj_fn_classifier_acc(data, target, model, criterion):
    def _obj_fn():
        outputs = model(data)
        _, predicted = outputs.max(1)
        total = target.size(0)
        correct = predicted.eq(target).sum().item()
        err = 1 - correct / total

        return outputs, err
    
    return _obj_fn

def build_obj_fn_classifier_layerwise(data, target, model, criterion, get_iterable_block_name=None, pre_block_forward=None, post_block_forward=None):
    if get_iterable_block_name is not None:
        iterable_block_name = get_iterable_block_name()
    else:
        iterable_block_name = None
    split_modules_list = split_model(model, iterable_block_name)
    
    # if no attribute for _obj_fn: same as build_obj_fn_classifier
    def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
        if ending_idx == None:
            ending_idx = len(split_modules_list)

        if starting_idx == 0:
            y = data
            ### pre_block_forward when start from image input
            if pre_block_forward is not None:
                y = pre_block_forward(model, y)
        else:
            assert input is not None
            y = input
        
        if detach_idx is not None and detach_idx < 0:
            detach_idx = len(split_modules_list) + detach_idx
        
        for i in range(starting_idx, ending_idx):
            y = split_modules_list[i](y)
            if detach_idx is not None and i == detach_idx:
                y = y.detach()
                y.requires_grad = True
            
        ### post_block_forward when end at classifier head
        if ending_idx == len(split_modules_list):
            if post_block_forward is not None:
                y = post_block_forward(model, y)
           
        if return_loss_reduction == 'mean':
            criterion.reduction = 'mean'
            return y, criterion(y, target)
        elif return_loss_reduction == 'none':
            criterion.reduction = 'none'
            loss = criterion(y, target)
            criterion.reduction = 'mean'
            return y, loss
        elif return_loss_reduction == 'no_loss':
            return y
        else:
            raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
    return _obj_fn

# def build_obj_fn_classifier_layerwise(data, target, model, criterion, iterable_block_name=None):
#     split_modules_list = split_model(model, iterable_block_name)
    
#     # if no attribute for _obj_fn: same as build_obj_fn_classifier
#     def _obj_fn(starting_idx=0, ending_idx=None, input=None, return_loss_reduction='mean', detach_idx=None):
#         if ending_idx == None:
#             ending_idx = len(split_modules_list)

#         if starting_idx == 0:
#             y = data
#         else:
#             assert input is not None
#             y = input
        
#         if detach_idx is not None and detach_idx < 0:
#             detach_idx = len(split_modules_list) + detach_idx
        
#         for i in range(starting_idx, ending_idx):
#             y = split_modules_list[i](y)
#             if detach_idx is not None and i == detach_idx:
#                 y = y.detach()
#                 y.requires_grad = True
           
#         if return_loss_reduction == 'mean':
#             criterion.reduction = 'mean'
#             return y, criterion(y, target)
#         elif return_loss_reduction == 'none':
#             criterion.reduction = 'none'
#             loss = criterion(y, target)
#             criterion.reduction = 'mean'
#             return y, loss
#         elif return_loss_reduction == 'no_loss':
#             return y
#         else:
#             raise NotImplementedError(f'Unknown {return_loss_reduction}')
    
#     return _obj_fn