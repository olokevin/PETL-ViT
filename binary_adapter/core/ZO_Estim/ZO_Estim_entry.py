import torch
import torch.nn as nn

class SplitedLayer(nn.Module):
    def __init__(self, idx, name, layer):
        super().__init__()
        self.idx = idx
        self.name = name
        self.layer = layer

class SplitedParam(nn.Module):
    def __init__(self, idx, name, param):
        super().__init__()
        self.idx = idx
        self.name = name
        assert isinstance(param, torch.Tensor)
        self.param = param

def split_model(model, iterable_block_name=None):
    modules = []
    # full model split
    if iterable_block_name is None:
        for m in model.children():
            if isinstance(m, (torch.nn.Sequential,)):
                modules += split_model(m)
            # elif hasattr(m, 'conv') and isinstance(m.conv, torch.nn.Sequential):
            #     modules += split_model(m.conv)
            else:
                modules.append(m)
    # only split iterable block
    else:
        iterable_block = getattr(model, iterable_block_name)
        assert isinstance(iterable_block, torch.nn.Sequential)
        for m in iterable_block.children():
            modules.append(m)
    return modules

def split_named_model(model, parent_name=''):
    named_modules = {}
    for name, module in model.named_children():
    # for name, module in model.named_modules():    # Error: non-stop recursion
        if isinstance(module, torch.nn.Sequential):
            named_modules.update(split_named_model(module, parent_name + name + '.'))
        # elif hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Sequential):
        #     named_modules.update(split_named_model(module.conv, parent_name + name + '.conv.'))
        else:
            named_modules[parent_name + name] = module
    return named_modules

from .ZO_Estim_MC import ZO_Estim_MC
from adaptformer import Adapter

opt_able_layers_dict = {
    'Adapter': Adapter,
}

def create_opt_layer_list(layer_list):
    if isinstance(layer_list, str):
        return opt_able_layers_dict[layer_list]
    elif isinstance(layer_list, list):
        opt_layers = []
        for layer_str in layer_list:
            opt_layers.append(opt_able_layers_dict[layer_str])
        return tuple(opt_layers)
    else:
        raise (ValueError("opt_layers_strs should either be a string of a list of strings"))

def build_ZO_Estim(config, model):
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
            actv_perturb_layer_list = create_opt_layer_list(config.actv_perturb_layer_list)
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

def vit_get_iterable_block_name():
    return 'blocks'

def vit_pre_block_forward(model, x):
    x = model.patch_embed(x)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    if model.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    return x

def vit_post_block_forward(model, x):
    x = model.norm(x)
    if model.dist_token is None:
        x = model.pre_logits(x[:, 0])
    else:
        x = x[:, 0], x[:, 1]
    
    if model.head_dist is not None:
        x, x_dist = model.head(x[0]), model.head_dist(x[1])  # x must be a tuple
        if model.training and not torch.jit.is_scripting():
            # during inference, return the average of both classifier predictions
            return x, x_dist
        else:
            return (x + x_dist) / 2
    else:
        x = model.head(x)
    return x

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