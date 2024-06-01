import torch
from torch import nn
from torch.nn import functional as F
from utils import QLinear
import timm


def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.adapter_mlp(x) * self.s
    return x

##### Bi-direction #####
class Adapter(nn.Module):
    def __init__(self, dim, bit):
        super().__init__()

        if bit == 32:
            self.adapter_down = nn.Linear(768, dim, bias=False)
            self.adapter_up = nn.Linear(dim, 768, bias=False)
            # nn.init.zeros_(self.adapter_up.weight)
            # nn.init.eye_(self.adapter_up.weight)
        else:
            self.adapter_down = QLinear(768, dim, bit)
            self.adapter_up = QLinear(dim, 768, bit)
            nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        self.perturb_forward_flag = False
        self.perturb_vec = None

        self.adapter_down_pre_actv = None
        self.adapter_up_pre_actv = None

        # Register forward hook to get output shape
        self.adapter_up.register_forward_hook(self.output_shape_hook)
        self.adapter_down.register_forward_hook(self.output_shape_hook)

    ### Bi-direction
    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)

        if self.perturb_forward_flag:
            self.adapter_down_pre_actv = x
            self.adapter_up_pre_actv = x_down
            self.binary_mask = x_down.ne(0)
            x_down = x_down + self.perturb_vec.reshape(x_down.shape) * self.binary_mask.int()

        x_up = self.adapter_up(x_down)

        return x_up
    
    @torch.no_grad()
    def local_backward(self, grad_output):
        ### test
        # print(f'cos_sim of out grad: {F.cosine_similarity(self.adapter_up.in_grad[0].view(-1), grad_output.view(-1), dim=0)}')
        # grad_output = self.adapter_up.in_grad[0]
        ### test
        U, S, Vh = torch.linalg.svd(self.adapter_up.weight.data, full_matrices=False)
        W_inv_T = U @ torch.diag(1 / S) @ Vh
        adapter_up_grad_output = torch.matmul(grad_output, W_inv_T.T)

        ### test
        # print(f'cos_sim of out grad: {F.cosine_similarity(self.adapter_up.out_grad[0].view(-1),adapter_up_grad_output.view(-1), dim=0)}')
        ### test

        self.adapter_up.weight.grad = torch.einsum('...i,...j->ij', adapter_up_grad_output, self.adapter_up_pre_actv)

        grad_output = grad_output * self.binary_mask.int()
        self.adapter_down.weight.grad = torch.einsum('...i,...j->ij', grad_output, self.adapter_down_pre_actv)
    
    def en_perturb_forward(self, u):
        self.perturb_forward_flag = True
        self.perturb_vec = u
    
    def disable_perturb_forward(self):
        self.perturb_forward_flag = False
        self.perturb_vec = None
        self.adapter_down_pre_actv = None
        self.adapter_up_pre_actv = None
    
    def output_shape_hook(self, module, input, output):
        # Get the shape of the output
        module.output_shape = output.shape

    def get_output_shape(self):
        return self.adapter_down.output_shape

###### perturb on adapter_up, backward update ######

# class Adapter(nn.Module):
#     def __init__(self, dim, bit):
#         super().__init__()

#         if bit == 32:
#             self.adapter_down = nn.Linear(768, dim, bias=False)
#             self.adapter_up = nn.Linear(dim, 768, bias=False)
#             # nn.init.zeros_(self.adapter_up.weight)
#             nn.init.eye_(self.adapter_up.weight)
#         else:
#             self.adapter_down = QLinear(768, dim, bit)
#             self.adapter_up = QLinear(dim, 768, bit)
#             nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
#         self.act = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim

#         self.perturb_forward_flag = False
#         self.perturb_vec = None

#         self.adapter_down_pre_actv = None
#         self.adapter_up_pre_actv = None

#         # Register forward hook to get output shape
#         self.register_forward_hook(self.output_shape_hook)
    
#     def forward(self, x):
#         B, N, C = x.shape
#         x_down = self.adapter_down(x)
#         x_down = self.act(x_down)
#         x_down = self.dropout(x_down)

#         if self.perturb_forward_flag:
#             self.adapter_down_pre_actv = x
#             self.adapter_up_pre_actv = x_down
#             self.binary_mask = x_down.ne(0)

#         x_up = self.adapter_up(x_down)

#         if self.perturb_forward_flag:
#             return x_up + self.perturb_vec.reshape(x_up.shape)
#         else:
#             return x_up
    
#     @torch.no_grad()
#     def local_backward(self, grad_output):
#         self.adapter_up.weight.grad.data = torch.einsum('...i,...j->ij', grad_output, self.adapter_up_pre_actv)
#         adapter_down_grad_output = torch.matmul(grad_output, self.adapter_up.weight)
#         self.adapter_down.weight.grad.data = torch.einsum('...i,...j->ij', adapter_down_grad_output, self.adapter_down_pre_actv)
    
#     def en_perturb_forward(self, u):
#         self.perturb_forward_flag = True
#         self.perturb_vec = u
    
#     def disable_perturb_forward(self):
#         self.perturb_forward_flag = False
#         self.perturb_vec = None
#         self.adapter_down_pre_actv = None
#         self.adapter_up_pre_actv = None
    
#     def output_shape_hook(self, module, input, output):
#         # Get the shape of the output
#         self.output_shape = output.shape

#     def get_output_shape(self):
#         return self.output_shape

def set_adapter(model, dim=32, s=1, bit=1):
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Block:
            layer.adapter_mlp = Adapter(dim, bit)
            layer.s = s
            bound_method = forward_block.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_adapter(layer, dim, s, bit)
