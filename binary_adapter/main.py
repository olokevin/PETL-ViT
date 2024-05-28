import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import save, load, load_config, set_seed, QLinear, AverageMeter, load_ZO_Estim_config
import adaptformer
import lora

from core.ZO_Estim.ZO_Estim_entry import build_ZO_Estim, build_obj_fn, split_model, split_named_model, SplitedLayer, SplitedParam



def train(args, model, dl, opt, scheduler, epoch, ZO_Estim=None):
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)

            if ZO_Estim is None:
                opt.zero_grad()
                loss.backward()
            else:
                obj_fn_type = args.ZO_Estim.obj_fn
                with torch.no_grad():
                    obj_fn = build_obj_fn(obj_fn_type, data=x, target=y, model=model, criterion=F.cross_entropy)
                    ZO_Estim.update_obj_fn(obj_fn)
                    outputs, loss, grads = ZO_Estim.estimate_grad()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(vit, test_dl)
            if acc > args.best_acc:
                args.best_acc = acc
                save(args, model)
            pbar.set_description('best_acc ' + str(args.best_acc))

    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = AverageMeter()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out, y)
    return acc.result().item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bit', type=int, default=1, choices=[1, 2, 4, 8, 32])
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='adaptformer',
                        choices=['adaptformer', 'adaptformer-bihead', 'lora', 'lora-bihead'])
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='.')
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--load_config', action='store_true', default=False)
    parser.add_argument('--ZO_Estim', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    if args.eval or args.load_config:
        load_config(args)
    
    if args.ZO_Estim:
        load_ZO_Estim_config(args)

    set_seed(args.seed)
    args.best_acc = 0
    vit = create_model(args.model, checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(args.dataset, normalize=False)

    if args.method == 'adaptformer':
        adaptformer.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    elif args.method == 'adaptformer-bihead':
        adaptformer.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.head = QLinear(768, get_classes_num(args.dataset), 1)
    elif args.method == 'lora':
        lora.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    elif args.method == 'lora-bihead':
        lora.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.head = QLinear(768, get_classes_num(args.dataset), 1)

    splited_named_modules = split_named_model(vit)
    for name, block in splited_named_modules.items():
        print(name)
    
    for name, layer in vit.named_modules():
        print(name)
    
    if not args.eval:
        trainable = []
        for n, p in vit.named_parameters():
            if ('adapter' in n or 'head' in n) and p.requires_grad:
                trainable.append(p)
            else:
                p.requires_grad = False
        opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        scheduler = CosineLRScheduler(opt, t_initial=100,
                                      warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
        
        ### Param perturb 
        param_perturb_list = args.ZO_Estim.param_perturb_list
        trainable_param_list = []
        assert isinstance(param_perturb_list, list)
        for name, param in vit.named_parameters():
            if any(s in name for s in param_perturb_list):
                trainable_param_list.append(SplitedParam(idx=int(name.split('.')[1])+2, name=name, param=param))
        
        ### Actv perturb 
        actv_perturb_list = args.ZO_Estim.actv_perturb_list
        trainable_layer_list = []
        assert isinstance(actv_perturb_list, list)
        for name, layer in vit.named_modules():
            if any(s in name for s in actv_perturb_list):
                trainable_layer_list.append(SplitedLayer(idx=int(name.split('.')[1])+2, name=name, layer=layer))
        
        if args.ZO_Estim.en is True:
            obj_fn = None
            ZO_Estim = build_ZO_Estim(args.ZO_Estim, model=vit, obj_fn=obj_fn, trainable_param_list=trainable_param_list, trainable_layer_list=trainable_layer_list )
        else:
            ZO_Estim = None
        
        vit = train(args, vit, train_dl, opt, scheduler, epoch=100, ZO_Estim=ZO_Estim)

    else:
        load(args, vit)
        args.best_acc = test(vit, test_dl)

    print('best_acc:', args.best_acc)