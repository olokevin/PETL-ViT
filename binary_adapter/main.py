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
from utils import save, load, load_config, set_seed, QLinear, AverageMeter, load_ZO_Estim_config, logger
import adaptformer
import lora

import time
from core.ZO_Estim.ZO_Estim_entry import build_ZO_Estim, build_obj_fn, SplitedLayer, SplitedParam, vit_get_iterable_block_name, vit_pre_block_forward, vit_post_block_forward

def train(args, model, dl, opt, scheduler, epoch, ZO_Estim=None):
    model.train()
    # pbar = tqdm(range(epoch))

    criterion = torch.nn.CrossEntropyLoss()
    
    # for ep in pbar:
    for ep in range(epoch):
        model.train()
        model = model.cuda()
        train_acc = AverageMeter()
        train_loss = 0

        ### Disable Dropout and DropPath for ZO training
        if ZO_Estim is not None:
            from timm.models.layers import DropPath
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0
                if isinstance(module, DropPath):
                    module.drop_prob = 0
            
        with tqdm(total=len(dl), desc='Train Epoch #{}'.format(ep)) as t:
            for i, batch in enumerate(dl):
                x, y = batch[0].cuda(), batch[1].cuda()
                    
                if ZO_Estim is None:
                    out = model(x)
                    loss = criterion(out, y)
                    opt.zero_grad()
                    loss.backward()
                else:
                    ##### Test #####
                    if args.debug:
                        out = model(x)
                        loss = criterion(out, y)
                        opt.zero_grad()
                        loss.backward()  

                        try:
                            block_idx = args.ZO_Estim.param_perturb_block_idx_list[-1]
                        except:
                            try:
                                block_idx = args.ZO_Estim.actv_perturb_block_idx_list[-1]
                            except:
                                block_idx = -1
                        splited_layer = SplitedLayer(idx=block_idx, name=f'blocks.{block_idx}.adapter_mlp', layer=model.blocks[block_idx].adapter_mlp)

                        # FO_grad = splited_layer.layer.out_grad[0].data
                        FO_adapter_up_grad_w = splited_layer.layer.adapter_up.weight.grad.data
                        FO_adapter_down_grad_w = splited_layer.layer.adapter_down.weight.grad.data
                    ##### Test #####

                    obj_fn_type = ZO_Estim.obj_fn_type
                    kwargs = {}
                    if obj_fn_type == 'classifier_layerwise':
                        kwargs = {'get_iterable_block_name': vit_get_iterable_block_name, "pre_block_forward": vit_pre_block_forward, "post_block_forward": vit_post_block_forward}
                    with torch.no_grad():
                        out = model(x)
                        loss = criterion(out, y)
                        obj_fn = build_obj_fn(obj_fn_type, data=x, target=y, model=model, criterion=criterion, **kwargs)
                        ZO_Estim.update_obj_fn(obj_fn)
                        ZO_Estim.estimate_grad()
                    
                    if args.debug:
                        # ZO_grad = splited_layer.layer.out_grad[0].data
                        ZO_adapter_up_grad_w = splited_layer.layer.adapter_up.weight.grad.data
                        ZO_adapter_down_grad_w = splited_layer.layer.adapter_down.weight.grad.data

                        # print('\n Grad output')
                        # print('cos sim grad_output', F.cosine_similarity(FO_grad.view(-1), ZO_grad.view(-1), dim=0))
                        # print('FO_grad:', torch.linalg.norm(FO_grad))
                        # print('ZO_grad:', torch.linalg.norm(ZO_grad))

                        # logger.info('Adapter_down')
                        # logger.info(f'weight cos sim {F.cosine_similarity(FO_adapter_down_grad_w.view(-1), ZO_adapter_down_grad_w.view(-1), dim=0)}')
                        # logger.info(f'FO_weight_grad norm: {torch.linalg.norm(FO_adapter_down_grad_w)}')
                        # logger.info(f'ZO_weight_grad norm: {torch.linalg.norm(ZO_adapter_down_grad_w)}')
                        # logger.info(f'ZO/FO: {torch.linalg.norm(ZO_adapter_down_grad_w)/torch.linalg.norm(FO_adapter_down_grad_w)}')   

                        logger.info('Adapter_up')
                        logger.info(f'weight cos sim {F.cosine_similarity(FO_adapter_up_grad_w.view(-1), ZO_adapter_up_grad_w.view(-1), dim=0)}')
                        logger.info(f'FO_weight_grad norm: {torch.linalg.norm(FO_adapter_up_grad_w)}')
                        logger.info(f'ZO_weight_grad norm: {torch.linalg.norm(ZO_adapter_up_grad_w)}')
                        logger.info(f'ZO/FO:  {torch.linalg.norm(ZO_adapter_up_grad_w)/torch.linalg.norm(FO_adapter_up_grad_w)}')
                
                opt.step()

                train_acc.update(out, y)
                train_loss += loss.item()

                train_info_dict = {
                    'train/acc': train_acc.result().item(),
                    'train/loss': loss.item(),
                    'train/lr': opt.param_groups[0]['lr'],
                }
                
                t.set_postfix(train_info_dict)
                t.update()
        
        ### Epoch training ends
        if scheduler is not None:
            scheduler.step(ep)
        
        train_loss = train_loss / len(dl)
        train_info_dict = {
            'train/acc': train_acc.result().item(),
            'train/loss': train_loss,
            'train/lr': opt.param_groups[0]['lr'],
        }
        logger.info(f'epoch {ep}: f{train_info_dict}')

        if ep % 10 == 9:
            acc = test(vit, test_dl)
            if acc > args.best_acc:
                args.best_acc = acc
                save(args, model)
            # pbar.set_description('best_acc ' + str(args.best_acc))

            test_info_dict = {
                'test/acc': acc,
                'test/best_acc': args.best_acc,
            }
            logger.info(f'epoch {ep}: f{test_info_dict}')

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
    # parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--model_path', type=str, metavar='DIR', help='run directory')
    parser.add_argument('--load_config', action='store_true', default=False)
    parser.add_argument('--ZO_Estim', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.eval or args.load_config:
        load_config(args)
    
    if args.ZO_Estim:
        load_ZO_Estim_config(args)

    set_seed(args.seed)

    if args.model_path is None:
        if args.ZO_Estim:
            training_method = 'ZO'
        else:
            training_method = 'FO'
        args.model_path = os.path.join(
            "./runs",
            args.method,
            args.dataset,
            training_method,
            time.strftime("%Y%m%d-%H%M%S")+'-'+str(os.getpid())
        )
    
    os.makedirs(args.model_path, exist_ok=True)
    logger.init(args, args.model_path)  # dump exp config
    logger.info(str(os.getpid()))
    logger.info(f'{args}')

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
    
    ### Model structure
    # for name, m in vit.named_children():
    #     print(name)
    #     if isinstance(m, (torch.nn.Sequential,)):
    #         for layer_name, layer in m.named_children():
    #             print(layer_name)

    # for name, layer in vit.named_parameters():
    #     print(name)
    
    ### Register backward hook for debugging
    if args.debug:
        def save_grad(module, grad_input, grad_output):
            module.in_grad = grad_input
            module.out_grad = grad_output

        from adaptformer import Adapter
        for name, layer in vit.named_modules():
            if type(layer) == Adapter:
                layer.adapter_up.register_full_backward_hook(save_grad)
    
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
        
        vit = vit.cuda()
        if args.ZO_Estim:
            ZO_Estim = build_ZO_Estim(args.ZO_Estim, model=vit)
        else:
            ZO_Estim = None

        vit = train(args, vit, train_dl, opt, scheduler, epoch=100, ZO_Estim=ZO_Estim)

        vit = vit.cpu()

    else:
        load(args, vit)
        args.best_acc = test(vit, test_dl)

    print('best_acc:', args.best_acc)