import torch
import os

from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except:        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from .rlp import RLP_NightRain
    ARCH_list = ['UNet', 'Uformer_T']

    arch = opt.arch
    use_rlp = opt.use_rlp
    use_rpim = opt.use_rpim
    rlp_on = "RLP" if use_rlp else "noRLP"
    rpim_on = "RPIM" if use_rpim else "noRPIM"
    dm_type = 'unet' if arch == 'UNet' else 'uformer'

    if arch in ARCH_list:
        print('You choose ' + arch + '...' + rlp_on + '...' + rpim_on)
        model_restoration = RLP_NightRain(in_c=3, out_c=3, use_rlp=use_rlp, use_rpim=use_rpim, rlp_feat=32, dm_type=dm_type, opt=opt)
    else:
        raise Exception("Arch error!")

    return model_restoration