import torch
import torch.nn as nn

from .prenet import *
from .unet import UNet
from .uformer import Uformer

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Rain Prior Injection Module
class RPIM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(RPIM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img
    

class InputProj_for_RPIM(nn.Module):
    def __init__(self, in_channel=3, out_channel=32, kernel_size=3, stride=1, act_layer=nn.LeakyReLU, dm_type='unet'):
        super().__init__()
        self.dm_type = dm_type
        if self.dm_type == 'unet': # Add a proj layer before UNet for RPIM
            self.proj = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
                nn.ReLU()
            )
        else:
            # The proj layer (input_proj) in Uformer is modified for RPIM
            self.proj = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
            )
            self.act = act_layer()
        
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, input, rlp, feat):
        x = torch.cat((input, rlp), 1)
        x = self.proj(x)
        x = torch.cat((x, feat), 1)
        if self.dm_type == 'uformer':
            x = self.act(x).flatten(2).transpose(1, 2).contiguous()
        
        return x


'''
To further facilitate the experiments of different RLP, RPIM and DM structures,
the model is re-organized into one general class to avoid repetitive definitions of same parts,
which may also benefit the fast substitution of exisiting models.

w/o RLP & RPIM, the model degenerates into the DM itself;
w/  RLP,        the RLP map is direcly concatenated with input;
w/  RPIM,       the RLP map and feature are modulated and injected at the input projection layer of DM.

Currently, the injection is implemented by    adding a new layer to UNet
                                        while modifying the "input_proj" layer of Uformer.
It makes the code ugly.

Ideally, it may be implemented universally for different DM, to keep the code tidy.
'''
class RLP_NightRain(nn.Module):
    def __init__(self, in_c=3, out_c=3, use_rlp=False, use_rpim=False, dm_type='', rlp_feat=32, bias=False, opt=None):
        super(RLP_NightRain, self).__init__()
        # use_rlp: whether to use Rain Location Prior Module (RLP)
        # use_rpim: whether to use Rain Prior Injection Module (RPIM), only valid when use_rlp is True
        # dm_type: choose a deraining module (DM) from 'unet' and 'uformer'
        
        self.in_c = in_c
        self.out_c = out_c
        self.dm_type = dm_type
        self.injection_dim = 0

        if use_rlp:
            # Currently, PReNet serves as the RLP after some modifications, i.e., initialize the RLP map
            self.recurrent_iter = 6 if self.dm_type == 'unet' else 4
            self.rlp = PReNet_for_RLP(recurrent_iter=self.recurrent_iter, use_GPU=True)
            self.in_c = self.in_c + 1
            
            if use_rpim:
                # Rain Prior Injection Module (RPIM), adapted from SAM of MPRNet
                self.rpim = RPIM(rlp_feat, kernel_size=1, bias=bias)
            
                # Projection layer for prior injection after RPIM
                if self.dm_type == 'unet':
                    self.proj = InputProj_for_RPIM(in_channel=self.in_c, out_channel=rlp_feat, kernel_size=5, stride=1, act_layer=nn.ReLU, dm_type=self.dm_type)
                    self.in_c = rlp_feat*2
                else:
                    self.injection_dim = rlp_feat
                
        # Deraining Module (DM)
        if self.dm_type == 'unet':
            self.dm = UNet(in_c=self.in_c, out_c=self.out_c, n_feat=32)
        elif self.dm_type == 'uformer':
            self.dm = Uformer(img_size=opt.train_ps, dd_in=self.in_c, embed_dim=opt.embed_dim, injection_dim=self.injection_dim, win_size=8, token_projection='linear', token_mlp='leff', modulator=True)
            if use_rpim is True:
                self.dm.input_proj = InputProj_for_RPIM(in_channel=self.in_c, out_channel=opt.embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU, dm_type=self.dm_type)

    def forward(self, input):
        # if use_rlp, extract the RLP map from Input
        if hasattr(self, 'rlp'):
            feat_rlp, rlp_out, rlp_list = self.rlp(input)

            # if use_rpim, modulate the RLP map and feature map with RPIM
            if hasattr(self, 'rpim'):
                feat_rpim, rlp_final = self.rpim(feat_rlp, rlp_out)
                rlp_list.append(rlp_final)

                # Currently, prior injection is implemented by adding a new layer for UNet
                if self.dm_type == 'unet':
                    x = self.proj(input, rlp_final, feat_rpim)
            else:
                # if no RPIM, then the RLP map is directly concatenated to the input
                x = torch.cat((input, rlp_out), 1)
        else:
            # no RLP, no RPIM, the vanilla Deraining Module itself
            x = input

        # Currently, prior injection is implemented at the input_proj layer of Uformer
        # Ideally, it should be unified for different deraining module to make the code simple
        if hasattr(self, 'rpim') and self.dm_type == 'uformer':
            out = self.dm(input, rlp=rlp_final, feat=feat_rpim)
        else:
            out = self.dm(x)

        return out, rlp_list if hasattr(self, 'rlp') else None

