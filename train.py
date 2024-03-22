import os

import torch
import torch.optim as optim
import random
import time
import numpy as np
import datetime
from tqdm import tqdm 

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from timm.utils import NativeScaler
#from pdb import set_trace as stx

from rlp.loss import CharbonnierLoss
from rlp.models import model_utils
from rlp.options import parse_options
from rlp.dataset import *

########## parser ##########
opt = parse_options().parse_args()
print(opt)

########## Set GPUs ##########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

torch.backends.cudnn.benchmark = True

########## Logs dir ##########
rlp_suffix = "_RLP" if opt.use_rlp else ""
rpim_suffix = "_RPIM" if opt.use_rpim else ""
arch = opt.arch + rlp_suffix + rpim_suffix + opt.env

log_dir = os.path.join(opt.save_dir, 'deraining', opt.dataset, arch)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

########## Set Seeds ##########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

########## Model ##########
model_restoration = model_utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

########## Optimizer ##########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


########## DataParallel ##########
if torch.cuda.device_count() > 1:    
    model_restoration = torch.nn.DataParallel(model_restoration, device_ids=[0,1])
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model_restoration.cuda()
     

########## Scheduler ##########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

########## Resume ##########
if opt.resume:
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+ path_chk_rest)
    model_utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = model_utils.load_start_epoch(path_chk_rest) + 1 
    lr = model_utils.load_optim(optimizer, path_chk_rest) 

    # for p in optimizer.param_groups: p['lr'] = lr 
    # warmup = False 
    # new_lr = lr 
    # print('------------------------------------------------------------------------------') 
    # print("==> Resuming Training with learning rate:",new_lr) 
    # print('------------------------------------------------------------------------------') 
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

########## Loss ##########
criterion = CharbonnierLoss().cuda()

########## DataLoader ##########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = DatasetTrain(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
                          num_workers=opt.train_workers, pin_memory=False, drop_last=False)

print("Sizeof training set: ", train_dataset.__len__())

######### train ###########
print('===> Start Epoch {}, End Epoch {}'.format(start_epoch, opt.nepoch))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch):
    epoch_start_time = time.time()
    epoch_loss = 0

    for i, data in enumerate(tqdm(train_loader), 0): 
        # zero_grad
        optimizer.zero_grad()

        input = data[0].cuda()
        gt    = data[1].cuda()
        input_name = data[2]

        with torch.cuda.amp.autocast():
            restored, _ = model_restoration(input)
            loss = criterion(restored, gt)
        loss_scaler(
            loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss +=loss.item()

    if opt.save_images:
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().numpy()
        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = restored_img*255
            restored_img = restored_img.astype(np.uint8)
            restored_img = Image.fromarray(restored_img)
            restored_img.save(os.path.join(result_dir, 'epoch{}_{}.png'.format(epoch, input_name)))
        
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')


    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ", datetime.datetime.now().isoformat())
