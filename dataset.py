import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import load_img, Augment_RGB_torch, random_add_jpg_compression
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DatasetTrain(Dataset):
    def __init__(self, data_dir, img_options=None):
        super(DatasetTrain, self).__init__()

        input_folder = 'rainy'
        gt_folder = 'gt'

        
        input_filenames = sorted(os.listdir(os.path.join(data_dir, input_folder)))
        #gt_filenames   = sorted(os.listdir(os.path.join(data_dir, gt_folder)))
        gt_filenames = [x[:-7]+'.png' for x in input_filenames]
        
        self.input_paths = [os.path.join(data_dir, input_folder, x) for x in input_filenames if is_image_file(x)]
        self.gt_paths    = [os.path.join(data_dir, gt_folder, x)    for x in gt_filenames    if is_image_file(x)]
        
        self.img_options = img_options

        self.img_num = len(self.input_paths)

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        tar_index = index % self.img_num
        input = torch.from_numpy(np.float32(load_img(self.input_paths[tar_index])))
        gt    = torch.from_numpy(np.float32(load_img(self.gt_paths[tar_index])))

        # input = torch.from_numpy(random_add_jpg_compression(input, [35,90]))
        
        input = input.permute(2,0,1)
        gt    = gt.permute(2,0,1)

        input_name = os.path.split(self.input_paths[tar_index])[-1]
        gt_name    = os.path.split(self.gt_paths[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = gt.shape[1]
        W = gt.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        input = input[:, r:r + ps, c:c + ps]
        gt    = gt[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        input = getattr(augment, apply_trans)(input)
        gt    = getattr(augment, apply_trans)(gt)

        return input, gt, input_name, gt_name


##################################################################################################
class DatasetTest(Dataset):
    def __init__(self, inp_dir):
        super(DatasetTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_paths = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_num = len(self.inp_paths)

    def __len__(self):
        return self.inp_num

    def __getitem__(self, index):

        inp_path = self.inp_paths[index]
        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
        inp = Image.open(inp_path)

        inp = TF.to_tensor(inp)
        return inp, filename
