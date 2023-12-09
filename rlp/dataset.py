import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from rlp.utils import load_img, random_add_jpg_compression
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor
    
##################################################################################################
class DatasetTrain(Dataset):
    def __init__(self, data_dir, img_options=None):
        super(DatasetTrain, self).__init__()

        input_folder = 'rainy'
        gt_folder = 'gt'
        self.augment   = Augment_RGB_torch()
        self.transforms_aug = [method for method in dir(self.augment) if callable(getattr(self.augment, method)) if not method.startswith('_')] 
        
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

        # Random Crop
        ps = self.img_options['patch_size']
        H = gt.shape[1]
        W = gt.shape[2]
        r = np.random.randint(0, H - ps) if not H-ps else 0
        c = np.random.randint(0, W - ps) if not H-ps else 0
        
        input = input[:, r:r + ps, c:c + ps]
        gt    = gt[:, r:r + ps, c:c + ps]

        apply_trans = self.transforms_aug[random.getrandbits(3)]

        input = getattr(self.augment, apply_trans)(input)
        gt    = getattr(self.augment, apply_trans)(gt)

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
