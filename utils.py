import torch
import math
import numpy as np
import cv2

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img



def expand2square(timg,factor=16.0):
    b, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)
      
    img = torch.ones(b,3,X,X).type_as(timg) # 3, h, w
    mask = torch.zeros(b,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h)//2):((X - h)//2 + h), ((X - w)//2):((X - w)//2 + w)] = timg
    mask[:, :, ((X - h)//2):((X - h)//2 + h), ((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask



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


### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for
            best quality. Default: 90.
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    img = np.clip(img, 0, 1)
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img


def random_add_jpg_compression(img, quality_range=(90, 100)):
    """Randomly add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality_range (tuple[float] | list[float]): JPG compression quality
            range. 0 for lowest quality, 100 for best quality.
            Default: (90, 100).
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    quality = np.random.uniform(quality_range[0], quality_range[1])
    return add_jpg_compression(img, quality)




def rgb_to_ycbcr(img):
    '''
    This function converts an RGB image to YCbCr color space,
    following the Matlab implementation of rgb2ycbcr and the standard ITU-R BT.601 conversion matrix.
    The result is slightly different from the Matlab implementation due to rounding errors.
    '''
    # Check if input is a single image or a batch of images
    if img.ndim == 3:  # Single image
        img = img.unsqueeze(0)  # Add a batch dimension
        is_single_image = True
    else:
        is_single_image = False

    if img.dtype == torch.uint8:
        img = img.float()  # Convert to float

    # Normalize if necessary
    if img.max() > 1.0:
        img = img / 255.0

    # Transformation matrix and offset
    T = torch.tensor([
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.000],
        [112.000, -93.786, -18.214]
    ], dtype=img.dtype, device=img.device) / 255
    offset = torch.tensor([16, 128, 128], dtype=img.dtype, device=img.device)

    # Prepare output tensor
    ycbcr_img = torch.zeros_like(img)

    # Apply the conversion for each channel
    for p in range(3):
        ycbcr_img[:, p, :, :] = T[p, 0] * img[:, 0, :, :] + T[p, 1] * img[:, 1, :, :] + T[p, 2] * img[:, 2, :, :] + offset[p] / 255

    
    # output dimension is (batch, channel, height, width)
    return ycbcr_img