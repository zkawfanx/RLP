### python script to evaluate PSNR and SSIM following evaluate_PSNR_SSIM.m from MPRNet
### results may be slightly different from Matlab script

import os
import torch
from torchvision.io import read_image
import kornia
from utils import rgb_to_ycbcr
from tqdm import tqdm

# Ensure CUDA is available for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    root_dir = '/path/to/results'
    datasets = ['']
    methods = ['UNet']

    for dataset in datasets:
        print(dataset)
        for method in methods:
            file_path = os.path.join(root_dir, dataset, method)

            image_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(('.jpg', '.png'))]
            gt_files = [image_file.replace(method, 'gt')[:-7]+'.png' for image_file in image_files]

            total_psnr = 0.0
            total_ssim = 0.0
            img_num = len(image_files)

            for (img_file, gt_file) in tqdm(zip(image_files, gt_files), 0):
                input_img = read_image(img_file).float().unsqueeze(0)
                gt_img = read_image(gt_file).float().unsqueeze(0)

                input_img, gt_img = input_img.to(device), gt_img.to(device)
                # get the Y channel from YCbCr
                input_img, gt_img = rgb_to_ycbcr(input_img)[:,0,:,:], rgb_to_ycbcr(gt_img)[:,0,:,:]


                psnr_val = kornia.metrics.psnr(input_img.unsqueeze(1), gt_img.unsqueeze(1), max_val=1.0)
                total_psnr += psnr_val

                ssim_val = kornia.metrics.ssim(input_img.unsqueeze(1), gt_img.unsqueeze(1), window_size=11, max_val=1.0).mean()
                total_ssim += ssim_val
                # print(psnr_val, ssim_val, img_file)
                

            avg_psnr = total_psnr / img_num
            avg_ssim = total_ssim / img_num
            print(f'For {method}, PSNR: {avg_psnr}, SSIM: {avg_ssim}')

if __name__ == "__main__":
    main()
