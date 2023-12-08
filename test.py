import argparse
import os
from models import model_utils
import torch
import numpy as np
import cv2
from dataset import DatasetTest
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import expand2square

parser = argparse.ArgumentParser(description='Image deraining inference on GTAV-NightRain')

parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--input_dir', default='/path/to/test/data', type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='/path/to/results', type=str, help='Directory for results')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')

parser.add_argument('--model_name', default='UNet_RLP_RPIM', type=str, help='arch')
parser.add_argument('--weights', default='/path/to/weights', type=str, help='Path to weights')

# args only for Uformer
parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
parser.add_argument('--embed_dim', type=int, default=16, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--query_embed', action='store_true', default=False, help='query embedding for the decoder')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

parser.add_argument('--tile', type=bool, default=False, help='whether to tile for test image of large size')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


if __name__ == "__main__":
    args.arch = ''.join([x if x in args.model_name else '' for x in ['UNet', 'Uformer_T']])
    args.use_rlp  = 'RLP'  in args.model_name
    args.use_rpim = 'RPIM' in args.model_name

    model_restoration = model_utils.get_arch(args)

    model_utils.load_checkpoint(model_restoration,args.weights)    
    print("===>Testing using weights: ",args.weights)
    
    model_restoration.cuda()
    model_restoration.eval()
    
    test_dataset = DatasetTest(args.input_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)
    
    result_dir = os.path.join(args.result_dir, args.arch)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for i, (input, filename) in enumerate(tqdm(test_loader)):
            input = input.cuda()
            
            if not args.tile:
                if 'Uformer' in args.arch:
                    b, _, h, w = input.size()
                    # Uformer accepts squared inputs
                    if not args.tile:
                        input, mask = expand2square(input)
                    
                restored, _ = model_restoration(input)

                if 'Uformer' in args.arch:
                    restored = torch.masked_select(restored, mask.bool()).reshape(b, 3, h, w)
                
            else:
                b, _, h, w = input.size()
                # for batch processing or large images, tiling it
                # currently used for large Uformer on GTAV-NightRain data
                tiles = []
                masks = []
                tile, mask = expand2square(input[:,:,:,:1280], factor=128)
                tiles.append(tile)
                masks.append(mask)
                tile, mask = expand2square(input[:,:,:,-1280:], factor=128)
                tiles.append(tile)
                masks.append(mask)

                restored = []
                for i in range(len(tiles)):
                    tile_restored, _ = model_restoration(tiles[i])
                    
                    tile_restored = torch.masked_select(tile_restored,(masks[i].bool())).reshape(b,3,h,1280)
                    restored.append(tile_restored)

                restored = torch.cat([restored[0][:,:,:,:960],restored[1][:,:,:,-960:]],3)
            
            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().numpy()
            for batch in range(len(restored)):
                restored_img = restored[batch]
                restored_img = np.uint8(restored_img * 255)
                restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(result_dir, filename[batch] + '.png'), restored_img)