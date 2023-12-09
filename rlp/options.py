import argparse

def parse_options():
    """docstring for training configuration"""
    parser = argparse.ArgumentParser(description='Image deraining training on GTAV-NightRain')

    # args for arch selection
    parser.add_argument('--mode', type=str, default ='deraining',  help='image restoration mode')
    parser.add_argument('--arch', type=str, default ='Uformer_B',  help='archtechture')
    parser.add_argument('--use_rlp', action='store_true', default=False, help='whether to use RLP')
    parser.add_argument('--use_rpim', action='store_true', default=False, help='whether to use RPIM')

    # args for training
    parser.add_argument('--train_dir', type=str, default ='/path/to/train/data',  help='dir of train data')
    parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--train_workers', type=int, default=8, help='train_dataloader workers')
    parser.add_argument('--gpu', type=str, default='0', help='GPUs')

    parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
    parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
    parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay of')        
    parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
    parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup')

    # args for logging
    parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
    parser.add_argument('--save_images', action='store_true',default=False)
    parser.add_argument('--env', type=str, default ='_',  help='env')
    parser.add_argument('--dataset', type=str, default ='GTAV-NightRain')
    parser.add_argument('--checkpoint', type=int, default=10, help='epochs to save checkpoint')

    # args for resuming training
    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--pretrain_weights',type=str, default='./log/Uformer_B/models/model_best.pth', help='path of pretrained_weights')

    # args for Uformer
    parser.add_argument('--dd_in', type=int, default=3, help='dd_in')
    parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
    parser.add_argument('--embed_dim', type=int, default=16, help='dim of emdeding features')
    parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
    parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
    parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
    parser.add_argument('--modulator', action='store_true', default=False, help='multi-scale modulator')
    
    return parser
