python train.py --arch Uformer_T \
                --batch_size 4 \
                --gpu 0 \
                --train_ps 256 \
                --train_dir /path/to/train/data \
                --save_dir ./logs \
                --dataset GTAV-NightRain \
                --warmup \
                --use_rlp \
                --use_rpim