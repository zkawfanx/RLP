# Learning Rain Location Prior for Nighttime Deraining (ICCV2023)

> [**Learning Rain Location Prior for Nighttime Deraining**]()  
> Fan Zhang, Shaodi You, Yu Li, Ying Fu  
> ICCV 2023

![framework](assets/framework.png)

This repository contains the official implementation and experimental data of the ICCV2023 paper "Learning Rain Location Prior for Nighttime Deraining", by Fan Zhang, Shaodi You, Yu Li, Ying Fu.

[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Learning_Rain_Location_Prior_for_Nighttime_Deraining_ICCV_2023_paper.pdf) | [Supp](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhang_Learning_Rain_Location_ICCV_2023_supplemental.pdf) | [Data](https://www.kaggle.com/datasets/zkawfanx/gtav-nightrain-rerendered-version)





## Update
- **2023.12.08:** Code release.
- **2023.12.03:** Initial release of experimental data.
- **2023.08.10:** Repo created.

## To Do
- [ ] Recollect misaligned data.
- [x] Code release.
- [x] Experimental data release.



## Data Release

![example](assets/example.gif)

The experimental data used in the paper is now publicly available at [Kaggle](https://www.kaggle.com/datasets/zkawfanx/gtav-nightrain-rerendered-version). It is based on [GTAV-NightRain](https://arxiv.org/pdf/2210.04708.pdf) dataset and increase the difficulty by enlarging the rain density.

In this new version, we collected 5000 rainy images paired with 500 clean images for the training set, and 500/100 for the test set. Each clean image corresponds to 10/5 rainy images. The image resolution is 1920x1080.

#### Note
Please note that this is the very data used in the experiments. 

However, after checking carefully, we find that there exist a few scenes with misalignments due to operation mistakes during collection. We filter out these scenes and there's about 0.5dB improvement in PSNR, which applys to all evaluated methods.

We plan to re-collect and update these misaligned scenes and provide the updated quantitative results later.



## Requirements
- [x] Python 3.6.13
- [x] Pytorch 1.10.2
- [x] Cudatoolkit 11.3

You can refer to [Uformer](https://github.com/ZhendongWang6/Uformer) and [MPRNet](https://github.com/swz30/MPRNe) for detailed dependency list. Necessary list will be updated later.

## Training
- Download the [Dataset](https://www.kaggle.com/datasets/zkawfanx/gtav-nightrain-rerendered-version) on Kaggle or prepare your own training dataset, then modify the `--train_dir` to corresponding directory.
- Train the model by simply run
```
bash train.sh
```
You can
- Select the Derainig Module (DM) by `--arch`, currently supporting `UNet` and `Uformer_T`.
- Enable the Rain Location Prior Module (RLP) by `--use_rlp`.
- Enable the Rain Prior Injection Module (RPIM) using `_use_rpim`, which should be considered when RLP is used.
- Change other parameters in `options.py`.


## Evaluation
- Prepare your test images or simply test on the downloaded data, by running
```
bash test.sh
```
- Modify `--input_dir` to your `/path/to/test/images` and `--result_dir` for saving results. 
- Modify `--weights` to the model checkpoint you have.
- Modify `--model_name` following the format of `{DM}{_RLP}{_RPIM}`, such as `Uformer_T_RLP_RPIM` in `weights` folder.

### Metrics
To calculate PSNR and SSIM metrics, you can use the Matlab script
```
evaluate_PSNR_SSIM.m
```
or the Python version
```
python evaluate_PSNR_SSIM.py
```
The results produced by `.py` script is slightly different from the `.m` script.


## Checkpoints


## License
MIT license.

CC BY-NC-SA 4.0 for data.

## Bibtex
If you find this repo useful, please give us a star and consider citing our papers:
```
@inproceedings{zhang2023learning,
  title={Learning Rain Location Prior for Nighttime Deraining},
  author={Zhang, Fan and You, Shaodi and Li, Yu and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13148--13157},
  year={2023}
}

@article{zhang2022gtav,
  title={GTAV-NightRain: Photometric Realistic Large-scale Dataset for Night-time Rain Streak Removal},
  author={Zhang, Fan and You, Shaodi and Li, Yu and Fu, Ying},
  journal={arXiv preprint arXiv:2210.04708},
  year={2022}
}
```

## Acknowledgement
The code is re-organized based on [Uformer](https://github.com/ZhendongWang6/Uformer) and [MPRNet](https://github.com/swz30/MPRNe). Thanks for their great works!