# Learning Rain Location Prior for Nighttime Deraining (ICCV2023)

> [**Learning Rain Location Prior for Nighttime Deraining**]()
> 
> Fan Zhang, Shaodi You, Yu Li, Ying Fu
> 
> ICCV 2023

This is the official implementation of the ICCV2023 paper "Learning Rain Location Prior for Nighttime Deraining", by Fan Zhang, Shaodi You, Yu Li, Ying Fu.

[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Learning_Rain_Location_Prior_for_Nighttime_Deraining_ICCV_2023_paper.pdf) | [Data](https://www.kaggle.com/datasets/zkawfanx/gtav-nightrain-rerendered-version)





## News
- **2023.12.03:** Initial release of experimental data.
- **2023.08.10:** Repo created.

## To Do
- [ ] Code release.
- [x] Experimental data release.



## Data Release
The experimental data used in the paper is now publicly available at [Kaggle](https://www.kaggle.com/datasets/zkawfanx/gtav-nightrain-rerendered-version). It is based on [GTAV-NightRain](https://arxiv.org/pdf/2210.04708.pdf) dataset and increase the difficulty by enlarging the rain density.

In this new version, we collected 5000 rainy images paired with 500 clean images for the training set, and 500/50 for the test set. Each clean image corresponds to 10 rainy images. The image resolution is 1920x1080.

#### Note
Please note that this is the very data used in the experiments. 

However, after careful check, we find that there are a few scenes containing misalignments due to operation mistakes during collection. We filter out these scenes and there's about 0.5dB improvement in PSNR, which applys to all evaluated methods.

We plan to re-collect and update these misaligned scenes and provide the updated quantitative results later.


## Citation
If you find this repo useful, please consider citing our papers:
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
