# Boost Video Frame Interpolation via Motion Adaptation (BMVC 2023 Oral)

This repository contains the official PyTorch implementation of VFI_Adapter: https://arxiv.org/abs/2306.13933/

## Some Information
[Project Page](https://haoningwu3639.github.io/VFI_Adapter_Webpage/)  $\cdot$ [PDF Download](https://arxiv.org/abs/2306.13933/) $\cdot$ [Checkpoints](https://drive.google.com/file/d/1NSPgTwQZMniGmMG-jRWFR2YGzdCThiiU/view?usp=drive_link)

## Requirements
This code has been tested with PyTorch 1.12 and CUDA 11.1. It should also be compatible with higher versions of PyTorch and CUDA. Several essential dependencies are as follows:
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.12](https://pytorch.org/)
- torchvision == 0.13.1
- cudatoolkit == 11.3.1
- cupy-cuda11x == 11.6.0

A suitable [conda](https://conda.io/) environment named `vfi_adapter` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate vfi_adapter
```

## Dataset Preparation

Following [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) and [VFIT](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer), we evaluate our proposed method on [Vimeo90K](http://toflow.csail.mit.edu/), [DAVIS](https://davischallenge.org/), and [SNU-FILM](https://myungsub.github.io/CAIN/) datasets.

If you want to train and benchmark our method, please download [Vimeo90K-Triplet](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip), [Vimeo90K-Septuplet](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip), [DAVIS](https://www.dropbox.com/s/9t6x7fi9ui0x6bt/davis-90.zip?dl=0), and [SNU-FILM](https://www.dropbox.com/s/32wpcpt5izkhoh8/snufilm-test.zip?dl=0). You can place the downloaded datasets in `./Dataset/` folder, where the index of frames has been given.

## Pre-trained Models Preparation

Our proposed plug-in Adapter is trained based on three different pre-trained VFI models, so you need to download the pre-trained models and put them into the corresponding directory for initialization. The pre-trained checkpoints can be downloaded from: [RIFE](https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view?usp=sharing), [IFRNet](https://www.dropbox.com/sh/hrewbpedd2cgdp3/AADbEivu0-CKDQcHtKdMNJPJa?dl=0), [UPRNet](https://github.com/srcn-ivl/UPR-Net/tree/master/checkpoints). Specially, for IFRNet and UPRNet, we use IFRNet_large and UPRNet-LARGE as backbones.

## Training

With the pre-trained backbones, you can freeze their parameters and train our plug-in Adapter now.

For RIFE_adapter, you can train via:
```
cd RIFE_adapter
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --world_size=2
```
For IFRNet_adapter, you can train via:
```
cd IFRNet_adapter
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --world_size=2
```
For UPRNet_adapter, you can train via:
```
cd UPRNet_adapter
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --world_size=2
```

## Benchmark
After training the plug-in Adapter based on pre-trained backbones, you can run benchmark tests in each subdirectory, here we take IFRNet as an example:
```
cd IFRNet_adapter
CUDA_VISIBLE_DEVICES=0 python benchmark/Vimeo90K_sep.py
CUDA_VISIBLE_DEVICES=0 python benchmark/DAVIS.py
CUDA_VISIBLE_DEVICES=0 python benchmark/SNU_FILM.py
```
In each script, there is a hyperparameter `adap_step`, that controls the test-time adaptation steps of the model. The default number is set to 10.

NOTE: If you want to reproduce the results of ene-to-end adaptation, you should load the original pre-trained backbone models and adapt all parameters. In addition, considering that the gradient descent of each adaptation has a certain degree of randomness, multiple experiments are expected to achieve desired results.

## TODO
- [x] Data Preparation
- [x] Model Code
- [x] Training Code
- [x] Benchmark Code
- [x] Release Checkpoints

## Citation
If you use this code for your research or project, please cite:
 
	@article{wu2023boost,
      title={Boost Video Frame Interpolation via Motion Adaptation}, 
      author={Haoning Wu and Xiaoyun Zhang and Weidi Xie and Ya Zhang and Yanfeng Wang},
      booktitle={British Machine Vision Conference (BMVC)},
      year={2023},
	}

## Acknowledgements
Many thanks to the code bases from [RIFE](https://github.com/megvii-research/ECCV2022-RIFE), [IFRNet](https://github.com/ltkong218/IFRNet), [UPRNet](https://github.com/srcn-ivl/UPR-Net).

## Contact
If you have any question, please feel free to contact haoningwu3639@gmail.com.
