# [ICCV'25 Highlight] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection

This repository contains the official PyTorch implementation of the paper **"[Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](https://iccv.thecvf.com/virtual/2025/poster/805)"** by Mian Zou, Nan Zhong, Baosheng Yu, Yibing Zhan, and Kede Ma.

☀️ If you find this work useful for your research, please kindly star our repo and cite our paper! ☀️

- [x] Release [arXiv paper](https://arxiv.org/abs/2507.22824)
- [x] [ICCV 2025 open access](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Bi-Level_Optimization_for_Self-Supervised_AI-Generated_Face_Detection_ICCV_2025_paper.html) 
- [x] Release inference codes
- [x] Release checkpoints
  - [x] OC-detector
  - [x] BC-detector
- [x] Release datasets
  - [x] Testing
  - [x] Training
- [x] Release training codes

## 📁 Datasets

| Datasets & Materials |                                                 Link                                                 |      
|:-------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|
|Training | [Baidu Disk](https://pan.baidu.com/s/126c1zY-zuc16Kkf6SVH43A?pwd=mbnb)
|Testing | [Baidu Disk](https://pan.baidu.com/s/1W9MG-pm-x4Kpkh-HrtHtYA?pwd=5dtw)

We use FDF ([CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/) version) for self-supervised training, and you can download it from the original project page. We also provide the augmented FDF by the proposed artificial face manipulations and the corresponding codes (See [dataset/training_dataset/preprocessing](https://github.com/MZMMSEC/AIGFD_BLO/tree/main/dataset/training_dataset/preprocessing)). After downloading all the necessary files, please put them into the ``data`` folder, with the data structure in the ``data`` folder as 
```
SSL_training_data
├── fdf_ccby2_exif_ordinal_categorical.json
├── FDF
│   ├── data
│   │   ├── face_dataaug_neg
│   │   │   ├── 1000002_0.png
│   │   │   ├──...
│   │   ├── fdf
│   │   │   ├── bounding_box
│   │   │   ├── landmarks
│   │   │   ├── fdf_metainfo.json
│   │   │   ├── images
│   │   │   │   ├── 128
│   │   │   │   │   ├──100000.png
│   │   │   │   │   ├──...
│   │   │   │   ├──...

```


For the test sets we used in the main experiments, we collected them from [DiffusionFace](https://github.com/Rapisurazurite/DiffFace) and [DiFF](https://github.com/xaCheng1996/DiFF), and put them together for testing. If you find them useful, please cite these two papers. We also generate and collect additional AI-synthesized faces produced by other generative models. They are **Diffusion Models**: [DDIM (2020)](https://github.com/ermongroup/ddim), [LDM (2022)](https://github.com/compvis/stable-diffusion), [SDv2.1 (2022)](https://github.com/Stability-AI/stablediffusion), [FreeDoM (2023)](https://github.com/yujiwen/FreeDoM), [HPS (2023)](https://tgxs002.github.io/align_sd_web/), [Midjourney (2023)](https://www.midjourney.com/home), [SDXL (2023)](https://github.com/stability-ai/generative-models), and **GANs**: [ProGAN (2018)](https://github.com/tkarras/progressive_growing_of_gans), [StarGAN (2018)](https://github.com/wkentaro/StarGAN), [StyleGAN2 (2020)](https://github.com/NVlabs/stylegan2), [pi-GAN (2021)](https://github.com/marcoamonteiro/pi-GAN), [VQ-GAN (2021)](https://github.com/KGML-lab/vq-gan).
During testing, please put the dataset in the ``data`` folder, and the data structure is as follows:
```
data
├── celeba-face
│   ├── train.txt
│   ├── test.txt
│   ├── imgs
├── ddim
├── freeDoM
├── hps
├── ldm
├── midjourney
├── sdxl
├── sdv21
├── stargan
├── stylegan2
├── pigan
├── progan
├── vqgan
```

## 🚀 Quick Start

### 1. Installation of base requirements
 - python == 3.8
 - PyTorch == 1.13
 - Miniconda
 - CUDA == 11.7

### 2. Download the pretrained models

|      Model       |                                                               Download                                                                 |
|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| OC-GMM | [Baidu Disk](https://pan.baidu.com/s/19aB91CXXBZEe9X9e3expnQ?pwd=6irr) |
| BC-MLP | [Baidu Disk](https://pan.baidu.com/s/1ti2IjDKX5W-KPax47Mqgtw?pwd=tmrk) |

After downloading these checkpoints, put them into the folder ``pretrained``.

### 3. Inference

**OC-GMM**
```
CUDA_VISIBEL_DEVICES=4 python OC_GMM.py --resume ./pretrained/OC.pth
```

**BC-MLP**
```
CUDA_VISIBEL_DEVICES=4 python BC_MLP.py --resume ./pretrained/BC.pth [--only_gan, --only_diffusion, --only_paper, optional for choosing test sets, including images from GANs, diffusion models, or those mentioned in the paper.]
```

**BC-MLP Complexity**
```
CUDA_VISIBEL_DEVICES=4 python BC_MLP.py --resume ./pretrained/BC.pth --complexity_cal
```


## ⚡ Self-Supervised Training via Bi-Level Optimization
```
CUDA_VISIBLE_DEVICES=5 python SSL_training.py --batch_size 48 --output ./output/train_SSL/ --save_model \
--name FaceCoarse-autol3e4_equalExFa-rn50exifinit_bz48_lr_1e5_FLfidelityV2-u-SM \
--lr 1e-5 --cosineTmax 20 --scheduler cosine --resume_text None \
--weight autol --weights_exif 1.0 --weights_face 1.0 --autol_lr 3e-4 --task face_coarse
```

## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{zou2025bi,
  title={Bi-Level Optimization for Self-Supervised AI-Generated Face Detection},
  author={Zou, Mian and Zhong, Nan and Yu, Baosheng and Zhan, Yibing and Ma, Kede},
  year={2025},
  booktitle={International Conference on Computer Vision},
  pages={18959--18968}
}
```

