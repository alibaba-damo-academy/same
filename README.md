# Medical image registration with SAM feature

This is the official Pytorch implementation of "SAME++: A Self-supervised Anatomical eMbeddings Enhanced medical image registration framework using stable sampling and regularized transformation".

This is the repo for SAME registration. It contains the following registration methods:
- Non-learning
    - SAM-Affine 
    - SAM-Coarse
    - SAM-InsOpt
- Learning
    - SAM-Deform


# Instructions
## Requirements
- Anaconda3 with python=3.7
- Pytorch=1.9.0
- SAM [[code]](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2) derived from SAM: Self-supervised Learning of Pixel-wise Anatomical Embeddings in Radiological Images (IEEE IMI 2022) [[pdf]](https://ieeexplore.ieee.org/document/9760421/) 

## Installation
First install SAM
```
pip install -U openmim
mim install mmcv-full==1.4.7
cd SAM
python -m pip install -e .
```
Then install SAMReg
```
cd SAME
pip install -e .
```

## Repo structures
This repo contains the following folders under root: 
- demos: demo code to test SAME on new data.
- scripts: traing script and training config files.


## Datasets
SAMReg provides a base dataset class. To use this dataset class, one needs to organize the folder structure of the dataset as follows with a ‘splits.json’ file attached.  

Example preprocess file can be found at [demos](demos/data_preprocess_example.py).
```
root/
|-- case1/
|   |-- case1_image.nii.gz
|   |-- case1_label.nii.gz
|   |-- case1_mask.nii.gz
|-- case2/
...
|-- pre_align/
|   |-- case1_case2_affine_inv.npy
|   |-- case1_case2_affine.npy
|   |-- case1_case2_coarse_phi.npy
|   |-- case1_case3_affine_inv.npy
|   |-- case1_case3_affine.npy
|   |-- case1_case3_coarse_phi.npy

```


## Train
Users can either use the training pipeline implemented in [scrips](scripts/) or implement their own training procedure. The included training pipeline provides the following features:
- Tensorboard integrated
- Well structed output folder include the tensorboard log, trained weights, plotted figures and a record of the experiment.

To use the included training pipeline, one only needs to implement the following functions in a training config class based on the specific training task.
- make_net(self, inshape)
- make_dataloader(self, data_path)
- _init_losses(self)
- train_kernel(self, model, train_batch, device, epoch, exp_folder)
- debug_kernel(self, model, batch, device, epoch, exp_folder)

Multiple sample training config files can be found at [here](scripts/train_config/).

## Inference
Users can either use the demo code to test.
```
python reg_sam_affine.py -o=results -d=data/processed/AbdomenCTCT --data_shape 192 160 256 --data_phase=val -e=SAME/affine -g=3 
python reg_sam_coarse.py -o=results -d=data/processed/AbdomenCTCT --data_shape 192 160 256 --data_phase=val -e=SAME/coarse -g=3 
python eval_NetInsO_abdomen.py
```


## Publication
If you find this repository useful, please cite:

- **SAME++: A Self-supervised Anatomical eMbeddings Enhanced medical image registration framework using stable sampling and regularized transformation**  
Lin Tian*, [Zi Li*](https://alison-brie.github.io/), Fengze Liu, Xiaoyu Bai, Jia Ge, Le Lu, Marc Niethammer, Xianghua Ye, Ke Yan, Daikai Jin. ArXiv 2023 [eprint arXiv:2311.14986](https://arxiv.org/abs/2311.14986 "eprint arXiv:2311.14986")

- **SAME: Deformable Image Registration based on Self-supervised Anatomical Embeddings**  
Fengze Liu, Ke Yan, Adam Harrison, Dazhou Guo, Le Lu, Alan Yuille, Lingyun Huang, Guotong Xie, Jing Xiao, Xianghua Ye, Dakai Jin.
MICCAI 2021 [eprint arXiv:2109.11572](https://arxiv.org/abs/2109.11572 "eprint arXiv:2109.11572")

