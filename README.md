# Medical image registration with SAM feature

This is the official Pytorch implementation of "SAME++: A Self-supervised Anatomical eMbeddings Enhanced medical image registration framework using stable sampling and regularized transformation". If you have any questions, please contact us at alisonbrielee@gmail.com.

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
First install SAM repo
```
pip install -U openmim
mim install mmcv-full==1.4.7
cd SAM
python -m pip install -e .
```
Then install the repo SAMReg
```
git clone https://github.com/alibaba-damo-academy/same.git
cd same
pip install -e .
```

## Repo structures
This repo contains the following folders under root: 
- demos: demo code to test SAME on new data.
- scripts: training script and training config files.



## Usage example on LearnReg abdomenCTCT dataset
```
wget https://learnreg.oss-cn-beijing.aliyuncs.com/dataset/AbdomenCTCTC.zip
unzip AbdomenCTCTC.zip -d ./data/
 ```


### Prepare dataset
SAMReg provides a base dataset class. To use this dataset class, one needs to organize the folder structure of the dataset as follows with a ‘splits.json’ file attached.  

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
Example files can be found at [data](data/AbdomenCTCT).

### Global Alignment via SAMAffine or SAMCoarse

```
python demos/reg_sam_affine.py -o=results/Affine -d=data/AbdomenCTCT --data_shape 192 160 256 --data_phase=train -e=SAMcoarse -g=0
python demos/reg_sam_coarse.py -o=results/Coarse -d=data/AbdomenCTCT --data_shape 192 160 256 --data_phase=train -e=SAMcoarse -g=0
```

### Train SAM-Deform from scratch
```
python scripts/train.py -o=results -d=data/AbdomenCTCT --data_shape 192 160 256  -e=SAMDeform --train_config=SAMReg/scripts/train_config/train_config_abdomen.py -g=0 --lr=5e-5 --epochs=200 --save_model_period=20
```
Or, users can either use the training pipeline implemented in [scrips](scripts/) or implement their own training procedure. The included training pipeline provides the following features:
- Tensorboard integrated
- Well-structured output folder includes the tensorboard log, trained weights, plotted figures, and a record of the experiment.

To use the included training pipeline, one only needs to implement the following functions in a training config class based on the specific training task.
- make_net(self, inshape)
- make_dataloader(self, data_path)
- _init_losses(self)
- train_kernel(self, model, train_batch, device, epoch, exp_folder)
- debug_kernel(self, model, batch, device, epoch, exp_folder)

Multiple sample training config files can be found at [here](scripts/train_config/).

### Inference based on trained SAM-Deform model and SAM-IO
Users can either use the demo code to test.
```
python eval_NetInsO_abdomen.py
```


## Publication
If you find this repository useful, please cite:

- **SAME++: A Self-supervised Anatomical eMbeddings Enhanced medical image registration framework using stable sampling and regularized transformation**  
Lin Tian*, [Zi Li*](https://alison-brie.github.io/), Fengze Liu, Xiaoyu Bai, Jia Ge, Le Lu, Marc Niethammer, Xianghua Ye, Ke Yan, Daikai Jin. ArXiv 2023 [eprint arXiv:2311.14986](https://arxiv.org/abs/2311.14986 "eprint arXiv:2311.14986")

- **SAME: Deformable Image Registration based on Self-supervised Anatomical Embeddings**  
Fengze Liu, Ke Yan, Adam Harrison, Dazhou Guo, Le Lu, Alan Yuille, Lingyun Huang, Guotong Xie, Jing Xiao, Xianghua Ye, Dakai Jin.
MICCAI 2021 [eprint arXiv:2109.11572](https://arxiv.org/abs/2109.11572 "eprint arXiv:2109.11572")

