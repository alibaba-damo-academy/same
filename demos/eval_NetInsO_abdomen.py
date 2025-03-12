import os
import time
import scipy.io as sio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from SAMReg.tools.utils.med import compute_folding_gpu, compute_dice_gpu, compute_dice_labels, get_SDlogJ_official, get_SDlogJ_gpu
from SAMReg.tools.utils.general import path_import, make_dir
from SAMReg.cores.functionals import spatial_transformer
from SAMReg.cores.SAMConvexAdam import SAMConvexAdam
from SAMReg.tools.interfaces import init_model
from datetime import datetime
from SAMReg.tools.utils.general import path_import, make_dir
from SAMReg.datasets.BaseDataset import BaseDataset
from SAMReg.cores.functionals import spatial_transformer, correlation, correlation_split, compose


gpu_id = 0
device = torch.cuda.set_device(gpu_id) if torch.cuda.is_available() else 'cpu'
device = torch.cuda.current_device()

inshape = [192, 160, 256] # AbdomenCT

data_folder = "data/AbdomenCTCT"
model_path = "results/AbdomenCTCT/SAME/deform/vm/unsupervised/svf/ncc/2023_08_24_16_34_14/checkpoints/final.pt" 


exp_path = "/".join(model_path.split("/")[:-2])
# Create experiment folder
timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
exp_folder = os.path.join(output_path, timestamp)
make_dir(exp_folder)
print(f"The experiment is recorded in {exp_folder}")

# Load network and dataset
train_config = path_import(f"{exp_path}/train_config.py").train_config()
model = train_config.make_net(inshape[::-1])

# load weight
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

dataset = BaseDataset(data_folder, shape=inshape, body_only=True, with_mask=True, with_label=True, phase="val", with_prealign_coarse=True, clamp=(-800,400), cfg=train_config.embed_cfg)
test_loader = DataLoader(
        dataset, 
        batch_size=1)

dices_before = []
dices_after = []
foldings = []
runnningTime = []
flow = {}

labels = list(range(1, 14))
dice_vals = np.zeros((len(labels), 45))
k = 0
save_file = f"{exp_folder}/SAM_Deform_Dices.mat"
SDlogJs_ins = []

### stage one ###
with torch.no_grad():
    for source, target, source_info, target_info, source_mask, target_mask, pre_align, source_label, target_label in test_loader:

        source = source.to(device)
        target = target.to(device)
        pre_align = pre_align.to(device)
        
        # run inputs through the model to produce a warped image and flow field
        start = time.time()
        warped, phi = model(source, target, pre_align=pre_align, registration=True)
        end = time.time()
        runnningTime.append(end - start)
        print(f"Running time: {end - start}")
        
        warped, phi = warped.detach(), phi.detach()

        flow[source_info['case_id'][0]+ target_info['case_id'][0]] = phi.cpu().numpy()

        foldings.append(compute_folding_gpu(phi[0].permute(1,2,3,0), target_mask[0, 0].to(device)).cpu())
        print(f"Foldings(%): {compute_folding_gpu(phi[0].permute(1,2,3,0), target_mask[0, 0].to(device)).cpu()}")

        with torch.no_grad():
           
            warped_label  = spatial_transformer(
                source_label.to(device).float(),
                phi,
                mode='nearest'
            )
           
        val_before, _ = compute_dice_labels(source_label[0, 0].cpu().numpy(), target_label[0, 0].cpu().numpy())
        val_after, _ = compute_dice_labels(warped_label[0, 0].cpu().numpy(), target_label[0, 0].cpu().numpy())
        dices_before.append(np.mean(val_before))
        dices_after.append(np.mean(val_after))
        print(f"Before registration: {np.mean(val_before)} | After registration: {np.mean(val_after)}")
        
        if len(val_after)==13:
            dice_vals[:, k] = val_after
            k += 1



print(f"Mean Running time: {np.mean(runnningTime)}")
print(f"Mean Foldings(%): {torch.tensor(foldings).mean()}")
print(f"Mean Before registration: {torch.tensor(dices_before).mean()} | After registration: {torch.tensor(dices_after).mean()}")
print(f"{torch.sum(torch.where(torch.tensor(dices_before)<torch.tensor(dices_after), 1., 0.))} cases are imrpoved.")
sio.savemat(save_file, {'dice_vals': dice_vals, 'labels': labels})

# exit()


### stage two ###
sam_config = "SAMReg/demos/configs/sam/sam_r18_i3d_fpn_1x_multisets_sgd_T_0.5_half_test.py"
sam_weight = "SAM/iter_38000.pth"
embed, cfg = init_model(sam_config, sam_weight)
for param in embed.parameters():
    param.requires_grad = False
embed.eval()
SAMmodel = SAMConvexAdam(embed=embed).cuda()

dices_before_ins = []
dices_after_ins = []
foldings_ins = []
runnningTime_ins = []

dice_vals_ins = np.zeros((len(labels), 45))
k = 0
save_file_ins = f"{exp_folder}/SAM_InsOpt_Dices.mat"



for source, target, source_info, target_info, source_mask, target_mask, pre_align, source_label, target_label in test_loader:

    source = source[:, 0:1, :, :, :].to(device)
    target = target[:, 0:1, :, :, :].to(device)
    pre_align =  torch.from_numpy(flow[source_info['case_id'][0]+ target_info['case_id'][0]]).to(device)

    source_ = spatial_transformer(source, pre_align, padding_mode="background")

    start = time.time()
    warped, phi = SAMmodel.instanceOptimization(source_, target)
    end = time.time()
    runnningTime_ins.append(end - start)
    print(f"Running time: {end - start}")

    with torch.no_grad():
        
        source_label_  = spatial_transformer(
                source_label.to(device).float(),
                pre_align,
                mode='nearest'
        )

        grid = (
                F.affine_grid(
                    torch.eye(3, 4).unsqueeze(0).cuda(),
                    source.shape,
                    align_corners=True,
                )
                .permute(0, 4, 1, 2, 3)
                .flip(1)
            )

        phi = grid + compose((pre_align - grid), phi - grid)

        foldings_ins.append(compute_folding_gpu(phi[0].permute(1,2,3,0), target_mask[0, 0].to(device)).cpu())
        print(f"Foldings(%): {compute_folding_gpu(phi[0].permute(1,2,3,0), target_mask[0, 0].to(device)).cpu()}")


        warped_label  = spatial_transformer(
                source_label.to(device).float(),
                phi,
                mode='nearest'
        )

    val_before, _ = compute_dice_labels(source_label_[0, 0].cpu().numpy(), target_label[0, 0].cpu().numpy())
    val_after, _ = compute_dice_labels(warped_label[0, 0].cpu().numpy(), target_label[0, 0].cpu().numpy())
    dices_before_ins.append(np.mean(val_before))
    dices_after_ins.append(np.mean(val_after))
    print(f"Before registration: {np.mean(val_before)} | After registration: {np.mean(val_after)}")
    if len(val_after)==13:
        dice_vals_ins[:, k] = val_after
        k += 1
    

print(f"-------------------------------------DeformNet on Abdomen-------------------------------------------")
print(f"Mean Running time: {np.mean(runnningTime)}")
print(f"Mean Foldings(%): {torch.tensor(foldings).mean()}")
print(f"Mean Before registration: {torch.tensor(dices_before).mean()} | After registration: {torch.tensor(dices_after).mean()}")
print(f"{torch.sum(torch.where(torch.tensor(dices_before)<torch.tensor(dices_after), 1., 0.))} cases are imrpoved.")
sio.savemat(save_file, {'dice_vals': dice_vals, 'labels': labels})

print(f"--------------------------------------InsOpt on Abdomen--------------------------------------------")
print(f"Mean ins Running time: {np.mean(runnningTime_ins)}")
print(f"Mean ins Foldings(%): {torch.tensor(foldings_ins).mean()}")
print(f"Mean ins Before registration: {torch.tensor(dices_before_ins).mean()} | ins After registration: {torch.tensor(dices_after_ins).mean()}")
print(f"{torch.sum(torch.where(torch.tensor(dices_before_ins)<torch.tensor(dices_after_ins), 1., 0.))} cases are imrpoved.")
sio.savemat(save_file_ins, {'dice_vals': dice_vals_ins, 'labels': labels})
