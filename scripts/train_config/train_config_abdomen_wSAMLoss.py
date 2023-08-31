import os
from collections import defaultdict

import torch
from mmcv import Config
from SAMReg.cores.losses import FeatSim, LNCCLoss
from SAMReg.cores.voxelmorph import VoxelMorph
from SAMReg.datasets.BaseDataset import BaseDataset
from SAMReg.tools.interfaces import init_model
from SAMReg.tools.utils.visualize import plot_comparison_at
from SAMReg.cores.layers import CorrLayer
from torch.utils.data import DataLoader

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
import voxelmorph as vxm


class train_config:
    def __init__(self):
        self.sam_config = "../demos/configs/sam/sam_r18_i3d_fpn_1x_multisets_sgd_T_0.5_half_test.py"
        self.sam_weight = "../demos/iter_38000.pth"

        self.embed_cfg = Config.fromfile(self.sam_config)

        self._init_losses()

    def make_net(self, inshape):
        embed, self.cfg = init_model(self.sam_config, self.sam_weight)
        # freeze embedding model
        for param in embed.parameters():
            param.requires_grad = False
        embed.eval()

        # unet architecture
        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]

        model = VoxelMorph(
            inshape=inshape,
            embed=embed,
            nb_unet_features=[enc_nf, dec_nf],
            src_feats=1,
            trg_feats=1,
            sam_feature_level=1,
            sam_feature_dim=256,
            sam_corr = CorrLayer(kernel_size=1, stride=1),
            integrate_step=7
        )
        return model
    
    def make_dataloader(self, data_path, inshape):
        train_loader = DataLoader(
            BaseDataset(data_path, shape=inshape, body_only=True, phase="train", with_mask=True, with_prealign_coarse=True, clamp=(-800,400), cfg=self.embed_cfg), 
            batch_size=1, shuffle=True, drop_last=True)
        debug_loader = DataLoader(
            BaseDataset(data_path, shape=inshape, body_only=True, phase="val", with_mask=True, with_prealign_coarse=True, clamp=(-800,400), cfg=self.embed_cfg), 
            batch_size=1)
        return train_loader, debug_loader
    
    def _init_losses(self):
        self.losses = {
            "sim": [LNCCLoss(win=21).loss, 0.01],
            "reg": [vxm.losses.Grad('l2', loss_mult=2).loss, 10],  
            "SAM_sim": [FeatSim().loss, 0.01]
            }

    def train_kernel(self, model, train_batch, device, epoch, exp_folder):
        (
            source, target, 
            source_info, target_info,
            source_mask, target_mask,
            coarse_phi
        ) = train_batch

        source = source.to(device)
        target = target.to(device)
        coarse_phi = coarse_phi.to(device)
        source_mask = source_mask.to(device)

        warped, warped_sam, target_sam, warped_mask, phi_param = model(source, target, source_label=source_mask, pre_align=coarse_phi)

        # calculate total loss
        loss = 0
        loss_dict = defaultdict(lambda: 0.)

        # compute sim
        mask = warped_mask[:, 1:2].detach()
        mask[mask > 0.8] = 1.0
        mask[mask <= 0.8] = 0.0
        sim_l = self.losses['sim'][0](target[:, 1:2], warped, mask)
        reg_l = self.losses['reg'][0](phi_param, phi_param)


        loss = self.losses['sim'][1] * sim_l + self.losses['reg'][1] * reg_l 
        
        loss_dict['sim'] = sim_l.item()
        loss_dict['reg'] = reg_l.item()
        loss_dict['total_loss'] = loss.item()

        return loss, loss_dict
    
    def debug_kernel(self, model, batch, device, epoch, exp_folder):
        (
            source, target, 
            source_info, target_info,
            source_mask, target_mask,
            pre_align
        ) = batch

        source = source.to(device)
        target = target.to(device)
        pre_align = pre_align.to(device)

        with torch.no_grad():
            warped, phi = model(source, target, pre_align=pre_align, registration=True)
            warped, phi = warped.detach(), phi.detach()
        plot_comparison_at((100, 100, 100), 15, 
                    phi[0].permute(1,2,3,0).cpu().numpy(),
                    source[0,1].cpu().numpy(),
                    warped[0,0].detach().cpu().numpy(),
                    target[0,1].cpu().numpy(),
                    save_to=os.path.join(os.path.join(exp_folder, "records"), 'epoch_%04d.png' % epoch))

