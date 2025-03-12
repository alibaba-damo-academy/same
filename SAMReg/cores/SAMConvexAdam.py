from syslog import LOG_SYSLOG
import torch
import torch.nn as nn
import torch.nn.functional as F

from SAMReg.cores.functionals import spatial_transformer, correlation, correlation_split, compose, neg_Jdet_loss, JacboianDet
from SAMReg.cores.losses import NCCLoss

class SAMConvexAdam(nn.Module):
    def __init__(self, embed, kernel_size=[6, 3]):
        # super(SAMConvexAdam, self).__init__()
        super().__init__()
        self.embed = embed
        self.kernel_size = kernel_size

    def forward(self, source, target, pre_align=None):
        if pre_align is not None:
            ### Warp source with affine
            # initial_phi = (
            #     F.affine_grid(pre_align, size=source.shape, align_corners=True)
            #     .permute(0, 4, 1, 2, 3)
            #     .flip(1)
            # )
            ### Warp source with coarse/flow
            initial_phi = (
                F.interpolate(pre_align, size=source.shape[2:], mode="trilinear", align_corners=True)
            )
            warped = spatial_transformer(source, initial_phi, mode='bilinear', padding_mode="background")
        else:
            warped = source

        with torch.no_grad():
            # Get SAM feature
            target_fine, target_coarse = self.embed.extract_feat(target)
            source_fine, source_coarse = self.embed.extract_feat(warped)

            # Register on coarse feature
            disp = self._adam_convex(source_coarse, target_coarse, self.kernel_size[0])

            # Update source image
            disp_hr = F.interpolate(
                disp, size=source.shape[2:], mode="trilinear", align_corners=True
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
            warped = spatial_transformer(
                warped,
                grid + disp_hr,
                mode="bilinear",
                padding_mode="background",
            )

            # Get new feature
            source_fine_, source_coarse_ = self.embed.extract_feat(warped)

            # Free memory
            del warped, disp_hr

            source_coarse_ = F.normalize(
                F.interpolate(
                    source_coarse_,
                    size=source_fine.shape[2:],
                    mode="trilinear",
                    align_corners=True
                ), dim=1
            )
            target_coarse = F.normalize(
                F.interpolate(
                    target_coarse,
                    size=source_fine.shape[2:],
                    mode="trilinear",
                    align_corners=True
                ), dim=1
            )

           
            disp_fine = self._adam_convex(
                torch.cat([source_fine_, source_coarse_], dim=1),
                torch.cat([target_fine, target_coarse], dim=1),
                self.kernel_size[1],
            )
            disp = compose(disp, disp_fine)

            disp_hr = F.interpolate(
                disp, size=source.shape[2:], mode="trilinear", align_corners=True
            )

        # Run instance optimization
        with torch.no_grad():
            source_coarse = F.normalize(
                F.interpolate(
                    source_coarse,
                    size=target_fine.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                ),
                1,
            )

        # create optimisable displacement grid
        disp_lr = F.interpolate(
            disp_hr, size=target_fine.shape[2:], mode="trilinear", align_corners=True
        )

        fitted_disp = self._instance_optimization(
            disp_lr,
            [source_coarse.float(), source_fine.float()],
            [target_coarse.float(), target_fine.float()],
        )

        disp_hr = F.interpolate(
            fitted_disp, size=source.shape[2:], mode="trilinear", align_corners=True
        )
        disp_smooth = F.avg_pool3d(
            F.avg_pool3d(
                F.avg_pool3d(disp_hr, 3, padding=1, stride=1), 3, padding=1, stride=1
            ),
            3,
            padding=1,
            stride=1,
        )

        if pre_align is not None:
            phi = grid + compose((initial_phi - grid), disp_smooth)
        else:
            phi = grid + disp_smooth
        warped = spatial_transformer(
            source, phi, mode="bilinear", padding_mode="background"
        )
        return warped, phi


    def _adam_convex(
        self,
        feat_source_coarse,
        feat_target_coarse,
        kernel_size,
    ):
        cost_volume = -correlation_split(
            feat_source_coarse.half(), feat_target_coarse.half(), kernel_size, 1, kernel="ssd"
        )

        disp_soft = self._coupled_convex(cost_volume, kernel_size).float()

        return disp_soft

    # solve two coupled convex optimisation problems for efficient global regularisation
    def _coupled_convex(self, cost, kernel_size):
        b, _, h, w, d = cost.shape
        assert b == 1, "SAMConvexAdam supports registration of one pair of images only!"

        cost = cost[0]
        cost_argmin = torch.argmin(cost, 0)

        disp_mesh_t = (
            F.affine_grid(
                kernel_size * torch.eye(3, 4).cuda().half().unsqueeze(0),
                (1, 1, kernel_size * 2 + 1, kernel_size * 2 + 1, kernel_size * 2 + 1),
                align_corners=True,
            )
            .permute(0, 4, 1, 2, 3)
            .flip(1)
            .reshape(3, -1, 1)
        )
        # Normalize to [-1,1]
        scale = torch.tensor([h - 1, w - 1, d - 1]).view(3, 1, 1).cuda() / 2.0
        disp_mesh_t = disp_mesh_t / scale

        disp_soft = F.avg_pool3d(
            disp_mesh_t.view(3, -1)[:, cost_argmin.view(-1)].reshape(1, 3, h, w, d),
            3,
            padding=1,
            stride=1,
        )

        coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
        for j in range(6):
            ssd_coupled_argmin = torch.zeros_like(cost_argmin)
            with torch.no_grad():
                for i in range(h):
                    coupled = cost[:, i, :, :] + coeffs[j] * (
                        disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1)
                    ).pow(2).sum(0).view(-1, w, d)
                    ssd_coupled_argmin[i] = torch.argmin(coupled, 0)

            disp_soft = F.avg_pool3d(
                disp_mesh_t.view(3, -1)[:, ssd_coupled_argmin.view(-1)].reshape(
                    1, 3, h, w, d
                ),
                3,
                padding=1,
                stride=1,
            )

        return disp_soft


    def _instance_optimization(self, param, feat_source, feat_target):
        param = nn.parameter.Parameter(param)  # B*3*H*W*D

        optimizer = torch.optim.Adam([param], lr=0.05, eps=1e-4)
        grid0 = (
            F.affine_grid(
                torch.eye(3, 4).unsqueeze(0).to(param.device),
                param.shape,
                align_corners=True,
            )
            .permute(0, 4, 1, 2, 3)
            .flip(1)
        )

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        lambda_weight = 1000  # with tps: .5, without:0.7
        for iter in range(50):  # 80
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(
                    F.avg_pool3d(param, 3, stride=1, padding=1),
                    3,
                    stride=1,
                    padding=1,
                ),
                3,
                stride=1,
                padding=1,
            )
            reg_loss = (
                lambda_weight
                * (
                    (disp_sample[0, :, :, 1:, :] - disp_sample[0, :, :, :-1, :]) ** 2
                ).mean()
                + lambda_weight
                * (
                    (disp_sample[0, :, 1:, :, :] - disp_sample[0, :, :-1, :, :]) ** 2
                ).mean()
                + lambda_weight
                * (
                    (disp_sample[0, :, :, :, 1:] - disp_sample[0, :, :, :, :-1]) ** 2
                ).mean()
            )

            phi = grid0 + disp_sample

            feat_dim = feat_source[0].shape[1]
            feat_warped = spatial_transformer(
                torch.cat(feat_source, 1),
                phi,
                mode="bilinear",
                padding_mode="border",
            )

            sampled_cost = -F.cosine_similarity(
                F.normalize(feat_warped[:, :feat_dim].half(), 1),
                feat_target[0].half(),
                1,
            ) - F.cosine_similarity(
                F.normalize(feat_warped[:, feat_dim:].half(), 1),
                feat_target[1].half(),
                1,
            )
            loss = sampled_cost.mean()
            (loss + reg_loss).backward()
            optimizer.step()


        return param.detach()
    

    def Image_Space_instance_optimization(self, param, source, target, lambda_weight=10, epoch=100):
        # delta_phi = torch.nn.parameter.Parameter(torch.zeros((1, 3) + source.shape[2:]).cuda()) # B*3*H*W*D
        # phi = delta_phi + param 

        phi = torch.nn.parameter.Parameter(param)  # B*3*H*W*D
        optimizer = torch.optim.Adam([phi], lr=0.008, eps=1e-4)
        sim = NCCLoss().loss

        grid0 = (
            F.affine_grid(
                torch.eye(3, 4).unsqueeze(0).to(param.device),
                param.shape,
                align_corners=True,
            )
            .permute(0, 4, 1, 2, 3)
            .flip(1)
        )

        # jac = JacboianDet(phi.permute(0, 2, 3, 4, 1)-grid0.permute(0, 2, 3, 4, 1), grid0.permute(0, 2, 3, 4, 1))  # B*H*W*D*3\n"
        # mask = F.interpolate( jac.unsqueeze(0), size=source.shape[2:], mode='nearest') < 0
        # mask = mask.cuda()
        # print(mask.shape)

        # from SAMReg.tools.utils.med import save_to_itk
        # import numpy as np
        # mask_data = np.array(mask.clone().detach().cpu().numpy())+0.0
        # print(mask_data )
        # save_to_itk(mask_data[0,0], "/home/lizi.li/DSWCOPY/data/LiverTestData/results/LiverCTCT/SAME/net_flow_JacMask.nii.gz")

        mask = torch.ones((1, 1) + source.shape[2:]).cuda()

        # print("start")
        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(epoch):  
            optimizer.zero_grad()

            warped = spatial_transformer(source, phi, mode="bilinear", padding_mode="background")

            sim_loss = sim(warped*mask, target*mask)

            # disp_sample = (phi) * mask
            # disp_sample = (phi)

            # reg_loss = (
            #     lambda_weight
            #     * (
            #         (disp_sample[0, :, :, 1:, :] - disp_sample[0, :, :, :-1, :]) ** 2
            #     ).mean()
            #     + lambda_weight
            #     * (
            #         (disp_sample[0, :, 1:, :, :] - disp_sample[0, :, :-1, :, :]) ** 2
            #     ).mean()
            #     + lambda_weight
            #     * (
            #         (disp_sample[0, :, :, :, 1:] - disp_sample[0, :, :, :, :-1]) ** 2
            #     ).mean()
            # )
            det_loss =  lambda_weight * neg_Jdet_loss(phi - grid0, grid0) 

            # loss = sim_loss + det_loss + reg_loss
            loss = sim_loss + det_loss
            # print(sim_loss.item(), reg_loss.item(), det_loss.item())

            loss.backward()
            optimizer.step()
        
        return phi.detach()


    def instanceOptimization(self, source, target, pre_align=None):
        if pre_align is not None:
            pre_align = F.interpolate( pre_align, size=source.shape[2:], mode='trilinear', align_corners=True)
            warped  = spatial_transformer(source, pre_align, mode='bilinear', padding_mode="background")
        else:
            warped = source
        # warped = source

        with torch.no_grad():
            # Get SAM feature
            target_emb = self.embed.extract_feat(target)
            source_emb = self.embed.extract_feat(warped)

            target_fine, target_coarse = target_emb[0], target_emb[1]
            source_fine, source_coarse = source_emb[0], source_emb[1]

            target_coarse = F.normalize(
                F.interpolate(
                    target_coarse,
                    size=source_fine.shape[2:],
                    mode="trilinear",
                    align_corners=True
                ), dim=1
            )
            source_coarse = F.normalize(
                F.interpolate(
                    source_coarse,
                    size=target_fine.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                ), 1,
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

        ### Run instance optimization
        # create optimisable displacement grid
        # if pre_align is not None:
        #     disp_init = F.interpolate(
        #     pre_align, size=source_fine.shape[2:], mode="trilinear", align_corners=True
        # )
        # else:
        #     disp_init = torch.zeros((1, 3) + source_fine.shape[2:]).cuda()

        disp_init = torch.zeros((1, 3) + source_fine.shape[2:]).cuda()

        fitted_disp = self._instance_optimization(
            disp_init,
            [source_coarse.float(), source_fine.float()],
            [target_coarse.float(), target_fine.float()],
        )
        # print(disp_init.shape, source_coarse.shape, source_fine.shape) # [1, 3, 56, 96, 96] [1, 128, 56, 96, 96] [1, 128, 56, 96, 96]

        disp_hr = F.interpolate(
            fitted_disp, size=source.shape[2:], mode="trilinear", align_corners=True
        )

        disp_smooth = F.avg_pool3d(
            F.avg_pool3d(
                F.avg_pool3d(disp_hr, 3, padding=1, stride=1), 3, padding=1, stride=1
            ),
            3,
            padding=1,
            stride=1,
        )

        if pre_align is not None:
            phi = grid + compose(pre_align - grid, disp_smooth)
        else:
            phi = grid + disp_smooth 
        

        warped = spatial_transformer(
            source, phi, mode="bilinear", padding_mode="background"
        )
        return warped, phi