from syslog import LOG_SYSLOG
import torch
import torch.nn as nn
import torch.nn.functional as F

from SAMReg.cores.functionals import spatial_transformer, correlation, correlation_split, compose, neg_Jdet_loss, JacboianDet


class SAMInsOpt(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

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
            )
            loss = sampled_cost.mean()
            (loss + reg_loss).backward()
            optimizer.step()


        return param.detach()
    

    def instanceOptimization(self, source, target, pre_align=None):
        if pre_align is not None:
            pre_align = F.interpolate(pre_align, size=source.shape[2:], mode='trilinear', align_corners=True)
            warped  = spatial_transformer(source, pre_align, mode='bilinear', padding_mode="background")
        else:
            warped = source
        # warped = source

        with torch.no_grad():
            # Get SAM feature
            target_fine = self.embed.extract_feat(target)[0]
            source_fine = self.embed.extract_feat(warped)[0]

            grid = (
                F.affine_grid(
                    torch.eye(3, 4).unsqueeze(0).cuda(),
                    source.shape,
                    align_corners=True,
                )
                .permute(0, 4, 1, 2, 3)
                .flip(1)
            )

        disp_init = torch.zeros((1, 3) + source_fine.shape[2:]).cuda()

        fitted_disp = self._instance_optimization(
            disp_init,
            [source_fine.float()],
            [target_fine.float()],
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
            phi = grid + compose(pre_align - grid, disp_smooth)
        else:
            phi = grid + disp_smooth 
        

        warped = spatial_transformer(
            source, phi, mode="bilinear", padding_mode="background"
        )
        return warped, phi