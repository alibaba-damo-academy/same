import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from SAMReg.tools.interfaces import find_point_in_vol, get_embedding, find_point_in_vol_stable
from SAMReg.tools.utils.med import (
    get_identity,
    points_to_canonical_space_cuda,
    select_points_with_stride,
)
from SAMReg.cores.functionals import affine_solver


class SAMCoarse(nn.Module):
    def __init__(self, embed, cfg, with_stable_selection=True):
        super(SAMCoarse, self).__init__()
        self.embed = embed
        self.cfg = cfg
        if with_stable_selection:
            self.match_points_fn = find_point_in_vol_stable
        else:
            self.match_points_fn = find_point_in_vol

    def forward(
        self,
        source,
        target,
        source_mask,
        target_mask,
        threshold=0.7,
        stride=8,
        batch_size=1000,
        ite=100,
        lr=5e-2
    ):
        shape = source.shape[2:]
        device = source.device
        with torch.no_grad():
            source_emb = get_embedding(self.embed, source)
            target_emb = get_embedding(self.embed, target)
            source_pts = torch.tensor(select_points_with_stride(np.array(shape), stride, source_mask[0, 0]), device=device)

            source_pts, target_pts, scores = self.match_points_fn(
                source_emb, target_emb, source_pts, self.cfg, batch_size=batch_size
            )
            
            idx = scores > threshold
            source_pts = source_pts[idx]
            target_pts = target_pts[idx]
            num_pts = source_pts.shape[0]
            
            idx = (
                (target_mask[0, 0, target_pts[:, 0], target_pts[:, 1], target_pts[:, 2]] *
                source_mask[0, 0, source_pts[:, 0], source_pts[:, 1], source_pts[:, 2]]) == 1
            )
            source_pts = source_pts[idx]
            target_pts = target_pts[idx]
            num_pts = source_pts.shape[0]
            print(f"{num_pts} are selected.")

            # Transform points to canonical coords [-1, 1]
            target_pts = points_to_canonical_space_cuda(target_pts, torch.tensor(shape, device=target_pts.device))
            source_pts = points_to_canonical_space_cuda(source_pts, torch.tensor(shape, device=target_pts.device))

            target_pts = F.pad(target_pts.flip(dims=[1]), (0,1), value=1).float()
            source_pts = F.pad(source_pts.flip(dims=[1]), (0,1), value=1).float()

            affine_matrix, affine_matrix_inv = affine_solver(source_pts, target_pts)
        coarse_phi = self._register_coarse(
            source_pts, target_pts, affine_matrix_inv, shape, stride*2, ite, lr
        )

        return (
            affine_matrix,
            affine_matrix_inv,
            coarse_phi,
        )

    def _register_coarse(
        self, source_pts, target_pts, affine_matrix, shape, stride, epochs=800, lr=5e-2
    ):
        # Optimize coarse field
        print("Start optimizing coarse field...")
        affine_grids = F.affine_grid(
            affine_matrix.unsqueeze(0),
            size=[1, 1] + [int(i / stride) for i in shape],
            align_corners=True,
        )
        disp = torch.zeros_like(
            affine_grids, device=source_pts.device, requires_grad=True
        )

        opt = torch.optim.Adam(params=[disp], lr=lr)

        identity = torch.tensor(
            get_identity([int(i / stride) for i in shape], True), device=source_pts.device
        ).unsqueeze(0)
        affine_disp = affine_grids - identity

        def _compose(disp1, disp2):
            """
            (identity+disp1) \circ (identity + disp2)
            return tensor 1x3xDxWxH
            """
            phi2 = identity + disp2
            return F.grid_sample(
                disp1.permute(0, 4, 1, 2, 3),
                phi2,
                mode="bilinear",
                align_corners=True,
                padding_mode="zeros",
            ) + phi2.permute(0, 4, 1, 2, 3)

        def _diff_reg(phi, norm='l2'):
            x_ = phi[:, 1:, :-1, :-1] - phi[:, :-1, :-1, :-1]
            y_ = phi[:, :-1, 1:, :-1] - phi[:, :-1, :-1, :-1]
            z_ = phi[:, :-1, :-1, 1:] - phi[:, :-1, :-1, :-1]

            if norm == 'l2':
                return (
                    torch.mean(x_**2) + torch.mean(y_**2) + torch.mean(z_**2)
                ) / 3.0
            else:
                return (
                    torch.mean(torch.abs(x_) + torch.abs(y_) + torch.abs(z_))
                ) / 3.0

        for i in range(epochs):
            opt.zero_grad()
            warped_pts_ = F.grid_sample(
                _compose(affine_disp, disp),
                target_pts[:, :3].unsqueeze(0).unsqueeze(1).unsqueeze(1),
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )[0, :, 0, 0].permute(1, 0)
            loss = torch.mean(
                torch.sum((warped_pts_ - source_pts[:, :3]) ** 2, dim=1)
            ) + 0.04 * _diff_reg(disp, norm='l2')
            loss.backward()
            opt.step()

        return _compose(affine_disp, disp.detach()).flip(1)
