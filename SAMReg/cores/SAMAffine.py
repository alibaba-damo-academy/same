import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from SAMReg.tools.interfaces import find_point_in_vol, get_embedding, find_point_in_vol_stable
from SAMReg.tools.utils.med import (
    points_to_canonical_space_cuda,
    select_points_with_stride,
)
from SAMReg.cores.functionals import affine_solver


class SAMAffine(nn.Module):
    """ """

    def __init__(self, embed, cfg, with_stable_selection=True):
        super(SAMAffine, self).__init__()
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
        batch_size=1000
    ):
        assert (
            source.shape[0] == 1
        ), "SAMAffine only supports one pair of images per call."
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
            print(f"{num_pts} are selected.")

            print("Remove points that are outside of mask...")
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

            # Extend to homogeneous coords
            # The reason we need to flip along dim 1 is that we want the computed affine
            # be aligned with affine_grid in pytorch. 
            # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AffineGridGenerator.cpp
            # shows that the grids is computed via matrix multiplication between kji grids and theta.
            target_pts = F.pad(target_pts.flip(dims=[1]), (0,1), value=1).float()
            source_pts = F.pad(source_pts.flip(dims=[1]), (0,1), value=1).float()

            affine_matrix, affine_matrix_inv = affine_solver(source_pts, target_pts)

        return (
            affine_matrix.cpu().numpy(),
            affine_matrix_inv.cpu().numpy(),
        )
