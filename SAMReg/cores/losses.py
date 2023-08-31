import torch
import torch.nn
import torch.nn.functional as F


class FeatSim:
    """
    Similarity measure between features.
    """

    def __init__(self) -> None:
        self.cosSim = torch.nn.CosineSimilarity(dim=1)

    def loss(self, x, y, mask=None):
        if mask is not None:
            return 1 - torch.sum(self.cosSim(x, y) * mask)/mask.sum()
        else:
            return torch.mean(1 - self.cosSim(x, y))


class NCCLoss:
    """
    A implementation of the normalized cross correlation (NCC)
    """

    def loss(self, x, y, mask=None):
        x = x.contiguous().view(x.shape[0], -1)
        y = y.contiguous().view(y.shape[0], -1)
        if mask is not None:
            mask = mask.view(mask.shape[0], -1)
            x = x * mask
            y = y * mask
            area = mask.sum(1)
            input_minus_mean = x - x.sum(1, keepdim=True) / area.unsqueeze(-1) + 1e-10
            target_minus_mean = y - y.sum(1, keepdim=True) / area.unsqueeze(-1) + 1e-10
            nccSqr = (
                (input_minus_mean * target_minus_mean * mask).sum(1) / area
            ) / torch.sqrt(
                ((input_minus_mean**2 * mask).sum(1) / area)
                * ((target_minus_mean**2 * mask).sum(1) / area)
            )
        else:
            input_minus_mean = x - torch.mean(x, 1).view(x.shape[0], 1) + 1e-10
            target_minus_mean = y - torch.mean(y, 1).view(y.shape[0], 1) + 1e-10
            nccSqr = ((input_minus_mean * target_minus_mean).mean(1)) / torch.sqrt(
                ((input_minus_mean**2).mean(1)) * ((target_minus_mean**2).mean(1))
            )
        nccSqr = nccSqr.mean()

        return 1 - nccSqr


class LNCCLoss:
    def __init__(self, win=9):
        self.win_size = win**3
        self.mean_op = torch.nn.AvgPool3d(
            win, stride=1, padding=win // 2, count_include_pad=False
        )

    def loss(self, I, J, mask=None):
        I2 = torch.mul(I, I)
        J2 = torch.mul(J, J)
        IJ = torch.mul(I, J)
        I_mean = self.mean_op(I)
        J_mean = self.mean_op(J)
        I2_mean = self.mean_op(I2)
        J2_mean = self.mean_op(J2)
        IJ_mean = self.mean_op(IJ)
        cross = IJ_mean - I_mean * J_mean
        I_var = I2_mean - I_mean * I_mean
        J_var = J2_mean - J_mean * J_mean
        cc = (cross**2) / (I_var * J_var + 1e-5)
        if mask is not None:
            return 1 - torch.sum(cc * mask) / mask.sum()
        else:
            return 1 - torch.mean(cc)


class MSELoss:
    def loss(self, x, y):
        return torch.mean((x - y) ** 2)


class AffineReg:
    def loss(self, affine_param, sched="l2"):
        """
        compute regularization loss of  affine parameters,
        l2: compute the l2 loss between the affine parameter and the identity parameter
        det: compute the determinant of the affine parameter, which prefers to rigid transformation
        :param sched: 'l2' , 'det'
        :return: the regularization loss on batch
        """
        weight_mask = torch.ones(4, 3).cuda()
        bias_factor = 1.0
        weight_mask[3, :] = bias_factor
        weight_mask = weight_mask.view(-1)
        if sched == "l2":
            return torch.sum((affine_param) ** 2 * weight_mask, dim=1).mean()
        elif sched == "det":
            mean_det = 0.0
            for i in range(affine_param.shape[0]):
                affine_matrix = affine_param[i, :9].contiguous().view(3, 3)
                mean_det += torch.det(affine_matrix)
            return mean_det


class DiceLoss:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


import numpy as np
import torch.nn as nn


def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor(
        [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]]
    ).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = (x > y).view(-1) & (dist == 2).view(-1)

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[
        torch.arange(12) * 27
        + idx_shift1[:, 0] * 9
        + idx_shift1[:, 1] * 3
        + idx_shift1[:, 2]
    ] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[
        torch.arange(12) * 27
        + idx_shift2[:, 0] * 9
        + idx_shift2[:, 1] * 3
        + idx_shift2[:, 2]
    ] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(
        rpad2(
            (
                F.conv3d(rpad1(img), mshift1, dilation=dilation)
                - F.conv3d(rpad1(img), mshift2, dilation=dilation)
            )
            ** 2
        ),
        kernel_size,
        stride=1,
    )

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind