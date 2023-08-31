import torch
import torch.nn as nn
import torch.nn.functional as F
from SAMReg.cores.functionals import correlation, spatial_transformer


class ConvBlock(nn.Module):
    """
    A convolutional block including conv, BN, nonliear activiation, residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ndims=3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        batchnorm=False,
        residual=False,
        nonlinear=nn.LeakyReLU(0.2),
    ):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(ConvBlock, self).__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        BatchN = getattr(nn, 'BatchNorm%dd' % ndims)
        self.bn = BatchN(out_channels) if batchnorm else None
        self.nonlinear = nonlinear
        if residual:
            self.residual = Conv(
                in_channels, out_channels, 1, stride=stride, bias=bias
            )
        else:
            self.residual = None

    def forward(self, x):
        x_1 = self.conv(x)
        if self.bn:
            x_1 = self.bn(x_1)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        if self.residual:
            x_1 = self.residual(x) + x_1
        return x_1


class FullyConnectBlock(nn.Module):
    """
    A fully connect block including fully connect layer, nonliear activiation
    """

    def __init__(
        self, in_channels, out_channels, bias=True, nonlinear=nn.LeakyReLU(0.2)
    ):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        :param batchnorm:
        :param residual:
        :param nonlinear:
        """

        super(FullyConnectBlock, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.nonlinear = nonlinear

    def forward(self, x):
        x_1 = self.fc(x)
        if self.nonlinear:
            x_1 = self.nonlinear(x_1)
        return x_1


class CorrLayer(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CorrLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, y):
        return correlation(x, y, self.kernel_size, self.stride)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring. 
    This layer is borrowed from VoxelMorph github repo: https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)

    def forward(self, vec):
        vec = vec * self.scale

        identity = (
            F.affine_grid(
                torch.eye(3, 4).unsqueeze(0).to(vec.device),
                (1,1) + tuple(vec.shape[2:]),
                align_corners=True,
            )
            .permute(0, 4, 1, 2, 3)
            .flip(1)
        )

        for _ in range(self.nsteps):
            vec = vec + spatial_transformer(vec, vec+identity, mode="bilinear")
        return vec


