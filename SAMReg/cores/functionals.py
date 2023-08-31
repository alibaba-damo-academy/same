import pathlib
from typing import Tuple
import torch
import torch.nn.functional as F
from SAMReg.tools.utils.med import compute_folding_gpu

def phi_to_grid(phi):
    """
    Transform phi to pytorch grid.

    Args:
        phi (tensor): BxDxWxHx3. The last dim stores the corresponding
        coordinate (d,w,h). The coordinate is in canonical space [-1,1].
    """
    return torch.flip(phi, [-1])


def grid_to_phi(grid):
    """
    Transform pytorch grid to phi.

    Args:
        grid (tensor): BxDxWxHx3. The last dim stores the corresponding
        coordinate in order of (h,w,d). The coordinate is in canonical space [-1,1].
    """
    return torch.flip(grid, [-1])


def custom_grid_sampler(
    src: torch.tensor, grids: torch.tensor, mode="bilinear", padding_mode="zeros"
) -> torch.tensor:
    """
    Sample the source image with grids.
    Args:
        src (torch.tensor): BxCxWxH or BxCxDxWxH tensor.
        grids (torch.tensor): BxWxHx2 or BxDxWxHx3 tensor. Grids should be in canonical space [-1,1].
        padding_mode (string): On top of the padding mode provided by pytorch grid_sample function, an extra
        padding_mode="background" is provided. This mode will pad the boundary with the background intensity.
        To be noted, the smallest value in src is assumed to be the backround intensity.

    Returns:
        torch.tensor: BxCxWxH or BxCxDxWxH tensor. The warped image.
    """
    dim = grids.shape[1:-1]

    if padding_mode == "background":
        shift_intensity = True
        padding_mode = "zeros"
    else:
        shift_intensity = False

    # move channels dim to last position
    if len(dim) == 2:
        if shift_intensity:
            bg = torch.amin(src, dim=[2, 3], keepdim=True)
            src -= bg

        warped = F.grid_sample(
            src,
            grids,
            align_corners=True,
            mode=mode,
            padding_mode=padding_mode,
        )

        if shift_intensity:
            warped += bg
        return warped
    elif len(dim) == 3:
        if shift_intensity:
            bg = torch.amin(src, dim=[2, 3, 4], keepdim=True).detach()
            src = src - bg

        warped = F.grid_sample(
            src,
            grids,
            align_corners=True,
            mode=mode,
            padding_mode=padding_mode,
        )

        if shift_intensity:
            return warped + bg
        else:
            return warped


def spatial_transformer(
    src: torch.tensor, phi: torch.tensor, mode="bilinear", padding_mode="zeros"
) -> torch.tensor:
    """
    Warping the source with phi.
    Args:
        src (torch.tensor): BxCxWxH or BxCxDxWxH tensor.
        phi (torch.tensor): Bx2xWxH or Bx3xDxWxH tensor. Phi should be in canonical space [-1,1].
        padding_mode (string): On top of the padding mode provided by pytorch grid_sample function, an extra
        padding_mode="background" is provided. This mode will pad the boundary with the background intensity.
        To be noted, the smallest value in src is assumed to be the backround intensity.

    Returns:
        torch.tensor: BxCxWxH or BxCxDxWxH tensor. The warped image.
    """
    dim = phi.shape[2:]

    if len(dim) == 2:
        return custom_grid_sampler(
            src, phi.flip(1).permute(0, 2, 3, 1), mode, padding_mode
        )
    else:
        return custom_grid_sampler(
            src, phi.flip(1).permute(0, 2, 3, 4, 1), mode, padding_mode
        )


def spatial_transformer_deprecate(
    src: torch.tensor, phi: torch.tensor, mode="bilinear", padding_mode="zeros"
) -> torch.tensor:
    """
    !!! This function should be deprecated. Need to refactor the old code to use the new spatial_transformer.
    Warping the source with phi.
    Args:
        src (torch.tensor): BxCxWxH or BxCxDxWxH tensor.
        phi (torch.tensor): Bx3xWxH or Bx3xDxWxH tensor. Phi should be in canonical space [-1,1].
        padding_mode (string): On top of the padding mode provided by pytorch grid_sample function, an extra
        padding_mode="background" is provided. This mode will pad the boundary with the background intensity.
        To be noted, the smallest value in src is assumed to be the backround intensity.

    Returns:
        torch.tensor: BxCxWxH or BxCxDxWxH tensor. The warped image.
    """
    dim = phi.shape[2:]

    if padding_mode == "background":
        shift_intensity = True
        padding_mode = "zeros"
    else:
        shift_intensity = False

    # move channels dim to last position
    if len(dim) == 2:
        if shift_intensity:
            bg = torch.amin(src, dim=[2, 3], keepdim=True)
            src -= bg

        warped = F.grid_sample(
            src,
            phi.permute(0, 2, 3, 1),
            align_corners=True,
            mode=mode,
            padding_mode=padding_mode,
        )

        if shift_intensity:
            warped += bg
        return warped
    elif len(dim) == 3:
        if shift_intensity:
            bg = torch.amin(src, dim=[2, 3, 4], keepdim=True)
            src -= bg

        warped = F.grid_sample(
            src,
            phi.permute(0, 2, 3, 4, 1),
            align_corners=True,
            mode=mode,
            padding_mode=padding_mode,
        )

        if shift_intensity:
            warped = warped + bg
        else:
            return warped


def correlation(
    x: torch.tensor, y: torch.tensor, kernel_size: int, stride: int, kernel="cosine"
) -> torch.tensor:
    """
    Compute the correlation between x and y within kernel_size*2+1 range.
    Args:
        x  (torch.tensor): BxCx*.
        y  (torch.tensor): BxCx*.

    Returns:
        torch.tensor: Bx(num of voxels in range)x*
    """
    shape = list(x.shape[2:])
    batch_size = x.shape[0]
    dim = len(shape)

    x = F.pad(x, (kernel_size,) * dim * 2, mode="replicate")
    for i in range(dim):
        x = x.unfold(2 + i, kernel_size * 2 + 1, stride)
    for i in range(dim):
        y = y.unsqueeze(-1)

    target_axis_order = (0, dim + 1, *[i + 1 for i in range(dim)])
    if kernel == "cosine":
        t = F.cosine_similarity(x, y, 1, 1e-4)
    else:
        # we use squared sum difference. This is the same as cosine sim when x and y are normalized.
        t = 1 - 0.5 * ((x-y)**2).sum(1)
    return t.reshape(batch_size, *shape, -1).permute(*target_axis_order)


def correlation_split(
    x: torch.tensor, y: torch.tensor, kernel_size: int, stride: int, kernel="cosine", batch_size=1
) -> torch.tensor:
    """
    Compute the correlation between x and y within kernel_size*2+1 range.
    Args:
        x  (torch.tensor): BxCx*.
        y  (torch.tensor): BxCx*.

    Returns:
        torch.tensor: Bx(num of voxels in range)x*
    """
    shape = list(x.shape[2:])
    batch_size = x.shape[0]
    chanel_size = x.shape[1]
    dim = len(shape)

    x = F.pad(x, (kernel_size,) * dim * 2, mode="replicate")
    res = []
    # for i in range(dim):
    #     y = y.unsqueeze(-1)
    for i in range(3):
        x = x.unfold(2 + i, kernel_size * 2 + 1, stride)
    
    # for j in range(kernel_size):
    #     if kernel == "cosine":
    #         res.append(F.cosine_similarity(x_[i], y, 1, 1e-4))
    #     else:
    #         # we use squared sum difference. This is the same as cosine sim when x and y are normalized.
    #         res.append(1 - 0.5 * ((x[i]-y)**2).sum(1))
    
    # y = y.unsqueeze(-1)
    # x = x.reshape(batch_size, chanel_size, *shape, -1).contiguous()

    target_axis_order = (0, dim + 1, *[i + 1 for i in range(dim)])
    
    for i in range(kernel_size * 2 + 1):
        for j in range(kernel_size * 2 + 1):
            for k in range(kernel_size * 2 + 1):
                if kernel == "cosine":
                    res.append(F.cosine_similarity(x[:,:,:,:,:,i,j,k], y, 1, 1e-4))
                else:
                    # we use squared sum difference. This is the same as cosine sim when x and y are normalized.
                    res.append(1 - 0.5 * ((x[:,:,:,:,:,i,j,k]-y)**2).sum(1))
    return torch.stack(res, dim=-1).reshape(batch_size, *shape, -1).permute(*target_axis_order)


def affine_solver(
    source_pts: torch.tensor, target_pts: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute the affine transform and inverse affine transform with given source and target points.
    Currently, this function only supports points in 3D. Noted, this function does not support batch computation.
    Args:
        source_pts (torch.tensor): Nx3
        target_pts (torch.tensor): Nx3

    Returns:
        torch.tensor: The 3x4 affine matrix.
        torch.tensor: The 3x4 inverse affine matrix.

    """

    affine_matrix = torch.linalg.lstsq(source_pts, target_pts).solution
    affine_matrix_inv = torch.linalg.inv(affine_matrix.transpose(0, 1))
    return affine_matrix.transpose(0, 1)[:3], affine_matrix_inv[:3]


def compose(disp1: torch.tensor, disp2: torch.tensor) -> torch.tensor:
    """
    return (d_1+identity) \circ (d_2+identity)
    Args:
        disp1: (torch.tensor): Bx3xDxHxW.
        disp2: (torch.tensor): Bx3xDxHxW.
    """
    identity = (
        F.affine_grid(
            torch.eye(3, 4).unsqueeze(0).to(disp2.device),
            (1,1) + tuple(disp2.shape[2:]),
            align_corners=True,
        )
        .permute(0, 4, 1, 2, 3)
        .flip(1)
    )
    return (
        spatial_transformer(
            disp1,
            disp2 + identity,
            mode="bilinear",
            padding_mode="border"
        )
        + disp2
    )

def inverse_consistency(disp_field1s, disp_field2s, identity, iter=20): ## disp_field1s disp_field2s 的区别？分别代表？
    #make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()
        for i in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5*(disp_field1s-F.grid_sample(disp_field2s,(identity+disp_field1s).permute(0,2,3,4,1)))
            disp_field2i = 0.5*(disp_field2s-F.grid_sample(disp_field1s,(identity+disp_field2s).permute(0,2,3,4,1)))

    return disp_field1i, disp_field2i


def instance_optimization(param, source, target, sim, reg, lamda=1, epoch=50):
    device = source.device
    phi = torch.nn.parameter.Parameter(param)  # B*3*H*W*D\n"
    
    optimizer = torch.optim.Adam([phi], lr=0.008)#, eps=1e-4)
    
    source = source.float()+50
    
    print("start")
    # run Adam optimisation with diffusion regularisation and B-spline smoothing
    for iter in range(epoch):  
        optimizer.zero_grad()

        warped = spatial_transformer(
                source,
                phi,
                mode="bilinear",
                padding_mode="background",
            )

        sim_loss = sim(warped, target)
        reg_loss = reg(phi,phi)
        loss = sim_loss + lamda*reg_loss
        print(sim_loss.item(), reg_loss.item())
        loss.backward()
        optimizer.step()
    
    return phi.detach()



def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet

def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    # det = JacboianDet(y_pred, sample_grid)
    # p = torch.mean(torch.where(det<0, 1., 0.))

    # phi = (y_pred + sample_grid)[0].permute(1,2,3,0)
    # loss = compute_folding_gpu(phi) 

    # return loss
    return torch.mean(selected_neg_Jdet)
    