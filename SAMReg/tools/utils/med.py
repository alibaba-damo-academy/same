from typing import List
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.cluster import KMeans
from skimage import measure, morphology
import scipy
import torch
import torch.nn.functional as F
import itk
import struct

def save_itkimage(img_numpy, out_path, src_sitk):
    """
    Save img_numpy using the same spacing, origin, and orientation as src_sitk
    """
    img_sitk = sitk.GetImageFromArray(img_numpy)
    img_sitk.SetSpacing(src_sitk.GetSpacing())
    img_sitk.SetOrigin(src_sitk.GetOrigin())
    img_sitk.SetDirection(src_sitk.GetDirection())
    sitk.WriteImage(img_sitk, out_path)



def save_image(img_numpy, out_path, im_info):
    """
    Save img_numpy using the spacing, origin, and orientation in im_info
    """
    img_sitk = sitk.GetImageFromArray(img_numpy)
    img_sitk.SetSpacing(im_info['spacing'])
    img_sitk.SetOrigin(im_info['origin'])
    img_sitk.SetDirection(im_info['direction'])
    sitk.WriteImage(img_sitk, out_path)


def save_to_itk(img, p):
    img_itk = itk.image_from_array(img)
    itk.imwrite(img_itk, p)


def read_image(im_path):
    vol = load_sitk_vol(im_path)
    spacing = vol.GetSpacing()
    origin = vol.GetOrigin()
    direc = vol.GetDirection()
    vol = sitk.GetArrayFromImage(vol)
    if not np.all(np.array(direc) == np.eye(3).ravel()):
        vol = adjust_direction(im_path, vol, direc).copy()
    return vol, dict(im_path=im_path, spacing=spacing, origin=origin, direction=np.eye(3).ravel(), origin_direction=direc)



def read_image_general(im_path):
    vol = load_sitk_vol(im_path)
    spacing = vol.GetSpacing()
    origin = vol.GetOrigin()
    direc = vol.GetDirection()
    vol = sitk.GetArrayFromImage(vol)
    return vol, dict(im_path=im_path, spacing=spacing, origin=origin, direction=direc)



def load_sitk_vol(im_path):
    if im_path.endswith('.nii') or im_path.endswith('.nii.gz'):
        vol = sitk.ReadImage(im_path)
    else:
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(im_path)
        reader.SetFileNames(series)
        vol = reader.Execute()
    return vol


def adjust_direction(fn, vol, direc, verbose=False):
    """
    Adjust the orientation of an image to RAI.
    We only handle if an axis is flipped. We can't handle if axes are rotated or tilted.
    Please also make sure that direc is consistent with the actual orientation of vol.
    Usage:
    vol = sitk.ReadImage(im_path)
    direc = im_sitk.GetDirection()
    vol = sitk.GetArrayFromImage(im_sitk)
    adjust_direction(im_path, vol, direc)
    """
    direc = np.reshape(direc, (3,3))
    if verbose:
        print(f"{fn} has direction {direc}.")
    assert np.all(np.abs(direc) == np.eye(3)), f'unsupported direction!'
    for axis in range(3):
        if direc[axis, axis] == -1:
            vol = np.flip(vol, 2-axis)
            if verbose:
                print(f"axis {axis} is flipped. Please remember to flip the prediction results back!")
    return vol


def visualize(im1, im2, norm_ratio_1, norm_ratio_2, pt1, pt2, score, save_path=None):
    # if im_val > -800:  # soft tissue window
    #     window_high = 200
    #     window_low = -200
    # else:  # lung
    #     window_high = 200
    #     window_low = -1200
    fig, ax = plt.subplots(3, 2, figsize=(20, 30))
    # q_img = im1.copy()#.transpose(2, 0, 1)
    # q_img[q_img < window_low] = window_low
    # q_img[q_img > window_high] = window_high
    # q_img = (q_img - window_low) / (window_high - window_low)

    q_img = im1.copy().astype(np.float32)

    slice = q_img[pt1[2], :, :]
    ax[0, 0].set_title('query')
    ax[0, 0].imshow(slice, cmap='gray')
    ax[0, 0].plot((pt1[0]), (pt1[1]), 'o', markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=12, markeredgewidth=2)
    slice = q_img[:, pt1[1], :]
    slice = slice[::-1, :]
    ax[1, 0].set_title('query')
    ax[1, 0].imshow(slice, cmap='gray', aspect=norm_ratio_1[2] / norm_ratio_1[0])
    ax[1, 0].plot((pt1[0]), (q_img.shape[0] - pt1[2] - 1), 'o',
                  markerfacecolor='none', markeredgecolor="red",
                  markersize=12, markeredgewidth=2)

    slice = q_img[:, :, pt1[0]]
    slice = slice[::-1, :]
    ax[2, 0].set_title('query')
    ax[2, 0].imshow(slice, cmap='gray', aspect=norm_ratio_1[2] / norm_ratio_1[1])
    ax[2, 0].plot((pt1[1]), (q_img.shape[0] - pt1[2] - 1), 'o',
                  markerfacecolor='none', markeredgecolor="red",
                  markersize=12, markeredgewidth=2)
    k_img = im2.copy()#.transpose(2, 0, 1)

    # k_img[k_img < window_low] = window_low
    # k_img[k_img > window_high] = window_high
    # k_img = (k_img - window_low) / (window_high - window_low)
    k_img = k_img.astype(np.float32)

    slice = k_img[pt2[2], :, :]
    ax[0, 1].set_title('key')
    ax[0, 1].imshow(slice, cmap='gray')
    ax[0, 1].plot((pt2[0]), (pt2[1]), 'o', markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=12, markeredgewidth=2)

    slice = k_img[:, pt2[1], :]
    slice = slice[::-1, :]

    ax[1, 1].set_title('key')
    ax[1, 1].imshow(slice, cmap='gray', aspect=norm_ratio_2[2] / norm_ratio_2[0])
    ax[1, 1].plot((pt2[0]), (k_img.shape[0] - pt2[2] - 1), 'o',
                  markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=12, markeredgewidth=2)
    slice = k_img[:, :, pt2[0]]
    slice = slice[::-1, :]
    ax[2, 1].set_title('key')
    ax[2, 1].imshow(slice, cmap='gray', aspect=norm_ratio_2[2] / norm_ratio_2[1])
    ax[2, 1].plot((pt2[1]), (k_img.shape[0] - pt2[2] - 1), 'o',
                  markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=12, markeredgewidth=2)
    plt.suptitle(f'score:{score}')
    plt.tight_layout()
    if save_path is None:
        plt.show()  # may be slow
    else:
        plt.savefig(save_path)
    plt.close()

def select_points_with_stride(shape, stride, mask):
    '''
    Generate grids with stride.
    Point match require coord axis order in kji.
    '''
    dim = len(shape)
    num_pts = ((shape - np.ceil(stride/2))/stride).astype(int)
    end = int(stride/2) + num_pts * stride
    if dim == 3:
        x_ = np.linspace(int(stride/2)-1, end[0]-1, num_pts[0]+1)
        y_ = np.linspace(int(stride/2)-1, end[1]-1, num_pts[1]+1)
        z_ = np.linspace(int(stride/2)-1, end[2]-1, num_pts[2]+1)
        x, y, z = np.meshgrid(z_, y_, x_)
        grids = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1).astype(int)
    return grids[mask[grids[:,2], grids[:,1], grids[:,0]] == True]


# def select_random_points(num_pts, im):
#     # pts = (np.random.rand(num_pts, 3) * np.array(im1.shape)[::-1]).astype(int)
#     pts, im_vals = [], []
#     while len(pts) < num_pts:
#         pt1 = (np.random.rand(3) * np.array(im.shape)[::-1]).astype(int)
#         if im[pt1[2], pt1[1], pt1[0]] > -800:
#             pts.append(pt1)
#             im_vals.append(im[pt1[2], pt1[1], pt1[0]])
#     return np.vstack(pts), im_vals



def select_random_points(num_pts, im):
    # pts = (np.random.rand(num_pts, 3) * np.array(im1.shape)[::-1]).astype(int)
    pts, im_vals = [], []
    while len(pts) < num_pts:
        pt1 = (np.random.rand(3) * np.array(im.shape)[::-1]).astype(int)
        if im[pt1[2], pt1[1], pt1[0]] > -800:
            pts.append(pt1)
            im_vals.append(im[pt1[2], pt1[1], pt1[0]])
    return np.vstack(pts), im_vals


def compute_dice(x, y):
    '''
    This function computes the mean dice across all the labels except the backgorund.
    Background is assumed to be equal to 0.
    Args:
        x (numpy): The array containing label ids for image X. DxHxW.
        y (numpy): The array containing label ids for image Y. DxHxW.
    
    Returns:
        dice (float): The mean dice over all the labels exept 0. 0 is assumed
        to be the label for background.
        dice_in_detail (numpy): The dice scores of each labels.
    '''
    eps = 1e-11
    n_labels = max(np.max(x), np.max(y))
    dices = {}

    def _compute_dice(x, y):
        return 2*(x & y).sum() / (x.sum() + y.sum())

    for i in range(1, n_labels+1):
        if (x==i).sum() > 0 and (y==i).sum() > 0:
            dices[str(i)] = _compute_dice(x==i, y==i)
    return np.array(list(dices.values())).mean(), dices



def compute_dice_gpu(x_, y_, labels=None):
    '''
    This function computes the mean dice across all the labels except the backgorund.
    Background is assumed to be equal to 0.
    Args:
        x (tensor): The array containing label ids for image X. DxHxW.
        y (tensor): The array containing label ids for image Y. DxHxW.
    
    Returns:
        dice (float): The mean dice over all the labels exept 0. 0 is assumed
        to be the label for background.
        dice_in_detail (numpy): The dice scores of each labels.
    '''
    dices = {}

    if labels == None:
        labels = np.intersect1d(torch.unique(x_).cpu().numpy(), torch.unique(y_).cpu().numpy())
    
    assert labels[0]>=0, 'Labels should be equal or greater than 0.'
    
    # Remove background
    if labels[0] == 0:
        labels = labels[1:]

    if len(labels) == 0:
        return torch.zeros(1), [0]
    else:
        x = F.one_hot(x_, -1)[:,:,:,labels]
        y = F.one_hot(y_, -1)[:,:,:,labels]

        x_sum = x.sum([0,1,2])
        y_sum = y.sum([0,1,2])

        dices = 2*(x & y).sum([0,1,2]) / (x_sum + y_sum)

        return dices.mean(), dices


def points_to_canonical_space(pts, shape):
    '''
    This function transform points in volume space to canonical space [-1,1].
    Thus, the transformed points align with pytorch grid_sample.
    Args:
        pts (numpy): Nx3
        shape (numpy): This specify the shape of the volume. 
    
    Returns:
        pts (numpy): Transformed points.
    '''
    return pts / (np.expand_dims(shape, 0)-1) * 2.0 - 1.

def points_to_canonical_space_cuda(pts, shape):
    '''
    This function transform points in volume space to canonical space [-1,1].
    Thus, the transformed points align with pytorch grid_sample.
    Args:
        pts (tensor): Nx3
        shape (numpy): This specify the shape of the volume. 
    
    Returns:
        pts (tensor): Transformed points.
    '''
    return pts / (shape.unsqueeze(0)-1) * 2.0 - 1.


def seg_bg_mask(img):
    """
    Calculate the segementation mask for the whole body.
    Assume the dimensions are in Superior/inferior, anterior/posterior, right/left order.
    :param img: a 3D image represented in a numpy array.
    :return: The segmentation Mask. BG = 0
    """
    (D,W,H) = img.shape

    img = np.copy(img)
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the body
    # to renormalize washed out images
    # middle = img[int(D/5):int(D/5*4),int(W/5):int(W/5*4),int(H/5):int(H/5*4)] 
    middle = img
    mean = np.mean(middle)  

    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # clear bg
    dilation = morphology.dilation(thresh_img,np.ones([4,4,4]))
    eroded = morphology.erosion(dilation,np.ones([4,4,4]))

    # Select the largest area besides the background
    labels = measure.label(eroded, background=1)
    regions = measure.regionprops(labels)
    roi_label = 0
    max_area = 0
    for region in regions:
        if region.label != 0 and region.area > max_area:
            max_area = region.area
            roi_label = region.label
    thresh_img = np.where(labels==roi_label, 1, 0)

    # bound the ROI. 
    # TODO: maybe should check for bounding box
    # thresh_img = 1 - eroded
    sum_over_traverse_plane = np.sum(thresh_img, axis=(1,2))
    top_idx = 0
    for i in range(D):
        if sum_over_traverse_plane[i] > 0:
            top_idx = i
            break
    bottom_idx = D-1
    for i in range(D-1, -1, -1):
        if sum_over_traverse_plane[i] > 0:
            bottom_idx = i
            break
    for i in range(top_idx, bottom_idx+1):
        thresh_img[i]  = morphology.convex_hull_image(thresh_img[i])

    labels = measure.label(thresh_img)
    
    bg_labels = []
    corners = [(0,0,0),(-1,0,0),(0,-1,0),(-1,-1,0),(0,-1,-1),(0,0,-1),(-1,0,-1),(-1,-1,-1)]
    for pos in corners:
        bg_labels.append(labels[pos])
    bg_labels = np.unique(np.array(bg_labels))
    
    mask = labels
    for l in bg_labels:
        mask = np.where(mask==l, -1, mask)
    mask = np.where(mask==-1, 0, 1)

    roi_labels = measure.label(mask, background=0)
    roi_regions = measure.regionprops(roi_labels)
    bbox = [0,0,0,D,W,H]
    for region in roi_regions:
        if region.label == 1:
            bbox = region.bbox
    
    return mask, bbox

def seg_bed(img):
    '''
    This is modified based on the code Yixiao provided.
    This script can remove the bed from nect CT.

    Args:
        img (numpy): DxWxH
    '''
    d, h, w = img.shape
    mask = np.zeros((d, h, w), dtype=bool)
    for i in range(d):
        bin_slice = img[i] > -800
        ccs = morphology.label(bin_slice)
        largest_cc = ccs == np.argmax(np.bincount(ccs.flat)[1:]) + 1
        largest_cc = scipy.ndimage.binary_fill_holes(largest_cc)
        mask[i] = largest_cc
    
    # Close hole
    dilation = morphology.dilation(mask,np.ones([4,4,4]))
    mask = morphology.erosion(dilation,np.ones([4,4,4]))

    # clear bg
    erosion = morphology.erosion(mask,np.ones([4,4,4]))
    mask = morphology.dilation(erosion,np.ones([4,4,4]))
    
    return mask

def compute_folding(phi, mask=None):
    '''
    Computes the percentage of foldings inside the phi. 
    The negative determinant jacobian at each voxel indicates whether 
    these is folding locally.

    Args:
        phi (numpy): DxWxHx3.
    '''
    D, W, H, _ = phi.shape
    jacob = np.zeros((D, W, H, 3, 3))
    jacob[:-1,:,:,:,0] = phi[1:, :, :] - phi[:-1, :, :]
    jacob[:,:-1,:,:,1] = phi[:, 1:, :] - phi[:, :-1, :]
    jacob[:,:,:-1,:,2] = phi[:, :, 1::] - phi[:, :, :-1]

    jacob = np.reshape(jacob[:-1, :-1, :-1], (-1, 3, 3))
    det = np.linalg.det(jacob)
    if mask is not None:
        mask = mask[:-1,:-1,:-1]
        return np.sum((det<0)*mask.reshape(-1))/np.sum(mask)*100
    return np.mean(det<0)*100

def compute_folding_gpu(phi, mask=None, return_map=False):
    '''
    Computes the percentage of foldings inside the phi. 
    The negative determinant jacobian at each voxel indicates whether 
    these is folding locally.

    Args:
        phi (tensor): DxWxHx3.
    '''
    a = (phi[1:, 1:, 1:] - phi[:-1, 1:, 1:]).detach()
    b = (phi[1:, 1:, 1:] - phi[1:, :-1, 1:]).detach()
    c = (phi[1:, 1:, 1:] - phi[1:, 1:, :-1]).detach()

    det = torch.sum(torch.cross(a, b, 3) * c, axis=3)

    if mask is not None:
        mask = mask[:-1,:-1,:-1]
        return torch.sum(torch.where(det<0, 1., 0.)*mask)/mask.sum()*100
    else:
        if return_map:
            return torch.mean(torch.where(det<0, 1., 0.))*100, F.pad(torch.where(det<0, 1., 0.), pad=(0,1,0,1,0,1), value=0)
        else:
            return torch.mean(torch.where(det<0, 1., 0.))*100


def compute_folding_map(phi):
    '''
    Computes the percentage of foldings inside the phi. 
    The negative determinant jacobian at each voxel indicates whether 
    these is folding locally.

    Args:
        phi (numpy): DxWxHx3.
    '''
    D, W, H, _ = phi.shape
    jacob = np.zeros((D, W, H, 3, 3))
    jacob[:-1,:,:,:,0] = phi[1:, :, :] - phi[:-1, :, :]
    jacob[:,:-1,:,:,1] = phi[:, 1:, :] - phi[:, :-1, :]
    jacob[:,:,:-1,:,2] = phi[:, :, 1::] - phi[:, :, :-1]

    jacob = np.reshape(jacob[:-1, :-1, :-1], (-1, 3, 3))
    det = np.linalg.det(jacob)
    return det.reshape(D-1, W-1, H-1)


def get_SDlogJ_gpu(phi, mask=None):
    '''
    Computes the percentage of foldings inside the phi. 
    The negative determinant jacobian at each voxel indicates whether 
    these is folding locally.

    Args:
        phi (tensor): DxWxHx3.
    '''
    a = (phi[1:, 1:, 1:] - phi[:-1, 1:, 1:]).detach()
    b = (phi[1:, 1:, 1:] - phi[1:, :-1, 1:]).detach()
    c = (phi[1:, 1:, 1:] - phi[1:, 1:, :-1]).detach()

    det = torch.sum(torch.cross(a, b, 3) * c, axis=3).clip(0.000000001, 1000000000)

    if mask is not None:
        mask = mask[:-1,:-1,:-1].int()
        return torch.std(torch.log(det*mask))
    else:
        return torch.std(torch.log(det))


##### metrics #####
def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet


def get_SDlogJ_official(disp, mask=None):
    disp_arr = disp.cpu().numpy()
    H, W, D, _ = disp_arr.shape
    jacdetval = jacobian_determinant(np.resize(disp_arr,(1, 3, H, W, D)))
    log_jac_det = np.log(jacdetval)
    if mask is not None:
        mask = mask.cpu().numpy()
        res = np.ma.MaskedArray(log_jac_det, 1-mask[2:-2, 2:-2, 2:-2]).std()
    else:
        res = log_jac_det.std()
    return res



def get_identity(shape: list, in_canonical: bool=False, inverse_coord: bool=True):
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    if in_canonical:
        x = x/(shape[0] -1) * 2.0 - 1.0
        y = y/(shape[1] -1) * 2.0 - 1.0
        z = z/(shape[2] -1) * 2.0 - 1.0
    if inverse_coord:
        return np.stack([z, y, x], axis=-1).astype(np.float32)
    else:
        return np.stack([x, y, z], axis=-1).astype(np.float32)




def compute_dice_labels(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return (dicem, labels)

def transformDisplacement(inputdatPath, H, W, D):

    with open(inputdatPath, 'rb') as content_file:
        content = content_file.read()
        
    grid_space = int((torch.pow(torch.Tensor([H*W*D])/(len(content)/12),0.334)))
    disp_field = torch.from_numpy(np.array(struct.unpack('f'*(len(content)//4),content))).reshape(1,3,D//grid_space,W//grid_space,H//grid_space).permute(0,1,4,3,2).float()
    disp_field = F.interpolate(disp_field,size=(H,W,D),mode='trilinear',align_corners=None).permute(0,2,3,4,1)[:,:,:,:,torch.Tensor([2,0,1]).long()].flip(4)
    identity = torch.from_numpy(np.array(np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij'))).permute(2,3,1,0)
    
    return disp_field[0] + identity


if __name__ == '__main__':

    # temp = np.random.rand(50, 40, 60, 3)
    # folding = compute_folding(temp)
    # print(folding)
    # print()

    # # temp = np.random.rand(1, 3, 50, 40, 60)
    # std = get_SDlogJ(temp)
    # print(std)

    # val = jacobian_determinant(np.resize(temp,(1, 3, 50, 40, 60)))
    # print(np.nanstd(np.log(val)))
    # print(val.shape)


    grids = get_identity((10, 20, 30), in_canonical=True)
    foldings = compute_folding(np.flip(grids,-1))
    # std = get_SDlogJ(np.flip(grids,-1))
    print(grids.shape)
    print(foldings)
    # print(std)

    # x = torch.randint(0, 10, (10,20,30)).to('cuda')
    # y = torch.randint(0, 10, (10,20,30)).to('cuda')
    # x = torch.where(x==4, 0, x)
    # d_g, _ = compute_dice_gpu(x, y)
    # d, _ = compute_dice(x.cpu().numpy(), y.cpu().numpy())
    # print(f"DICE from cpu: {d}, DICE from gpu: {d_g}")
    



