import numpy as np

import torch
from mmcv import Config
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmdet.models import build_detector
import torch.nn.functional as F
import torchio as tio
from sam.datasets.piplines import Resample, RescaleIntensity
import torch.cuda

def init_model(config, checkpoint=""):
    print('Initializing model ...')
    cfg = Config.fromfile(config)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if checkpoint is not "":
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    # model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    # model.eval()
    return model, cfg

def load_cfg(config):
    cfg = Config.fromfile(config)
    return cfg

def proc_image(im, im_info, cfg, label=None, mask=None, pad_shape=None):
    assert np.all(np.reshape(im_info['direction'], (3, 3)) == np.eye(3)), f'unsupported direction!'

    img_data = torch.from_numpy(im).permute(2, 1, 0)[None]
    tio_affine = np.hstack((np.diag(im_info['spacing']), np.array(im_info['origin'])[:, None]))
    tio_affine = np.vstack((tio_affine, [0, 0, 0, 1]))
    norm_ratio = np.array(im_info['spacing']) / np.array(cfg.norm_spacing)

    # Prepare subject data
    subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_data, affine=tio_affine)
        )

    if mask is not None:
        subject["mask"] = tio.LabelMap(
                        tensor=torch.from_numpy(mask.astype(np.int8)).permute(2,1,0)[None],
                        affine=tio_affine)

    if label is not None:
        subject['label'] = tio.LabelMap(
                tensor=torch.from_numpy(label.astype(np.int8)).permute(2,1,0)[None],
                affine=tio_affine)

    data = {}
    data['image_fn'] = im_info['im_path']
    data['subject'] = subject

    # Change the orientation to canonical orientation (RAS+)
    # Since SAM model ask fro LPS+, when passed to SAM
    # one needs to flip the first two orientations.
    # to_canonical = tio.transforms.ToCanonical()
    # data["subject"] = to_canonical(data["subject"])
    
    # Resample
    resample = Resample()
    data = resample(data)


    # # pad or crop
    # if pad_shape is not None:
    #     if not np.array_equal(pad_shape, im.shape):
    #         transform = tio.CropOrPad(pad_shape, mask_name="mask")
    #         data["subject"] = transform(data["subject"])


    # pad or crop
    if pad_shape is not None:
        if not np.array_equal(pad_shape, subject.shape[1:]):
            transform = tio.CropOrPad(pad_shape, mask_name="mask", padding_mode=-1000)
            transform_ = tio.CropOrPad(pad_shape, mask_name="mask", padding_mode=0)
            data["subject"]["image"] = transform(data["subject"]["image"])
            if mask is not None:
                data["subject"]["mask"] = transform_(data["subject"]["mask"])
            if label is not None:
                data["subject"]['label'] = transform_(data["subject"]['label'])

    
    # Scale intensity
    rescale = RescaleIntensity()
    data = rescale(data)

    return data['subject'], norm_ratio


def proc_image_without_rescale(im, im_info, cfg, label=None, mask=None, pad_shape=None):

    assert np.all(np.reshape(im_info['direction'], (3, 3)) == np.eye(3)), f'unsupported direction!'

    img_data = torch.from_numpy(im).permute(2, 1, 0)[None]
    tio_affine = np.hstack((np.diag(im_info['spacing']), np.array(im_info['origin'])[:, None]))
    tio_affine = np.vstack((tio_affine, [0, 0, 0, 1]))
    norm_ratio = np.array(im_info['spacing']) / np.array(cfg.norm_spacing)

    # Prepare subject data
    subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_data, affine=tio_affine)
        )

    if mask is not None:
        subject["mask"] = tio.LabelMap(
                        tensor=torch.from_numpy(mask.astype(np.int8)).permute(2,1,0)[None],
                        affine=tio_affine)

    if label is not None:
        subject['label'] = tio.LabelMap(
                tensor=torch.from_numpy(label.astype(np.int8)).permute(2,1,0)[None],
                affine=tio_affine)

    data = {}
    data['image_fn'] = im_info['im_path']
    data['subject'] = subject

    # Change the orientation to canonical orientation (RAS+)
    # Since SAM model ask fro LPS+, when passed to SAM
    # one needs to flip the first two orientations.
    # to_canonical = tio.transforms.ToCanonical()
    # data["subject"] = to_canonical(data["subject"])
    
    # Resample
    resample = Resample()
    data = resample(data)

    # pad or crop
    if pad_shape is not None:
        if not np.array_equal(pad_shape, subject.shape[1:]):
            transform = tio.CropOrPad(pad_shape, mask_name="mask")
            data["subject"] = transform(data["subject"])

    return data['subject'], norm_ratio




def proc_image_general(im, im_info, norm_spacing, label=None, mask=None, pad_shape=None):

    assert np.all(np.reshape(im_info['direction'], (3, 3)) == np.eye(3)), f'unsupported direction!'

    img_data = torch.from_numpy(im).permute(2, 1, 0)[None]
    tio_affine = np.hstack((np.diag(im_info['spacing']), np.array(im_info['origin'])[:, None]))
    tio_affine = np.vstack((tio_affine, [0, 0, 0, 1]))
    norm_ratio = np.array(im_info['spacing']) / np.array(norm_spacing)

    # Prepare subject data
    subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_data, affine=tio_affine)
        )

    if mask is not None:
        subject["mask"] = tio.LabelMap(
                        tensor=torch.from_numpy(mask.astype(np.int8)).permute(2,1,0)[None],
                        affine=tio_affine)

    if label is not None:
        subject['label'] = tio.LabelMap(
                tensor=torch.from_numpy(label.astype(np.int8)).permute(2,1,0)[None],
                affine=tio_affine)

    data = {}
    data['image_fn'] = im_info['im_path']
    data['subject'] = subject

    # Change the orientation to canonical orientation (RAS+)
    # Since SAM model ask fro LPS+, when passed to SAM
    # one needs to flip the first two orientations.
    # to_canonical = tio.transforms.ToCanonical()
    # data["subject"] = to_canonical(data["subject"])
    
    # Resample
    resample = Resample(norm_spacing=norm_spacing)
    data = resample(data)

    # pad or crop
    if pad_shape is not None:
        if not np.array_equal(pad_shape, subject.shape[1:]):
            transform = tio.CropOrPad(pad_shape, mask_name="mask")
            data["subject"] = transform(data["subject"])

    return data['subject'], norm_ratio



    



def get_embedding(model, im):
    with torch.no_grad():
        result = model.extract_feat(im)
    result = dict(coarse_emb=result[1], fine_emb=result[0], im_shape=im.shape[2:])
    return result

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode='floor')
    return tuple(reversed(out))

def match(query_vol, key_vol, query_points, batch_size):
    query_vec = query_vol[0, :, query_points[:, 0], query_points[:, 1], query_points[:, 2]].T

    pts, scores = [], []
    for i in range(int((query_points.shape[0] - 1) / batch_size) + 1):
        sim = F.conv3d(key_vol, query_vec[i*batch_size:(i+1)*batch_size, :, None, None, None])
        sim = sim.view(sim.shape[1], -1)
        max_sims, ind = torch.max(sim, dim=1)
        xyz = torch.stack(unravel_index(ind, key_vol.shape[2:])).permute(1, 0)

        pts.append(xyz)
        scores.append(max_sims)
    return torch.cat(pts, dim=0), torch.cat(scores)

def find_point_in_vol(query_data, key_data, query_points, cfg, batch_size=10000):
    with torch.no_grad():
        query_data = torch.cat([
            F.normalize(
                F.interpolate(query_data['coarse_emb'], query_data['fine_emb'].shape[2:], mode='trilinear', align_corners=True), 
                dim=1),
            query_data['fine_emb']], dim=1)
        key_data = torch.cat([
            F.normalize(
                F.interpolate(key_data['coarse_emb'], key_data['fine_emb'].shape[2:], mode='trilinear', align_corners=True), 
                dim=1),
            key_data['fine_emb']], dim=1)

        query_points = torch.div(query_points, cfg.local_emb_stride, rounding_mode='floor').flip(dims=[1])
        match_points, sims = match(query_data, key_data, query_points, batch_size)
        
        match_points = match_points * cfg.local_emb_stride
        query_points = query_points * cfg.local_emb_stride
    return query_points, match_points, sims/2.


def find_point_in_vol_stable(query_data, key_data, query_points, cfg, batch_size=10000, ite=5):
    with torch.no_grad():
        query_data = torch.cat([
            F.normalize(
                F.interpolate(query_data['coarse_emb'], query_data['fine_emb'].shape[2:], mode='trilinear', align_corners=True), 
                dim=1),
            query_data['fine_emb']], dim=1)
        key_data = torch.cat([
            F.normalize(
                F.interpolate(key_data['coarse_emb'], key_data['fine_emb'].shape[2:], mode='trilinear', align_corners=True), 
                dim=1),
            key_data['fine_emb']], dim=1)

        query_points = torch.div(query_points, cfg.local_emb_stride, rounding_mode='floor').flip(dims=[1])
        query_points_ = query_points
        
        for _ in range(ite):
            query_points = query_points_
            match_points, sims = match(query_data, key_data, query_points, batch_size)
            query_points_, sims = match(key_data, query_data, match_points, batch_size)

        ind = ((query_points_-query_points)**2).sum(1) < 1
        match_points = match_points[ind]
        query_points = query_points[ind]
        sims = sims[ind].unsqueeze(1)

        # Remove redudent data
        re = torch.unique(torch.cat([match_points, query_points, sims], dim=1), dim=0)
        match_points, query_points, sims = re[:,:3].long(), re[:,3:6].long(), re[:,6]

        query_points = query_points * cfg.local_emb_stride
        match_points = match_points * cfg.local_emb_stride
    return query_points, match_points, sims/2.


def find_point_in_vol_mind(query_data, key_data, query_points, cfg, batch_size=10000):
    fine_query_vec = extract_point_emb_mind(query_data, query_points, cfg)
    print("fine_query_vec.shape:", fine_query_vec.shape)
    pts, scores = [], []
    for i in range(int((query_points.shape[0] - 1) / batch_size) + 1):
        with torch.no_grad():
            pts_, scores_ = match_vec_in_vol_mind(
                fine_query_vec[i*batch_size:(i+1)*batch_size],
                key_data, cfg)
            pts.append(pts_)
            scores.append(scores_[:, None])
    return np.vstack(pts), np.vstack(scores)[:,0]






def extract_point_emb_mind(query_data, query_points, cfg):
    query_points = np.array(query_points)
    query_points = np.floor(query_points / cfg.local_emb_stride).astype(int)
    fine_query_vol = query_data['fine_emb']
    # coarse_query_vol, fine_query_vol = query_data['coarse_emb'], query_data['fine_emb']
    # coarse_query_vol = F.interpolate(coarse_query_vol, fine_query_vol.shape[2:], mode='trilinear', align_corners=True)
    # coarse_query_vol = F.normalize(coarse_query_vol, dim=1)
    # coarse_query_vec = coarse_query_vol[0, :, query_points[:, 2], query_points[:, 1], query_points[:, 0]].T
    fine_query_vec = fine_query_vol[0, :, query_points[:, 2], query_points[:, 1], query_points[:, 0]].T
    return fine_query_vec


def match_vec_in_vol_mind(fine_query_vec, key_data, cfg):
    fine_key_vol = key_data['fine_emb']
    # coarse_key_vol, fine_key_vol = key_data['coarse_emb'], key_data['fine_emb']

    # is it correct to interpolate embeddings? Will it mix neighboring pixels?
    # coarse_key_vol = F.interpolate(coarse_key_vol, fine_key_vol.shape[2:], mode='trilinear', align_corners=True)
    # coarse_key_vol = F.normalize(coarse_key_vol, dim=1)

    # change to convolution operator in GPU, similar speed w mat mul
    sim = (fine_key_vol - fine_query_vec[:, :, None, None, None]).pow(2).sum(1)
    # sim_fine = F.conv3d(fine_key_vol, fine_query_vec[:, :, None, None, None])
    # sim_coarse = F.conv3d(coarse_key_vol, coarse_query_vec[:, :, None, None, None])

    # instead of interp emb, interp sim. Its speed and accuracy is similar to interp emb, but has lower sim scores
    # sim_coarse = F.interpolate(sim_coarse, sim_fine.shape[2:], mode='trilinear')

    # sim = (sim_fine[0] + sim_coarse[0])/2
    # sim = sim_fine[0]
    sim = sim.view(sim.shape[0], -1)

    # compute sim by mat mul
    # dim = coarse_query_vec.shape[1]
    # fine_key_vec = fine_key_vol[0, :, :, :, :].reshape(dim, -1)
    # coarse_key_vec = coarse_key_vol[0, :, :, :, :].reshape(dim, -1)
    # sim_fine = torch.einsum("nc,ck->nk", fine_query_vec, fine_key_vec)
    # sim_coarse = torch.einsum("nc,ck->nk", coarse_query_vec, coarse_key_vec)
    # sim = (sim_fine + sim_coarse)/2

    # don't interp sim to ori image size, but rescale matched points
    ind = torch.argmin(sim, dim=1).cpu().numpy()
    zyx = np.unravel_index(ind, fine_key_vol.shape[2:])
    xyz = np.vstack(zyx)[::-1] * cfg.local_emb_stride
    xyz = xyz.T
    xyz = np.minimum(np.round(xyz.astype(int)), np.array(key_data['im_shape'])[::-1]-1)

    # interp sim to ori image size, no need to rescale points, maybe more accurate, similar speed, more memory
    # sim = (sim_fine + sim_coarse)/2
    # sim = F.interpolate(sim, key_data['im_shape'], mode='trilinear', align_corners=False)[0]
    # sim = sim.view(sim.shape[0], -1)
    # ind = torch.argmax(sim, dim=1).cpu().numpy()
    # zyx = np.unravel_index(ind, key_data['im_shape'])
    # xyz = np.vstack(zyx)[::-1].T

    min_sims, _ = sim.min(dim=1)

    return xyz, min_sims.cpu().numpy()
