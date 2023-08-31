import json
import progressbar as pb
import numpy as np
import torch
from torch.utils.data import Dataset

from SAMReg.tools.utils.med import read_image
from SAMReg.tools.interfaces import proc_image


class BaseDataset(Dataset):

    def __init__(self, data_path, shape=None, with_prealign_affine=False, with_prealign_coarse=False, with_mask=False, with_label=False, body_only=False, clamp=None, phase="train", cfg=None):
        """
        the dataloader for registration task, to avoid frequent disk communication, all pairs are compressed into memory
        :param data_path:  string, path to the data
            the data should be preprocessed and saved into txt
        :param phase:  string, 'train'/'val'/ 'test'/ 'debug' ,    debug here means a subset of train data, to check if model is overfitting
        :param transform: function,  apply transform on data
        : seg_option: pars,  settings for segmentation task, None for segmentation task
        : reg_option:  pars, settings for registration task, None for registration task

        """

        self.data_path = data_path
        self.phase = phase
        self.cfg = cfg
        self.with_prealign_affine = with_prealign_affine
        self.with_prealign_coarse = with_prealign_coarse
        self.with_mask = with_mask
        self.with_label = with_label
        self.body_only = body_only
        self.shape = np.array(shape) if shape is not None else None
        self.clamp = clamp

        self.paired_id_list = []
        self.img_id_list = []
        self.img_list = {}
        self.img_mask_list = {}
        self.img_label_list = {}
        self.img_info_list = {}
        self.prealign_affine_list = []
        self.prealign_coarse_list = []
        
        self._get_file_list()
        if self.with_prealign_affine:
            self._load_prealign_affine()
        if self.with_prealign_coarse:
            self._load_prealign_coarse()

        self.init_img_pool()
    

    def _load_prealign_affine(self):
        """
        Used to load pre align phi
        """
        for p in self.paired_id_list:
            self.prealign_affine_list.append(np.load(f"{self.data_path}/pre_align/{p[0]}_{p[1]}_affine_inv.npy"))
    
    def _load_prealign_coarse(self):
        """
        Used to load pre align phi
        """
        for p in self.paired_id_list:
            self.prealign_coarse_list.append(np.load(f"{self.data_path}/pre_align/{p[0]}_{p[1]}_coarse_phi.npy"))

    def _get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        setting = json.load(open(f"{self.data_path}/splits.json"))
        
        if self.phase == "debug":
            phase = "train"
            setting[phase] = setting[phase][:2]
        else:
            phase = self.phase
        
        # Generate the list
        id_set = set()
        for p in setting[phase]:
            id_set.add(p[0])
            id_set.add(p[1])
            self.paired_id_list.append([p[0], p[1]])
        for case in id_set:
            self.img_id_list.append(case)

    def _read_case(self, case_id):
        """
        The logic used to process one single case
        """
        d = {}
        img, img_info = read_image(f"{self.data_path}/{case_id}/{case_id}_image.nii.gz")
        
        if self.with_mask or self.body_only:
            img_mask = np.load(f"{self.data_path}/{case_id}/{case_id}_mask.npy")
        else:
            img_mask = None

        if self.with_label:
            label, _ = read_image(f"{self.data_path}/{case_id}/{case_id}_label.nii.gz")
        else:
            label = None

        sub, norm_ratio = proc_image(
            img, 
            img_info, self.cfg, 
            mask=img_mask,
            label=label,
            pad_shape=self.shape)


        # proc_image() will standardize the orientation to RAS+
        # Thus, here we need to flip the first two dimensions according 
        # to the requirement of SAM embed model (SPL+). 
        # But in practice, I found just permute the orientation to SAR+ works the best
        # TODO: Need to sort out the orientation. Ideally, everything should happen in
        # cononical orientation.
        # d["img"] = torch.flip(sub["image"].data, dims=(1,2)).permute(0, 3, 2, 1)
        d["img"] = sub["image"].data.permute(0, 3, 2, 1)
        if self.with_mask or self.body_only:
            d["mask"] = sub["mask"].data.permute(0, 3, 2, 1)
            if self.body_only:
                d["img"][d["mask"]==0] = torch.min(d["img"])
                
        if self.with_label:
            d["label"] = sub["label"].data.permute(0, 3, 2, 1)

        d['img_info'] = {
            'case_id': case_id
        }
        
        return d
    
    

    def init_img_pool(self):
        """
            Load the image in parallel.
        """
        def _split_dict(dict_to_split, split_num):
            index_list = list(range(len(dict_to_split)))
            index_split = np.array_split(np.array(index_list), split_num)
            split_dict = []
            for i in range(split_num):
                dj = dict_to_split[index_split[i][0]:index_split[i][-1]+1]
                split_dict.append(dj)
            return split_dict


        img_dict = {}
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(self.img_id_list)).start()
        count = 0
        for c in self.img_id_list:
            img_dict[c] = self._read_case(c)
            count += 1
            pbar.update(count)
        pbar.finish()

        print("the loading phase finished, total {} img and labels have been loaded".format(len(img_dict)))

        for case_name in self.img_id_list:
            case = img_dict[case_name]
            if self.clamp is not None:
                self.img_list[case_name] = torch.cat([case['img'], self._renorm_image(case['img'], clamp=self.clamp)], dim=0)
            else:
                self.img_list[case_name] = case['img']
            self.img_info_list[case_name] = case['img_info']
            if self.with_mask:
                self.img_mask_list[case_name] = case['mask']
            if self.with_label:
                self.img_label_list[case_name] = case['label']
    

    # Since SAM require HU be clamped from (-1024, 3071) to [-50, 205]
    # And to make the LNCC numerically stable, we need to clamp the image from (-800, 400) to [-1,1]. 
    def _renorm_image(self, x, clamp=(-800, 400)):
        assert clamp[0]<clamp[1], "Error! Clamp range is not correct."
        clamp = [(i+1024.)*255/4095.-50. for i in clamp]
        return (torch.clamp(x, min=clamp[0], max=clamp[1])-clamp[0])/(clamp[1]-clamp[0])*2.-1.

    def __len__(self):
        return len(self.paired_id_list)

    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic

        """
        idx = idx % len(self.paired_id_list)

        s_id, t_id = self.paired_id_list[idx]

        res = [
            self.img_list[s_id],
            self.img_list[t_id],
            self.img_info_list[s_id],
            self.img_info_list[t_id]
        ]
        if self.with_mask:
            res.append(self.img_mask_list[s_id])
            res.append(self.img_mask_list[t_id])
        if self.with_prealign_affine:
            res.append(self.prealign_affine_list[idx])
        if self.with_prealign_coarse:
            res.append(self.prealign_coarse_list[idx])
        if self.with_label:
            res.append(self.img_label_list[s_id])
            res.append(self.img_label_list[t_id])
        return tuple(res)
