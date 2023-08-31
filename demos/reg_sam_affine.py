import os
import sys
from datetime import datetime
from stat import S_IREAD
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from SAMReg.datasets.BaseDataset import BaseDataset
from SAMReg.tools.Config import Config
from SAMReg.tools.interfaces import init_model
from SAMReg.tools.utils.general import (
    get_git_revisions_hash,
    make_dir,
    set_seed_for_demo,
)
from SAMReg.cores.SAMAffine import SAMAffine


def prepare(args):
    output_path = args.output_path
    exp_name = args.exp_name
    data_path = args.data_path
    dataset_name = data_path.split("/")[-1]

    # Create experiment folder
    timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    exp_folder_path = os.path.join(output_path, dataset_name, exp_name, timestamp)
    make_dir(exp_folder_path)
    print(f"The experiment is recorded in {exp_folder_path}")

    test_path = os.path.join(exp_folder_path, "tests")
    make_dir(test_path)

    setting = Config()

    # Update setting file with command input
    setting["dataset"]["data_path"] = data_path
    setting["train"]["output_path"] = exp_folder_path
    setting["train"]["gpu_ids"] = args.gpu_id

    # Write the commit hash for current codebase
    label = get_git_revisions_hash()
    setting["exp"]["git_commit"] = label

    # Write the command argument list to the setting file
    setting["exp"]["command_line"] = " ".join(sys.argv)

    task_output_path = os.path.join(exp_folder_path, "setting.json")
    setting.save_json(task_output_path)

    # Make the setting file read-only
    os.chmod(task_output_path, S_IREAD)

    return setting, exp_folder_path


if __name__ == "__main__":
    """
    Run Affine registration.
    Arguments:
        --output_path/ -o: the path of output folder
        --data_path/ -d: the path to the dataset folder
        --exp_name/ -e: the name of the experiment
        --gpu_id/ -g: gpu_id to use
        --sim_threshold: threshold used to filter the matched points.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        type=str,
        default=None,
        help="the path of output folder",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        required=True,
        type=str,
        default="",
        help="the path to the data folder",
    )
    parser.add_argument(
        "--data_shape",
        required=True,
        nargs="+",
        type=int,
        help="the shape the image should be cropped to by the program.",
    )
    parser.add_argument(
        "--data_phase",
        required=False,
        type=str,
        default="val",
        help="which phase in the dataset should we run on. {train, val, test, debug}",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        required=True,
        type=str,
        default=None,
        help="the name of the experiment",
    )
    parser.add_argument(
        "-g", "--gpu_id", required=False, type=str, default="0", help="gpu_id to use"
    )

    # SAM parameters
    parser.add_argument(
        "--sim_threshold",
        required=False,
        type=float,
        default="0.7",
        help="threshold used to filter the matched points",
    )
    parser.add_argument(
        "--sam_config",
        required=False,
        type=str,
        default="../demos/configs/sam/sam_r18_i3d_fpn_1x_multisets_sgd_T_0.5_half_test.py",
        help="path to the config file of SAM",
    )
    parser.add_argument(
        "--sam_weight",
        required=False,
        type=str,
        default="../demos/iter_38000.pth",
        help="path to the weight file of SAM",
    )

    # Parameters
    parser.add_argument(
        "--with_stable_selection",
        required=False,
        type=int,
        default=1,
        help="Indicating whether stable selection is used. 0 means w/o stable selection.",
    )

    args = parser.parse_args()
    print(args)

    set_seed_for_demo()
    setting, output_folder = prepare(args)
    exp_folder = f"{output_folder}/tests"

    # Check arguments
    shape = list(args.data_shape)
    assert len(shape)==3, "The shape of the image is not properly set."

    # Set gpu
    gpus = args.gpu_id.split(",")
    gpus = [int(g) for g in gpus]
    n_gpu = len(gpus)
    device = "cuda" if n_gpu > 0 else "cpu"
    if n_gpu > 0:
        torch.cuda.set_device(gpus[0])
        torch.backends.cudnn.benchmark = True
        device = torch.cuda.current_device()
    print(f"The current device: {torch.cuda.current_device()}")

    # Save training script
    shutil.copy(__file__, output_folder + f"/{os.path.basename(__file__)}")

    # embed, cfg = init_model(args.sam_config, args.sam_weight)
    embed, cfg = init_model(args.sam_config)
    for param in embed.parameters():
        param.requires_grad = False
    embed.eval()

    model = SAMAffine(embed, cfg, with_stable_selection=(args.with_stable_selection!=0))
    model = model.cuda()

    train_loader = DataLoader(
        BaseDataset(
            setting["dataset"]["data_path"],
            shape=shape,
            with_mask=True,
            body_only=True,
            with_label=True,
            phase=args.data_phase,
            cfg=cfg
        ),
        batch_size=1,
    )
    runnningTime = []
    for (
        source,
        target,
        source_info,
        target_info,
        source_mask,
        target_mask,
        source_label,
        target_label,
    ) in train_loader:
        print(
            f"Start register {source_info['case_id'][0]} to {target_info['case_id'][0]}"
        )

        source = source.to(device).float()
        target = target.to(device).float()
        source_label = source_label.to(device)
        target_label = target_label.to(device)

        start = time.time()

        affine_matrix, affine_matrix_inv = model(
            source,
            target,
            source_mask,
            target_mask,
            args.sim_threshold
        )
        end = time.time()
        runnningTime.append(end - start)
        print(f"Running time: {end - start}")

        np.save(
            f"{exp_folder}/{source_info['case_id'][0]}_{target_info['case_id'][0]}_affine.npy",
            affine_matrix,
        )
        np.save(
            f"{exp_folder}/{source_info['case_id'][0]}_{target_info['case_id'][0]}_affine_inv.npy",
            affine_matrix_inv,
        )
        
    print(f"Finish running experiment at {output_folder}")
    print(f"Mean Running time: {np.mean(runnningTime)}")
