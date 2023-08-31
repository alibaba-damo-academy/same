import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from stat import S_IREAD

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from SAMReg.tools.Config import Config
from SAMReg.tools.utils.general import (
    get_git_revisions_hash,
    make_dir,
    set_seed_for_demo,
    path_import,
)
from torch.utils.tensorboard import SummaryWriter


def train(
    train_config,
    data_path,
    data_shape,
    epochs,
    lr,
    use_lr_schduler,
    device,
    writer,
    exp_folder,
    save_model_period,
    load_model="",
):
    # Check the train_config is valid
    assert hasattr(
        train_config, "make_net"
    ), "The train config python file should contain a method named make_net."
    assert hasattr(
        train_config, "make_dataloader"
    ), "The train config python file should contain a method named make_dataloader."
    assert hasattr(
        train_config, "train_kernel"
    ), "The train config python file should contain a method named train_kernel."

    # Init dataloader
    train_loader, debug_loader = train_config.make_dataloader(data_path, data_shape)

    inshape = list(next(iter(train_loader))[0].shape[2:])
    model = train_config.make_net(inshape)
    if load_model is not "":
        checkpoint = torch.load(load_model, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    # set learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    # resume optimizer and lr
    if load_model is not "":
        load_model_split = load_model.split("/")
        load_model_split[-1] = "optim_" + load_model_split[-1]
        checkpoint = torch.load("".join(load_model_split), map_location="cpu")
        optimizer.load_state_dict(checkpoint["opt"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    # Creates a GradScaler once at the beginning of training.
    # scaler = torch.cuda.amp.GradScaler()

    ite = 0
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}...")

        # We only have one group of parameter, thus take [0].
        writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch) 

        loss_dict = defaultdict(lambda: 0.0)
        for train_batch in train_loader:
            loss, loss_log = train_config.train_kernel(
                model, train_batch, device, epoch, exp_folder
            )

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss

            for k, v in loss_log.items():
                loss_dict[k] += v
                writer.add_scalar(f"Train_running/{k}", v, ite)
            
            ite += 1
        
        # Update learning rate
        if use_lr_schduler:
            lr_scheduler.step()

        # print epoch info
        batch_num = len(train_loader)
        for k, v in loss_dict.items():
            writer.add_scalar(f"Train/{k}", v / batch_num, epoch)
        
        # save model checkpoint
        if save_model_period > 0 and epoch % save_model_period == 0:
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(
                    os.path.join(exp_folder, "checkpoints"), "%04d.pt" % epoch
                ),
            )
            torch.save(
                {"opt": optimizer.state_dict(),
                 "lr_scheduler": lr_scheduler.state_dict(),
                 "epoch":epoch},
                os.path.join(
                    os.path.join(exp_folder, "checkpoints"), "optim_%04d.pt" % epoch
                ),
            )

            # debug and validate model
            with torch.no_grad():
                debug_batch = next(iter(debug_loader))
                train_config.debug_kernel(model, debug_batch, device, epoch, exp_folder)

    # final model save
    torch.save(
        {"state_dict": model.state_dict()},
        os.path.join(os.path.join(exp_folder, "checkpoints"), "final.pt"),
    )


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

    # Create checkpoint path, record path and log path
    checkpoint_path = os.path.join(exp_folder_path, "checkpoints")
    make_dir(checkpoint_path)
    record_path = os.path.join(exp_folder_path, "records")
    make_dir(record_path)
    log_path = os.path.join(exp_folder_path, "logs")
    make_dir(log_path)
    test_path = os.path.join(exp_folder_path, "tests")
    make_dir(test_path)

    setting_path = args.setting_path
    # assert os.path.isfile(setting_path), "Setting file is not found."
    setting = Config(setting_path)

    # Update setting file with command input
    setting["dataset"]["data_path"] = data_path
    setting["train"]["output_path"] = exp_folder_path
    setting["train"]["continue_from"] = args.continue_from
    setting["train"]["gpu_ids"] = args.gpu_id

    # # Write the commit hash for current codebase
    # label = get_git_revisions_hash()
    # setting["exp"]["git_commit"] = label

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
        --setting_path/ -s: the path to the folder where settings are saved
        --continue_from: the path to the checkpoint file where we should resume the training
        --gpu_id/ -g: gpu_id to use
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="An easy interface for training registration models"
    )
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
        "-e",
        "--exp_name",
        required=True,
        type=str,
        default=None,
        help="the name of the experiment",
    )
    parser.add_argument(
        "--train_config",
        required=True,
        type=str,
        default=None,
        help="the path to the train config python file.",
    )
    parser.add_argument(
        "-s",
        "--setting_path",
        required=False,
        type=str,
        default="",
        help="path of the folder where settings are saved,should include cur_task_setting.json",
    )
    parser.add_argument(
        "--continue_from",
        required=False,
        type=str,
        default="",
        help="Which checkpoint we should continue train from",
    )
    parser.add_argument(
        "-g", "--gpu_id", required=False, type=str, default="0", help="gpu_id to use"
    )

    # Train parameters
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default=100,
        help="the number of epochs to train.",
    )
    parser.add_argument(
        "--lr",
        required=False,
        type=float,
        default=1e-3,
        help="the learning rate used to train.",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        required=False,
        type=int,
        default=0,
        help="whether to use learning rate scheduler. 0 means no and 1 means yes.",
    )

    # Train log settings
    parser.add_argument(
        "--save_model_period",
        required=False,
        type=int,
        default=20,
        help="set the period of time to save model weight. Any number smaller \
                than 1 will turn off the function.",
    )

    args = parser.parse_args()
    print(args)

    set_seed_for_demo()
    setting, exp_folder = prepare(args)

    # Set gpu
    gpus = args.gpu_id.split(",")
    gpus = [int(g) for g in gpus]
    n_gpu = len(gpus)
    device = "cuda" if n_gpu > 0 else "cpu"
    if n_gpu > 0:
        torch.cuda.set_device(gpus[0])
        device = torch.cuda.current_device()
        torch.backends.cudnn.benchmark = True
    print(f"The current device: {torch.cuda.current_device()}")

    writer = SummaryWriter(
        os.path.join(exp_folder, "logs")
        + "/"
        + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )

    assert args.train_config is not None, "Need specify train config python file."
    train_config = path_import(args.train_config).train_config()

    # Save training config
    shutil.copy(args.train_config, f"{exp_folder}/train_config.py")

    train(
        train_config,
        args.data_path,
        args.data_shape,
        args.epochs,
        args.lr,
        args.use_lr_scheduler,
        device,
        writer,
        exp_folder,
        save_model_period=args.save_model_period,
        load_model=args.continue_from,
    )
