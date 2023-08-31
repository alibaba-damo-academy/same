import os
from glob import glob
import subprocess
import torch
import numpy as np
import random
import importlib


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii'))
    return hashes

def set_seed_for_demo():
    """ reproduce the training demo"""
    seed = 2021
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def path_import(absolute_path):
   '''implementation taken from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly'''
   spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
   module = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(module)
   return module

def gpu_usage():
    print(
        "gpu usage (current/max): {:.2f} / {:.2f} GB".format(
            torch.cuda.memory_allocated() * 1e-9,
            torch.cuda.max_memory_allocated() * 1e-9,
        )
    )

