#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###########################################################
# Cell 4: Set command line arguments
###########################################################
from __future__ import print_function

# Simulate command line arguments
# cmd_args = "--dataset=perm-mnist --model=der --buffer_size=10 --alpha=0.5 --lr=1e-4".split()
import os
import sys

# Force CUDA invisible, disable GPU
gpu_device="" # Set to empty string to disable all GPUs
# gpu_device="0"
gpu_device="1"

dataset_name='seq-survival'
backbone_name='mome'

# model_name='sgd'
# model_name='ewc_on'
# model_name='lwf'
# model_name='ms_moe'
# model_name='lora4cl'

# model_config='best'

### Rehearsal
# model_name='er'
# model_name='der'
# model_name='derpp'
model_name='consurv'
# model_name='imex_reg'
# model_name='mose_wsi'


model_config='default' # Rehearsal use default, since can adjust the buffer size

buffer_size=32 # total 16 class
# buffer_size=64


cmd_args_change=[
    '--fold=0', # setup fold
    f'--model={model_name}',
    f'--model_config={model_config}',
    '--n_epochs=1',  # 1 for running quickly, default is 20
    '--debug_mode=1', # 1 for debug mode, 0 for normal mode
    '--n_classes=16',
    '--dataset_config=test_short',
    # '--dataset_config=default',
    # '--dataset_config=joint',
    '--enable_other_metrics=0', # skip other metrics for testing
    # '--warm_up_epochs=5',
    # '--moe_mode=ensemble',
    '--gc=5'
]

rehearsal_cmd_args=[
    f'--buffer_size={buffer_size}',
]

cmd_args_default=[
    f'--dataset={dataset_name}',
    # '--dataset_config=default',
    f'--backbone={backbone_name}',
    f'--model_type={backbone_name}',
    # '--wandb_entity=<your-wandb-entity>',
    # '--wandb_project=<your-wandb-project>',
    '--savecheck=task',
    # '--gc=1', # gradient accumulation steps, should be 32 for default, 1 currently
    '--eval_epochs=1',
    '--seed=42',
    '--optim_mom=0.9', # for sgd
    '--optim_wd=1e-5', # for sgd and adam
    '--optimizer=adam',
    '--lr=2e-4',
    '--bs_micro_fix_shuffle',
    '--num_workers=24',
    # '--num_workers=0' # for debug convenience, not needed for training
]


# # Save original sys.argv
# original_argv = sys.argv
# ewc_on_cmd_args=[
#     '--lr=0.1',
#     '--e_lambda=10',
#     '--gamma=1'
# ]



from IPython import get_ipython
def in_ipynb():
    try:
        return ipython is not None and 'IPKernelApp' in ipython.config
    except Exception as e:
        # raise e
        return False


# sys.argv = [sys.argv[0]] + cmd_args
# Set new sys.argv, simulate command line input
if in_ipynb():
    print("in ipynb")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
    sys.argv = [sys.argv[0]] + cmd_args_change + cmd_args_default
    if model_name in ['er', 'der', 'consurv', 'imex_reg', 'mose_wsi'] or 'derpp' in model_name:
        sys.argv += rehearsal_cmd_args
else:
    sys.argv += cmd_args_default
# otherwise, use the original sys.argv

# Set new sys.argv, simulate command line input
# if model == 'ewc_on':
#     sys.argv = [sys.argv[0]] + cmd_args + ewc_on_cmd_args
# else:
#     sys.argv = [sys.argv[0]] + cmd_args

print(sys.argv)


# In[ ]:


###########################################################
# Cell 0: Basic imports and environment setup
###########################################################
# %matplotlib inline
# %cd /path/to/your/repo/
# automatically reload edited modules

import os

current_dir = os.getcwd()
print(current_dir)
import sys

# Setup paths
if current_dir not in sys.path:
    sys.path.append(current_dir)
# %cd /path/to/your/repo/
mammoth_path = os.getcwd()
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

import os

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("Number of GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available")

sys.path

###########################################################
# Cell 1: Basic imports and environment setup
###########################################################
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# needed (don't change it)
import os
import sys
import time
import importlib
import socket
import datetime
import uuid
import argparse
import torch
import numpy
import logging
from utils.conf import base_path, get_device
from utils import setup_logging

# Set the number of threads Torch will use
torch.set_num_threads(2)


# Import and set up logging configuration
from utils import setup_logging
setup_logging()

###########################################################
# Cell 2: Main entry point
###########################################################
# Main imports
if __name__ == '__main__':
    # Log the start of Mammoth! on the current host, helping to identify logging output from different runs
    logging.info(f"Running Mammoth! on {socket.gethostname()}. (if you see this message more than once, you are probably importing something wrong)")

    # Import the warn_once function to handle warnings that should only be shown once
    from utils.conf import warn_once
    try:
        # Load environment variables from a .env file, unless the MAMMOTH_TEST environment variable is set to '0'
        if os.getenv('MAMMOTH_TEST', '0') == '0':
            from dotenv import load_dotenv
            load_dotenv()
        else:
            # Warn that the script is running in test mode and will ignore the .env file
            warn_once("Running in test mode. Ignoring .env file.")
    except ImportError:
        # Warn that python-dotenv is not installed and the .env file will be ignored
        warn_once("Warning: python-dotenv not installed. Ignoring .env file.")


###########################################################
# Cell 3: Import required modules and functions

"""
Main entry point for training a model on a continual learning dataset.

This function orchestrates the entire training process by:
1. Setting up the environment and configurations
2. Initializing the dataset, backbone, and model
3. Configuring distributed training if requested
4. Launching the training process

Args:
    args: Optional command line arguments. If None, they will be parsed.
"""
from main_utils import lecun_fix, check_args, parse_args, extend_args, load_configs, add_help
from utils.conf import base_path, get_device
from models import get_model
from datasets import get_dataset
from utils.training import train
from models.utils.future_model import FutureModel
from backbone import get_backbone

# Fix for accessing LeCun's website
lecun_fix()


###########################################################
# Cell 4: Parse command line arguments
###########################################################
args = None
if args is None:
    args = parse_args()


# Setup device configuration
device = get_device(avail_devices=args.device)
args.device = device

# Configure base path for data/checkpoints
base_path(args.base_path)

# Setup precision and optimization settings
if args.code_optimization != 0:
    torch.set_float32_matmul_precision('high' if args.code_optimization == 1 else 'medium')
    logging.info(f"Code_optimization is set to {args.code_optimization}")
    logging.info(f"Using {torch.get_float32_matmul_precision()} precision for matmul.")

    if args.code_optimization == 2:
        if not torch.cuda.is_bf16_supported():
            raise NotImplementedError('BF16 is not supported on this machine.')

# modified backbone for imex_reg
if args.model == 'imex_reg':
    args.backbone = 'mome_imex_reg'

# modified backbone for mose
if args.model == 'mose_wsi':
    args.backbone = 'mome_mose'

print(f"gc: {args.gc}")
print(f"device: {args.device}")
print(f"model: {args.model}")
print(args)


# In[ ]:


# Data preparation
# Initialize dataset and extend arguments
dataset = get_dataset(args)


# In[ ]:


# Model preparation
from backbone.model_mome import MoMETransformer
### Specify the input dimension size if using genomic features.
if 'omic' in args.mode or args.mode == 'cluster' or args.mode == 'graph' or args.mode == 'pyramid':
    args.omic_input_dim = dataset.omic_input_dim
    print("Genomic Dimension from dataset", args.omic_input_dim)
elif 'coattn' in args.mode:
    args.omic_sizes = dataset.omic_sizes
    print('Genomic Dimensions from dataset', args.omic_sizes)
else:
    args.omic_input_dim = 0


extend_args(args, dataset)
check_args(args, dataset=dataset)

# Setup model backbone
backbone = get_backbone(args)
logging.info(f"Using backbone: {args.backbone}, backbone.device: {backbone.device}")

###########################################################
# Cell 5: Configure torch.compile optimization
###########################################################

# Configure torch.compile optimization if requested
if args.code_optimization == 3:
    # check if the model is compatible with torch.compile
    # from https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    if torch.cuda.get_device_capability()[0] >= 7 and os.name != 'nt':
        print("================ Compiling model with torch.compile ================")
        logging.warning("`torch.compile` may break your code if you change the model after the first run!")
        print("This includes adding classifiers for new tasks, changing the backbone, etc.")
        print("ALSO: some models CHANGE the backbone during initialization. Remember to call `torch.compile` again after that.")
        print("====================================================================")
        backbone = torch.compile(backbone)
    else:
        if torch.cuda.get_device_capability()[0] < 7:
            raise NotImplementedError('torch.compile is not supported on this machine.')
        else:
            raise Exception(f"torch.compile is not supported on Windows. Check https://github.com/pytorch/pytorch/issues/90768 for updates.")

# Initialize model with loss function and transformations
kwargs = {'args': args}
loss = dataset.get_loss(**kwargs)
model = get_model(args, backbone, loss, dataset.get_transform(), dataset=dataset)
args.gpu_device = model.device
print(f'model.device: {model.device}')
# print(args)
assert isinstance(model, FutureModel) or not args.eval_future, "Model does not support future_forward."

# Setup distributed training if requested
if args.distributed == 'dp':
    from utils.distributed import make_dp

    if args.batch_size < torch.cuda.device_count():
        raise Exception(f"Batch too small for DataParallel (Need at least {torch.cuda.device_count()}).")

    model.net = make_dp(model.net)
    model.to('cuda:0')
    args.conf_ngpus = torch.cuda.device_count()
elif args.distributed == 'ddp':
    # DDP breaks the buffer, it has to be synchronized.
    raise NotImplementedError('Distributed Data Parallel not supported yet.')

# Set process title for easier identification
try:
    import setproctitle
    buffer_str = '_buffer_'+ str(args.buffer_size) if 'buffer_size' in args else ''
    joint_str = '_joint' if args.joint_training else ''
    setproctitle.setproctitle('{}_fold_{}{}{}_{}'.format(args.model, args.fold, buffer_str, joint_str, args.dataset))
except ImportError:
    logging.info("setproctitle package not installed. Process title will not be set.")


# In[ ]:


# Clean dataset
dataset.train_loader=None
dataset.train_loaders = []
dataset.train_loaders_dict = {}
dataset.test_loaders = []
dataset.test_loaders_dict = {}

# Start training
from utils.training_wsi import train
from utils.evaluate_wsi import evaluate_wsi
from utils.evaluate_wsi_joint import evaluate_wsi_joint
from backbone.model_mome import MoMETransformer
import warnings

# Ignore specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*(Series.__getitem__|torch.load).*")

# Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

train(model, dataset, args)


# In[ ]:


if 'ms_moe' in args.model:
    print(f'classifier.expert_usage_counts: {model.net.classifier.expert_usage_counts}')
    print(f'classifier.expert_weight_sums: {model.net.classifier.expert_weight_sums}')
    print(f'model.net.MoME_patho2.moe.expert_usage_counts: {model.net.MoME_patho2.moe.expert_usage_counts}')
    print(f'model.net.MoME_patho2.moe.expert_weight_sums: {model.net.MoME_patho2.moe.expert_weight_sums}')
    print(f'model.net.MoME_genom2.moe.expert_usage_counts: {model.net.MoME_genom2.moe.expert_usage_counts}')
    print(f'model.net.MoME_genom2.moe.expert_weight_sums: {model.net.MoME_genom2.moe.expert_weight_sums}')


# In[ ]:





# In[ ]:


model.net


# In[ ]:


# Example: compute total model parameters
total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from utils.analysis_utils import detailed_parameter_count

# Reload the autoreload extension to ensure it picks up the new file

# Calculate the parameters for the current model
# Make sure the training is finished and the `model` object is available.
print("Calculating parameters for the model `model.net`...")
if 'model' in locals() and hasattr(model, 'net'):
    detailed_parameter_count(model.net)
else:
    print("Model object 'model' not found or does not have 'net' attribute. Please ensure the training cell has been run.")


# In[ ]:


# detailed_parameter_count(model.net)


# In[ ]:





# In[ ]:





# In[ ]:




