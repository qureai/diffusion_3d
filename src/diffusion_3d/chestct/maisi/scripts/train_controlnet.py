import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import lightning as L
import torch
import torch.distributed as dist
import torch.nn.functional as F
from clearml import Task
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter
from munch import munchify
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from diffusion_3d.chestct.maisi.configs.config_controlnet_training import config_controlnet_training
from diffusion_3d.chestct.maisi.configs.config_data import config_data
from diffusion_3d.chestct.maisi.configs.config_model import config_model

from .utils import binarize_labels, define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp

# Get config
config = munchify(
    {
        "model": config_model,
        "training": config_controlnet_training,
        "data": config_data,
    }
)


# Seed everything
L.seed_everything(config.training.seed)


# Start ClearML Task if applicable
task = None
if not config.training.fast_dev_run and config.training.log_on_clearml:
    task: Task = Task.init(project_name=config.training.clearml_project, task_name=config.training.task_name)
    task.connect_configuration(config, name="config")
    task.add_tags(config.training.clearml_tags)


# Initiliaze model
unet = define_instance(config_model, "diffusion_unet_def")


# swin3d = Swin3DMIMDownstream(config.model, config.training)
# swin3d.swin.load_state_dict(state_dict)
# swin3d.swin.eval()  # Freeze encoder
# datamodule = InfarctSegmentationDataModule(config.data)
