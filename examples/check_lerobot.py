import shutil
import os
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro
import h5py
import torch
import tqdm
import numpy as np

# os.system('export XDG_CACHE_HOME=/pfstem/likaiyu/resources/.cache')
REPO_NAME = "20250214_Y_AL05_zhuyaoshibiao_GY_ai_biaozhu"
dataset = LeRobotDataset(repo_id=REPO_NAME, local_files_only=True)