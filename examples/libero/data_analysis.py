import os
import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import ffmpeg
import ipdb
import torch
import tqdm
import matplotlib.pyplot as plt
# XDG_CACHE_HOME=/pfstem/likaiyu/resources/.cache



def plot_robot_xyplane_stats(points, save_name):
    points_np = points.numpy()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(points_np[:, 0], points_np[:, 1],
                s=2,               # dot size
                c='blue',          # dot color
                alpha=0.05)        # transparency to emphasize density

    plt.title('Robot eef on XY plane')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'/root/PI_Official/data/visualization/{save_name}.png', dpi=300, bbox_inches='tight')


repo_id='MultiTask_7task_0422'
sample_num = 10_000

dataset = LeRobotDataset(repo_id, local_files_only=True)
action_points = []
state_points = []
for i in tqdm.tqdm(range(sample_num)):

    item = dataset[i]
    action_points.append(item['actions'][:2])
    action_points.append(item['actions'][7:9])
    state_points.append(item['observation.state'][:2])
    state_points.append(item['observation.state'][7:9])

action_points = torch.stack(action_points)
state_points = torch.stack(state_points)

print('Plotting action stats...')
plot_robot_xyplane_stats(action_points, repo_id + '_action')
print('Plotting state stats...')
plot_robot_xyplane_stats(state_points, repo_id + '_state')
