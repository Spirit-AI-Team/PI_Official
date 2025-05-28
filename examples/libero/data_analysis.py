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
# import seaborn as sns
import matplotlib.patches as patches
from random import sample 
# !export XDG_CACHE_HOME=/pfstem/likaiyu/resources/.cache

#XY平面分布
def plot_robot_xyplane_stats(action_points, state_points, save_name):
    points_np = action_points.numpy()[...,:2]
    points_np2 = state_points.numpy()[...,:2]
    # Plot
    plt.figure(figsize=(8, 6))
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    plt.scatter(points_np[:, 0], points_np[:, 1],
                s=2,               # dot size
                c='red',          # dot color
                alpha=.1)        # transparency to emphasize density
    plt.scatter(points_np2[:, 0], points_np2[:, 1],
                s=2,               # dot size
                c='blue',          # dot color
                alpha=.1)        # transparency to emphasize density
    # patches.Rectangle((-0.6,))
    plt.axis([-0.6,0.6,-1.295,-0.095])
    plt.title('Robot EEF on XY plane')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    legend = plt.legend(['action','state'])

    for handle in legend.legend_handles:
        handle.set_alpha(1)
    plt.savefig(f'/root/PI_Official/data/visualization/XYPlane/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
#XYZ分布
def plot_singledim_stats(action_points, state_points, save_name):
    points_np = action_points.numpy()
    points_np2 = state_points.numpy()
    data = np.concatenate([points_np2[::2], points_np2[1::2]],1)
    # points = np.concatenate([points_np, points_np2], axis = 0)
    # data = points.T
    print(f'data shape {data.shape}')

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, patch_artist=True)
    plt.title("Boxplot of Each EEF Dimension")
    plt.ylabel("EEF Value")
    plt.xticks(range(1, 7), ['L_X','L_Y','L_Z','R_X','R_Y','R_Z'])  # label x-axis from Dim 0 to Dim 13
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('/root/PI_Official/data/visualization/XYZ_BoxPlot', exist_ok = True)
    plt.savefig(f'/root/PI_Official/data/visualization/XYZ_BoxPlot/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def cal_cmd_vel(state_points, delta = 3):
    state_points = state_points.numpy()
    left_points = state_points[::2]
    right_points = state_points[1::2]
    diff = left_points[delta:] - left_points[:-delta]
    left_mean = np.sqrt(np.sum(diff*diff, axis=1)).mean()
    diff = right_points[delta:] - right_points[:-delta]
    right_mean = np.sqrt(np.sum(diff*diff, axis=1)).mean()
    return (left_mean + right_mean)/2

def cal_cmd_state_diff(action_points, state_points):
    action_points = action_points.numpy()
    state_points = state_points.numpy()
    diff = action_points - state_points
    return np.sqrt(np.sum(diff*diff, axis=1)).mean()

#臂角限制
def plot_angle_limit_violation(state_points, save_name, threshold = 5, sample_num = 10_000):
    left_angle_limit = [
    [-180,60],[-180,3],[-175,175],[-129,90],[-175,175],[-95,95],[-90,90]
    ]
    right_angle_limit = [
        [-60,180],[-180,3],[-175,175],[-90,129],[-175,175],[-95,95],[-90,90]
    ]
    joint_names = [f'LJ{i}' for i in range(1,8)] + [f'RJ{i}' for i in range(1,8)]
    left_angle_limit = np.array(left_angle_limit)
    right_angle_limit = np.array(right_angle_limit)
    angle_limit = np.concatenate([left_angle_limit,right_angle_limit])

    state_points = 180 * state_points/np.pi
    state_points = state_points.numpy()
    up_violation = np.sum(np.int8(state_points >= angle_limit[:,1]-threshold),axis=0)
    down_violation = np.sum(np.int8(state_points <= angle_limit[:,0]+threshold),axis=0)
    plt.figure(figsize=(8, 6))
    plt.bar(joint_names, up_violation*100/sample_num,color='skyblue',alpha=.5)
    plt.bar(joint_names, down_violation*100/sample_num,color='yellow',alpha=.5)
    legend = plt.legend(['up-limit','down-limit'])
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    plt.title('Angle Limit Violation By Joint')
    # plt.xlabel('X')
    plt.ylabel('Probability of Violation')
    plt.savefig(f'/root/PI_Official/data/visualization/Joint_Limit/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 按条数plot
def plot_variant_comparison(action_points, state_points, title="Left Arm", save_name=None):
    """
    Plot comparison of two time-series variants over 7 dimensions.
    
    Args:
        action_points (np.ndarray): Shape (t, 7), first variant.
        state_points (np.ndarray): Shape (t, 7), second variant.
        title (str): Overall title of the plot.
        save_name (str or None): If provided, saves the figure to this path.
    """
    assert action_points.shape == state_points.shape, "Variants must have the same shape"
    assert action_points.shape[1] == 7, "Each variant must have 7 dimensions"
    axis_names = ['X','Y','Z','r','p','y','G']
    max_frame = 900
    action_points = action_points[:max_frame]
    state_points = state_points[:max_frame]

    t = action_points.shape[0]
    time = np.arange(t)/30

    fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True, dpi = 300)
    fig.suptitle(title, fontsize=14)

    for i in range(7):
        axes[i].plot(time, action_points[:, i], 'b--o', markersize=2,
               markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.5,
               label='action', alpha=0.6)
        axes[i].plot(time, state_points[:, i], 'r--o', markersize=2,
               markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.5,
               label='state', alpha=0.6)
        axes[i].set_ylabel(axis_names[i])
        axes[i].grid(True)
        if i == 0:
            axes[i].legend(loc='upper right')

    axes[-1].set_xlabel('Time Step(s)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_name:
        plt.savefig(f'/root/PI_Official/data/visualization/episode_demo/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

#解决rot vec突变
def preprocess_rotvec(points):
    assert points.shape[0] > 1
    rot_vec = points[...,3:6]
    rot_vec = rot_vec.numpy()
    for i in range(1, points.shape[0]):
        if np.dot(rot_vec[i], rot_vec[i-1]) < 0:
            rot_vec[i] *= -1
            print(f'norm is {np.sqrt(np.sum(rot_vec[i]**2))}')
    points[..., 3:6] = torch.tensor(rot_vec)
    return points

if __name__ == "__main__":
    #加载数据集
    repo_id='HRPI_MultiTask_PutPlateOnRack_0423_29'
    sample_num = 10_000

    dataset = LeRobotDataset(repo_id, local_files_only=True)
    import ipdb;ipdb.set_trace()
    action_points = []
    state_points = []
    #sample 10000个点
    for i in tqdm.tqdm(sample(range(len(dataset)),sample_num)):

        item = dataset[i]
        action_points.append(item['actions'][2:])
        state_points.append(item['observation.state'][2:])

    action_points = torch.stack(action_points)
    state_points = torch.stack(state_points)

    #可视化
    plot_angle_limit_violation(state_points, repo_id)
    # plot_robot_xyplane_stats(action_points, state_points, repo_id)
    # plot_singledim_stats(action_points, state_points, repo_id)

    #按条数加载数据集
    repo_id='YC_MeetingRoom_PutPenInBox_0514'
    # repo_id = 'MultiTask_7task_0422'
    # repo_id = 'HRPI_MultiTask_PutPlateOnRack_0423_29'
    dataset = LeRobotDataset(repo_id, episodes = list(range(100)),local_files_only=True)
    episode_id = 5
    start,end = dataset.episode_data_index['from'][episode_id], dataset.episode_data_index['to'][episode_id]

    l_action_points, r_action_points = [],[]
    l_state_points, r_state_points = [],[]
    for i in tqdm.tqdm(range(start,end)):
        item = dataset[i]
        l_action_points.append(item['actions'][:7])
        r_action_points.append(item['actions'][7:])
        l_state_points.append(item['observation.state'][:7])
        r_state_points.append(item['observation.state'][7:])

    l_action_points = torch.stack(l_action_points)
    r_action_points = torch.stack(r_action_points)
    l_state_points = torch.stack(l_state_points)
    r_state_points = torch.stack(r_state_points)
    l_action_points = preprocess_rotvec(l_action_points)
    l_state_points = preprocess_rotvec(l_state_points)
    plot_variant_comparison(l_action_points, l_state_points, save_name = repo_id + '_' + str(episode_id))