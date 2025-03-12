"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import os
import sys
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro
import h5py
import torch
import tqdm
import numpy as np
import time
import ipdb
'''
uv run examples/libero/convert_ours_to_lerobot.py --data-dir /hy-tmp/lmz/pi0_data/example --create-from-scratch
param:
create-from-scratch: create lerobot dataset from scratch. this will Clean up any existing dataset in the output directory
'''
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13 
FPS = 30
REPO_NAME = "Fold_Shirt_0312"  # Name of the output dataset, also used for the Hugging Face Hub
#XDG_CACHE_HOME=/pfstem/likaiyu/resources/.cache

JOINT_MAPPING = {
    "observation.images.cam_high": 'camera0_rgb', 
    "observation.images.cam_left_wrist": 'camera1_rgb', 
    "observation.images.cam_right_wrist": 'camera2_rgb', 
    "observation.state": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle'], 
    "actions": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle'],
}

EEF_MAPPING = {
    "observation.images.cam_high": 'camera0_rgb', 
    "observation.images.cam_left_wrist": 'camera1_rgb', 
    "observation.images.cam_right_wrist": 'camera2_rgb',
    "observation.state":['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot1_eef_pos', 'robot1_eef_rot_axis_angle', 'robot1_gripper_width'],
    "actions":['robot0_cmd_eef_pos', 'robot0_cmd_eef_rot_axis_angle', 'robot0_gripper_width', 'robot1_cmd_eef_pos', 'robot1_cmd_eef_rot_axis_angle', 'robot1_gripper_width'],
}

dataset_paths = [
                "/mnt/pfs-chihiro/20250311",
                "/mnt/pfs-chihiro/20250310"
                ]
dataset_files = []
for dataset_path in dataset_paths:
    dataset_files += [os.path.join(dataset_path, p) for p in os.listdir(dataset_path) if 'QUICK_HF' in p]

DATASET_TASK = {}
# '/pfstem/likaiyu/resources/hdf5/0_1new/20250225_Y_AL02_DYF03_PI0STEP01FINE_CXJ_ai_hdf5':'Flatten the shirt',
for p in dataset_files:
    DATASET_TASK[p] = 'Flatten the shirt' if 'QUICK_HF' not in p else 'Fold the shirt'


# ipdb.set_trace()
def main(data_dir: str = '/pfstem/likaiyu/resources/hdf5', *, 
         push_to_hub: bool = False, 
         create_from_scratch: bool = True,
         mapping:dict = EEF_MAPPING,
         ):
    # Clean up any existing dataset in the output directory
    if create_from_scratch:
        output_path = LEROBOT_HOME / REPO_NAME
        print (f'remove old path: {output_path}')
        if output_path.exists():
            shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    if create_from_scratch:
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="aloha",
            fps=FPS,
            features={
                "observation.images.cam_high": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.cam_left_wrist": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.cam_right_wrist": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=40,
            image_writer_processes=30,
        )
    else:
        dataset = LeRobotDataset(repo_id=REPO_NAME, local_files_only=True)

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    # H5 Action Format:
    #   tensor Nx22
    #   1      1       7                7               6      
    #   l_grip r_grip  l_joints+l_grip  r_joints+r_grip padding 
    #
    # Aloha Action Format:
    #   tensor Nx14
    #   6        1       6         1
    #   l_joints l_grip  r_joints  r_grip

    for raw_dataset_name, task_instruction in DATASET_TASK.items():
        # raw_dataset_name, task_instruction = pair    
        hdf5s_path = os.path.join(data_dir, raw_dataset_name)
        hdf5_file_names = os.listdir(hdf5s_path)
        hdf5_file_names.sort()
        for hdf5_file_name in tqdm.tqdm(hdf5_file_names, total=len(hdf5_file_names)):
            hdf5_file_path = os.path.join(hdf5s_path, hdf5_file_name)
            value_dict = {"observation.images.cam_high": None, "observation.images.cam_left_wrist": None, "observation.images.cam_right_wrist": None, "observation.state": None, "actions": None}
            with h5py.File(hdf5_file_path, "r") as ep:

                for i, (key, item) in enumerate(value_dict.items()):
                    # print(key, item.shape)
                    if type(mapping[key]) == list:
                        v = []
                        for each_key in mapping[key]:
                            # print(each_key, ep[each_key].shape, type(ep[each_key]))
                            try:
                                v.append(torch.from_numpy(np.array(ep[each_key])))
                            except:
                                import ipdb; ipdb.set_trace()
                        v = torch.cat(v, dim=1)
                        if mapping == JOINT_MAPPING:
                            v = v[..., 2:16]
                        value_dict[key] = v
                    else:
                        value_dict[key] = torch.from_numpy(np.array(ep[mapping[key]]))

                ### norm gripper
                gripper = value_dict['observation.state'][..., LEFT_GRIPPER:LEFT_GRIPPER+1]
                min_value = torch.min(gripper, dim=0, keepdim=True)[0]
                normed_value = gripper - min_value
                normed_value = normed_value / (torch.max(normed_value, dim=0, keepdim=True)[0]+1e-6)
                value_dict['observation.state'][..., LEFT_GRIPPER:LEFT_GRIPPER+1] = normed_value * 5.
                gripper = value_dict['observation.state'][..., RIGHT_GRIPPER:RIGHT_GRIPPER+1]
                min_value = torch.min(gripper, dim=0, keepdim=True)[0]
                normed_value = gripper - min_value
                normed_value = normed_value / (torch.max(normed_value, dim=0, keepdim=True)[0]+1e-6) 
                value_dict['observation.state'][..., RIGHT_GRIPPER:RIGHT_GRIPPER+1] = normed_value * 5.
                ##################
                # value_dict['action'] = value_dict['observation.state']

                len_traj = value_dict["observation.state"].shape[0]
                for i in range(len_traj):
                    dataset.add_frame(
                        {
                            "observation.images.cam_high": value_dict['observation.images.cam_high'][i],
                            "observation.images.cam_left_wrist": value_dict['observation.images.cam_left_wrist'][i],
                            "observation.images.cam_right_wrist": value_dict['observation.images.cam_right_wrist'][i],
                            "observation.state": value_dict['observation.state'][i],
                            "actions": value_dict["actions"][i],
                        }
                    )
                dataset.save_episode(task=task_instruction)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False, keep_image_files = False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tic = time.time()
    tyro.cli(main)
    toc = time.time()
    print(f'Convert lerobot in {(toc-tic)//60} mins {(toc-tic)%60} secs')


