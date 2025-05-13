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
import glob
import json
'''
uv run examples/libero/convert_ours_to_lerobot.py --data-dir /hy-tmp/lmz/pi0_data/example --create-from-scratch
param:
create-from-scratch: create lerobot dataset from scratch. this will Clean up any existing dataset in the output directory
'''
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13 
FPS = 30
REPO_NAME = "YC_MultiTask_PutPlateOnRack_0512"#"ALLShirt_EEF_0307_19"  # Name of the output dataset, also used for the Hugging Face Hub
#XDG_CACHE_HOME=/pfstem/likaiyu/resources/.cache

JOINT_MAPPING = {
    "observation.images.cam_high": 'camera0_rgb', 
    "observation.images.cam_left_wrist": 'camera1_rgb', 
    "observation.images.cam_right_wrist": 'camera2_rgb', 
    "observation.state": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle'], 
    "actions": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle'],
}

EEF_MAPPING_CMD = {
    "observation.images.cam_high": 'camera0_rgb', 
    "observation.images.cam_left_wrist": 'camera1_rgb', 
    "observation.images.cam_right_wrist": 'camera2_rgb',
    "observation.state":['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot1_eef_pos', 'robot1_eef_rot_axis_angle', 'robot1_gripper_width'],
    "actions":['robot0_cmd_eef_pos', 'robot0_cmd_eef_rot_axis_angle', 'robot0_gripper_width', 'robot1_cmd_eef_pos', 'robot1_cmd_eef_rot_axis_angle', 'robot1_gripper_width'],
}

EEF_MAPPING_CMD_GRIPPER = {
    "observation.images.cam_high": 'camera0_rgb', 
    "observation.images.cam_left_wrist": 'camera1_rgb', 
    "observation.images.cam_right_wrist": 'camera2_rgb',
    "observation.state":['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot1_eef_pos', 'robot1_eef_rot_axis_angle', 'robot1_gripper_width'],
    "actions":['robot0_cmd_eef_pos', 'robot0_cmd_eef_rot_axis_angle', 'robot0_gripper_cmd_width', 'robot1_cmd_eef_pos', 'robot1_cmd_eef_rot_axis_angle', 'robot1_gripper_cmd_width'],
}

EEF_MAPPING = {
    "observation.images.cam_high": 'camera0_rgb', 
    "observation.images.cam_left_wrist": 'camera1_rgb', 
    "observation.images.cam_right_wrist": 'camera2_rgb',
     "observation.state":['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot1_eef_pos', 'robot1_eef_rot_axis_angle', 'robot1_gripper_width'],
     "actions":['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width', 'robot1_eef_pos', 'robot1_eef_rot_axis_angle', 'robot1_gripper_width'],
}

device = '_Y_M1'#'HRPI' '_Y_AL' '_Y_M1'
dataset_paths = [
                    # '/mnt/pfs-chihiro/20250509',
                    # '/mnt/pfs-chihiro/20250510'
                    '/mnt/pfs-chihiro/20250512',
                ]
dataset_files = []
for dataset_path in dataset_paths:
    dataset_files += [os.path.join(dataset_path, p) for p in os.listdir(dataset_path) if device in p]

BLACK_LIST = set([
    '20250422_Y_AL05_PICKPLACE_AUG01_PutBottleInOrganizer_FMS',
    '20250421_Y_AL09_PICKPLACE_AUG02_PutBowInOrganizer_LYB',
    '20250421_Y_AL05_PICKPLACE_AUG01_PutBowInOrganizer_ZYD',
    '20250417_Y_AL0X_PICKPLACE_AUG02_PutTissueInOrganizer_SZH',
    '20250417_Y_AL04_PICKPLACE_AUG01_PutTissueInOrganizer_FMS',
    '20250415_Y_AL05_PICKPLACE_PutEgglnEggrack_ZWS',
    '20250414_Y_AL05_PICKPLACE_PutEgglnEggrack_ZWS01',
    '20250414_Y_AL04_PICKPLACE_PutEggInEggrack_HZY01',
    '20250414_Y_AL04_PICKPLACE_PutEggInEggrack_HZY',
    '20250414_Y_AL05_PICKPLACE_PutEgglnEggrack_ZWS',
])

DATASET_TASK = {}
for p in dataset_files:

    if os.path.basename(p) in BLACK_LIST:
        print(f'Find {os.path.basename(p)} in black list, remove the path')
        continue

    # if 'PutEggInEggrack' in p or 'PutEgglnEggrack' in p:
        # DATASET_TASK[p] = 'There is an egg and an egg tray on the table. Locate the egg, pick up the egg, and place the egg onto the egg tray.'
    # if 'PutToyInOrganizer' in p:
    #     DATASET_TASK[p] = 'There is a toy and an organizer on the table. Locate the toy, pick up the toy, and place the toy into the organizer.'
    # if 'PutTissueInOrganizer' in p:
    #     DATASET_TASK[p] = 'There is a tissue and an organizer on the table. Locate the tissue, pick up the tissue, and place the tissue into the organizer.'
    if 'PutPlateOnRack' in p:
        DATASET_TASK[p] = 'There is a plate and a dish drainer on the table. Locate the plate, pick up the plate, and place the plate onto the dish drainer.'
    # if 'PutAnimalInOrganizer' in p:
    #     DATASET_TASK[p] = 'There is an animal and an organizer on the table. Locate the animal, pick up the animal, and place the animal into the organizer.'
    # if 'PutCanInOrganizer' in p:
    #     DATASET_TASK[p] = 'There is a can and an organizer on the table. Locate the can, pick up the can, and place the can into the organizer.'
    # if 'PutPenInPenhold' in p:
    #     DATASET_TASK[p] = 'There is a pen and a penhold on the table. Locate the pen, pick up the pen, and place the pen into the penhold.'
    # if 'PutBowInOrganizer' in p:
    #     DATASET_TASK[p] = 'There is a bow and an organizer on the table. Locate the bow, pick up the bow, and place the bow into the organizer.'
    # if 'PutBottleInOrganizer' in p:
    #     DATASET_TASK[p] = 'There is a bottle and a organizer on the table. Locate the bottle, pick up the bottle, and place the bottle into the organizer.'
    # if 'PutGlueInPenhold' in p:
    #     DATASET_TASK[p] = 'There is a glue stick and a penhold on the table. Locate the glue stick, pick up the glue stick, and place the glue stick into the penhold.'
    # if 'STEP01AUG06' in p:
        # DATASET_TASK[p] = "Flatten the shirt"
        # DATASET_TASK[p] = 'Flatten the shirt: pick a shirt from the basket, put it on the table and then flatten the shirt and make it horizontal to your side of table'
    # if 'PI0STACK_HF' in p:
    #     # DATASET_TASK[p] = "Fold up again and stack the shirt to the corner"
        # DATASET_TASK[p] = 'Stack the shirt: fold up the shirt again and stack the shirt to the corner'
    # if '15QUICK_HF' in p:
    #     # DATASET_TASK[p] = "Fold the shirt"
        #  DATASET_TASK[p] = "Fold the shirt: fold up once on both sides of the shirt, then rotate the shirt to vertical state, and finally fold the bottom part to the top"

def main(data_dir: str = '', *, 
         push_to_hub: bool = False, 
         create_from_scratch: bool = True,
         mapping:dict = EEF_MAPPING_CMD_GRIPPER,
         ):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if create_from_scratch:
        # print (f'remove old path: {output_path}')
        if output_path.exists():
            content = input('Dataset already exists! Do you want to remove it?: Y/N')
            if content == 'Y':
                shutil.rmtree(output_path)
            else:
                return

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
            image_writer_threads=20,
            image_writer_processes=10,
        )
    else:
        dataset = LeRobotDataset(repo_id=REPO_NAME, local_files_only=True)
        dataset.start_image_writer(num_threads=20, num_processes=10)

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
    cnt = len(glob.glob(os.path.join(output_path, 'data_info*.json')))
    with open(f'{output_path}/data_info{cnt}.json','w') as f:
        json.dump(DATASET_TASK, f)
    for raw_dataset_name, task_instruction in DATASET_TASK.items():
        # raw_dataset_name, task_instruction = pair  
        # try:
        hdf5s_path = os.path.join(data_dir, raw_dataset_name)
        hdf5_file_names = glob.glob(os.path.join(raw_dataset_name, '*.hdf5'))
        hdf5_file_names.sort(key = lambda x:int(x.split('/')[-1][:-5]))
        json_file = glob.glob(os.path.join(raw_dataset_name, 'info.json'))[0]
        with open(json_file, 'r') as f:
            content = json.load(f)
            valid_inds = []
            for i, data in enumerate(content['datasets']):
                if data['validity'] == 1:
                    valid_inds.append(i)
        # import ipdb;ipdb.set_trace()
        hdf5_file_names = [hdf5_file_names[i] for i in valid_inds]
        for hdf5_file_name in tqdm.tqdm(hdf5_file_names, total=len(hdf5_file_names)):
            hdf5_file_path = os.path.join(hdf5s_path, hdf5_file_name)
            value_dict = {"observation.images.cam_high": None, "observation.images.cam_left_wrist": None, "observation.images.cam_right_wrist": None, "observation.state": None, "actions": None}
            with h5py.File(hdf5_file_path, "r") as ep:
                for i, (key, item) in enumerate(value_dict.items()):
                    if type(mapping[key]) == list:
                        v = []
                        for each_key in mapping[key]:
                            # print(each_key, ep[each_key].shape, type(ep[each_key]))
                            v.append(torch.from_numpy(np.array(ep[each_key])))
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
                # normed_value = normed_value / (torch.max(normed_value, dim=0, keepdim=True)[0]+1e-6)
                value_dict['observation.state'][..., LEFT_GRIPPER:LEFT_GRIPPER+1] = normed_value #* 5.
                gripper = value_dict['observation.state'][..., RIGHT_GRIPPER:RIGHT_GRIPPER+1]
                min_value = torch.min(gripper, dim=0, keepdim=True)[0]
                normed_value = gripper - min_value
                # normed_value = normed_value / (torch.max(normed_value, dim=0, keepdim=True)[0]+1e-6) 
                value_dict['observation.state'][..., RIGHT_GRIPPER:RIGHT_GRIPPER+1] = normed_value #* 5.

                gripper = value_dict['actions'][..., LEFT_GRIPPER:LEFT_GRIPPER+1]
                min_value = torch.min(gripper, dim=0, keepdim=True)[0]
                normed_value = gripper - min_value
                # normed_value = normed_value / (torch.max(normed_value, dim=0, keepdim=True)[0]+1e-6)
                value_dict['actions'][..., LEFT_GRIPPER:LEFT_GRIPPER+1] = normed_value #* 5.
                gripper = value_dict['actions'][..., RIGHT_GRIPPER:RIGHT_GRIPPER+1]
                min_value = torch.min(gripper, dim=0, keepdim=True)[0]
                normed_value = gripper - min_value
                # normed_value = normed_value / (torch.max(normed_value, dim=0, keepdim=True)[0]+1e-6) 
                value_dict['actions'][..., RIGHT_GRIPPER:RIGHT_GRIPPER+1] = normed_value #* 5.

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
        # except Exception as e:
        #     log = open('/root/PI_Official/data/error.txt', 'w')  
        #     log.write(f'{raw_dataset_name} does not have cmd eef\n')
        #     log.close()
        #     continue

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

