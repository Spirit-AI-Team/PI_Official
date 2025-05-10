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

import tos
import io



'''
uv run examples/libero/convert_ours_to_lerobot.py --data-dir /hy-tmp/lmz/pi0_data/example --create-from-scratch
param:
create-from-scratch: create lerobot dataset from scratch. this will Clean up any existing dataset in the output directory
'''
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13 
FPS = 30
REPO_NAME = "HRPI_PutObjectlnDrawer_0509_test"#"ALLShirt_EEF_0307_19"  # Name of the output dataset, also used for the Hugging Face Hub
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

def load_hdf5_file(client: tos.TosClientV2, bucket_name: str, object: str) -> h5py.File:
    response = client.get_object(bucket_name, object)
    return h5py.File(io.BytesIO(response.content.read()), 'r')

def load_file(client: tos.TosClientV2, bucket_name: str, object: str) -> bytes:
    response = client.get_object(bucket_name, object)
    return response.content.read()

def main(data_dir: str = '', *, 
         push_to_hub: bool = False, 
         create_from_scratch: bool = False,
         mapping:dict = EEF_MAPPING_CMD_GRIPPER,
         ):
    
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    print (output_path)
    import pdb;pdb.set_trace()
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
                    "dtype": "float64",
                    "shape": (14,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float64",
                    "shape": (14,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=40,
            image_writer_processes=10,
        )
    else:
        dataset = LeRobotDataset(repo_id=REPO_NAME, local_files_only=True)
        dataset.start_image_writer(num_threads=40, num_processes=10)

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
    # cnt = len(glob.glob(os.path.join(output_path, 'data_info*.json')))
    # with open(f'{output_path}/data_info{cnt}.json','w') as f:
    #     json.dump(DATASET_TASK, f)


    ak = os.getenv('TOS_ACCESS_KEY')
    sk = os.getenv('TOS_SECRET_KEY')

    endpoint = "https://tos-cn-beijing.ivolces.com"
    region = "cn-beijing"

    client = tos.TosClientV2(ak, sk, endpoint, region)
    bucket_name = "chihiro"

    dataset_paths = [
        # '20250402',
        # '20250403',
        # '20250507/20250507_Z_HRPI03_MULTI_PullOutDrawer_LJM/',   # 加后面的/避免相同的结果
        # '20250507/20250507_Z_HRPI03_MULTI_PullOutDrawer_LJM01/',
        # '20250507/20250507_Z_HRPI02_MULTI_PullOutDrawer_CH01/',
        '20250509/20250509_Z_HRPI01_MULTI_PutObjectInDrawer_LJ/',
        '20250509/20250509_Z_HRPI02_MULTI_PutObjectInDrawer_LJ/',
        '20250509/20250509_Z_HRPI03_MULTI_PutObjectInDrawer_LTJ/',
        '20250509/20250509_Z_HRPI04_MULTI_PutObjectlnDrawer_WJY/',
    ]

    for dataset_path in dataset_paths:
        list_response = client.list_objects_type2(bucket_name, prefix=dataset_path)
        hdf5_files = []
        other_files = []
        for obj in list_response.contents:
            name = os.path.basename(obj.key)
            if name.endswith(".hdf5"):
                hdf5_files.append(obj.key)
            else:
                other_files.append(obj.key)

        DATASET_TASK = {}
        for p in hdf5_files:
            if 'action1' in p:
                DATASET_TASK[p] = 'action1:place the cup on the coffee machine'
            elif 'action2' in p:
                DATASET_TASK[p] = 'action2:Click the espresso button to making the coffee'
            elif 'action3' in p:
                DATASET_TASK[p] = 'action3:Take the cup off the coffee machine and place it on the table'
            elif 'StackCup1To3' in p:
                DATASET_TASK[p] = 'Stack the cups from one to three.'
            # elif 'StackCup1To3_AUG02' in p:
            # elif 'AUG01' in p:
            #     DATASET_TASK[p] = 'Stack the cups from one to three. Step two: put the second cup near the first cup.'
            # elif 'AUG02' in p:
            #     DATASET_TASK[p] = 'Stack the cups from one to three. Step three: put the third cup near the first and second cup to make them form a triangle.'
            elif 'PickUpCup' in p:
                DATASET_TASK[p] = 'Start by positioning the cup correctly, then secure it with a clamp.'
            elif 'PickUpCup2' in p:
                DATASET_TASK[p] = 'Pick up the fallen cup and place it back in an upright position.'
            elif 'PullOutDrawer' in p:
                DATASET_TASK[p] = 'Pull out the drawer and take out all the objects inside.'
            elif 'PutObjectlnDrawer' in p:
                DATASET_TASK[p] = 'Put the object in the drawer.'
            else:
                DATASET_TASK[p] = 'Rip off last piece of tissue from the toilet roll, then use the sticker to seal the toilet roll.'

        info_path = os.path.join(dataset_path, 'info.json')
        assert info_path in other_files, f"Error: {info_path} not found in other_files." 
        
        info_data = json.loads(load_file(client, bucket_name, info_path)) 
        validities = {}
        for dataset_item in info_data['datasets']:
            key = dataset_item['ID']
            value = dataset_item['validity']
            validities[key] = value

        if len(hdf5_files) != len(validities.keys()):
            print(f'len(hdf5_files): {len(hdf5_files)} != len(validities.keys()) {len(validities.keys())}')
            continue

        for raw_dataset_name, task_instruction in tqdm.tqdm(DATASET_TASK.items()):
            file_idx = int (os.path.basename(raw_dataset_name).split('.')[0])
            # if file_idx in fail_list:
            #     continue
            
            assert file_idx in validities.keys()
            
            if not validities[file_idx]:
                print (f'{raw_dataset_name} not valid')
                continue
            
            hdf5_file = load_file(client, bucket_name, raw_dataset_name)
            value_dict = {"observation.images.cam_high": None, "observation.images.cam_left_wrist": None, "observation.images.cam_right_wrist": None, "observation.state": None, "actions": None}
            with h5py.File(io.BytesIO(hdf5_file), "r") as ep:

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
                # import ipdb;ipdb.set_trace()
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


