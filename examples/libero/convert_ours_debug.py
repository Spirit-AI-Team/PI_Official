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
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro
import h5py
import torch
import tqdm
import numpy as np

REPO_NAME = "debug"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "Dieyifu_1_3_0214",
]  # For simplicity we will combine multiple Libero datasets into one training dataset
LEFT_GRIPPER = 0
RIGHT_GRIPPER = 13

def main(data_dir: str = '/hy-tmp/likaiyu/resources/ours', *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="aloha",
        fps=10,
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
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    # H5 Action Format:
    #   tensor Nx22
    #   1      1       7                7               6      
    #   l_grip r_grip  l_joints+l_grip  r_joints_r_grip padding 
    #
    # Aloha Action Format:
    #   tensor Nx14
    #   6        1       6         1
    #   l_joints l_grip  r_joints  r_grip

    for raw_dataset_name in RAW_DATASET_NAMES:
        # raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        hdf5s_path = os.path.join(data_dir, raw_dataset_name)
        hdf5_file_names = os.listdir(hdf5s_path)
        hdf5_file_names.sort()
        for hdf5_file_name in tqdm.tqdm(hdf5_file_names, total=len(hdf5_file_names)):
            hdf5_file_path = os.path.join(hdf5s_path, hdf5_file_name)
            if hdf5_file_path != "/hy-tmp/likaiyu/resources/ours/Dieyifu_1_3_0214/episode_146.hdf5":
                continue
            mapping = {"observation.images.cam_high": 'camera0_rgb', "observation.images.cam_left_wrist": 'camera1_rgb', "observation.images.cam_right_wrist": 'camera2_rgb', "observation.state": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle'], "actions": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle']}
            value_dict = {"observation.images.cam_high": None, "observation.images.cam_left_wrist": None, "observation.images.cam_right_wrist": None, "observation.state": None, "actions": None}
            with h5py.File(hdf5_file_path, "r") as ep:

                for i, (key, item) in enumerate(value_dict.items()):
                    if type(mapping[key]) == list:
                        v = []
                        for each_key in mapping[key]:
                            # print(each_key, ep[each_key].shape, type(ep[each_key]))
                            v.append(torch.from_numpy(np.array(ep[each_key])))
                        v = torch.cat(v, dim=1)
                        # v = v[..., 2:16]
                        v = torch.cat([v[..., :8], v[..., 9:15]], dim=1)
                        value_dict[key] = v
                    else:
                        value_dict[key] = torch.from_numpy(np.array(ep[mapping[key]]))

                for key, value in value_dict.items():
                    print(key, value.shape)

                ### norm gripper
                gripper = value_dict['observation.state'][..., LEFT_GRIPPER:LEFT_GRIPPER+1]
                print(gripper)
                min_value = torch.min(gripper, dim=0, keepdim=True)[0]
                print(min_value)
                normed_value = gripper - min_value
                tmp = torch.max(normed_value, dim=0, keepdim=True)[0]
                if tmp[0][0] < 0.01:
                    print(tmp[0][0], hdf5_file_path)
                normed_value = normed_value / (torch.max(normed_value, dim=0, keepdim=True)[0])
                value_dict['observation.state'][..., LEFT_GRIPPER:LEFT_GRIPPER+1] = normed_value * 5.
                gripper = value_dict['observation.state'][..., RIGHT_GRIPPER:LEFT_GRIPPER+1]
                min_value = torch.min(gripper, dim=0, keepdim=True)[0]
                normed_value = gripper - min_value
                normed_value = normed_value / (torch.max(normed_value, dim=0, keepdim=True)[0])
                value_dict['observation.state'][..., RIGHT_GRIPPER:LEFT_GRIPPER+1] = normed_value * 5.
                # print(value_dict['observation.state'][:5])
                ##################
                value_dict['action'] = value_dict['observation.state']

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
                dataset.save_episode(task="fold the shirt")

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
