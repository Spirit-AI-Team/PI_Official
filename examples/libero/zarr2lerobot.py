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

'''
uv run examples/libero/zarr2lerobot.py --data-dir /pfstem/likaiyu/resources/raw_data/4steps --create-from-scratch
param:
create-from-scratch: create lerobot dataset from scratch. this will Clean up any existing dataset in the output directory
'''

import sys
sys.path.append("/root/shuo/universal_manipulation_interface")

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
import time

import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer

REPO_NAME = "exp_zarrzip_input3"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = {
    "20250218_Y_AL06_DYF02_4STEPS_WXY_ai.zarr.zip": "fold the shirt3.",
}  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False, create_from_scratch: bool = True):
    # Clean up any existing dataset in the output directory
    if create_from_scratch:
        output_path = LEROBOT_HOME / REPO_NAME
        print(f'remove old path: {output_path}')
        if output_path.exists():
            shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    if create_from_scratch:
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
            image_writer_threads=1,
            image_writer_processes=1,
        )
    else:
        dataset = LeRobotDataset(repo_id=REPO_NAME, local_files_only=True)

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name, task in RAW_DATASET_NAMES.items():
        print(f"Converting {raw_dataset_name}...")
        zarr_path = os.path.join(data_dir, raw_dataset_name)
        mapping = {"observation.images.cam_high": 'camera0_rgb', "observation.images.cam_left_wrist": 'camera1_rgb', "observation.images.cam_right_wrist": 'camera2_rgb', "observation.state": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle'], "action": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle']}

        with zarr.ZipStore(zarr_path, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store,
                store=zarr.MemoryStore()
            )

        episode_size = replay_buffer.meta['episode_ends']

        for key, value in replay_buffer.data.items():
            if 'gripper' in key:
                min_value = np.min(value, axis=0, keepdims=True)
                normed_value = value - min_value
                normed_value = normed_value / np.max(normed_value, axis=0, keepdims=True)
                replay_buffer.data[key] = normed_value * 5.

        episode_size = episode_size[:5]
        for idx in tqdm.tqdm(range(len(episode_size)), desc="episode"):
            start_idx = 0 if idx == 0 else episode_size[idx - 1]
            end_idx = episode_size[idx]
            if end_idx < start_idx:
                raise ValueError(f'episode {idx} out of range')
            value_dict = {"observation.images.cam_high": None, "observation.images.cam_left_wrist": None,
                          "observation.images.cam_right_wrist": None, "observation.state": None, "action": None}
            for i, (key, item) in enumerate(value_dict.items()):
                if type(mapping[key]) == list:
                    v = []
                    for each_key in mapping[key]:
                        v.append(torch.from_numpy(np.array(replay_buffer.data[each_key][start_idx:end_idx])))
                    v = torch.cat(v, dim=1)
                    v = torch.cat([v[..., :8], v[..., 9:15]], dim=1)
                    value_dict[key] = v
                else:
                    value_dict[key] = torch.from_numpy(np.array(replay_buffer.data[mapping[key]][start_idx:end_idx]))

            value_dict['action'] = value_dict['observation.state']

            len_traj = value_dict["observation.state"].shape[0]
            for i in range(len_traj):
                dataset.add_frame(
                    {
                        "observation.images.cam_high": value_dict['observation.images.cam_high'][i],
                        "observation.images.cam_left_wrist": value_dict['observation.images.cam_left_wrist'][i],
                        "observation.images.cam_right_wrist": value_dict['observation.images.cam_right_wrist'][i],
                        "observation.state": value_dict['observation.state'][i],
                        "actions": value_dict["action"][i],
                    }
                )
            dataset.save_episode(task=task)

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
    tic = time.time()
    tyro.cli(main)
    toc = time.time()
    print(f'Convert lerobot in {(toc - tic) // 60} mins {(toc - tic) % 60} secs')
