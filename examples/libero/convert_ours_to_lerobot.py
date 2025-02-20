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

'''
uv run examples/libero/convert_ours_to_lerobot.py --data-dir /hy-tmp/lmz/pi0_data/example --create-from-scratch
param:
create-from-scratch: create lerobot dataset from scratch. this will Clean up any existing dataset in the output directory
'''

REPO_NAME = "aloha_ours_lerobot3"  # Name of the output dataset, also used for the Hugging Face Hub
DATASET_TASK = {
    # "aloha_ours" : "fold the shirt.",
    # 'aloha_ours_6steps': 'fold the shirt in 6 step.',
    '20250117_Y_AL06_dieyifu_01_pretra_6steps_WXY_ai': 'fold the shirt in 6 step.',
}

def main(data_dir: str, *, 
         push_to_hub: bool = False, 
         create_from_scratch: bool = False,
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
                "action": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    else:
        dataset = LeRobotDataset(repo_id=REPO_NAME, local_files_only=True)

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name, task_instruction in DATASET_TASK.items():
        # raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        hdf5s_path = os.path.join(data_dir, raw_dataset_name)
        hdf5_file_names = os.listdir(hdf5s_path)
        hdf5_file_names.sort()
        for hdf5_file_name in tqdm.tqdm(hdf5_file_names, total=len(hdf5_file_names)):
            hdf5_file_path = os.path.join(hdf5s_path, hdf5_file_name)
            mapping = {"observation.images.cam_high": 'camera0_rgb', "observation.images.cam_left_wrist": 'camera1_rgb', "observation.images.cam_right_wrist": 'camera2_rgb', "observation.state": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle'], "action": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle']}
            value_dict = {"observation.images.cam_high": None, "observation.images.cam_left_wrist": None, "observation.images.cam_right_wrist": None, "observation.state": None, "action": None}
            with h5py.File(hdf5_file_path, "r") as ep:

                for i, (key, item) in enumerate(value_dict.items()):
                    # print(key, item.shape)
                    if type(mapping[key]) == list:
                        v = []
                        for each_key in mapping[key]:
                            # print(each_key, ep[each_key].shape, type(ep[each_key]))
                            v.append(torch.from_numpy(np.array(ep[each_key])))
                        v = torch.cat(v, dim=1)
                        v = torch.cat([v[..., :8], v[..., 9:15]], dim=1)
                        value_dict[key] = v
                    else:
                        value_dict[key] = torch.from_numpy(np.array(ep[mapping[key]]))

                # print(value_dict['observation.state'][10:20, 1:2])
                # print("min max, ", torch.max(value_dict['observation.state'][..., 1:2]), torch.min(value_dict['observation.state'][..., 0:1]))
                # exit()
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
                            "action": value_dict["action"][i],
                        }
                    )
                dataset.save_episode(task=task_instruction)

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
