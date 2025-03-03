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
import cv2
import ffmpeg

'''
uv run examples/libero/convert_ours_to_lerobot.py --data-dir /hy-tmp/lmz/pi0_data/example --create-from-scratch
param:
create-from-scratch: create lerobot dataset from scratch. this will Clean up any existing dataset in the output directory
'''
POOL_SIZE = 9
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13 
REPO_NAME = "temp"  # Name of the output dataset, also used for the Hugging Face Hub
DATASET_TASK = {
# '/pfstem/likaiyu/resources/hdf5/0_1new/20250225_Y_AL02_DYF03_PI0STEP01FINE_CXJ_ai_hdf5':'Flatten the shirt',
'/pfstem/likaiyu/resources/hdf5/0_1new/20250225_Y_AL03_DYF03_PI0STEPS01_WHJ_ai_hdf5':'Flatten the shirt',
'/pfstem/likaiyu/resources/hdf5/0_1new/20250225_Y_AL04_DYF03_PI0STEP01_GY_ai_hdf5':'Flatten the shirt',
'/pfstem/likaiyu/resources/hdf5/0_1new/20250225_Y_AL05_DYF03_PI0STEPS01_WYJ_ai_hdf5':'Flatten the shirt',
'/pfstem/likaiyu/resources/hdf5/0_1new/20250225_Y_AL07_DYF03_PI0STEP01_TCZ_ai_hdf5':'Flatten the shirt',
'/pfstem/likaiyu/resources/hdf5/0_1new/20250225_Y_AL08_DYF03_PI0STEPS01_LTJ_ai_hdf5':'Flatten the shirt',
# '/pfstem/likaiyu/resources/hdf5/0_1new/20250225_Y_AL09_DYF03_PIOSTEP01FINE_SZH_ai_hdf5':'Flatten the shirt',
}

def vidwrite(filename, images, framerate=10, vcodec='libx264'):
    """
    Writes a sequence of NumPy arrays as a video file.

    Args:
        filename (str): The name of the output video file.
        images (np.ndarray): A 4D NumPy array of shape (num_frames, height, width, channels).
        framerate (int, optional): The frame rate of the video. Defaults to 30.
        vcodec (str, optional): The video codec to use. Defaults to 'libx264'.
    """
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    
    n, height, width, channels = images.shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(filename, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def draw_line_on_images(image, points, color=(81, 55, 255)):
    num_frames = len(points)
    if num_frames < 2:
        cv2.circle(image, (points[0, 0, 0], points[0, 0, 1]), radius=2, color=color, thickness=2)
    else:
        cv2.polylines(image, [points], False, color, 2)
    return image

def main(data_dir: str = '/pfstem/likaiyu/resources/hdf5', *, 
         push_to_hub: bool = False, 
         create_from_scratch: bool = True,
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
                "actions": {
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

    # visualize config
    output_video_dir = os.path.join(LEROBOT_HOME, REPO_NAME, 'visualization')
    os.system(f'mkdir -p {output_video_dir}')
    total_visualize_episodes = 20
    total_number_of_tasks = len(DATASET_TASK)
    number_of_visualize_per_task = total_visualize_episodes // total_number_of_tasks

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
    
    for raw_dataset_name, task_instruction in DATASET_TASK.items():
        # raw_dataset_name, task_instruction = pair 
        cur_dataset_name = raw_dataset_name.split('/')[-1]   
        hdf5s_path = os.path.join(data_dir, raw_dataset_name)
        hdf5_file_names = os.listdir(hdf5s_path)
        hdf5_file_names.sort()
        episode_index_to_viz = np.random.choice(len(hdf5_file_names), size=number_of_visualize_per_task)
        for epi_idx, hdf5_file_name in tqdm.tqdm(enumerate(hdf5_file_names), total=len(hdf5_file_names)):
            hdf5_file_path = os.path.join(hdf5s_path, hdf5_file_name)
            mapping = {"observation.images.cam_high": 'camera0_rgb', "observation.images.cam_left_wrist": 'camera1_rgb', "observation.images.cam_right_wrist": 'camera2_rgb', "observation.state": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle'], "actions": ['robot0_gripper_width', 'robot1_gripper_width', 'robot_rjoint_rot_axis_angle']}
            value_dict = {"observation.images.cam_high": None, "observation.images.cam_left_wrist": None, "observation.images.cam_right_wrist": None, "observation.state": None, "actions": None}
            with h5py.File(hdf5_file_path, "r") as ep:

                for i, (key, item) in enumerate(value_dict.items()):
                    # print(key, item.shape)
                    if type(mapping[key]) == list:
                        v = []
                        for each_key in mapping[key]:
                            # print(each_key, ep[each_key].shape, type(ep[each_key]))
                            v.append(torch.from_numpy(np.array(ep[each_key])))
                        v = torch.cat(v, dim=1)
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
                dataset.save_episode(task=task_instruction)

                # visualize data
                # import ipdb; ipdb.set_trace()
                if epi_idx in episode_index_to_viz:
                    gripper_widths_images = None
                    full_images = []
                    total_number_of_frames = len(value_dict['observation.images.cam_left_wrist'])
                    for frame_idx in range(total_number_of_frames):
                        camera0_rgb = value_dict['observation.images.cam_high'][frame_idx]
                        camera1_rgb = value_dict['observation.images.cam_left_wrist'][frame_idx]
                        camera2_rgb = value_dict['observation.images.cam_right_wrist'][frame_idx]
                        camera_images = np.concatenate([camera1_rgb, camera0_rgb, camera2_rgb], axis=1)

                        height, width, channel = camera_images.shape

                        gripper_image_height = 50
                        gripper_widths_images = 255 * np.ones((gripper_image_height, width, channel))
                        robot0_gripper_widths = gripper_image_height * value_dict['actions'][:frame_idx+1, 6] / 5.0
                        robot1_gripper_widths = gripper_image_height * value_dict['actions'][:frame_idx+1, 13] / 5.0
                        coords_x = width * 1.0 * np.arange(frame_idx + 1) / (total_number_of_frames)

                        robot0_points = np.stack([coords_x, robot0_gripper_widths], axis=-1).reshape((-1, 1, 2)).astype(np.int32)
                        robot1_points = np.stack([coords_x, robot1_gripper_widths], axis=-1).reshape((-1, 1, 2)).astype(np.int32)

                        gripper_widths_images = draw_line_on_images(gripper_widths_images, robot0_points, color=(81, 55, 255))
                        gripper_widths_images = draw_line_on_images(gripper_widths_images, robot1_points, color=(107, 255, 51))

                        full_image = np.concatenate([camera_images, gripper_widths_images], axis=0)
                        full_images.append(full_image)

                    output_video_path = os.path.join(output_video_dir, f"{cur_dataset_name}_episode_{epi_idx}.mp4")
                    vidwrite(output_video_path, full_images, framerate=10)


    # pool = ThreadPool(POOL_SIZE)
    # pool.map(func, [(k,v) for k,v in DATASET_TASK.items()]) 
        # raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False, keep_image_files = True)

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


