"""
Visualize lerobot data
you need install ffmpeg-python 
"""

import os
import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import ffmpeg
from pathlib import Path

# set the LEROBOT_HOME and REPO_NAME to the dataset you need.
REPO_NAME = "MultiTask_EggInEggrack_0414"

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

def draw_value_on_image(images, values, val_range=[0, 1], color=(81, 55, 255)):
    height, width, channle = images.shape

    # scale value(y) according to image height
    values[:, 1] = height * (values[:, 1] - val_range[0]) / (val_range[1] - val_range[0])
    values = values.reshape((-1, 1, 2)).astype(np.int32)

    images = draw_line_on_images(images, values, color)
    return images

from PIL import ImageFont, ImageDraw, Image
def put_chinese_text(img, text, position, font_path='assets/SimHei.ttf', font_size=10, text_color=(255, 0, 0)):
    font = ImageFont.truetype(font_path, font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=text_color)
    img_np = np.array(img_pil)
    return img_np
    
def main(num_episodes_to_visualize=20, episodes_idx_to_visualize = None):
    # set the LEROBOT_HOME and REPO_NAME to the dataset you need.
    # LEROBOT_HOME = Path('/pfstem/wenxuan/resources/lerobot_15steps')
    dataset = LeRobotDataset(root=LEROBOT_HOME / REPO_NAME, repo_id=REPO_NAME, local_files_only=True)
    fps = dataset.fps
    num_episodes = dataset.num_episodes

    # visualize config
    if episodes_idx_to_visualize is None:
        episodes_idx_to_visualize = np.random.choice(num_episodes, size=num_episodes_to_visualize)

    output_video_dir = os.path.join(LEROBOT_HOME, REPO_NAME, 'visualization_test')
    os.system(f'mkdir -p {output_video_dir}')

    for episodes_idx in episodes_idx_to_visualize:
        ep_start = dataset.episode_data_index["from"][episodes_idx]
        ep_end = dataset.episode_data_index["to"][episodes_idx]
        total_number_of_frames = ep_end - ep_start + 1
        full_images = []
        actions = np.empty((0, 14))
        # states = np.empty((0, 14))
        for data_idx in range(ep_start, ep_end):
            value_dict = dataset[data_idx]
            actions = np.concatenate((actions, value_dict['actions'][None,...].numpy()), axis=0)

            camera0_rgb = (value_dict['observation.images.cam_high'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            camera1_rgb = (value_dict['observation.images.cam_left_wrist'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            camera2_rgb = (value_dict['observation.images.cam_right_wrist'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            camera_images = np.concatenate([camera1_rgb, camera0_rgb, camera2_rgb], axis=1)
            camera_images = np.ascontiguousarray(camera_images, dtype=np.uint8)
            height, width, channel = camera_images.shape

            # draw prompt
            task_index = value_dict['task_index'].item()
            prompt = dataset.meta.tasks[task_index]
            # cv2.putText(camera_images, prompt, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)
            camera_images = put_chinese_text(camera_images, prompt, (5, 20), font_size=20, text_color=(255, 0, 0))
            cv2.putText(camera_images, str(fps) + " fps", (width - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)

            frame_idx = data_idx - ep_start.item()

            # draw gripper widths
            gripper_image_height = 100
            gripper_widths_images = 255 * np.ones((gripper_image_height, width, channel))
            coords_x = width * 1.0 * np.arange(frame_idx + 1) / (total_number_of_frames)
            robot0_points = np.stack([coords_x, 5.0 - actions[:frame_idx+1, 6]], axis=-1)
            robot1_points = np.stack([coords_x, 5.0 - actions[:frame_idx+1, 13]], axis=-1)
            gripper_widths_images = draw_value_on_image(gripper_widths_images, robot0_points, val_range=[0, 5.], color=(81, 55, 255))
            gripper_widths_images = draw_value_on_image(gripper_widths_images, robot1_points, val_range=[0, 5.], color=(107, 255, 51))
            cv2.putText(gripper_widths_images, "gripper", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # draw first rot
            rot_range = [-np.pi, np.pi]
            rot_height = 100
            rot_images0 = 255 * np.ones((rot_height, width, channel))
            robot0_points = np.stack([coords_x, - actions[:frame_idx+1, 0]], axis=-1)
            robot1_points = np.stack([coords_x, - actions[:frame_idx+1, 7]], axis=-1)
            rot_images0 = draw_value_on_image(rot_images0, robot0_points, val_range=rot_range, color=(81, 55, 255))
            rot_images0 = draw_value_on_image(rot_images0, robot1_points, val_range=rot_range, color=(107, 255, 51))
            cv2.putText(rot_images0, "1st rot", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # draw second rot
            rot_images1 = 255 * np.ones((rot_height, width, channel))
            robot0_points = np.stack([coords_x, - actions[:frame_idx+1, 1]], axis=-1)
            robot1_points = np.stack([coords_x, - actions[:frame_idx+1, 8]], axis=-1)
            rot_images1 = draw_value_on_image(rot_images1, robot0_points, val_range=rot_range, color=(81, 55, 255))
            rot_images1 = draw_value_on_image(rot_images1, robot1_points, val_range=rot_range, color=(107, 255, 51))
            cv2.putText(rot_images1, "2nd rot", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # draw third rot
            rot_images2 = 255 * np.ones((rot_height, width, channel))
            robot0_points = np.stack([coords_x, - actions[:frame_idx+1, 2]], axis=-1)
            robot1_points = np.stack([coords_x, - actions[:frame_idx+1, 9]], axis=-1)
            rot_images2 = draw_value_on_image(rot_images2, robot0_points, val_range=rot_range, color=(81, 55, 255))
            rot_images2 = draw_value_on_image(rot_images2, robot1_points, val_range=rot_range, color=(107, 255, 51))
            cv2.putText(rot_images2, "3rd rot", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # draw forth rot
            rot_images3 = 255 * np.ones((rot_height, width, channel))
            robot0_points = np.stack([coords_x, - actions[:frame_idx+1, 3]], axis=-1)
            robot1_points = np.stack([coords_x, - actions[:frame_idx+1, 10]], axis=-1)
            rot_images3 = draw_value_on_image(rot_images3, robot0_points, val_range=rot_range, color=(81, 55, 255))
            rot_images3 = draw_value_on_image(rot_images3, robot1_points, val_range=rot_range, color=(107, 255, 51))
            cv2.putText(rot_images3, "4th rot", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # draw fifth rot
            rot_images4 = 255 * np.ones((rot_height, width, channel))
            robot0_points = np.stack([coords_x, - actions[:frame_idx+1, 4]], axis=-1)
            robot1_points = np.stack([coords_x, - actions[:frame_idx+1, 11]], axis=-1)
            rot_images4 = draw_value_on_image(rot_images4, robot0_points, val_range=rot_range, color=(81, 55, 255))
            rot_images4 = draw_value_on_image(rot_images4, robot1_points, val_range=rot_range, color=(107, 255, 51))
            cv2.putText(rot_images4, "5th rot", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # draw sixth rot
            rot_images5 = 255 * np.ones((rot_height, width, channel))
            robot0_points = np.stack([coords_x, - actions[:frame_idx+1, 5]], axis=-1)
            robot1_points = np.stack([coords_x, - actions[:frame_idx+1, 12]], axis=-1)
            rot_images5 = draw_value_on_image(rot_images5, robot0_points, val_range=rot_range, color=(81, 55, 255))
            rot_images5 = draw_value_on_image(rot_images5, robot1_points, val_range=rot_range, color=(107, 255, 51))
            cv2.putText(rot_images5, "6th rot", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            rot_images = np.concatenate([rot_images0, rot_images1, rot_images2, rot_images3, rot_images4, rot_images5], axis=0)

            full_image = np.concatenate([camera_images, gripper_widths_images, rot_images], axis=0)
            full_images.append(full_image)

        output_video_path = os.path.join(output_video_dir, f"episode_{episodes_idx}.mp4")
        vidwrite(output_video_path, full_images, framerate=fps)

if __name__ == "__main__":
    main()