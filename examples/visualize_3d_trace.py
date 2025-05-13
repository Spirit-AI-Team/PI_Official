#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:
python lerobot/scripts/eval.py --policy.path=/hy-tmp/checkpoints/pi0_6steps_pytorch/ --env.type=pusht --dataset.repo_id=lerobot/FoldShirt6steps --dataset.root=/hy-tmp/lerobot/MULTI_DW1/

"""

import json
import logging
import time
from tqdm import tqdm
from copy import deepcopy
from dataclasses import asdict
from pprint import pformat

import numpy as np
import torch
from termcolor import colored
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/pfstem/likaiyu/mozbrain')
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
)
from lerobot.common.datasets.utils import cycle
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

# from lerobot.common.datasets.spirit_dataset import SpiritDataset
from lerobot.common.datasets.transformed_dataset import TransformedDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key == "actions" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: EvalPipelineConfig) -> TransformedDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Returns:
        TransformedDataset
    """
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = TransformedDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        state_mask_thres=cfg.dataset.state_mask_thres,
        norm_samples=cfg.dataset.norm_samples,
    )

    return dataset

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    
    cfg.policy.use_delta_joint_actions_aloha = True
    cfg.dataset.norm_samples=2

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    import ipdb; ipdb.set_trace()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making dataset.")
    dataset = make_dataset(cfg)

    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")

    # create dataloader for offline training
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        sampler=None,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    logging.info("Start offline eval on a fixed dataset")

    # make stats empty to load stats from pretrained model
    dataset.meta.stats = {}
    start_time = time.perf_counter()
    logging.info("Making policy.")
    # import ipdb; ipdb.set_trace()
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    policy.config.use_delta_joint_actions_aloha = True
    policy.config.state_action_format = "eef6_gripper1_eef6_gripper1"
    logging.info(f'time: {time.perf_counter() - start_time}')
    policy.eval()
  
    res = []
    skip_count = 60
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # validation set:
    # epi 0: 0~250
    # epi 1: 540~790
    # epi 2: 900~1100
    # epi 3: 1200~1450
    # train set:
    # epi 0: 0~250
    # epi 1: 800~1050
    # epi 2: 1450~1700
    for idx in range(0, 2000):
        batch = next(dl_iter)
        if idx < 0:
            continue
        if idx > 250:
            break
        if skip_count == 60:
            skip_count = 0
        else:
            batch = next(dl_iter)
            skip_count += 1
            continue

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        action_raw = batch['action']
        # import ipdb; ipdb.set_trace()
        abs_action_raw = action_raw # batch['observation.state'][:, None, :] # + 
        abs_action_raw[:, :, [6, 13]] = action_raw[:, :, [6, 13]]
        gt_left_wrist = abs_action_raw[0][:, :3].cpu().numpy()
        gt_right_wrist = abs_action_raw[0][:, 7:10].cpu().numpy()

        ax.plot(gt_left_wrist[:, 0], gt_left_wrist[:, 1], gt_left_wrist[:, 2], color='red')
        ax.plot(gt_right_wrist[:, 0], gt_right_wrist[:, 1], gt_right_wrist[:, 2], color='red')

        # run infer multiple times
        infer_times = 5
        all_actions = []

        # losses = F.mse_loss(action_res, action_raw[0, :cfg.policy.n_action_steps], reduction="none")
        # res.append(losses[:, 2:].mean().item())
        for _ in tqdm(range(infer_times)):
            action_res = []
            while len(action_res) < cfg.policy.n_action_steps:
                with torch.inference_mode():
                    action = policy.select_action(batch)
                    action_res.append(action)

            action_res = torch.stack(action_res, dim=0).squeeze(dim=1)
            # abs_action_res = action_res # batch['observation.state'] + 
            # abs_action_res[:, [6, 13]] = action_res[:, [6, 13]]
            left_wrist = action_res[:, :3].cpu().numpy()
            right_wrist = action_res[:, 7:10].cpu().numpy()
            ax.plot(left_wrist[:, 0], left_wrist[:, 1], left_wrist[:, 2], color='blue')
            ax.plot(right_wrist[:, 0], right_wrist[:, 1], right_wrist[:, 2], color='green')

            all_actions.append(action_res)


        # # Convert to CPU / numpy.
        # action = action.to("cpu").numpy()
        # assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

    # plt.savefig("train_episode_0_lerobotpi0_ft32.png")
    plt.savefig("/root/PI_Official/examples/demo.png")
    # plt.savefig("episode_0_officialpi0_ft32.png")

    # print (f'loss: {np.mean(np.array(res))}')
    # import pdb;pdb.set_trace()
    print ('end')

if __name__ == "__main__":
    init_logging()
    eval_main()






