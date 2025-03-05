from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any
import os
import signal

import jax
import jax.numpy as jnp
from typing_extensions import override
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class PolicyConfig:
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    model_type: _model.ModelType = _model.ModelType.PI0
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )

class MultiPolicyV2(_policy.Policy):
    def __init__(self, policy1_setup, policy2_setup):
        self.policy1_setup = policy1_setup
        self.policy2_setup = policy2_setup
        policy1 = create_trained_policy(
            _config.get_config(policy1_setup['config']), policy1_setup['dir'], default_prompt=policy1_setup['default_prompt1']
        )
        self.num_polices = 2
        self.cur_policy_index = 0
        self.cur_policy = policy1

        def switch_handler(signum, frame):
            print("switch policy!!!")
            self.cur_policy_index = (self.cur_policy_index + 1) % self.num_polices 
            if self.cur_policy_index == 0:
                policy = create_trained_policy(
                    _config.get_config(self.policy1_setup['config']), self.policy1_setup['dir'], default_prompt=self.policy1_setup['default_prompt1']
                )
            else:
                policy = create_trained_policy(
                    _config.get_config(self.policy2_setup['config']), self.policy2_setup['dir'], default_prompt=self.policy2_setup['default_prompt1']
                )
            self.cur_policy = policy

            print("cur policy: ", self.cur_policy._input_transform.transforms[0])
        
        signal.signal(signal.SIGCONT, switch_handler)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.cur_policy._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self.cur_policy._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self.cur_policy._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self.cur_policy._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self.cur_policy._output_transform(outputs)

