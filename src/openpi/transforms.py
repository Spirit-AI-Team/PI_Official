from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import os
import random
import re
from pathlib import Path
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None
    delta_base: str 
    debug_prob: float = 0.1  # Probability of triggering debugger

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        if self.delta_base == 'state':
            actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        elif self.delta_base == 'actions':
            actions[..., :dims] -= np.expand_dims(np.where(mask, actions[..., 0, :dims], 0), axis=-2)
        else:
            raise Exception('Invalid delta base!')
        data["actions"] = actions
        
        # HACK
        # if np.random.random() < self.debug_prob and data['progress'] > 0.0:
        #     data["actions"][:, :6] *= 0
        #     data["actions"][:, 7:13] *= 0
        #     if data['prompt'] == 'Fetch pink bottle.':
        #         data['prompt'] = 'Fetch yellow bottle.'
        #     elif data['prompt'] == 'Fetch yellow bottle.':
        #         data['prompt'] = 'Fetch pink bottle.'

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None
    delta_base: str

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        if self.delta_base == 'state':
            actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        elif self.delta_base == 'actions':
            actions[..., :dims] += p.expand_dims(np.where(mask, actions[..., 0, :dims], 0), axis=-2)
        else:
            raise Exception('Invalid delta base!')
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class AugmentedPromptFromLeRobotTask(DataTransformFn):
    """Extracts and augments prompts from LeRobot dataset tasks with categorized drink prompts."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]
    # Path to the drink prompts JSON file
    drink_prompts_path: str = "./QA/drink_prompts.json"
    # Whether to use augmented prompts (if False, behaves like original PromptFromLeRobotTask)
    use_augmentation: bool = True
    # Random seed for reproducibility (optional)
    random_seed: int | None = None

    def __post_init__(self):
        """Load drink prompts and set up drink mapping."""
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        # Load the categorized drink prompts
        try:
            with open(self.drink_prompts_path, 'r', encoding='utf-8') as f:
                object.__setattr__(self, '_drink_prompts', json.load(f))
        except FileNotFoundError:
            print(f"Warning: Drink prompts file not found at {self.drink_prompts_path}. Falling back to original prompts.")
            object.__setattr__(self, '_drink_prompts', {})
        
        # Create mapping from English drink names to Chinese names for easier lookup
        drinks_mapping = {
            "冰红茶": "red tea",
            "青提绿茶": "green tea", 
            "乳酸菌水乐": "lactic drink",
            "劲凉冰红茶": "cool blue tea"
        }
        object.__setattr__(self, '_drinks_mapping', drinks_mapping)
        
        # Reverse mapping for quick lookup
        english_to_chinese = {v: k for k, v in drinks_mapping.items()}
        object.__setattr__(self, '_english_to_chinese', english_to_chinese)
        
        # Create keyword mapping for prompt classification
        prompt_keywords = {
            "冰红茶": ["red tea", "red", "kangshifu red"],
            "青提绿茶": ["green tea", "green", "kangshifu green"],
            "乳酸菌水乐": ["lactic drink", "lactic", "pink", "kangshifu lactic"],
            "劲凉冰红茶": ["cool blue tea", "cool tea", "blue", "kangshifu cool"]
        }
        object.__setattr__(self, '_prompt_keywords', prompt_keywords)

    def _classify_prompt_by_drink(self, prompt: str) -> str | None:
        """Classify a prompt to determine which drink category it belongs to."""
        prompt_lower = prompt.lower()
        
        # Check for each drink's keywords
        for chinese_name, keywords in self._prompt_keywords.items():
            for keyword in keywords:
                if keyword.lower() in prompt_lower:
                    return chinese_name
        
        return None

    def _get_random_augmented_prompt(self, original_prompt: str, episode_index: int) -> str:
        """Get a random augmented prompt from the same drink category, seeded by episode_index."""
        if not self.use_augmentation or not self._drink_prompts:
            return original_prompt
        
        if original_prompt is None or original_prompt not in self._drink_prompts:
            # If we can't classify or category doesn't exist, return original
            return original_prompt
        
        # Use episode_index as seed for reproducible randomness
        rng = random.Random(episode_index.item())
        
        # Get random prompt from the same category
        category_prompts = self._drink_prompts[original_prompt]
        if category_prompts:
            # return rng.choice(category_prompts)
            return random.choice(category_prompts)
        else:
            return original_prompt

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (original_prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        # Get augmented prompt if augmentation is enabled
        if self.use_augmentation:
            augmented_prompt = self._get_random_augmented_prompt(original_prompt, data["episode_index"])
            return {**data, "prompt": augmented_prompt, "original_prompt": original_prompt}
        else:
            return {**data, "prompt": original_prompt}


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
