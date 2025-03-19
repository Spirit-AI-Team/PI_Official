import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_TOWEL="aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    SPI0_ARX = "spi0_arx"
    SPI0_ARX_LoRA = 'spi0_arx_lora'
    SPI0_ARX_LoRA2 = 'spi0_arx_lora2'
    SPI0_ARX_LoRA3 = 'spi0_arx_lora3'
    SPI0_ARX_LoRA_Multi = 'spi0_arx_lora_multi'
    SPI0_ARX_FULL = 'spi0_arx_multi'
    SPI0_ARX_FULL2 = 'SPI0_ARX_FULL2'
    SPI0_ARX_FULL3 = 'SPI0_ARX_FULL3'
    SPI0_ARX_FULL4 = 'SPI0_ARX_FULL4'
@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_TOWEL: Checkpoint(
        config="pi0_aloha_towel",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_towel",
    ),
    EnvMode.SPI0_ARX: Checkpoint(
        config="spi0_aloha_eef_full",
        # dir="checkpoints/shirtflatten_0225_27_bases2_full/29999",
        dir="checkpoints/FlattenShirt_EEF_0306_07_s2eef/29999",
    ),
    EnvMode.SPI0_ARX_FULL: Checkpoint(
        config="spi0_aloha_eef_full",
        # dir="checkpoints/shirtflatten_EEF_06_07_full/29999",
        # dir="checkpoints/spi0_aloha_eef_15quick_full_0312/25000",
        dir="checkpoints/FlattenShirt_EEF_0306_15_LTJ/19999",
    ),
    EnvMode.SPI0_ARX_FULL2: Checkpoint(
        config="spi0_aloha_eef_full2",
        # dir="checkpoints/shirtflatten_EEF_06_07_full/29999",
        dir="checkpoints/FlattenShirt_EEF_0306_15_WHJ/19999",
    ),
    EnvMode.SPI0_ARX_FULL3: Checkpoint(
        config="spi0_aloha_eef_full3",
        # dir="checkpoints/shirtflatten_EEF_06_07_full/29999",
        dir="checkpoints/FlattenShirt_EEF_0306_15_WW/29999",
    ),
    EnvMode.SPI0_ARX_FULL4: Checkpoint(
        config="spi0_aloha_eef_full4",
        # dir="checkpoints/shirtflatten_EEF_06_07_full/29999",
        dir="checkpoints/FlattenShirt_EEF_0306_11_full/29999",
    ),
    EnvMode.SPI0_ARX_LoRA: Checkpoint(
        config="spi0_aloha_eef_lora",
        dir="checkpoints/shirtflatten_EEF_06_07_lora/29999",
    ),
    EnvMode.SPI0_ARX_LoRA2: Checkpoint(
        config="spi0_aloha_eef_lora",
        dir="checkpoints/shirtflatten_EEF_06_07_pi0base_lora/29999",
    ),
    EnvMode.SPI0_ARX_LoRA3: Checkpoint(
        config="spi0_aloha_eef_lora",
        dir="checkpoints/shirtfold3step_eefmvp_0305/29999",
    ),
    EnvMode.SPI0_ARX_LoRA_Multi: Checkpoint(
        config="spi0_aloha_finetune_lora",
        dir="checkpoints/pi0mvp_multi_0220_lora/19999",
    ),#Put the brown cup into the blue plate
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
