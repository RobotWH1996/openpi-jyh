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
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    CR100_OPEN_DOOR = "cr100_open_door"
    CR100_OPEN_DOOR_PI0 = "cr100_open_door_pi0"

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

    # ===== [RTC] 新增参数 =====
    rtc: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_schedule: str = "LINEAR"
    # ===== [RTC] 新增参数结束 =====

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.CR100_OPEN_DOOR: Checkpoint(
        config="pi05_cr100_open_door_lora",
        dir="/home/x100/wh/openpi/checkpoints/pi05_cr100_open_door_lora/pi05_cr100_open_door_lora/45000",
    ),
    EnvMode.CR100_OPEN_DOOR_PI0: Checkpoint(
        config="pi0_cr100_full_finetune",
        dir="/home/x100/wh/openpi/checkpoints/pi0/checkpoints/pi0_cr100_full_finetune/pi0_cr100_full_finetune/20000",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


# ===== [RTC] 新增函数 =====
def _apply_rtc_config(train_config: _config.TrainConfig, args: Args) -> _config.TrainConfig:
    """Apply RTC configuration to the train config if --rtc is enabled."""
    if not args.rtc:
        return train_config

    from openpi.rtc.configuration_rtc import RTCConfig, RTCAttentionSchedule

    rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_schedule),
        max_guidance_weight=args.rtc_max_guidance_weight,
        execution_horizon=args.rtc_execution_horizon,
    )

    new_model_config = dataclasses.replace(train_config.model, rtc_config=rtc_config)
    return dataclasses.replace(train_config, model=new_model_config)
# ===== [RTC] 新增函数结束 =====


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            # ===== [RTC] 原始代码 =====
            # return _policy_config.create_trained_policy(
            #     _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            # )
            # ===== [RTC] 新代码: 在创建 policy 前注入 RTC config =====
            config = _config.get_config(args.policy.config)
            config = _apply_rtc_config(config, args)
            return _policy_config.create_trained_policy(
                config, args.policy.dir, default_prompt=args.default_prompt
            )
            # ===== [RTC] 新代码结束 =====
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
