# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from dataclasses import dataclass, field

from verl.trainer.config.actor.actor import OptimizerConfig, PPOActorConfig


@dataclass
class DPOptimizerConfig(OptimizerConfig):
    r"""
    DP-specific optimizer configuration.

    Args:
        lr_warmup_steps (int, optional): Warmup steps; negative value delegates to lr_warmup_steps_ratio
            (default: ``-1``)
        min_lr_ratio (float, optional): Minimum LR ratio for cosine schedule (default: ``0.0``)
        num_cycles (float, optional): Number of cosine cycles in LR schedule (default: ``0.5``)
        warmup_style (str, optional): LR warmup style: ``"constant"`` or ``"cosine"`` (default: ``"constant"``)
    """

    lr_warmup_steps: int = -1
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5
    warmup_style: str = "constant"


@dataclass
class FSDPWrapPolicyConfig:
    r"""
    Configuration for FSDP wrap policy.

    Args:
        min_num_params (int, optional): Minimum number of parameters to trigger wrapping a layer with FSDP
            (default: ``0``)
    """

    min_num_params: int = 0


@dataclass
class FSDPConfig:
    r"""
    Configuration for FSDP (Fully Sharded Data Parallel).

    Args:
        wrap_policy (FSDPWrapPolicyConfig, optional): Policy for wrapping the model
            (default: ``FSDPWrapPolicyConfig()``)
        param_offload (bool, optional): Whether to offload model parameters to CPU (trades speed for memory)
            (default: ``False``)
        optimizer_offload (bool, optional): Whether to offload optimizer state to CPU (default: ``False``)
        offload_policy (bool, optional): Only for FSDP2: offload param/grad/optimizer during train (default: ``False``)
        reshard_after_forward (bool, optional): Only for FSDP2: Reshard after forward pass to reduce memory footprint
            (default: ``True``)
        fsdp_size (int, optional): Number of GPUs in each FSDP shard group; -1 means auto (default: ``-1``)
        forward_prefetch (bool, optional): Only for FSDP1: FSDP1 configuration, prefetch the next forward-pass
            all-gather before the current forward computation (default: ``False``)
    """

    wrap_policy: FSDPWrapPolicyConfig = field(default_factory=FSDPWrapPolicyConfig)
    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False


@dataclass
class DPActorConfig(PPOActorConfig):
    r"""
    DP-specific PPO actor configuration.

    Args:
        strategy (str, optional): The abstract actor configs. ``"fsdp"``, ``"fsdp2"`` or ``"megatron"``. must be set.
            (default: ``"fsdp"``)
        grad_clip (float, optional): Gradient clipping for actor updates, specific to the strategy (default: ``1.0``)
        ulysses_sequence_parallel_size (int, optional): Sequence parallelism size for Ulysses-style model parallelism
            (default: ``1``)
        entropy_from_logits_with_chunking (bool, optional): Calculate entropy with chunking to reduce memory peak
            (default: ``False``)
        entropy_checkpointing (bool, optional): Recompute entropy (default: ``False``)
        optim (DPOptimizerConfig, optional): DP-specific optimizer configuration (default: ``DPOptimizerConfig()``)
        fsdp_config (FSDPConfig, optional): FSDP configuration (default: ``FSDPConfig()``)
    """

    # TODO(haibin.lin): switch to fsdp2
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    optim: DPOptimizerConfig = field(default_factory=DPOptimizerConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
