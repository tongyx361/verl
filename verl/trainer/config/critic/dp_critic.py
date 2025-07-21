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

from verl.trainer.config.critic.critic import CriticConfig


@dataclass
class DPCriticOptimConfig:
    r"""
    Configuration for DP critic optimizer.

    Args:
        lr (float, optional): Learning rate (default: ``1e-5``)
        min_lr_ratio (float | None, optional): Minimum LR ratio for cosine schedule
            (default: ``None``)
        warmup_style (str, optional): LR warmup style: ``"constant"`` or ``"cosine"``
            (default: ``"constant"``)
    """

    lr: float = 1e-5
    min_lr_ratio: float | None = None
    warmup_style: str = "constant"


@dataclass
class FSDPWrapPolicyConfig:
    r"""
    Configuration for FSDP wrap policy.

    Args:
        min_num_params (int, optional): Minimum number of parameters to trigger wrapping
            (default: ``0``)
    """

    min_num_params: int = 0


@dataclass
class FSDPConfig:
    r"""
    Configuration for FSDP strategy.

    Args:
        param_offload (bool, optional): Whether to offload model parameters to CPU
            (default: ``False``)
        optimizer_offload (bool, optional): Whether to offload optimizer state to CPU
            (default: ``False``)
        offload_policy (bool, optional): Only for FSDP2: offload param/grad/optimizer during train
            (default: ``False``)
        reshard_after_forward (bool, optional): Only for FSDP2: Reshard after forward pass to reduce
            memory footprint (default: ``True``)
        wrap_policy (FSDPWrapPolicyConfig, optional): Policy for wrapping layers with FSDP
            (default: ``FSDPWrapPolicyConfig()``)
        fsdp_size (int, optional): Number of GPUs in each FSDP shard group; -1 means auto
            (default: ``-1``)
        forward_prefetch (bool, optional): Only for FSDP1: FSDP1 configuration, prefetch the next
            forward-pass all-gather before the current forward computation (default: ``False``)
    """

    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    reshard_after_forward: bool = True
    wrap_policy: FSDPWrapPolicyConfig = field(default_factory=FSDPWrapPolicyConfig)
    fsdp_size: int = -1
    forward_prefetch: bool = False


@dataclass
class DPCriticModelConfig:
    r"""
    Configuration for DP critic model.

    Args:
        use_shm (bool, optional): Whether to use shared memory for loading the model
            (default: ``False``)
        enable_activation_offload (bool, optional): Offload activations to CPU to reduce GPU memory
            usage (default: ``False``)
        use_remove_padding (bool, optional): Use remove padding optimization (saves compute)
            (default: ``False``)
        fsdp_config (FSDPConfig, optional): DP-specific config (default: ``FSDPConfig()``)
        lora_rank (int, optional): Set to positive value to enable LoRA (e.g., 32)
            (default: ``0``)
        lora_alpha (int, optional): LoRA scaling factor (default: ``16``)
        target_modules (str, optional): LoRA target modules: ``"all-linear"`` or list of linear
            projection layers (default: ``"all-linear"``)
    """

    use_shm: bool = False
    enable_activation_offload: bool = False
    use_remove_padding: bool = False
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str = "all-linear"


@dataclass
class DPCriticConfig(CriticConfig):
    r"""
    Configuration for DP critic.

    Args:
        strategy (str, optional): FSDP strategy (default: ``"fsdp"``)
        optim (DPCriticOptimConfig, optional): Optimizer configs (default: ``DPCriticOptimConfig()``)
        model (DPCriticModelConfig, optional): Model config for the critic (default: ``DPCriticModelConfig()``)
        forward_micro_batch_size (int | None, optional): Forward-only batch size during inference
            (global) (default: ``"${oc.select:.ppo_micro_batch_size,null}"``)
        forward_micro_batch_size_per_gpu (int | None, optional): Forward-only batch size during
            inference (per GPU) (default: ``"${oc.select:.ppo_micro_batch_size_per_gpu,null}"``)
        ulysses_sequence_parallel_size (int, optional): Sequence parallelism size for Ulysses-style
            model parallelism (default: ``1``)
        grad_clip (float, optional): Gradient clipping for critic updates (default: ``1.0``)
    """

    strategy: str = "fsdp"
    optim: DPCriticOptimConfig = field(default_factory=DPCriticOptimConfig)
    model: DPCriticModelConfig = field(default_factory=DPCriticModelConfig)
    forward_micro_batch_size: int | None = "${oc.select:.ppo_micro_batch_size,null}"
    forward_micro_batch_size_per_gpu: int | None = "${oc.select:.ppo_micro_batch_size_per_gpu,null}"
    ulysses_sequence_parallel_size: int = 1
    grad_clip: float = 1.0
