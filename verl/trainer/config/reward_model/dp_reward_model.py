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

from verl.trainer.config.reward_model.reward_model import RewardModelConfig


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
        wrap_policy (FSDPWrapPolicyConfig, optional): Policy for wrapping layers with FSDP
            (default: ``FSDPWrapPolicyConfig()``)
        param_offload (bool, optional): Whether to offload model parameters to CPU
            (default: ``False``)
        reshard_after_forward (bool, optional): Only for FSDP2: Reshard after forward pass to reduce
            memory footprint (default: ``True``)
        fsdp_size (int, optional): Number of GPUs in each FSDP shard group; -1 means auto
            (default: ``-1``)
        forward_prefetch (bool, optional): Only for FSDP1: FSDP1 configuration, prefetch the next
            forward-pass all-gather before the current forward computation (default: ``False``)
    """

    wrap_policy: FSDPWrapPolicyConfig = field(default_factory=FSDPWrapPolicyConfig)
    param_offload: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False


@dataclass
class DPRewardModelModelConfig:
    r"""
    Configuration for DP reward model.

    Args:
        use_shm (bool, optional): Whether to use shared memory for loading the model
            (default: ``False``)
        use_remove_padding (bool, optional): Use remove padding optimization (saves compute)
            (default: ``False``)
        use_fused_kernels (bool, optional): Whether to use fused reward kernels for speedup
            (default: ``"${actor_rollout_ref.model.use_fused_kernels}"``)
        fsdp_config (FSDPConfig, optional): DP-specific config (default: ``FSDPConfig()``)
    """

    use_shm: bool = False
    use_remove_padding: bool = False
    use_fused_kernels: bool = "${actor_rollout_ref.model.use_fused_kernels}"
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class DPRewardModelConfig(RewardModelConfig):
    r"""
    Configuration for DP reward model.

    Args:
        strategy (str, optional): FSDP strategy (default: ``"fsdp"``)
        model (DPRewardModelModelConfig, optional): Model configuration with DP-specific settings
            (default: ``DPRewardModelModelConfig()``)
        ulysses_sequence_parallel_size (int, optional): Sequence parallelism size for Ulysses-style
            model parallelism (default: ``1``)
    """

    strategy: str = "fsdp"
    model: DPRewardModelModelConfig = field(default_factory=DPRewardModelModelConfig)
    ulysses_sequence_parallel_size: int = 1
