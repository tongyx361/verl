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


@dataclass
class FSDPWrapPolicyConfig:
    r"""
    Configuration for FSDP wrap policy.

    Args:
        min_num_params (int, optional): Minimum number of params in a wrapped module (default: ``0``)
    """

    min_num_params: int = 0


@dataclass
class FSDPConfig:
    r"""
    Configuration for FSDP strategy.

    Args:
        param_offload (bool, optional): Whether to offload parameters in FSDP (default: ``False``)
        reshard_after_forward (bool | int, optional): Whether to perform reshard after model forward
            to save memory. Only for fsdp2, [True, False, int between 1 and fsdp_size] (default: ``True``)
        forward_prefetch (bool, optional): Only for FSDP1: FSDP1 configuration, prefetch the next
            forward-pass all-gather before the current forward computation (default: ``False``)
        wrap_policy (FSDPWrapPolicyConfig, optional): The wrap policy for FSDP model
            (default: ``FSDPWrapPolicyConfig()``)
    """

    param_offload: bool = False
    reshard_after_forward: bool | int = True
    forward_prefetch: bool = False
    wrap_policy: FSDPWrapPolicyConfig = field(default_factory=FSDPWrapPolicyConfig)


@dataclass
class DPReferenceConfig:
    r"""
    Configuration for DP reference model.

    Args:
        fsdp_config (FSDPConfig, optional): Config for FSDP strategy (default: ``FSDPConfig()``)
        ulysses_sequence_parallel_size (int, optional): Sequence parallel size. Same as
            actor_rollout_ref.actor.ulysses_sequence_parallel_size if it exists, otherwise 1
            (default: ``1``)
        entropy_from_logits_with_chunking (bool, optional): Calculate entropy with chunking to reduce
            memory peak (default: ``False``)
        entropy_checkpointing (bool, optional): Recompute entropy (default: ``False``)
    """

    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
