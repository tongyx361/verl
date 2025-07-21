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
from dataclasses import dataclass


@dataclass
class ReferenceConfig:
    r"""
    Configuration for reference model.

    Args:
        strategy (str, optional): FSDP config same as actor. For models larger than 7B, it's recommended
            to turn on offload for ref by default (default: ``"${actor_rollout_ref.actor.strategy}"``)
        use_torch_compile (bool, optional): Whether to enable torch.compile. Same as
            actor_rollout_ref.actor.use_torch_compile if it exists, otherwise true (default: ``True``)
        log_prob_micro_batch_size (int | None, optional): The batch size for one forward pass in the
            computation of log_prob. Global batch size. [Will be deprecated, use
            log_prob_micro_batch_size_per_gpu] (default: ``None``)
        log_prob_micro_batch_size_per_gpu (int | None, optional): The batch size for one forward pass
            in the computation of log_prob. Local batch size per GPU (default: ``None``)
        log_prob_use_dynamic_bsz (bool, optional): Enable dynamic batch size (sequence packing) for
            log_prob computation. Same as actor_rollout_ref.actor.use_dynamic_bsz if it exists,
            otherwise false (default: ``False``)
        log_prob_max_token_len_per_gpu (int, optional): The max token length per GPU. Same as
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu if it exists, otherwise 16384
            (default: ``16384``)
    """

    strategy: str = "${actor_rollout_ref.actor.strategy}"
    use_torch_compile: bool = True
    log_prob_micro_batch_size: int | None = None
    log_prob_micro_batch_size_per_gpu: int | None = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384
