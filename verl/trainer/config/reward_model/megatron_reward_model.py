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
class MegatronConfig:
    r"""
    Configuration for Megatron parallelism & checkpointing.

    Args:
        param_offload (bool, optional): Whether to offload model parameters to CPU
            (default: ``False``)
        tensor_model_parallel_size (int, optional): Number of GPUs in tensor model parallel group
            (default: ``1``)
        expert_model_parallel_size (int, optional): Number of GPUs in expert model parallel group
            (default: ``1``)
        expert_tensor_parallel_size (int | None, optional): Expert tensor parallel size
            (default: ``None``)
        pipeline_model_parallel_size (int, optional): Number of pipeline model parallel stages
            (default: ``1``)
        virtual_pipeline_model_parallel_size (int | None, optional): Change VPP interface for
            parallelism tests (default: ``None``)
        context_parallel_size (int, optional): Context parallel size (default: ``1``)
        sequence_parallel (bool, optional): Whether to use sequence parallelism (default: ``True``)
        use_distributed_optimizer (bool, optional): Whether to use distributed optimizer
            (default: ``False``)
        use_dist_checkpointing (bool, optional): Whether to enable distributed checkpointing
            (default: ``False``)
        dist_checkpointing_path (str | None, optional): Path for distributed checkpoints
            (default: ``None``)
        seed (int, optional): RNG seed for megatron
            (default: ``"${oc.select:actor_rollout_ref.actor.megatron.seed,42}"``)
        override_transformer_config (dict, optional): Any overrides to transformer config
            (default: ``{}``)
        use_mbridge (bool, optional): Whether to use mbridge for faster comms
            (default: ``"${oc.select:actor_rollout_ref.actor.megatron.use_mbridge,False}"``)
    """

    param_offload: bool = False
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = False
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: str | None = None
    seed: int = "${oc.select:actor_rollout_ref.actor.megatron.seed,42}"
    override_transformer_config: dict = field(default_factory=dict)
    use_mbridge: bool = "${oc.select:actor_rollout_ref.actor.megatron.use_mbridge,False}"


@dataclass
class MegatronRewardModelConfig(RewardModelConfig):
    r"""
    Configuration for Megatron reward model.

    Args:
        strategy (str, optional): Megatron strategy (default: ``"megatron"``)
        nccl_timeout (int, optional): Seconds, default is 10 minutes for torch, you can set it to a
            larger value if you have long-running operations like 32B or 72B model using megatron
            (default: ``600``)
        megatron (MegatronConfig, optional): Megatron parallelism & checkpointing config
            (default: ``MegatronConfig()``)
        load_weight (bool, optional): Whether to load weights (default: ``True``)
    """

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    load_weight: bool = True
