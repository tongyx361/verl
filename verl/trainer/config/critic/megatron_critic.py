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
class MegatronCriticOptimConfig:
    r"""
    Configuration for Megatron critic optimizer.

    Args:
        optimizer (str, optional): Select optimizer, default is Adam (default: ``"adam"``)
        lr (float, optional): Learning rate (default: ``1e-6``)
        clip_grad (float, optional): Clip gradients norm (default: ``1.0``)
        lr_warmup_init (float, optional): Initial learning rate for warmup, default to 0.0
            (default: ``0.0``)
        lr_warmup_steps (int | None, optional): Prioritized. None, 0 or Negative values mean
            delegating to lr_warmup_steps_ratio (default: ``None``)
        lr_decay_steps (int | None, optional): Learning rate decay steps (default: ``None``)
        lr_decay_style (str, optional): Select from constant/linear/cosine/inverse_square_root
            (default: ``"linear"``)
        min_lr (float, optional): Minimum learning rate, default to 0.0 (default: ``0.0``)
        weight_decay_incr_style (str, optional): Select from constant/linear/cosine
            (default: ``"constant"``)
        lr_wsd_decay_style (str, optional): Select from constant/exponential/cosine
            (default: ``"exponential"``)
        lr_wsd_decay_steps (int | None, optional): Number of steps for weight std decay
            (default: ``None``)
        use_checkpoint_opt_param_scheduler (bool, optional): Use checkpoint optimizer parameter
            scheduler (default: ``False``)
    """

    optimizer: str = "adam"
    lr: float = 1e-6
    clip_grad: float = 1.0
    lr_warmup_init: float = 0.0
    lr_warmup_steps: int | None = None
    lr_decay_steps: int | None = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: int | None = None
    use_checkpoint_opt_param_scheduler: bool = False


@dataclass
class MegatronCriticModelConfig:
    r"""
    Configuration for Megatron critic model.

    Args:
        override_config (dict, optional): Override default empty mapping
            (default: ``{"model_config": {}, "moe_config": {"freeze_moe_router": False}}``)
        enable_gradient_checkpointing (bool, optional): Enable gradient checkpointing to save memory
            (default: ``False``)
        gradient_checkpointing_kwargs (dict, optional): Activation Checkpointing settings
            (default: ``{}``)
    """

    override_config: dict = field(
        default_factory=lambda: {"model_config": {}, "moe_config": {"freeze_moe_router": False}}
    )
    enable_gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict = field(default_factory=dict)


@dataclass
class MegatronConfig:
    r"""
    Configuration for Megatron parallelism settings.

    Args:
        param_offload (bool, optional): Whether to offload model parameters to CPU
            (default: ``False``)
        grad_offload (bool, optional): Whether to offload gradients to CPU (default: ``False``)
        optimizer_offload (bool, optional): Whether to offload optimizer state to CPU
            (default: ``False``)
        tensor_model_parallel_size (int, optional): Size of tensor model parallel group
            (default: ``1``)
        expert_model_parallel_size (int, optional): Size of expert model parallel group
            (default: ``1``)
        expert_tensor_parallel_size (int | None, optional): Size of expert tensor parallel group
            (default: ``None``)
        pipeline_model_parallel_size (int, optional): Size of pipeline model parallel group
            (default: ``1``)
        virtual_pipeline_model_parallel_size (int | None, optional): Size of virtual pipeline model
            parallel group (default: ``None``)
        context_parallel_size (int, optional): Size of context parallel group (default: ``1``)
        sequence_parallel (bool, optional): Whether to use sequence parallelism (default: ``True``)
        use_distributed_optimizer (bool, optional): Whether to use distributed optimizer
            (default: ``True``)
        use_dist_checkpointing (bool, optional): Whether to use distributed checkpointing
            (default: ``False"``)
        dist_checkpointing_path (str | None, optional): Path for distributed checkpointing
            (default: ``None``)
        seed (int, optional): Random seed for Megatron
            (default: ``"${oc.select:actor_rollout_ref.actor.megatron.seed,42}"``)
        override_ddp_config (dict, optional): Allow to override Distributed Data Parallel (DDP) config
            (default: ``"${oc.select:actor_rollout_ref.actor.megatron.override_ddp_config,{}}"``)
        override_transformer_config (dict, optional): Transformer config overrides for Megatron
            (default: ``"${oc.select:actor_rollout_ref.actor.megatron.override_transformer_config,{}}"``)
        use_mbridge (bool, optional): Whether to use mBridge communications
            (default: ``"${oc.select:actor_rollout_ref.actor.megatron.use_mbridge,False}"``)
    """

    param_offload: bool = False
    grad_offload: bool = False
    optimizer_offload: bool = False
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: str | None = None
    seed: int = "${oc.select:actor_rollout_ref.actor.megatron.seed,42}"
    override_ddp_config: dict = "${oc.select:actor_rollout_ref.actor.megatron.override_ddp_config,{}}"
    override_transformer_config: dict = "${oc.select:actor_rollout_ref.actor.megatron.override_transformer_config,{}}"
    use_mbridge: bool = "${oc.select:actor_rollout_ref.actor.megatron.use_mbridge,False}"


@dataclass
class MegatronCheckpointConfig:
    r"""
    Configuration for Megatron checkpoint.

    Args:
        async_save (bool, optional): Asynchronous checkpoint saving (default: ``False``)
    """

    async_save: bool = False


@dataclass
class MegatronCriticConfig(CriticConfig):
    r"""
    Configuration for Megatron critic.

    Args:
        strategy (str, optional): Megatron strategy (default: ``"megatron"``)
        nccl_timeout (int, optional): Seconds, default is 10 minutes for torch, you can set it to a
            larger value if you have long-running operations like 32B or 72B model using megatron
            (default: ``600``)
        optim (MegatronCriticOptimConfig, optional): Optimizer configs (default: ``MegatronCriticOptimConfig()``)
        model (MegatronCriticModelConfig, optional): Model config for the critic
            (default: ``MegatronCriticModelConfig()``)
        megatron (MegatronConfig, optional): Megatron-specific parallelism settings (default: ``MegatronConfig()``)
        load_weight (bool, optional): Whether to load initial weights (default: ``True``)
        data_loader_seed (int | None, optional): Seed for data loader
            (default: ``"${oc.select:actor_rollout_ref.actor.data_loader_seed,null}"``)
        checkpoint (MegatronCheckpointConfig, optional): Checkpoint configuration
            (default: ``MegatronCheckpointConfig()``)
    """

    strategy: str = "megatron"
    nccl_timeout: int = 600
    optim: MegatronCriticOptimConfig = field(default_factory=MegatronCriticOptimConfig)
    model: MegatronCriticModelConfig = field(default_factory=MegatronCriticModelConfig)
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    load_weight: bool = True
    data_loader_seed: int | None = "${oc.select:actor_rollout_ref.actor.data_loader_seed,null}"
    checkpoint: MegatronCheckpointConfig = field(default_factory=MegatronCheckpointConfig)
