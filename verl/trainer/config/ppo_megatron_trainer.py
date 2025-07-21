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

from verl.trainer.config.algorithm import AlgoConfig
from verl.utils.profiler.config import ProfilerConfig


@dataclass
class CustomRewardFunctionConfig:
    r"""
    Configuration for custom reward function.

    Args:
        path (str | None, optional): Path to custom reward function (default: ``None``)
        name (str, optional): Name of custom reward function (default: ``"compute_score"``)
    """

    path: str | None = None
    name: str = "compute_score"


@dataclass
class PPOMegatronModelConfig:
    r"""
    Configuration for PPO Megatron model.

    Args:
        path (str, optional): Model path (default: ``"~/models/deepseek-llm-7b-chat"``)
        custom_chat_template (str | None, optional): Custom chat template (default: ``None``)
        external_lib (str | None, optional): External library (default: ``None``)
        override_config (dict, optional): Override configuration (default: ``{}``)
        enable_gradient_checkpointing (bool, optional): Whether to enable gradient checkpointing
            (default: ``False``)
        gradient_checkpointing_kwargs (dict, optional): Gradient checkpointing kwargs
            (default: ``{}``)
        use_fused_kernels (bool, optional): Whether to use custom fused kernels (PostProcessing, for
            memory efficiency) (default: ``False``)
        trust_remote_code (bool, optional): Whether to trust remote code (default: ``False``)
    """

    path: str = "~/models/deepseek-llm-7b-chat"
    custom_chat_template: str | None = None
    external_lib: str | None = None
    override_config: dict = field(default_factory=dict)
    enable_gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict = field(default_factory=dict)
    use_fused_kernels: bool = False
    trust_remote_code: bool = False


@dataclass
class PPOMegatronRolloutConfig:
    r"""
    Configuration for PPO Megatron rollout.

    Args:
        enable_chunked_prefill (bool, optional): May get higher throughput when set to True. When
            activated, Please increase max_num_batched_tokens or decrease max_model_len
            (default: ``False``)
        load_format (str, optional): Load format (default: ``"dummy_megatron"``)
        tensor_model_parallel_size (int, optional): Tensor model parallel size (default: ``1``)
        layer_name_map (dict, optional): Layer name map
            (default: ``{"qkv_layer_name": "qkv", "gate_proj_layer_name": "gate_up"}``)
    """

    enable_chunked_prefill: bool = False
    load_format: str = "dummy_megatron"
    tensor_model_parallel_size: int = 1
    layer_name_map: dict = field(default_factory=lambda: {"qkv_layer_name": "qkv", "gate_proj_layer_name": "gate_up"})


@dataclass
class PPOMegatronActorRolloutRefConfig:
    r"""
    Configuration for PPO Megatron actor rollout reference.

    Args:
        hybrid_engine (bool, optional): Whether to use hybrid engine (default: ``True``)
        nccl_timeout (int, optional): Seconds, default is 10 minutes for torch, you can set it to a
            larger value if you have long-running operations like 32B or 72B model using megatron
            (default: ``600``)
        model (PPOMegatronModelConfig, optional): Model configuration (default: ``PPOMegatronModelConfig()``)
        rollout (PPOMegatronRolloutConfig, optional): Rollout configuration (default: ``PPOMegatronRolloutConfig()``)
        profiler (ProfilerConfig, optional): Profiler configuration (default: ``ProfilerConfig()``)
    """

    hybrid_engine: bool = True
    nccl_timeout: int = 600
    model: PPOMegatronModelConfig = field(default_factory=PPOMegatronModelConfig)
    rollout: PPOMegatronRolloutConfig = field(default_factory=PPOMegatronRolloutConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


@dataclass
class PPOMegatronNsightOptions:
    r"""
    Configuration for PPO Megatron Nsight options.

    Args:
        trace (str, optional): Trace options (default: ``"cuda,nvtx,cublas,ucx"``)
        cuda_memory_usage (str, optional): CUDA memory usage (default: ``"true"``)
        cuda_graph_trace (str, optional): CUDA graph trace (default: ``"graph"``)
        capture_range (str | None, optional): Capture range (default: ``None``)
        capture_range_end (str | None, optional): Capture range end (default: ``None``)
        kill (str | None, optional): Kill option (default: ``None``)
    """

    trace: str = "cuda,nvtx,cublas,ucx"
    cuda_memory_usage: str = "true"
    cuda_graph_trace: str = "graph"
    capture_range: str | None = None
    capture_range_end: str | None = None
    kill: str | None = None


@dataclass
class PPOMegatronTrainerConfig:
    r"""
    Configuration for PPO Megatron trainer.

    Args:
        balance_batch (bool, optional): Whether to balance batch (default: ``True``)
        total_epochs (int, optional): Total epochs (default: ``30``)
        total_training_steps (int | None, optional): Total training steps (default: ``None``)
        profile_steps (list[int] | None, optional): Profile steps (default: ``None``)
        project_name (str, optional): Project name (default: ``"verl_examples"``)
        experiment_name (str, optional): Experiment name (default: ``"gsm8k"``)
        logger (list[str], optional): Logger (default: ``["console", "wandb"]``)
        log_val_generations (int, optional): Log validation generations (default: ``0``)
        nnodes (int, optional): Number of nodes (default: ``1``)
        n_gpus_per_node (int, optional): Number of GPUs per node (default: ``8``)
        save_freq (int, optional): Save frequency (default: ``-1``)
        esi_redundant_time (int, optional): ESI redundant time (default: ``0``)
        resume_mode (str, optional): Resume mode: auto, disable, or resume_path if resume_from_path
            is set (default: ``"auto"``)
        resume_from_path (str | None, optional): Resume from path (default: ``None``)
        del_local_ckpt_after_load (bool, optional): Whether to delete local checkpoint after load
            (default: ``False``)
        val_before_train (bool, optional): Whether to validate before train (default: ``True``)
        test_freq (int, optional): Test frequency (default: ``-1``)
        critic_warmup (int, optional): Critic warmup (default: ``0``)
        default_hdfs_dir (str | None, optional): Default HDFS directory (default: ``None``)
        default_local_dir (str, optional): Default local directory
            (default: ``"checkpoints/${trainer.project_name}/${trainer.experiment_name}"``)
        max_actor_ckpt_to_keep (int | None, optional): Maximum actor checkpoints to keep
            (default: ``None``)
        max_critic_ckpt_to_keep (int | None, optional): Maximum critic checkpoints to keep
            (default: ``None``)
        ray_wait_register_center_timeout (int, optional): Ray wait register center timeout
            (default: ``300``)
        device (str, optional): Device (default: ``"cuda"``)
        controller_nsight_options (PPOMegatronNsightOptions, optional): Controller Nsight options
            (default: ``PPOMegatronNsightOptions()``)
        worker_nsight_options (PPOMegatronNsightOptions, optional): Worker Nsight options
            (default: ``PPOMegatronNsightOptions()``)
    """

    balance_batch: bool = True
    total_epochs: int = 30
    total_training_steps: int | None = None
    profile_steps: list[int] | None = None
    project_name: str = "verl_examples"
    experiment_name: str = "gsm8k"
    logger: list[str] = field(default_factory=lambda: ["console", "wandb"])
    log_val_generations: int = 0
    nnodes: int = 1
    n_gpus_per_node: int = 8
    save_freq: int = -1
    esi_redundant_time: int = 0
    resume_mode: str = "auto"
    resume_from_path: str | None = None
    del_local_ckpt_after_load: bool = False
    val_before_train: bool = True
    test_freq: int = -1
    critic_warmup: int = 0
    default_hdfs_dir: str | None = None
    default_local_dir: str = "checkpoints/${trainer.project_name}/${trainer.experiment_name}"
    max_actor_ckpt_to_keep: int | None = None
    max_critic_ckpt_to_keep: int | None = None
    ray_wait_register_center_timeout: int = 300
    device: str = "cuda"
    controller_nsight_options: PPOMegatronNsightOptions = field(default_factory=PPOMegatronNsightOptions)
    worker_nsight_options: PPOMegatronNsightOptions = field(default_factory=PPOMegatronNsightOptions)


@dataclass
class RayInitConfig:
    r"""
    Configuration for Ray initialization.

    Args:
        num_cpus (int | None, optional): Number of CPUs for Ray. None means using all CPUs, which
            might cause hang if limited in systems like SLURM. Please set to a number allowed then
            (default: ``None``)
        timeline_json_file (str | None, optional): Timeline JSON file (default: ``None``)
    """

    num_cpus: int | None = None
    timeline_json_file: str | None = None


@dataclass
class PPOMegatronTrainerConfig:
    r"""
    Main configuration for PPO Megatron trainer.

    Args:
        actor_rollout_ref (PPOMegatronActorRolloutRefConfig, optional): Actor rollout reference
            configuration (default: ``PPOMegatronActorRolloutRefConfig()``)
        custom_reward_function (CustomRewardFunctionConfig, optional): Custom reward function
            configuration (default: ``CustomRewardFunctionConfig()``)
        algorithm (AlgoConfig, optional): Algorithm configuration (default: ``AlgoConfig()``)
        trainer (PPOMegatronTrainerConfig, optional): Trainer configuration (default: ``PPOMegatronTrainerConfig()``)
        ray_init (RayInitConfig, optional): Ray initialization configuration (default: ``RayInitConfig()``)
    """

    actor_rollout_ref: PPOMegatronActorRolloutRefConfig = field(default_factory=PPOMegatronActorRolloutRefConfig)
    custom_reward_function: CustomRewardFunctionConfig = field(default_factory=CustomRewardFunctionConfig)
    algorithm: AlgoConfig = field(default_factory=AlgoConfig)
    trainer: PPOMegatronTrainerConfig = field(default_factory=PPOMegatronTrainerConfig)
    ray_init: RayInitConfig = field(default_factory=RayInitConfig)
