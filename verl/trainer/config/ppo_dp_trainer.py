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

from hydra.core.config_store import ConfigStore

from verl.trainer.config.actor.dp_actor import DPActorConfig
from verl.trainer.config.rollout.rollout import RolloutConfig
from verl.utils.profiler.config import ProfilerConfig


@dataclass
class AlgoConfig:
    r"""
    Configuration for PPO algorithm parameters.

    Args:
        gamma (float, optional): Discount factor for future rewards (default: ``1.0``)
        lam (float, optional): Trade-off between bias and variance in the GAE estimator (default: ``1.0``)
        adv_estimator (str, optional): Advantage estimator type: ``"gae"``, ``"grpo"``, ``"reinforce_plus_plus"``, etc.
            (default: ``"gae"``)
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by std (specific to GRPO)
            (default: ``True``)
        use_kl_in_reward (bool, optional): Whether to enable in-reward KL penalty (default: ``False``)
        kl_penalty (str, optional): How to estimate KL divergence: ``"kl"``, ``"abs"``, ``"mse"``, ``"low_var_kl"``, or
            ``"full"`` (default: ``"kl"``)
        use_pf_ppo (bool, optional): Whether to enable preference feedback PPO (default: ``False``)
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    use_pf_ppo: bool = False


@dataclass
class KLControlConfig:
    r"""
    Configuration for KL divergence control.

    Args:
        type (str, optional): KL control type: ``"fixed"`` or ``"adaptive"`` (default: ``"fixed"``)
        kl_coef (float, optional): Initial coefficient for KL penalty (default: ``0.001``)
        horizon (int, optional): Horizon value for adaptive controller (if enabled) (default: ``10000``)
        target_kl (float, optional): Target KL divergence (used for adaptive controller) (default: ``0.1``)
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class PFPPOConfig:
    r"""
    Configuration for preference feedback PPO.

    Args:
        reweight_method (str, optional): Method for reweighting samples: ``"pow"``, ``"max_min"``, or ``"max_random"``
            (default: ``"pow"``)
        weight_pow (float, optional): Power used for weight scaling in ``"pow"`` method (default: ``2.0``)
    """

    reweight_method: str = "pow"
    weight_pow: float = 2.0


@dataclass
class CustomRewardFunctionConfig:
    r"""
    Configuration for custom reward functions.

    Args:
        path (str | None, optional): The path to the file containing your customized reward function.
            If not specified, pre-implemented reward functions will be used (default: ``None``)
        name (str, optional): The name of the reward function within the specified file (default: ``"compute_score"``)
    """

    path: str | None = None
    name: str = "compute_score"


@dataclass
class TrainerConfig:
    r"""
    Configuration for the PPO trainer.

    Args:
        balance_batch (bool, optional): Whether to balance batch sizes across distributed workers (default: ``True``)
        total_epochs (int, optional): Number of epochs in training (default: ``30``)
        total_training_steps (int | None, optional): Total training steps (can be set explicitly or derived from
            epochs) (default: ``None``)
        profile_steps (list[int] | None, optional): The steps that will be profiled. null means no profiling
            (default: ``None``)
        project_name (str, optional): Project name for experiment tracking (e.g., wandb) (default: ``"verl_examples"``)
        experiment_name (str, optional): Experiment name for run identification in tracking tools (default: ``"gsm8k"``)
        logger (list[str], optional): Logging backends to use: ``"console"``, ``"wandb"``, etc.
            (default: ``["console", "wandb"]``)
        log_val_generations (int, optional): Number of generations to log during validation (default: ``0``)
        rollout_data_dir (str | None, optional): Directory for logging rollout data; no dump if null (default: ``None``)
        validation_data_dir (str | None, optional): Directory for logging validation data; no dump if null
            (default: ``None``)
        nnodes (int, optional): Number of nodes used in the training (default: ``1``)
        n_gpus_per_node (int, optional): Number of GPUs per node (default: ``8``)
        save_freq (int, optional): Save frequency (by iteration) for model checkpoints (default: ``-1``)
        esi_redundant_time (int, optional): ESI redundant time for checkpoint saving before shutdown (default: ``0``)
        resume_mode (str, optional): Resume mode: ``"auto"``, ``"disable"``, or ``"resume_path"`` (default: ``"auto"``)
        resume_from_path (str | None, optional): Path to resume training from (only used when resume_mode is
            ``"resume_path"``) (default: ``None``)
        val_before_train (bool, optional): Whether to run validation before training begins (default: ``True``)
        val_only (bool, optional): Whether to run validation only (default: ``False``)
        test_freq (int, optional): Validation frequency (in training iterations) (default: ``-1``)
        critic_warmup (int, optional): Number of iterations to warm up the critic before updating policy
            (default: ``0``)
        default_hdfs_dir (str | None, optional): Default path to distributed filesystem for saving checkpoints
            (default: ``None``)
        del_local_ckpt_after_load (bool, optional): Whether to delete local checkpoints after loading
            (default: ``False``)
        default_local_dir (str, optional): Default local directory for saving checkpoints
            (default: ``"checkpoints/${trainer.project_name}/${trainer.experiment_name}"``)
        max_actor_ckpt_to_keep (int | None, optional): Maximum number of actor checkpoints to keep (default: ``None``)
        max_critic_ckpt_to_keep (int | None, optional): Maximum number of critic checkpoints to keep (default: ``None``)
        ray_wait_register_center_timeout (int, optional): Timeout (in seconds) for Ray worker to wait for registration
            (default: ``300``)
        device (str, optional): Device to run training on (e.g., ``"cuda"``, ``"cpu"``) (default: ``"cuda"``)
        use_legacy_worker_impl (str, optional): Whether to use legacy worker implementation:
            ``"auto"``, ``"enable"``, or ``"disable"`` (default: ``"auto"``)
    """

    balance_batch: bool = True
    total_epochs: int = 30
    total_training_steps: int | None = None
    profile_steps: list[int] | None = None
    project_name: str = "verl_examples"
    experiment_name: str = "gsm8k"
    logger: list[str] = field(default_factory=lambda: ["console", "wandb"])
    log_val_generations: int = 0
    rollout_data_dir: str | None = None
    validation_data_dir: str | None = None
    nnodes: int = 1
    n_gpus_per_node: int = 8
    save_freq: int = -1
    esi_redundant_time: int = 0
    resume_mode: str = "auto"
    resume_from_path: str | None = None
    val_before_train: bool = True
    val_only: bool = False
    test_freq: int = -1
    critic_warmup: int = 0
    default_hdfs_dir: str | None = None
    del_local_ckpt_after_load: bool = False
    default_local_dir: str = "checkpoints/${trainer.project_name}/${trainer.experiment_name}"
    max_actor_ckpt_to_keep: int | None = None
    max_critic_ckpt_to_keep: int | None = None
    ray_wait_register_center_timeout: int = 300
    device: str = "cuda"
    use_legacy_worker_impl: str = "auto"


@dataclass
class RayInitConfig:
    r"""
    Configuration for Ray initialization.

    Args:
        num_cpus (int | None, optional): Number of CPUs for Ray. Use a fixed number instead of null when using SLURM
            (default: ``None``)
        timeline_json_file (str | None, optional): Path to save Ray timeline JSON for performance profiling
            (default: ``None``)
    """

    num_cpus: int | None = None
    timeline_json_file: str | None = None


@dataclass
class FusedKernelOptions:
    r"""
    Configuration for fused kernel options.

    Args:
        impl_backend (str, optional): Implementation backend for fused kernels: ``"triton"`` or ``"torch"``
            (default: ``"torch"``)
    """

    impl_backend: str = "torch"


@dataclass
class PPODPModelConfig:
    r"""
    Configuration for PPO DP model.

    Args:
        path (str, optional): Huggingface model path. This can be either local path or HDFS path
            (default: ``"~/models/deepseek-llm-7b-chat"``)
        custom_chat_template (str | None, optional): Custom chat template for the model (default: ``None``)
        use_shm (bool, optional): Whether to use shared memory (SHM) for accelerating the loading of model weights
            (default: ``False``)
        external_lib (str | None, optional): Additional Python packages to register huggingface models/tokenizers
            (default: ``None``)
        override_config (dict, optional): Used to override model's original configurations, mainly dropout
            (default: ``{}``)
        enable_gradient_checkpointing (bool, optional): Enable gradient checkpointing for actor (default: ``True``)
        enable_activation_offload (bool, optional): Enable activation offloading for actor (default: ``False``)
        use_remove_padding (bool, optional): Whether to remove padding tokens in inputs during training
            (default: ``False``)
        lora_rank (int, optional): Set to positive value to enable LoRA (e.g., 32) (default: ``0``)
        lora_alpha (int, optional): LoRA scaling factor (default: ``16``)
        target_modules (str | list[str], optional): Target modules to apply LoRA. Options: ``"all-linear"`` (not
            recommended for VLMs) or ``[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]``
            (default: ``"all-linear"``)
        exclude_modules (str | None, optional): Exclude modules from applying Lora. Similar usage to target_modules
            and Peft. Example: ``'.*visual.*'`` for excluding the ViT in Qwen2.5-VL (default: ``None``)
        use_liger (bool, optional): Whether to use Liger for linear layer fusion (default: ``False``)
        use_fused_kernels (bool, optional): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP)
            (default: ``False``)
        fused_kernel_options (FusedKernelOptions, optional): Options for fused kernels. If use_fused_kernels is true,
            this will be used (default: ``FusedKernelOptions()``)
        trust_remote_code (bool, optional): Whether to enable loading a remote code model (default: ``False``)
    """

    path: str = "~/models/deepseek-llm-7b-chat"
    custom_chat_template: str | None = None
    use_shm: bool = False
    external_lib: str | None = None
    override_config: dict = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False
    use_remove_padding: bool = False
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str | list[str] = "all-linear"
    exclude_modules: str | None = None
    use_liger: bool = False
    use_fused_kernels: bool = False
    fused_kernel_options: FusedKernelOptions = field(default_factory=FusedKernelOptions)
    trust_remote_code: bool = False


@dataclass
class DPRolloutConfig(RolloutConfig):
    r"""
    Configuration for PPO DP rollout.

    Args:
        enable_chunked_prefill (bool, optional): Whether to enable chunked prefill (default: ``True``)
        load_format (str, optional): Format for loading model weights:
            ``"dummy_dtensor"``, ``"hf"``, ``"megatron"``, etc.
            ``"safetensors"`` (for huge model, and set use_shm=True); ``"dummy_dtensor"``: randomly init model weight
            (default: ``"dummy_dtensor"``)
        layered_summon (bool, optional): Whether to use layered summon for model loading (default: ``False``)
    """

    enable_chunked_prefill: bool = True
    load_format: str = "dummy_dtensor"
    layered_summon: bool = False


@dataclass
class PPODPActorRolloutRefConfig:
    r"""
    Configuration for PPO FSDP actor rollout reference.

    Args:
        hybrid_engine (bool, optional): Whether to use hybrid engine (default: ``True``)
        actor (DPActorConfig, optional): DP-specific PPO actor configuration (default: ``DPActorConfig()``)
        model (PPODPModelConfig, optional): Model configuration (default: ``PPODPModelConfig()``)
    """

    hybrid_engine: bool = True
    actor: DPActorConfig = field(default_factory=DPActorConfig)
    model: PPODPModelConfig = field(default_factory=PPODPModelConfig)
    rollout: DPRolloutConfig = field(default_factory=DPRolloutConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


@dataclass
class PPODPTrainerConfig:
    r"""
    Main configuration for PPO FSDP trainer.

    Args:
        actor_rollout_ref (PPODPActorRolloutRefConfig, optional): Actor rollout reference configuration
            (default: ``PPODPActorRolloutRefConfig()``)
        algorithm (AlgoConfig, optional): Algorithm configuration (default: ``AlgoConfig()``)
        custom_reward_function (CustomRewardFunctionConfig, optional): Custom reward function configuration
            (default: ``CustomRewardFunctionConfig()``)
        trainer (TrainerConfig, optional): Trainer configuration (default: ``TrainerConfig()``)
        ray_init (RayInitConfig, optional): Ray initialization configuration (default: ``RayInitConfig()``)
        kl_ctrl (KLControlConfig, optional): KL control configuration (default: ``KLControlConfig()``)
        pf_ppo (PFPPOConfig, optional): Preference feedback PPO configuration (default: ``PFPPOConfig()``)
    """

    actor_rollout_ref: PPODPActorRolloutRefConfig = field(default_factory=PPODPActorRolloutRefConfig)
    algorithm: AlgoConfig = field(default_factory=AlgoConfig)
    custom_reward_function: CustomRewardFunctionConfig = field(default_factory=CustomRewardFunctionConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ray_init: RayInitConfig = field(default_factory=RayInitConfig)
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    pf_ppo: PFPPOConfig = field(default_factory=PFPPOConfig)


cs = ConfigStore.instance()
PPO_DP_TRAINER_CONFIG_NAME = "ppo_dp_trainer"
cs.store(name=PPO_DP_TRAINER_CONFIG_NAME, node=PPODPTrainerConfig)
