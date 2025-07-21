# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket
from dataclasses import dataclass, field

import hydra
import ray
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type


@dataclass
class PolicyLossConfig:
    r"""
    Configuration for PPO policy loss function.

    Args:
        loss_mode (str, optional): Loss function mode: ``"vanilla"`` / ``"clip-cov"`` / ``"kl-cov"`` / ``"gpg"`` from
            https://arxiv.org/abs/2505.22617 (default: ``"vanilla"``)
        clip_cov_ratio (float, optional): Ratio of tokens to be clipped for clip-cov loss (default: ``0.0002``)
        clip_cov_lb (float, optional): Lower bound for clip-cov loss (default: ``1.0``)
        clip_cov_ub (float, optional): Upper bound for clip-cov loss (default: ``5.0``)
        kl_cov_ratio (float, optional): Ratio of tokens to be applied kl penalty for kl-cov loss (default: ``0.0002``)
        ppo_kl_coef (float, optional): KL divergence penalty coefficient (default: ``0.1``)
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class CheckpointConfig:
    r"""
    Configuration for model checkpointing.

    Args:
        save_contents (list[str], optional): What to include in saved checkpoints.
            With 'hf_model' you can save whole model as hf format,
            now only use sharded model checkpoint to save space (default: ["model", "optimizer", "extra"])
        load_contents (list[str], optional): For more flexibility,
            you can specify the contents to load from the checkpoint (default: ["model", "optimizer", "extra"])
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])


@dataclass
class OptimConfig:
    r"""
    Base optimizer configuration.

    Args:
        lr (float, optional): Learning rate (default: 1e-6)
        lr_warmup_steps_ratio (float, optional): Warmup steps ratio (used if lr_warmup_steps is negative) (default: 0.0)
        total_training_steps (int, optional): Total training steps (must be overridden at runtime) (default: -1)
        weight_decay (float, optional): Weight decay (default: 0.01)
    """

    lr: float = 1e-6
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01


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
class PPOActorConfig:
    r"""
    Abstract actor configuration for PPO training.

    Args:
        strategy (str): The abstract actor configs. ``"fsdp"``, ``"fsdp2"`` or ``"megatron"``. must be set.
        ppo_mini_batch_size (int, optional): Split each sample into sub-batches of this size for PPO (default: ``256``)
        ppo_micro_batch_size (int | None, optional): [Deprecated] Global micro batch size (default: ``None``)
        ppo_micro_batch_size_per_gpu (int | None, optional): Local per-GPU micro batch size (default: ``None``)
        use_dynamic_bsz (bool, optional): Whether to automatically adjust batch size at runtime (default: ``False``)
        ppo_max_token_len_per_gpu (int, optional): Max tokens per GPU in one PPO batch; affects gradient accumulation
            (default: ``16384``)
        clip_ratio (float, optional): PPO clip ratio (default: ``0.2``)
        clip_ratio_low (float, optional): Lower bound for asymmetric clipping (used in dual-clip PPO) (default: ``0.2``)
        clip_ratio_high (float, optional): Upper bound for asymmetric clipping (used in dual-clip PPO)
            (default: ``0.2``)
        policy_loss (PolicyLossConfig, optional): Policy loss configuration (default: ``PolicyLossConfig()``)
        clip_ratio_c (float, optional): Constant C in Dual-clip PPO; clips when advantage < 0 and ratio > C
            (default: ``3.0``)
        loss_agg_mode (str, optional): Loss aggregation mode: ``"token-mean"``, ``"seq-mean-token-sum"``, or
            ``"seq-mean-token-mean"`` (default: ``"token-mean"``)
        entropy_coeff (float, optional): Entropy regularization coefficient in PPO loss (default: ``0.0``)
        use_kl_loss (bool, optional): Whether to use KL loss instead of KL reward penalty. True for GRPO
            (default: ``False``)
        use_torch_compile (bool, optional): Whether to use ``torch.compile()`` (default: ``True``)
        kl_loss_coef (float, optional): KL loss coefficient when use_kl_loss is enabled. For GRPO (default: ``0.001``)
        kl_loss_type (str, optional): Type of KL divergence loss. Options: ``"kl"``(k1), ``"abs"``, ``"mse"``(k2),
            ``"low_var_kl"``(k3), ``"full"`` (default: ``"low_var_kl"``)
        ppo_epochs (int, optional): Number of PPO epochs per batch (default: ``1``)
        shuffle (bool, optional): Shuffle training data across PPO epochs (default: ``False``)
        checkpoint (CheckpointConfig, optional): Checkpoint configuration (default: ``CheckpointConfig()``)
        optim (OptimConfig, optional): Optimizer configuration (default: ``OptimConfig()``)
    """

    strategy: str
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: int | None = None
    ppo_micro_batch_size_per_gpu: int | None = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0.0
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class FSDPOptimConfig(OptimConfig):
    r"""
    FSDP-specific optimizer configuration.

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
class PPOFSDPActorConfig(PPOActorConfig):
    r"""
    FSDP-specific PPO actor configuration.

    Args:
        strategy (str, optional): The abstract actor configs. ``"fsdp"``, ``"fsdp2"`` or ``"megatron"``. must be set.
            (default: ``"fsdp"``)
        grad_clip (float, optional): Gradient clipping for actor updates, specific to the strategy (default: ``1.0``)
        ulysses_sequence_parallel_size (int, optional): Sequence parallelism size for Ulysses-style model parallelism
            (default: ``1``)
        entropy_from_logits_with_chunking (bool, optional): Calculate entropy with chunking to reduce memory peak
            (default: ``False``)
        entropy_checkpointing (bool, optional): Recompute entropy (default: ``False``)
        optim (FSDPOptimConfig, optional): FSDP-specific optimizer configuration (default: ``FSDPOptimConfig()``)
        fsdp_config (FSDPConfig, optional): FSDP configuration (default: ``FSDPConfig()``)
    """

    # TODO(haibin.lin): switch to fsdp2
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    optim: FSDPOptimConfig = field(default_factory=FSDPOptimConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class PPOFSDPActorRolloutRefConfig:
    r"""
    Configuration for PPO FSDP actor rollout reference.

    Args:
        actor (PPOFSDPActorConfig, optional): FSDP-specific PPO actor configuration (default: ``PPOFSDPActorConfig()``)
    """

    actor: PPOFSDPActorConfig = field(default_factory=PPOFSDPActorConfig)


@dataclass
class PPOFSDPTrainerConfig:
    r"""
    Main configuration for PPO FSDP trainer.

    Args:
        actor_rollout_ref (PPOFSDPActorRolloutRefConfig, optional): Actor rollout reference configuration
            (default: ``PPOFSDPActorRolloutRefConfig()``)
        algorithm (AlgoConfig, optional): Algorithm configuration (default: ``AlgoConfig()``)
        custom_reward_function (CustomRewardFunctionConfig, optional): Custom reward function configuration
            (default: ``CustomRewardFunctionConfig()``)
        trainer (TrainerConfig, optional): Trainer configuration (default: ``TrainerConfig()``)
        ray_init (RayInitConfig, optional): Ray initialization configuration (default: ``RayInitConfig()``)
        kl_ctrl (KLControlConfig, optional): KL control configuration (default: ``KLControlConfig()``)
        pf_ppo (PFPPOConfig, optional): Preference feedback PPO configuration (default: ``PFPPOConfig()``)
    """

    actor_rollout_ref: PPOFSDPActorRolloutRefConfig = field(default_factory=PPOFSDPActorRolloutRefConfig)
    algorithm: AlgoConfig = field(default_factory=AlgoConfig)
    custom_reward_function: CustomRewardFunctionConfig = field(default_factory=CustomRewardFunctionConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ray_init: RayInitConfig = field(default_factory=RayInitConfig)
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    pf_ppo: PFPPOConfig = field(default_factory=PFPPOConfig)


cs = ConfigStore.instance()
PPO_TRAINER_CONFIG_NAME = "ppo_trainer"
cs.store(name=PPO_TRAINER_CONFIG_NAME, node=PPOFSDPTrainerConfig)


@hydra.main(version_base=None, config_name=PPO_TRAINER_CONFIG_NAME)
def main(config: PPOFSDPTrainerConfig):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration object containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env=PPO_RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # import warnings
                # warnings.warn(f"Legacy worker impl is going to be deprecated, will be removed in the future. \
                #   Please set trainer.use_legacy_worker_impl = false to switch to the new worker implementation.")
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
