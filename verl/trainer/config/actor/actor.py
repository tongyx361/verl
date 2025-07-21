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
class CheckpointingConfig:
    r"""
    Configuration for model checkpointing.

    Args:
        save_contents (list[str], optional): What to include in saved checkpoints.
            With 'hf_model' you can save whole model as hf format,
            now only use sharded model checkpoint to save space (default: ["model", "optimizer", "extra"])
        load_contents (list[str], optional): For more flexibility,
            you can specify the contents to load from the checkpoint. If None, defaults to save_contents
            (default: None)
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] | None = None

    def __post_init__(self):
        """Set load_contents to save_contents if not explicitly provided."""
        if self.load_contents is None:
            self.load_contents = self.save_contents.copy()


@dataclass
class OptimizerConfig:
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
        checkpoint (CheckpointingConfig, optional): Checkpoint configuration (default: ``CheckpointingConfig()``)
        optim (OptimizerConfig, optional): Optimizer configuration (default: ``OptimizerConfig()``)
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
    checkpoint: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
