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
class SFTDataConfig:
    r"""
    Configuration for SFT data.

    Args:
        train_batch_size (int, optional): Training batch size (default: ``256``)
        micro_batch_size (int | None, optional): Micro batch size. Will be deprecated, use
            micro_batch_size_per_gpu (default: ``None``)
        micro_batch_size_per_gpu (int, optional): Micro batch size per GPU, this is also val batch
            size (default: ``4``)
        train_files (str, optional): Training files path (default: ``"~/data/gsm8k/train.parquet"``)
        val_files (str, optional): Validation files path (default: ``"~/data/gsm8k/test.parquet"``)
        prompt_key (str, optional): Single-turn settings - prompt key (default: ``"question"``)
        response_key (str, optional): Single-turn settings - response key (default: ``"answer"``)
        prompt_dict_keys (str | None, optional): Prompt dictionary keys (default: ``None``)
        response_dict_keys (str | None, optional): Response dictionary keys (default: ``None``)
        max_length (int, optional): Maximum length (default: ``1024``)
        truncation (str, optional): Truncation method (default: ``"error"``)
        balance_dp_token (bool, optional): Whether to balance DP token (default: ``False``)
        chat_template (str | None, optional): Chat template (default: ``None``)
        use_shm (bool, optional): Whether to use shared memory (default: ``False``)
    """

    train_batch_size: int = 256
    micro_batch_size: int | None = None
    micro_batch_size_per_gpu: int = 4
    train_files: str = "~/data/gsm8k/train.parquet"
    val_files: str = "~/data/gsm8k/test.parquet"
    prompt_key: str = "question"
    response_key: str = "answer"
    prompt_dict_keys: str | None = None
    response_dict_keys: str | None = None
    max_length: int = 1024
    truncation: str = "error"
    balance_dp_token: bool = False
    chat_template: str | None = None
    use_shm: bool = False


@dataclass
class SFTMultiturnConfig:
    r"""
    Configuration for SFT multi-turn settings.

    Args:
        enable (bool, optional): Set to true to use multi-turn dataset (default: ``False``)
        messages_key (str, optional): Key for messages list in multi-turn mode
            (default: ``"messages"``)
        tools_key (str, optional): Key for tools list in multi-turn mode (default: ``"tools"``)
        enable_thinking_key (str, optional): Whether to enable thinking in multi-turn mode
            (default: ``"enable_thinking"``)
    """

    enable: bool = False
    messages_key: str = "messages"
    tools_key: str = "tools"
    enable_thinking_key: str = "enable_thinking"


@dataclass
class SFTCustomClsConfig:
    r"""
    Configuration for SFT custom class.

    Args:
        path (str | None, optional): Path to custom class (default: ``None``)
        name (str | None, optional): Name of custom class (default: ``None``)
    """

    path: str | None = None
    name: str | None = None


@dataclass
class SFTFSDPWrapPolicyConfig:
    r"""
    Configuration for SFT FSDP wrap policy.

    Args:
        min_num_params (int, optional): Minimum number of parameters (default: ``0``)
    """

    min_num_params: int = 0


@dataclass
class SFTFSDPConfig:
    r"""
    Configuration for SFT FSDP.

    Args:
        model_dtype (str, optional): Model data type (default: ``"fp32"``)
        wrap_policy (SFTFSDPWrapPolicyConfig, optional): Wrap policy configuration
            (default: ``SFTFSDPWrapPolicyConfig()``)
        cpu_offload (bool, optional): Whether to offload to CPU (default: ``False``)
        offload_params (bool, optional): Whether to offload parameters (default: ``False``)
    """

    model_dtype: str = "fp32"
    wrap_policy: SFTFSDPWrapPolicyConfig = field(default_factory=SFTFSDPWrapPolicyConfig)
    cpu_offload: bool = False
    offload_params: bool = False


@dataclass
class SFTModelConfig:
    r"""
    Configuration for SFT model.

    Args:
        partial_pretrain (str, optional): Partial pretrain path (default: ``"~/models/gemma-1.1-7b-it"``)
        use_shm (bool, optional): Whether to use shared memory (default: ``False``)
        fsdp_config (SFTFSDPConfig, optional): FSDP configuration (default: ``SFTFSDPConfig()``)
        external_lib (str | None, optional): External library (default: ``None``)
        enable_gradient_checkpointing (bool, optional): Whether to enable gradient checkpointing
            (default: ``True``)
        trust_remote_code (bool, optional): Whether to trust remote code (default: ``False``)
        lora_rank (int, optional): Set to positive value to enable LoRA (e.g., 32)
            (default: ``0``)
        lora_alpha (int, optional): LoRA scaling factor (default: ``16``)
        target_modules (str, optional): Target modules for LoRA adaptation
            (default: ``"all-linear"``)
        use_liger (bool, optional): Whether to use Liger (default: ``False``)
        strategy (str, optional): Strategy (default: ``"fsdp2"``)
    """

    partial_pretrain: str = "~/models/gemma-1.1-7b-it"
    use_shm: bool = False
    fsdp_config: SFTFSDPConfig = field(default_factory=SFTFSDPConfig)
    external_lib: str | None = None
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str = "all-linear"
    use_liger: bool = False
    strategy: str = "fsdp2"


@dataclass
class SFTOptimConfig:
    r"""
    Configuration for SFT optimizer.

    Args:
        lr (float, optional): Learning rate (default: ``1e-5``)
        betas (list[float], optional): Beta values (default: ``[0.9, 0.95]``)
        weight_decay (float, optional): Weight decay (default: ``0.01``)
        warmup_steps_ratio (float, optional): Warmup steps ratio (default: ``0.1``)
        clip_grad (float, optional): Gradient clipping (default: ``1.0``)
        lr_scheduler (str, optional): Learning rate scheduler (default: ``"cosine"``)
    """

    lr: float = 1e-5
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.01
    warmup_steps_ratio: float = 0.1
    clip_grad: float = 1.0
    lr_scheduler: str = "cosine"


@dataclass
class SFTCheckpointConfig:
    r"""
    Configuration for SFT checkpoint.

    Args:
        save_contents (list[str], optional): What to include in saved checkpoints. With 'hf_model' you
            can save whole model as hf format, now only use sharded model checkpoint to save space
            (default: ``["model", "optimizer", "extra"]``)
        load_contents (list[str], optional): For more flexibility, you can specify the contents to
            load from the checkpoint (default: ``["model", "optimizer", "extra"]``)
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])


@dataclass
class SFTTrainerConfig:
    r"""
    Configuration for SFT trainer.

    Args:
        default_local_dir (str, optional): Default local directory
            (default: ``"checkpoints/${trainer.project_name}/${trainer.experiment_name}"``)
        default_hdfs_dir (str | None, optional): Default HDFS directory (default: ``None``)
        project_name (str, optional): Project name (default: ``"gsm8k-sft"``)
        experiment_name (str, optional): Experiment name (default: ``"test"``)
        total_epochs (int, optional): Total epochs (default: ``4``)
        total_training_steps (int | None, optional): Total training steps (default: ``None``)
        logger (list[str], optional): Logger (default: ``["console", "wandb"]``)
        seed (int, optional): Seed (default: ``1``)
        save_freq (int, optional): Save frequency (default: ``-1``)
        test_freq (int, optional): Test frequency (default: ``-1``)
        nnodes (int, optional): Number of nodes (default: ``1``)
        n_gpus_per_node (int, optional): Number of GPUs per node (default: ``8``)
        max_ckpt_to_keep (int | None, optional): Maximum number of checkpoints to keep, set to null
            to keep all (default: ``None``)
        resume_mode (str, optional): Resume mode: ``"auto"``, ``"disable"``, or ``"resume_path"``
            (default: ``"auto"``)
        resume_from_path (str | None, optional): Path to resume training from (used when resume_mode
            is ``"resume_path"`` or ``"auto"``) (default: ``None``)
        checkpoint (SFTCheckpointConfig, optional): Checkpoint configuration
            (default: ``SFTCheckpointConfig()``)
        device (str, optional): Device (default: ``"cuda"``)
    """

    default_local_dir: str = "checkpoints/${trainer.project_name}/${trainer.experiment_name}"
    default_hdfs_dir: str | None = None
    project_name: str = "gsm8k-sft"
    experiment_name: str = "test"
    total_epochs: int = 4
    total_training_steps: int | None = None
    logger: list[str] = field(default_factory=lambda: ["console", "wandb"])
    seed: int = 1
    save_freq: int = -1
    test_freq: int = -1
    nnodes: int = 1
    n_gpus_per_node: int = 8
    max_ckpt_to_keep: int | None = None
    resume_mode: str = "auto"
    resume_from_path: str | None = None
    checkpoint: SFTCheckpointConfig = field(default_factory=SFTCheckpointConfig)
    device: str = "cuda"


@dataclass
class SFTTrainerConfig:
    r"""
    Main configuration for SFT trainer.

    Args:
        data (SFTDataConfig, optional): Data configuration (default: ``SFTDataConfig()``)
        multiturn (SFTMultiturnConfig, optional): Multi-turn configuration
            (default: ``SFTMultiturnConfig()``)
        custom_cls (SFTCustomClsConfig, optional): Custom class configuration
            (default: ``SFTCustomClsConfig()``)
        model (SFTModelConfig, optional): Model configuration (default: ``SFTModelConfig()``)
        optim (SFTOptimConfig, optional): Optimizer configuration (default: ``SFTOptimConfig()``)
        ulysses_sequence_parallel_size (int, optional): Ulysses sequence parallel size
            (default: ``1``)
        use_remove_padding (bool, optional): Whether to use remove padding (default: ``False``)
        trainer (SFTTrainerConfig, optional): Trainer configuration (default: ``SFTTrainerConfig()``)
    """

    data: SFTDataConfig = field(default_factory=SFTDataConfig)
    multiturn: SFTMultiturnConfig = field(default_factory=SFTMultiturnConfig)
    custom_cls: SFTCustomClsConfig = field(default_factory=SFTCustomClsConfig)
    model: SFTModelConfig = field(default_factory=SFTModelConfig)
    optim: SFTOptimConfig = field(default_factory=SFTOptimConfig)
    ulysses_sequence_parallel_size: int = 1
    use_remove_padding: bool = False
    trainer: SFTTrainerConfig = field(default_factory=SFTTrainerConfig)
