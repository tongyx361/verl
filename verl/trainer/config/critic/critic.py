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

from verl.utils.profiler.config import ProfilerConfig


@dataclass
class CriticOptimConfig:
    r"""
    Configuration for critic optimizer.

    Args:
        lr_warmup_steps_ratio (float, optional): Warmup steps ratio; total steps will be injected at
            runtime (default: ``0.0``)
        total_training_steps (int, optional): Total training steps (must be overridden at runtime)
            (default: ``-1``)
        weight_decay (float, optional): Weight decay (default: ``0.01``)
    """

    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01


@dataclass
class CriticModelConfig:
    r"""
    Configuration for critic model.

    Args:
        path (str, optional): Path to pretrained model weights (default: ``"~/models/deepseek-llm-7b-chat"``)
        tokenizer_path (str, optional): Tokenizer path (defaults to actor's model path)
            (default: ``"${oc.select:actor_rollout_ref.model.path,"~/models/deepseek-llm-7b-chat"}"``)
        override_config (dict, optional): Hugging Face config override (default: ``{}``)
        external_lib (str | None, optional): External model implementation (optional)
            (default: ``"${oc.select:actor_rollout_ref.model.external_lib,null}"``)
        enable_gradient_checkpointing (bool, optional): Enable gradient checkpointing to save memory
            (default: ``True``)
        trust_remote_code (bool, optional): Whether to trust remote code from Hugging Face models
            (default: ``"${oc.select:actor_rollout_ref.model.trust_remote_code,false}"``)
    """

    path: str = "~/models/deepseek-llm-7b-chat"
    tokenizer_path: str = '${oc.select:actor_rollout_ref.model.path,"~/models/deepseek-llm-7b-chat"}'
    override_config: dict = field(default_factory=dict)
    external_lib: str | None = "${oc.select:actor_rollout_ref.model.external_lib,null}"
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = "${oc.select:actor_rollout_ref.model.trust_remote_code,false}"


@dataclass
class CriticCheckpointConfig:
    r"""
    Configuration for critic checkpoint.

    Args:
        save_contents (list[str], optional): What to include in saved checkpoints. With 'hf_model' you
            can save whole model as hf format, now only use sharded model checkpoint to save space
            (default: ``["model", "optimizer", "extra"]``)
        load_contents (list[str], optional): What to include when loading checkpoints
            (default: ``"${.save_contents}"``)
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = "${.save_contents}"


@dataclass
class CriticConfig:
    r"""
    Configuration for critic.

    Args:
        rollout_n (int, optional): Number of rollouts per update (mirrors actor rollout_n)
            (default: ``"${oc.select:actor_rollout_ref.rollout.n,1}"``)
        strategy (str, optional): FSDP or FSDP2 strategy used for critic model training
            (default: ``"???"``)
        optim (CriticOptimConfig, optional): Optimizer configs (default: ``CriticOptimConfig()``)
        model (CriticModelConfig, optional): Model config for the critic (default: ``CriticModelConfig()``)
        ppo_mini_batch_size (int, optional): PPO mini-batch size per update
            (default: ``"${oc.select:actor_rollout_ref.actor.ppo_mini_batch_size,256}"``)
        ppo_micro_batch_size (int | None, optional): Global micro batch size. [Deprecated]
            (default: ``None``)
        ppo_micro_batch_size_per_gpu (int | None, optional): Local per-GPU micro batch size
            (default: ``"${oc.select:.ppo_micro_batch_size,null}"``)
        use_dynamic_bsz (bool, optional): Whether to automatically adjust batch size at runtime
            (default: ``"${oc.select:actor_rollout_ref.actor.use_dynamic_bsz,false}"``)
        ppo_max_token_len_per_gpu (int, optional): Max tokens per GPU in one PPO batch (doubled for
            critic) (default: ``32768``)
        forward_max_token_len_per_gpu (int, optional): Max token length per GPU in forward pass
            (default: ``"${.ppo_max_token_len_per_gpu}"``)
        ppo_epochs (int, optional): Number of PPO epochs per batch
            (default: ``"${oc.select:actor_rollout_ref.actor.ppo_epochs,1}"``)
        shuffle (bool, optional): Shuffle training data across PPO epochs
            (default: ``"${oc.select:actor_rollout_ref.actor.shuffle,false}"``)
        cliprange_value (float, optional): PPO value function clipping range (default: ``0.5``)
        loss_agg_mode (str, optional): Loss aggregation mode: ``"token-mean"``, ``"seq-mean-token-sum"``,
            or ``"seq-mean-token-mean"`` (default: ``"${oc.select:actor_rollout_ref.actor.loss_agg_mode,token-mean}"``)
        checkpoint (CriticCheckpointConfig, optional): Checkpoint configs (default: ``CriticCheckpointConfig()``)
        profiler (ProfilerConfig, optional): Profiler configs (default: ``ProfilerConfig()``)
    """

    rollout_n: int = "${oc.select:actor_rollout_ref.rollout.n,1}"
    strategy: str = "???"
    optim: CriticOptimConfig = field(default_factory=CriticOptimConfig)
    model: CriticModelConfig = field(default_factory=CriticModelConfig)
    ppo_mini_batch_size: int = "${oc.select:actor_rollout_ref.actor.ppo_mini_batch_size,256}"
    ppo_micro_batch_size: int | None = None
    ppo_micro_batch_size_per_gpu: int | None = "${oc.select:.ppo_micro_batch_size,null}"
    use_dynamic_bsz: bool = "${oc.select:actor_rollout_ref.actor.use_dynamic_bsz,false}"
    ppo_max_token_len_per_gpu: int = 32768
    forward_max_token_len_per_gpu: int = "${.ppo_max_token_len_per_gpu}"
    ppo_epochs: int = "${oc.select:actor_rollout_ref.actor.ppo_epochs,1}"
    shuffle: bool = "${oc.select:actor_rollout_ref.actor.shuffle,false}"
    cliprange_value: float = 0.5
    loss_agg_mode: str = "${oc.select:actor_rollout_ref.actor.loss_agg_mode,token-mean}"
    checkpoint: CriticCheckpointConfig = field(default_factory=CriticCheckpointConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
