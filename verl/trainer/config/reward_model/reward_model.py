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
class RewardModelModelConfig:
    r"""
    Configuration for reward model.

    Args:
        input_tokenizer (str, optional): Input tokenizer. If the reward model's chat template is
            inconsistent with the policy, we need to first decode to plaintext, then apply the rm's
            chat_template. Then score with RM. If chat_templates are consistent, it can be set to null
            (default: ``"${actor_rollout_ref.model.path}"``)
        path (str, optional): RM's HDFS path or local path. Note that RM only supports
            AutoModelForSequenceClassification. Other model types need to define their own
            RewardModelWorker and pass it from the code (default: ``"~/models/FsfairX-LLaMA3-RM-v0.1"``)
        external_lib (str | None, optional): External model implementation (optional)
            (default: ``"${actor_rollout_ref.model.external_lib}"``)
        trust_remote_code (bool, optional): Whether to enable loading a remote code model, default to
            False (default: ``False``)
    """

    input_tokenizer: str = "${actor_rollout_ref.model.path}"
    path: str = "~/models/FsfairX-LLaMA3-RM-v0.1"
    external_lib: str | None = "${actor_rollout_ref.model.external_lib}"
    trust_remote_code: bool = False


@dataclass
class SandboxFusionConfig:
    r"""
    Configuration for cloud/local sandbox fusion.

    Args:
        url (str | None, optional): Cloud/local function URL for sandbox execution (default: ``None``)
        max_concurrent (int, optional): Max concurrent requests allowed to sandbox (default: ``64``)
        memory_limit_mb (int, optional): Max memory limit for each sandbox process in MB
            (default: ``1024``)
    """

    url: str | None = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024


@dataclass
class RewardModelConfig:
    r"""
    Configuration for reward model.

    Args:
        enable (bool, optional): Whether to enable reward model. If False, we compute the reward only
            with the user-defined reward functions. In GSM8K and Math examples, we disable reward
            model. For RLHF alignment example using full_hh_rlhf, we utilize reward model to assess
            the responses. If False, the following parameters are not effective (default: ``False``)
        strategy (str, optional): FSDP strategy: ``"fsdp"`` or ``"fsdp2"`` (default: ``"???"``)
        model (RewardModelModelConfig, optional): Model config for reward scoring
            (default: ``RewardModelModelConfig()``)
        micro_batch_size (int | None, optional): Global micro batch size. [Deprecated] will be
            deprecated, use micro_batch_size_per_gpu (default: ``None``)
        micro_batch_size_per_gpu (int | None, optional): Local per-GPU micro batch size
            (default: ``None``)
        max_length (int | None, optional): Maximum sequence length to process for scoring
            (default: ``None``)
        use_dynamic_bsz (bool, optional): Whether to dynamically adjust batch size at runtime
            (default: ``"${critic.use_dynamic_bsz}"``)
        forward_max_token_len_per_gpu (int, optional): Maximum number of tokens per GPU in one
            forward pass (default: ``"${critic.forward_max_token_len_per_gpu}"``)
        reward_manager (str, optional): Reward Manager. This defines the mechanism of computing
            rule-based reward and handling different reward sources. Default is naive. If all
            verification functions are multiprocessing-safe, the reward manager can be set to prime
            for parallel verification (default: ``"naive"``)
        launch_reward_fn_async (bool, optional): Whether to launch custom reward function
            asynchronously during log_prob. Custom reward function executed async on CPU, during
            log_prob (default: ``False``)
        sandbox_fusion (SandboxFusionConfig, optional): Cloud/local sandbox fusion configuration for
            custom reward logic (default: ``SandboxFusionConfig()``)
        profiler (ProfilerConfig, optional): Profiler configs (default: ``ProfilerConfig()``)
    """

    enable: bool = False
    strategy: str = "???"
    model: RewardModelModelConfig = field(default_factory=RewardModelModelConfig)
    micro_batch_size: int | None = None
    micro_batch_size_per_gpu: int | None = None
    max_length: int | None = None
    use_dynamic_bsz: bool = "${critic.use_dynamic_bsz}"
    forward_max_token_len_per_gpu: int = "${critic.forward_max_token_len_per_gpu}"
    reward_manager: str = "naive"
    launch_reward_fn_async: bool = False
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
