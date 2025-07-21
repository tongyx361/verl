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
from typing import Any


@dataclass
class EngineKwargsConfig:
    r"""
    Configuration for extra inference engine arguments.

    Args:
        vllm (dict[str, Any], optional): vLLM-specific engine arguments
            (default: ``{}``)
        sglang (dict[str, Any], optional): SGLang-specific engine arguments
            (default: ``{}``)
    """

    vllm: dict[str, Any] = field(default_factory=dict)
    sglang: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValKwargsConfig:
    r"""
    Configuration for sampling parameters used during validation.

    Args:
        top_k (int, optional): Top-k sampling parameter. -1 for vLLM rollout,
            0 for HF rollout (default: ``-1``)
        top_p (float, optional): Top-p sampling parameter (default: ``1.0``)
        temperature (float, optional): Sampling temperature for rollout
            (default: ``0``)
        n (int, optional): Whether to repeat n times for validation
            (default: ``1``)
        do_sample (bool, optional): Whether to sample during training rollout.
            False uses greedy sampling (default: ``False``)
    """

    top_k: int = -1
    top_p: float = 1.0
    temperature: float = 0
    n: int = 1
    do_sample: bool = False


@dataclass
class MultiTurnConfig:
    r"""
    Configuration for multi-turn interaction config for tools or chat.

    Args:
        enable (bool, optional): Set to True for multi-turn tool interaction tasks;
            should set rollout.name to sglang as well (default: ``False``)
        max_assistant_turns (int | None, optional): null for no limit
            (default max_length // 3) (default: ``None``)
        tool_config_path (str | None, optional): null for no tool
            (default: ``None``)
        max_user_turns (int | None, optional): null for no limit
            (default max_length // 3) (default: ``None``)
        max_parallel_calls (int, optional): max parallel call for tools in single
            turn (default: ``1``)
        max_tool_response_length (int, optional): max length of tool response
            (default: ``256``)
        tool_response_truncate_side (str, optional): truncate side of tool
            response: left, middle, right (default: ``"middle"``)
        interaction_config_path (str | None, optional): null for no interaction
            (default: ``None``)
        completion_callback (str | None, optional): null for default callback
            (default: ``None``)
        use_inference_chat_template (bool, optional): When set to True, the
            model's default chat template is used for multi-turn rollout, which
            typically matches production behavior. When set to False, the token
            ids recorded for training are used instead; unlike the default chat
            template, these always include the model's full output, which may
            contain additional content such as reasoning content. This maintains
            the consistency between training and rollout, but it will lead to
            longer prompts (default: ``False``)
        tokenization_sanity_check_mode (str, optional): Tokenization sanity
            check mode: disable, strict, ignore_strippable (default: ``"strict"``)
        format (str, optional): Format of the multi-turn interaction. Options:
            hermes, llama3_json, ... (default: ``"hermes"``)
    """

    enable: bool = False
    max_assistant_turns: int | None = None
    tool_config_path: str | None = None
    max_user_turns: int | None = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: str | None = None
    completion_callback: str | None = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"


@dataclass
class CustomAsyncServerConfig:
    r"""
    Configuration for custom async server configs.

    Args:
        path (str | None, optional): Path to the custom async server
            implementation (default: ``None``)
        name (str | None, optional): Class name of the custom async server
            class (e.g. AsyncvLLMServer) (default: ``None``)
    """

    path: str | None = None
    name: str | None = None


@dataclass
class AgentConfig:
    r"""
    Configuration for agent loop based rollout configs.

    Args:
        num_workers (int, optional): Number of agent loop workers
            (default: ``8``)
        agent_loop_config_path (str | None, optional): custom agent loop
            config path, which should contain list of configs to intialize
            AgentLoop instances (default: ``None``)
        custom_async_server (CustomAsyncServerConfig, optional): custom async
            server configs (default: ``CustomAsyncServerConfig()``)
    """

    num_workers: int = 8
    agent_loop_config_path: str | None = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)


@dataclass
class TraceConfig:
    r"""
    Configuration for trace rollout data.

    Args:
        backend (str | None, optional): trace backend, support mlflow, weave
            (default: ``None``)
        token2text (bool, optional): whether translate token id to text in
            output (default: ``False``)
    """

    backend: str | None = None
    token2text: bool = False


@dataclass
class RolloutConfig:
    r"""
    Configuration for rollout.

    Args:
        name (str, optional): actor_rollout_ref.rollout.name: hf/vllm/sglang.
            The default value will be removed in the future (default: ``"vllm"``)
        mode (str, optional): sync: LLM, async: AsyncLLM (default: ``"sync"``)
        temperature (float, optional): Sampling temperature for rollout
            (default: ``1.0``)
        top_k (int, optional): Top-k sampling parameter. -1 for vLLM rollout,
            0 for HF rollout (default: ``-1``)
        top_p (float, optional): Top-p sampling parameter. Default 1.0
            (default: ``1``)
        prompt_length (int, optional): typically the same as data max prompt
            length, same as data.max_prompt_length if it exists (default: ``512``)
        response_length (int, optional): typically the same as data max response
            length, same as data.max_response_length if it exists (default: ``512``)
        dtype (str, optional): for vllm rollout, Rollout model parameters type.
            Align with actor model's FSDP/Megatron type (default: ``"bfloat16"``)
        gpu_memory_utilization (float, optional): Fraction of GPU memory used
            by vLLM/SGLang for KV cache (default: ``0.5``)
        ignore_eos (bool, optional): Whether to ignore EOS and continue
            generating after EOS is hit (default: ``False``)
        enforce_eager (bool, optional): Whether to disable CUDA graph. Default
            True to allow cache freeing (default: ``True``)
        free_cache_engine (bool, optional): Whether to free engine KVCache
            after generation. Set enforce_eager=True when enabled
            (default: ``True``)
        tensor_model_parallel_size (int, optional): TP size for rollout. Not
            effective for hf (default: ``2``)
        max_num_batched_tokens (int, optional): max number of tokens in a batch
            (default: ``8192``)
        max_model_len (int | None, optional): max length for rollout
            (default: ``None``)
        max_num_seqs (int, optional): max length of sequences (default: ``1024``)
        log_prob_micro_batch_size (int | None, optional): [Will be deprecated,
            use log_prob_micro_batch_size_per_gpu] The batch size for one
            forward pass in the computation of log_prob. Global batch size
            (default: ``None``)
        log_prob_micro_batch_size_per_gpu (int | None, optional): The batch
            size for one forward pass in the computation of log_prob. Local
            batch size per GPU (default: ``None``)
        log_prob_use_dynamic_bsz (bool, optional): enable dynamic batch size
            (sequence packing) for log_prob computation, same as
            actor_rollout_ref.actor.use_dynamic_bsz if it exists, otherwise
            false (default: ``False``)
        log_prob_max_token_len_per_gpu (int, optional): max token length for
            log_prob computation, same as
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu if it exists,
            otherwise 16384 (default: ``16384``)
        disable_log_stats (bool, optional): disable logging statistics
            (default: ``True``)
        do_sample (bool, optional): for hf rollout, Whether to sample during
            training rollout. False uses greedy sampling (default: ``True``)
        n (int, optional): number of responses (i.e. num sample times). > 1 for
            grpo (default: ``1``)
        multi_stage_wake_up (bool, optional): Whether to wake up inference
            engine in multi-stage. (Wake up model weights first, then resume
            kv cache) (default: ``False``)
        engine_kwargs (EngineKwargsConfig, optional): Extra inference engine
            arguments (vllm, sglang) (default: ``EngineKwargsConfig()``)
        val_kwargs (ValKwargsConfig, optional): Sampling parameters used
            during validation (default: ``ValKwargsConfig()``)
        multi_turn (MultiTurnConfig, optional): Multi-turn interaction config
            for tools or chat (default: ``MultiTurnConfig()``)
        calculate_log_probs (bool, optional): support logging rollout prob for
            debugging purpose (default: ``False``)
        agent (AgentConfig, optional): [Experimental] agent loop based rollout
            configs (default: ``AgentConfig()``)
        update_weights_bucket_megabytes (int, optional): Specifies the tensor
            bucket size (in megabytes) for batch weight updates during rollout
            operations (default: ``512``)
        trace (TraceConfig, optional): trace rollout data
            (default: ``TraceConfig()``)
    """

    name: str = "vllm"
    mode: str = "sync"
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1
    prompt_length: int = 512
    response_length: int = 512
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 2
    max_num_batched_tokens: int = 8192
    max_model_len: int | None = None
    max_num_seqs: int = 1024
    log_prob_micro_batch_size: int | None = None
    log_prob_micro_batch_size_per_gpu: int | None = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384
    disable_log_stats: bool = True
    do_sample: bool = True
    n: int = 1
    multi_stage_wake_up: bool = False
    engine_kwargs: EngineKwargsConfig = field(default_factory=EngineKwargsConfig)
    val_kwargs: ValKwargsConfig = field(default_factory=ValKwargsConfig)
    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)
    calculate_log_probs: bool = False
    agent: AgentConfig = field(default_factory=AgentConfig)
    update_weights_bucket_megabytes: int = 512
    trace: TraceConfig = field(default_factory=TraceConfig)
