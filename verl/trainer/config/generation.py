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
class GenerationDataConfig:
    r"""
    Configuration for generation data.

    Args:
        path (str, optional): Data path (default: ``"~/data/rlhf/math/test.parquet"``)
        prompt_key (str, optional): Prompt key (default: ``"prompt"``)
        n_samples (int, optional): Number of samples (default: ``5``)
        output_path (str, optional): Output path (default: ``"/opt/tiger/math_Qwen2-7B-Instruct.parquet"``)
        batch_size (int, optional): Batch size (default: ``128``)
    """

    path: str = "~/data/rlhf/math/test.parquet"
    prompt_key: str = "prompt"
    n_samples: int = 5
    output_path: str = "/opt/tiger/math_Qwen2-7B-Instruct.parquet"
    batch_size: int = 128


@dataclass
class GenerationModelConfig:
    r"""
    Configuration for generation model.

    Args:
        path (str, optional): Model path (default: ``"~/models/Qwen2-7B-Instruct"``)
        external_lib (str | None, optional): External library (default: ``None``)
    """

    path: str = "~/models/Qwen2-7B-Instruct"
    external_lib: str | None = None


@dataclass
class GenerationRolloutConfig:
    r"""
    Configuration for generation rollout.

    Args:
        name (str, optional): Rollout name (default: ``"vllm"``)
        mode (str, optional): Mode: sync for LLM, async for AsyncLLM (default: ``"sync"``)
        temperature (float, optional): Temperature (default: ``1.0``)
        top_k (int, optional): Top-k, 0 for hf rollout, -1 for vllm rollout (default: ``50``)
        top_p (float, optional): Top-p (default: ``0.7``)
        prompt_length (int, optional): Prompt length (default: ``1536``)
        response_length (int, optional): Response length (default: ``512``)
        dtype (str, optional): Data type, should align with FSDP (default: ``"bfloat16"``)
        gpu_memory_utilization (float, optional): GPU memory utilization (default: ``0.5``)
        ignore_eos (bool, optional): Whether to ignore EOS (default: ``False``)
        enforce_eager (bool, optional): Whether to enforce eager (default: ``True"``)
        free_cache_engine (bool, optional): Whether to free cache engine (default: ``True"``)
        load_format (str, optional): Load format (default: ``"dummy_dtensor"``)
        tensor_model_parallel_size (int, optional): Tensor model parallel size (default: ``1``)
        max_num_batched_tokens (int, optional): Maximum number of batched tokens (default: ``8192``)
        max_model_len (int | None, optional): Maximum model length (default: ``None``)
        max_num_seqs (int, optional): Maximum number of sequences (default: ``1024``)
        log_prob_micro_batch_size (int | None, optional): Log probability micro batch size, will be
            deprecated, use log_prob_micro_batch_size_per_gpu (default: ``None``)
        log_prob_micro_batch_size_per_gpu (int, optional): Log probability micro batch size per GPU
            (default: ``8``)
        do_sample (bool, optional): Whether to do sample, for hf rollout (default: ``True``)
        disable_log_stats (bool, optional): Whether to disable log stats, for hf rollout
            (default: ``True``)
        enable_chunked_prefill (bool, optional): Whether to enable chunked prefill, for hf rollout
            (default: ``True``)
        n (int, optional): N, for hf rollout (default: ``1``)
        calculate_log_probs (bool, optional): Support logging rollout prob for debugging purpose
            (default: ``False``)
    """

    name: str = "vllm"
    mode: str = "sync"
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.7
    prompt_length: int = 1536
    response_length: int = 512
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    free_cache_engine: bool = True
    load_format: str = "dummy_dtensor"
    tensor_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_model_len: int | None = None
    max_num_seqs: int = 1024
    log_prob_micro_batch_size: int | None = None
    log_prob_micro_batch_size_per_gpu: int = 8
    do_sample: bool = True
    disable_log_stats: bool = True
    enable_chunked_prefill: bool = True
    n: int = 1
    calculate_log_probs: bool = False


@dataclass
class GenerationActorConfig:
    r"""
    Configuration for generation actor.

    Args:
        strategy (str, optional): Strategy, this is for backward-compatibility (default: ``"fsdp"``)
        ulysses_sequence_parallel_size (int, optional): SP size (default: ``1``)
        entropy_from_logits_with_chunking (bool, optional): Calculate entropy with chunking to reduce
            memory peak (default: ``False``)
        entropy_checkpointing (bool, optional): Recompute entropy (default: ``False``)
        fsdp_config (dict, optional): FSDP configuration (default: ``{"fsdp_size": -1, "forward_prefetch": False}``)
    """

    strategy: str = "fsdp"
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: dict = field(default_factory=lambda: {"fsdp_size": -1, "forward_prefetch": False})


@dataclass
class GenerationTrainerConfig:
    r"""
    Configuration for generation trainer.

    Args:
        nnodes (int, optional): Number of nodes (default: ``1``)
        n_gpus_per_node (int, optional): Number of GPUs per node (default: ``8``)
        device (str, optional): Device (default: ``"cuda"``)
    """

    nnodes: int = 1
    n_gpus_per_node: int = 8
    device: str = "cuda"


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
class GenerationConfig:
    r"""
    Main configuration for generation.

    Args:
        trainer (GenerationTrainerConfig, optional): Trainer configuration (default: ``GenerationTrainerConfig()``)
        data (GenerationDataConfig, optional): Data configuration (default: ``GenerationDataConfig()``)
        model (GenerationModelConfig, optional): Model configuration (default: ``GenerationModelConfig()``)
        rollout (GenerationRolloutConfig, optional): Rollout configuration (default: ``GenerationRolloutConfig()``)
        actor (GenerationActorConfig, optional): Actor configuration (default: ``GenerationActorConfig()``)
        ray_init (RayInitConfig, optional): Ray initialization configuration (default: ``RayInitConfig()``)
    """

    trainer: GenerationTrainerConfig = field(default_factory=GenerationTrainerConfig)
    data: GenerationDataConfig = field(default_factory=GenerationDataConfig)
    model: GenerationModelConfig = field(default_factory=GenerationModelConfig)
    rollout: GenerationRolloutConfig = field(default_factory=GenerationRolloutConfig)
    actor: GenerationActorConfig = field(default_factory=GenerationActorConfig)
    ray_init: RayInitConfig = field(default_factory=RayInitConfig)
