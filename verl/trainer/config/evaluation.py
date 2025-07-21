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
class EvaluationDataConfig:
    r"""
    Configuration for evaluation data.

    Args:
        path (str, optional): Data path (default: ``"/tmp/math_Qwen2-7B-Instruct.parquet"``)
        prompt_key (str, optional): Prompt key (default: ``"prompt"``)
        response_key (str, optional): Response key (default: ``"responses"``)
        data_source_key (str, optional): Data source key (default: ``"data_source"``)
        reward_model_key (str, optional): Reward model key (default: ``"reward_model"``)
    """

    path: str = "/tmp/math_Qwen2-7B-Instruct.parquet"
    prompt_key: str = "prompt"
    response_key: str = "responses"
    data_source_key: str = "data_source"
    reward_model_key: str = "reward_model"


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
class EvaluationConfig:
    r"""
    Main configuration for evaluation.

    Args:
        data (EvaluationDataConfig, optional): Data configuration (default: ``EvaluationDataConfig()``)
        custom_reward_function (CustomRewardFunctionConfig, optional): Custom reward function
            configuration (default: ``CustomRewardFunctionConfig()``)
        ray_init (RayInitConfig, optional): Ray initialization configuration (default: ``RayInitConfig()``)
    """

    data: EvaluationDataConfig = field(default_factory=EvaluationDataConfig)
    custom_reward_function: CustomRewardFunctionConfig = field(default_factory=CustomRewardFunctionConfig)
    ray_init: RayInitConfig = field(default_factory=RayInitConfig)
