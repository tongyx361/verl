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

from dataclasses import dataclass, field
from typing import Any

from verl.base_config import BaseConfig
from verl.utils.custom import CustomFunctionConfig

__all__ = ["AlgoConfig", "DynamicGroupFilterConfig", "KLControlConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class DynamicGroupFilterConfig(BaseConfig):
    """Configuration for group filtering (used in DAPO, etc.).

    NOTE: This feature is not compatible with asynchronous reward computation (``launch_reward_fn_async`=True`) yet.

    Attributes:
        enable (bool): Whether to enable group filtering.
        metric (str): Metric to use for filtering like "acc" and "seq_reward". Defaults to ``"seq_reward"``.
        max_num_gen_batches (int): Maximum number of batch generation attempts when collecting candidate responses.
            Non-positive values mean no upper limit (use with caution).
        filter_function (str): Path to the filter function (e.g., "my_package.my_module.my_filter_func").
            Required when ``filter_groups=True``. By default the original mixed rewards in DAPO is used
            ("verl.utils.filter.dynamic_group_filter.filter_for_mixed").
        filter_kwargs (dict): Additional arguments for the filter function.
    """

    enable: bool = False
    metric: str = "seq_reward"
    max_num_gen_batches: int = 0
    filter_function: CustomFunctionConfig = field(
        default_factory=lambda: CustomFunctionConfig(
            path="verl.utils.filter.dynamic_group_filter", name="filter_for_mixed"
        )
    )


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (DynamicGroupFilterConfig): Configuration for group filtering (used in DAPO, etc.).
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: DynamicGroupFilterConfig = field(default_factory=DynamicGroupFilterConfig)
