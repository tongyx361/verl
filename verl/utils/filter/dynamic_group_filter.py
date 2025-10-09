# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import importlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.base_config import BaseConfig
from verl.trainer.config.algorithm import DynamicGroupFilterConfig
from verl.utils.custom import CustomFunctionConfig, get_custom_fn
from verl.utils.debug import metrics

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", logging.WARN))


class DynamicGroupFilter:
    """Dynamic group filter (in DAPO).

    Args:
        config (DynamicGroupFilterConfig): Configration for dynamic group filter.

    References:
        Yu and Zhang et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale.
            NeurIPS 2025. https://neurips.cc/virtual/2025/poster/120129.
        Tong and Sheng et al. The open-source implementation of DAPO.
            https://github.com/volcengine/verl/tree/main/recipe/dapo.
    """

    def __init__(self, full_config: DictConfig):
        filter_config: DynamicGroupFilterConfig = full_config.algorithm.dynamic_group_filter
        self.metric = filter_config.metric

        max_num_gen_batches = filter_config.max_num_gen_batches
        self.max_num_gen_batches = max_num_gen_batches if max_num_gen_batches > 0 else float("inf")

        self.filter_fn = get_custom_fn(filter_config.filter_function)
        assert self.filter_fn, f"Failed to load custom filter function from `{filter_config.filter_function=}`."

        group_train_batch_size = full_config.data.train_batch_size
        group_size = full_config.actor_rollout_ref.rollout.n
        self.traj_train_batch_size = int(group_train_batch_size * group_size)

        self.reset()

    def reset(self) -> None:
        # Raw
        self.num_gen_batches = 0
        self.raw_metric_values = []
        # Qualified
        # NOTE: Maintain a full batch instead of batch list to minimize the peak memory usage.
        self.full_qualified_batch = None

    def filter(self, batch: DataProto, group_train_batch_size: int, group_size: int) -> DataProto | None:
        """Filter and accumulate a candidate batch.

        Args:
            batch (DataProto): A candidate batch to filter and accumulate.
            group_train_batch_size (int): The train batch size of prompt groups.
            group_size (int): The group size, i.e. the number of trajectories in a prompt group.

        Returns:
            DataProto | None: The batch ready for training, ``None`` otherwise.
        """
        self.num_gen_batches += 1

        # Collect raw metric values
        if self.metric == "seq_final_reward":
            final_rewards = batch.batch["token_level_rewards"]
            metric_values = final_rewards.sum(dim=-1).tolist()
        elif self.metric == "seq_reward":
            rewards = batch.batch["token_level_rewards"]
            metric_values = rewards.sum(dim=-1).tolist()
        else:
            metric_values = batch.batch[self.metric].tolist()
        assert isinstance(metric_values, list)
        self.raw_metric_values.extend(metric_values)

        # Collect metric values by group ID
        group_ids = batch.non_tensor_batch["uid"]
        group_id_to_metric_values = defaultdict(list)
        for gid, metric_val in zip(group_ids, metric_values, strict=True):
            group_id_to_metric_values[gid].append(metric_val)
        # Keep group IDs that pass the filter
        qualified_group_ids = [gid for gid, gmetrics in group_id_to_metric_values.items() if self.filter_fn(gmetrics)]
        # Find indices of qualified trajectories
        qualified_traj_idxs = [i for i, gid in enumerate(group_ids) if gid in qualified_group_ids]
        qualified_batch = batch[qualified_traj_idxs]
        self.full_qualified_batch = (
            DataProto.concat([self.full_qualified_batch, qualified_batch])
            if self.full_qualified_batch is not None
            else qualified_batch
        )

        logger.info(f"After {self.num_gen_batches=}, {len(self.full_qualified_batch)=}, {self.traj_train_batch_size=}")
        if len(self.full_qualified_batch) >= self.traj_train_batch_size:
            logger.info(
                "Collected enough qualified trajectories. "
                f"Aligning {len(self.full_qualified_batch)=} to {self.traj_train_batch_size=}"
            )
            return self.full_qualified_batch[: self.traj_train_batch_size]
        elif self.num_gen_batches >= self.max_num_gen_batches:
            raise ValueError(
                "Exceeded max number of generation batches. "
                "Consider increasing `max_num_gen_batches` to allow more attempts "
                "or using data easier for the policy."
            )
        else:
            logger.info("Not enough qualified trajectories.")
            return None


def filter_for_non_uniform(metric_values: list[float | int], **kwargs) -> bool:
    """Filter for non-uniform metrics, e.g., a group of samples with non-uniform ``seq_reward``s.

    Args:
        metric_values: List of metric values for samples from the same prompt
        **kwargs: Additional arguments (unused)

    Returns:
        bool: True if prompt should be kept (has mixed rewards), False otherwise
    """
    return len(set(metric_values)) > 1
