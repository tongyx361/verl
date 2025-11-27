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

import warnings
from enum import Enum, auto

from omegaconf import DictConfig

from verl.single_controller.base import Worker
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    TODO: Decouple "capability" and "usage".
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()
    Env = auto()
    Validator = auto()

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
            Role.Env: "env",
            Role.Validator: "validator",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
            "env": cls.Env,
            "validator": cls.Validator,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


def need_reference_policy(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """Given a role worker mapping, do we need ref policy."""
    return Role.RefPolicy in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping


def need_reward_model(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """Given a role worker mapping, do we need reward model."""
    return Role.RewardModel in role_worker_mapping


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False
