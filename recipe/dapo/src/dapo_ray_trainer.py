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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import logging
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


@dataclass
class UpdatingState:
    gen_round_cnt: int = 0
    gen_prompt_cnt: int = 0
    gen_traj_cnt: int = 0
    qualified_rate: float = 0.0
    batch: Optional[DataProto] = None
    timing_raw: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    metrics: dict[str, float] = field(default_factory=lambda: defaultdict(float))


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self) -> None:
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        updating_state = UpdatingState()
        # Generation-level state
        prompt_batch = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                data_state = self.train_dataloader.state_dict()
                print(f"{data_state=}")

                prompt_batch = (
                    prompt_batch.concat(DataProto.from_single_dict(batch_dict))
                    if prompt_batch is not None
                    else DataProto.from_single_dict(batch_dict)
                )

                prompt_bsz = self.config.data.train_batch_size
                estim_num_prompt_needed = (
                    -int(-1 / updating_state.qualified_rate) * prompt_bsz if updating_state.qualified_rate > 0 else 0
                )
                updating_state.gen_prompt_cnt += len(prompt_batch)
                # Ceiling + at least one batch more for tolerance
                if updating_state.gen_prompt_cnt <= estim_num_prompt_needed:
                    print(f"{updating_state.gen_prompt_cnt=} <= {estim_num_prompt_needed=}. Keep loading...")
                    continue

                updating_state.gen_round_cnt += 1

                # pop those keys for generation
                if "multi_modal_inputs" in prompt_batch.non_tensor_batch.keys():
                    gen_batch = prompt_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                else:
                    gen_batch = prompt_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", updating_state.timing_raw):
                    # generate a batch
                    with _timer("gen", updating_state.timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", updating_state.timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            prompt_batch = prompt_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(prompt_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            prompt_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            prompt_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    prompt_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(prompt_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = prompt_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    prompt_batch = None
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", updating_state.timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            # TODO: This will be cleared if we use multiple genenration batches
                            updating_state.metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        updating_state.batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        updating_state.batch = (
                            DataProto.concat([updating_state.batch, new_batch])
                            if updating_state.batch is not None
                            else new_batch
                        )

                        # Ceiling
                        updating_state.gen_traj_cnt += len(gen_batch_output)
                        updating_state.qualified_rate = len(updating_state.batch) / updating_state.gen_traj_cnt
                        traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
                        if len(updating_state.batch) < traj_bsz:
                            print(f"{len(updating_state.batch)=} < {traj_bsz=}. Keep generating...")
                            continue
                        else:  # Align the batch
                            updating_state.batch = updating_state.batch[:traj_bsz]

                    assert updating_state.batch is not None
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(updating_state.batch, metrics=updating_state.metrics)

                    # compute global_valid tokens
                    updating_state.batch.meta_info["global_token_num"] = torch.sum(
                        updating_state.batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", updating_state.timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(updating_state.batch)
                        updating_state.batch = updating_state.batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", updating_state.timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(updating_state.batch)
                            updating_state.batch = updating_state.batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", updating_state.timing_raw):
                            values = self.critic_wg.compute_values(updating_state.batch)
                            updating_state.batch = updating_state.batch.union(values)

                    with _timer("adv", updating_state.timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        updating_state.batch = compute_advantage(
                            updating_state.batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", updating_state.timing_raw):
                            critic_output = self.critic_wg.update_critic(updating_state.batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        updating_state.metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", updating_state.timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(updating_state.batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        updating_state.metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", updating_state.timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        updating_state.metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", updating_state.timing_raw):
                            self._save_checkpoint()

                # collect metrics
                assert updating_state.batch is not None
                updating_state.metrics.update(
                    compute_data_metrics(batch=updating_state.batch, use_critic=self.use_critic)
                )
                updating_state.metrics.update(
                    compute_timing_metrics(batch=updating_state.batch, timing_raw=updating_state.timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                updating_state.metrics.update(
                    compute_throughout_metrics(
                        batch=updating_state.batch, timing_raw=updating_state.timing_raw, n_gpus=n_gpus
                    )
                )

                updating_state.metrics.update(
                    {
                        "train/gen_round_cnt": updating_state.gen_round_cnt,
                        "train/gen_prompt_cnt": updating_state.gen_prompt_cnt,
                        "train/gen_traj_cnt": updating_state.gen_traj_cnt,
                        "train/qualified_rate": updating_state.qualified_rate,
                    }
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=updating_state.metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                updating_state = UpdatingState()
