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

import json
import logging
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


@dataclass
class UpdatingState:
    gen_round_cnt: int = 0
    gen_prompt_cnt: int = 0
    gen_traj_cnt: int = 0
    batch: Optional[DataProto] = None
    timing_raw: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    metrics: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    # Leaving for the next generation stage
    qualified_rate: float = 0.0

    def reset(self):
        self.gen_round_cnt = 0
        self.gen_prompt_cnt = 0
        self.gen_traj_cnt = 0

        self.batch = None
        self.timing_raw = defaultdict(float)
        self.metrics = defaultdict(float)


@dataclass
class GenerationState:
    prompt_batch: Optional[DataProto] = None
    sampling_params: dict[str, Any] = field(default_factory=dict)

    def reset(self):
        self.prompt_batch = None


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # load extra info
        extra_info_path = os.path.join(global_step_folder, "extra_info.json")
        if os.path.exists(extra_info_path):
            with open(extra_info_path) as f:
                extra_info = json.load(f)
            if "qualified_rate" in extra_info:
                self.updating_state.qualified_rate = extra_info["qualified_rate"]
            if "sampling_params" in extra_info:
                self.generation_state.sampling_params = extra_info["sampling_params"]

    def _save_checkpoint(self):
        """Overridden method from parent class to add state tracking to checkpoint"""

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        # latest checkpointed iteration tracker (for atomic usage)
        extra_info = {
            # "latest_checkpointed_iteration": self.global_steps,
            "qualified_rate": self.updating_state.qualified_rate,
            "sampling_params": self.generation_state.sampling_params,
        }
        with open(os.path.join(local_global_step_folder, "extra_info.json"), "w") as f:
            json.dump(extra_info, f)

    def fit(self) -> None:
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.updating_state = UpdatingState()
        self.generation_state = GenerationState(
            sampling_params={"max_tokens": self.config.actor_rollout_ref.rollout.response_length}
        )
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_timing: dict[str, float] = defaultdict(float)
            with _timer("testing", val_timing):
                val_metrics = self._validate()
            val_metrics["timing_s/testing"] = val_timing["testing"]
            pprint(f"Initial validation metrics: {val_metrics}")
            tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                data_state = self.train_dataloader.state_dict()
                logger.debug("%s", f"{data_state=}")

                new_prompt_batch = DataProto.from_single_dict(batch_dict)

                # Initalize generation-level state
                self.generation_state.prompt_batch = (
                    DataProto.concat([self.generation_state.prompt_batch, new_prompt_batch])
                    if self.generation_state.prompt_batch is not None
                    else new_prompt_batch
                )

                prompt_bsz = self.config.data.train_batch_size
                estim_num_prompt_needed = (
                    -int(-1 / self.updating_state.qualified_rate) * prompt_bsz
                    if self.updating_state.qualified_rate > 0
                    else 0
                )
                estim_num_remaining_prompt_needed = estim_num_prompt_needed - self.updating_state.gen_prompt_cnt
                logger.info(
                    "[step:%d/gen_round_cnt:%d] # of Prompts to Gen. (%d) <= # of Remaining Prompts Needed (%d)?",
                    self.global_steps,
                    self.updating_state.gen_round_cnt,
                    len(self.generation_state.prompt_batch),
                    int(estim_num_remaining_prompt_needed * self.config.algorithm.filter_groups.oversampling_factor),
                )
                if len(self.generation_state.prompt_batch) <= int(
                    estim_num_remaining_prompt_needed * self.config.algorithm.filter_groups.oversampling_factor
                ):
                    logger.info("Keep loading...")
                    continue

                self.updating_state.gen_prompt_cnt += len(self.generation_state.prompt_batch)
                self.updating_state.gen_round_cnt += 1

                # pop those keys for generation
                batch_keys = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys = ["raw_prompt_ids"]
                if "multi_modal_inputs" in self.generation_state.prompt_batch.non_tensor_batch.keys():
                    batch_keys.extend(["multi_modal_data", "multi_modal_inputs"])
                gen_batch = self.generation_state.prompt_batch.pop(
                    batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys
                )
                if gen_batch.meta_info.get("sampling_params") is None:
                    gen_batch.meta_info["sampling_params"] = {}
                sampling_params = gen_batch.meta_info["sampling_params"]
                if sampling_params.get("max_tokens") is None:
                    sampling_params["max_tokens"] = self.config.actor_rollout_ref.rollout.response_length

                is_last_step = self.global_steps >= self.total_training_steps

                gen_batch.meta_info["sampling_params"] = self.generation_state.sampling_params
                with _timer("step", self.updating_state.timing_raw):
                    # generate a batch
                    with _timer("gen", self.updating_state.timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", self.updating_state.timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            self.generation_state.prompt_batch = self.generation_state.prompt_batch.union(
                                gen_baseline_output
                            )
                            reward_baseline_tensor = self.reward_fn(self.generation_state.prompt_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            self.generation_state.prompt_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            self.generation_state.prompt_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    self.generation_state.prompt_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(self.generation_state.prompt_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = self.generation_state.prompt_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    self.generation_state.reset()
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", self.updating_state.timing_raw):
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

                        logger.info("%s", f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                    if not self.config.algorithm.filter_groups.enable:
                        self.updating_state.batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        # KL rewards are not supported yet
                        if metric_name == "seq_reward":
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
                        self.updating_state.batch = (
                            DataProto.concat([self.updating_state.batch, new_batch])
                            if self.updating_state.batch is not None
                            else new_batch
                        )
                        if self.config.data.dynamic_max_resp_len.enable:
                            response_mask = compute_response_mask(self.updating_state.batch)
                            resp_lens = response_mask.sum(dim=-1)
                            gen_max_resp_len = resp_lens.max().item()
                            max_non_truncated_resp_len = resp_lens[resp_lens < gen_max_resp_len].max().item()
                            logger.info(
                                "[step:%d/gen_round_cnt:%d] Max non-truncated response length: %d"
                                " / Max response length: %d",
                                self.global_steps,
                                self.updating_state.gen_round_cnt,
                                max_non_truncated_resp_len,
                                gen_max_resp_len,
                            )
                            max_tokens = int(
                                max_non_truncated_resp_len
                                * (1 + self.config.data.dynamic_max_resp_len.extending_tolerance)
                            )
                            self.generation_state.sampling_params["max_tokens"] = max_tokens

                        # Ceiling
                        self.updating_state.gen_traj_cnt += len(gen_batch_output)
                        self.updating_state.qualified_rate = (
                            len(self.updating_state.batch) / self.updating_state.gen_traj_cnt
                        )
                        traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
                        if len(self.updating_state.batch) < traj_bsz:
                            logger.info(
                                "[step:%d/gen_round_cnt:%d] # of Traj. (%d) < Traj. Batch Size (%d)."
                                " Keep generating...",
                                self.global_steps,
                                self.updating_state.gen_round_cnt,
                                len(self.updating_state.batch),
                                traj_bsz,
                            )
                            continue
                        else:  # Align the batch
                            self.updating_state.batch = self.updating_state.batch[:traj_bsz]

                    assert self.updating_state.batch is not None
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(self.updating_state.batch, metrics=self.updating_state.metrics)

                    # compute global_valid tokens
                    self.updating_state.batch.meta_info["global_token_num"] = torch.sum(
                        self.updating_state.batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    self.updating_state.batch.batch["response_mask"] = compute_response_mask(self.updating_state.batch)

                    # recompute old_log_probs
                    with _timer("old_log_prob", self.updating_state.timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(self.updating_state.batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = self.updating_state.batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        self.updating_state.metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        self.updating_state.batch = self.updating_state.batch.union(old_log_prob)

                    token_level_rewards = self.updating_state.batch.batch["token_level_scores"]
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", self.updating_state.timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(self.updating_state.batch)
                            self.updating_state.batch = self.updating_state.batch.union(ref_log_prob)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            token_level_rewards, kl_metrics = apply_kl_penalty(
                                self.updating_state.batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            # TODO: This will be cleared if we use multiple genenration batches
                            self.updating_state.metrics.update(kl_metrics)
                    self.updating_state.batch.batch["token_level_rewards"] = token_level_rewards

                    # compute values
                    if self.use_critic:
                        with _timer("values", self.updating_state.timing_raw):
                            values = self.critic_wg.compute_values(self.updating_state.batch)
                            self.updating_state.batch = self.updating_state.batch.union(values)

                    with _timer("adv", self.updating_state.timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        self.updating_state.batch = compute_advantage(
                            self.updating_state.batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", self.updating_state.timing_raw):
                            critic_output = self.critic_wg.update_critic(self.updating_state.batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        self.updating_state.metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", self.updating_state.timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(self.updating_state.batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        self.updating_state.metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with _timer("testing", self.updating_state.timing_raw):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    self.updating_state.metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", self.updating_state.timing_raw):
                        self._save_checkpoint()

                # collect metrics
                assert self.updating_state.batch is not None
                self.updating_state.metrics.update(
                    compute_data_metrics(batch=self.updating_state.batch, use_critic=self.use_critic)
                )
                self.updating_state.metrics.update(
                    compute_timing_metrics(batch=self.updating_state.batch, timing_raw=self.updating_state.timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                self.updating_state.metrics.update(
                    compute_throughout_metrics(
                        batch=self.updating_state.batch, timing_raw=self.updating_state.timing_raw, n_gpus=n_gpus
                    )
                )

                self.updating_state.metrics.update(
                    {
                        "train/gen_round_cnt": self.updating_state.gen_round_cnt,
                        "train/gen_prompt_cnt": self.updating_state.gen_prompt_cnt,
                        "train/gen_traj_cnt": self.updating_state.gen_traj_cnt,
                        "train/qualified_rate": self.updating_state.qualified_rate,
                    }
                )

                # TODO: make a canonical logger that supports various backend
                tracker.log(data=self.updating_state.metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.updating_state.reset()
