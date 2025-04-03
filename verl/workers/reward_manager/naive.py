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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from omegaconf import DictConfig
import math
import time
from collections import defaultdict


def zipngram_tokens(tokens: list[int], ngram_size: int):
    """
    c.f. https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
    """
    return zip(*[tokens[i:] for i in range(ngram_size)])


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, config: DictConfig = None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.config = config

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        rep_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        accs: list[float] = []

        already_print_data_sources = defaultdict(int)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            max_resp_len = self.config.data.max_response_length
            exceed_reward = self.config.reward_model.exceeding_reward
            if exceed_reward is not None and valid_response_length >= max_resp_len:
                final_reward = exceed_reward
            else:
                # score_start = time.time()
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                # print(f"{i=}: score time: {time.time() - score_start}")
                acc = score == self.config.reward_model.correct_score
                accs.append(acc)

                cos_len_scaling_reward_cfg = self.config.reward_model.cosine_length_scaling
                if not cos_len_scaling_reward_cfg.enabled:
                    final_reward = score
                else:
                    if acc:
                        min_value = cos_len_scaling_reward_cfg.correct_reward.min
                        max_value = cos_len_scaling_reward_cfg.correct_reward.max
                    else:
                        # Yes, they are swapped. This is required for the cosine formula below
                        # to work with negative numbers.
                        min_value = cos_len_scaling_reward_cfg.wrong_reward.max
                        max_value = cos_len_scaling_reward_cfg.wrong_reward.min

                    progress = valid_response_length / max_resp_len
                    cosine = math.cos(progress * math.pi)
                    cos_len_scaling_reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
                    final_reward = cos_len_scaling_reward

            reward_tensor[i, valid_response_length - 1] = final_reward

            rep_cfg = self.config.reward_model.repetition
            if rep_cfg.enabled:
                # rep_start = time.time()
                resp_tok_ids = response_ids[:int(valid_response_length)]
                repeated = []
                ngrams = set()
                for start_idx, ng in enumerate(zipngram_tokens(resp_tok_ids, rep_cfg.ngram_size)):
                    if ng in ngrams:
                        repeated.append(start_idx)
                    ngrams.add(ng)

                tok_rewards = torch.zeros(response_ids.shape[-1], dtype=torch.float32)
                curr_end_idx = -1
                for start_idx in repeated:
                    if not rep_cfg.only_start or start_idx > curr_end_idx:
                        for i in range(start_idx, start_idx + rep_cfg.ngram_size):
                            tok_rewards[i] = rep_cfg.reward

                    curr_end_idx = start_idx + rep_cfg.ngram_size

                rep_reward_tensor[i] += tok_rewards
                # print(f"{i=}: rep time: {time.time() - rep_start}")

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[data_source]", data_source)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[final_reward]", final_reward)
                print("[rep_reward]", rep_reward_tensor[i].sum().item())

        return {
            "reward_tensor": reward_tensor,
            "rep_reward_tensor": rep_reward_tensor,
            "accs": accs,
        }
