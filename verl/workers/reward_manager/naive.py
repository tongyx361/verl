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


def get_repetition_penalty(ngram_size: int, max_penalty: float, resp_text: str) -> float:
    """
    c.f. https://github.com/eddycmu/demystify-long-cot/blob/7cec7017d52444798b39496efa8759aaeafdd125/openrlhf/openrlhf/reward/repetition.py#L56-L70
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if max_penalty == 0:
        return 0

    ngrams = set()
    total = 0
    for ng in zipngram(resp_text, ngram_size):
        ngrams.add(ng)
        total += 1

    scaling = 1 - len(ngrams) / total
    return scaling * max_penalty


def zipngram(text: str, ngram_size: int):
    """
    c.f. https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
    """
    words = text.lower().split()
    return zip(*[words[i:] for i in range(ngram_size)])


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, config: DictConfig) -> None:
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

        already_print_data_sources = {}

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
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                cos_len_scaling_reward_cfg = self.config.reward_model.cosine_length_scaling
                if not cos_len_scaling_reward_cfg.enabled:
                    final_reward = score
                else:
                    if score == self.config.reward_model.correct_score:
                        min_value = cos_len_scaling_reward_cfg.correct_reward.min
                        max_value = cos_len_scaling_reward_cfg.correct_reward.max
                    else:
                        # Yes, they are swapped. This is required for the cosine formula below
                        # to work with negative numbers.
                        min_value = cos_len_scaling_reward_cfg.wrong_reward.min
                        max_value = cos_len_scaling_reward_cfg.wrong_reward.max

                    progress = valid_response_length / max_resp_len
                    cosine = math.cos(progress * math.pi)
                    cos_len_scaling_reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
                    final_reward = cos_len_scaling_reward

            reward_tensor[i, valid_response_length - 1] = final_reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
