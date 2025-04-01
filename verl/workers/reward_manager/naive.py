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
import multiprocessing as mp
from functools import partial
import os


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


def zipngram_tokens(tokens: list[int], ngram_size: int):
    """
    c.f. https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
    """
    return zip(*[tokens[i:] for i in range(ngram_size)])


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(
            self,
            tokenizer,
            num_examine,
            compute_score=None,
            config: DictConfig = None,
            num_processes: int = int(os.cpu_count() * 0.8),
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.config = config
        self.pool = mp.Pool(processes=num_processes)  # Create process pool during initialization

    def process_single_item(self, i, data_item, max_resp_len, exceed_reward):
        """Process a single data item and return its rewards"""
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

        if exceed_reward is not None and valid_response_length >= max_resp_len:
            final_reward = exceed_reward
            acc = None
        else:
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            acc = score == self.config.reward_model.correct_score

            cos_len_scaling_reward_cfg = self.config.reward_model.cosine_length_scaling
            if not cos_len_scaling_reward_cfg.enabled:
                final_reward = score
            else:
                if acc:
                    min_value = cos_len_scaling_reward_cfg.correct_reward.min
                    max_value = cos_len_scaling_reward_cfg.correct_reward.max
                else:
                    min_value = cos_len_scaling_reward_cfg.wrong_reward.min
                    max_value = cos_len_scaling_reward_cfg.wrong_reward.max

                progress = valid_response_length / max_resp_len
                cosine = math.cos(progress * math.pi)
                final_reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)

        # Calculate repetition penalty if enabled
        rep_rewards = None
        if self.config.reward_model.repetition.enabled:
            rep_cfg = self.config.reward_model.repetition
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
                    for j in range(start_idx, start_idx + rep_cfg.ngram_size):
                        tok_rewards[j] = rep_cfg.reward
                curr_end_idx = start_idx + rep_cfg.ngram_size
            rep_rewards = tok_rewards

        return {
            'index': i,
            'final_reward': final_reward,
            'valid_response_length': valid_response_length,
            'acc': acc,
            'rep_rewards': rep_rewards,
            'debug_info': {
                'prompt': prompt_str,
                'response': response_str,
                'ground_truth': ground_truth,
                'score': score if 'score' in locals() else None,
                'data_source': data_source
            }
        }

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        rep_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        accs: list[float] = []

        # Prepare parameters for parallel processing
        max_resp_len = self.config.data.max_response_length
        exceed_reward = self.config.reward_model.exceeding_reward

        # Process items in parallel
        process_fn = partial(self.process_single_item, max_resp_len=max_resp_len, exceed_reward=exceed_reward)
        results = self.pool.starmap(process_fn, [(i, data[i]) for i in range(len(data))])

        # Collect results
        already_print_data_sources = {}
        for result in results:
            i = result['index']
            reward_tensor[i, result['valid_response_length'] - 1] = result['final_reward']

            if result['acc'] is not None:
                accs.append(result['acc'])

            if result['rep_rewards'] is not None:
                rep_reward_tensor[i] += result['rep_rewards']

            # Handle debug printing
            debug = result['debug_info']
            data_source = debug['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", debug['prompt'])
                print("[response]", debug['response'])
                print("[ground_truth]", debug['ground_truth'])
                print("[score]", debug['score'])
                print("[rep_reward]", rep_reward_tensor[i].sum().item())

        return {
            "reward_tensor": reward_tensor,
            "rep_reward_tensor": rep_reward_tensor,
            "accs": accs,
        }

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()
