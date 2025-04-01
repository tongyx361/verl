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
import ray
from concurrent.futures import TimeoutError
import time


def zipngram_tokens(tokens: list[int], ngram_size: int):
    """
    c.f. https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
    """
    return zip(*[tokens[i:] for i in range(ngram_size)])


@ray.remote
def process_single_item(args):
    """Standalone processing function that can be pickled"""
    i, batch_row, non_tensor_batch_row, tokenizer, compute_score, config = args

    prompt_ids = batch_row['prompts']
    prompt_length = prompt_ids.shape[-1]
    attention_mask = batch_row['attention_mask']

    valid_prompt_length = attention_mask[:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = batch_row['responses']
    valid_response_length = attention_mask[prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    # decode
    prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    ground_truth = non_tensor_batch_row['reward_model']['ground_truth']
    data_source = non_tensor_batch_row['data_source']
    extra_info = non_tensor_batch_row.get('extra_info', None)

    max_resp_len = config.data.max_response_length
    exceed_reward = config.reward_model.exceeding_reward

    if exceed_reward is not None and valid_response_length >= max_resp_len:
        final_reward = exceed_reward
        acc = 0
        score = 0
    else:
        score = compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        acc = score == config.reward_model.correct_score

        cos_len_scaling_reward_cfg = config.reward_model.cosine_length_scaling
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
    if config.reward_model.repetition.enabled:
        rep_cfg = config.reward_model.repetition
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
            'score': score,
            'data_source': data_source
        }
    }


@ray.remote
def process_batch(args_batch):
    """Process a batch of items in parallel"""
    results = []
    for args in args_batch:
        i, batch_row, non_tensor_batch_row, tokenizer, compute_score, config = args
        try:
            result = process_single_item((i, batch_row, non_tensor_batch_row, tokenizer, compute_score, config))
            if result is not None:
                results.append(result)
        except Exception as error:
            print(f"Error processing item {i}: {error}")
    return results


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(
            self,
            tokenizer,
            num_examine,
            compute_score=None,
            config: DictConfig = None,
            timeout: int = 300,
            chunk_size: int = 32,  # Renamed to better reflect its purpose
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.config = config
        self.timeout = timeout
        self.chunk_size = chunk_size

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        print("Entering NaiveRewardManager")
        print(f"{len(data.batch)=}")

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        rep_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        accs: list[float] = []

        # Prepare data for parallel processing
        print("Collecting process_args")
        process_args = []
        for i in range(len(data)):
            batch_row = {}
            for k, v in data.batch.items():
                if isinstance(v, torch.Tensor):
                    batch_row[k] = v[i].cpu()
                else:
                    batch_row[k] = v[i]

            non_tensor_batch_row = {}
            for k, v in data.non_tensor_batch.items():
                if isinstance(v, torch.Tensor):
                    non_tensor_batch_row[k] = v[i].cpu()
                elif isinstance(v, dict):
                    non_tensor_batch_row[k] = v[i]
                else:
                    non_tensor_batch_row[k] = v[i] if hasattr(v, '__getitem__') else v

            process_args.append((i, batch_row, non_tensor_batch_row, self.tokenizer, self.compute_score, self.config))
        print(f"Collected {len(process_args)=}.")

        # Process items in parallel using Ray with chunked result collection
        print("Starting parallel processing with Ray")
        results = []

        # Create all futures at once for true parallel processing
        futures = [process_single_item.remote(args) for args in process_args]

        # Collect results in chunks to avoid overwhelming the system
        for i in range(0, len(futures), self.chunk_size):
            chunk_futures = futures[i:i + self.chunk_size]
            try:
                # Get results for this chunk in parallel
                chunk_results = ray.get(chunk_futures, timeout=self.timeout)
                # Filter out None results and extend
                results.extend([r for r in chunk_results if r is not None])
            except TimeoutError:
                print(f"Processing timed out for chunk starting at index {i}")
            except Exception as error:
                print(f"Error processing chunk: {error}")

        print(f"Finished parallel processing with {len(results)=}")

        # Process results
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

        print("Finished NaiveRewardManager")
        print(f"{reward_tensor.shape=}")
        print(f"{rep_reward_tensor.shape=}")
        print(f"Accuracy: {sum(accs) / len(accs)}")
        return {
            "reward_tensor": reward_tensor,
            "rep_reward_tensor": rep_reward_tensor,
            "accs": accs,
        }
