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
class CustomClsConfig:
    r"""
    Configuration for custom dataset class.

    Args:
        path (str | None, optional): The path to the file containing your customized dataset class.
            If not specified, pre-implemented dataset will be used (default: ``None``)
        name (str | None, optional): The name of the dataset class within the specified file
            (default: ``None``)
    """

    path: str | None = None
    name: str | None = None


@dataclass
class SamplerConfig:
    r"""
    Configuration for data sampler settings.

    Args:
        class_path (str | None, optional): The path to the module containing a curriculum class which
            implements the AbstractSampler interface (default: ``None``)
        class_name (str | None, optional): The name of the curriculum class like ``"MySampler"``
            (default: ``None``)
    """

    class_path: str | None = None
    class_name: str | None = None


@dataclass
class DatagenConfig:
    r"""
    Configuration for data generation.

    Args:
        path (str | None, optional): The path to the file containing your customized data generation
            class. E.g. ``"pkg://verl.experimental.dynamic_dataset.dynamicgen_dataset"`` (default: ``None``)
        name (str | None, optional): The class name of the data generation class within the specified
            file. E.g. ``"MockDataGenerator"`` (default: ``None``)
    """

    path: str | None = None
    name: str | None = None


@dataclass
class DataConfig:
    r"""
    Configuration for data loading and processing.

    Args:
        tokenizer (str | None, optional): Tokenizer class or path. If null, it will be inferred from
            the model (default: ``None``)
        use_shm (bool, optional): Whether to use shared memory for data loading (default: ``False``)
        train_files (list[str], optional): Training set parquet. The program will read all files into memory,
            so it can't be too large (< 100GB). The path
            can be either a local path or an HDFS path (default: ``"~/data/rlhf/gsm8k/train.parquet"``)
        val_files (list[str], optional): Validation parquet. Can be a list or a single file
            (default: ``"~/data/rlhf/gsm8k/test.parquet"``)
        prompt_key (str, optional): The field in the dataset where the prompt is located
            (default: ``"prompt"``)
        reward_fn_key (str, optional): The field used to select the reward function (if using different
            ones per example) (default: ``"data_source"``)
        max_prompt_length (int, optional): Maximum prompt length. All prompts will be left-padded to
            this length. An error will be reported if the length is too long (default: ``512``)
        max_response_length (int, optional): Maximum response length. Rollout in RL algorithms (e.g.
            PPO) generates up to this length (default: ``512``)
        train_batch_size (int, optional): Batch size sampled for one training iteration of different
            RL algorithms (default: ``1024``)
        val_batch_size (int | None, optional): Batch size used during validation. Can be null
            (default: ``None``)
        return_raw_input_ids (bool, optional): Whether to return the original input_ids without adding
            chat template. This is used when the reward model's chat template differs from the policy
            (default: ``False``)
        return_raw_chat (bool, optional): Whether to return the original chat (prompt) without applying
            chat template (default: ``False``)
        return_full_prompt (bool, optional): Whether to return the full prompt with chat template
            (default: ``False``)
        shuffle (bool, optional): Whether to shuffle the data in the dataloader (default: ``True``)
        dataloader_num_workers (int, optional): Number of dataloader workers (default: ``8``)
        validation_shuffle (bool, optional): Whether to shuffle the validation set (default: ``False``)
        filter_overlong_prompts (bool, optional): Whether to filter overlong prompts (default: ``False``)
        filter_overlong_prompts_workers (int, optional): Number of workers for filtering overlong
            prompts. For large-scale datasets, filtering can be time-consuming. Use multiprocessing to
            speed up (default: ``1``)
        truncation (str, optional): Truncate the input_ids or prompt if they exceed max_prompt_length.
            Options: ``"error"``, ``"left"``, ``"right"``, ``"middle"`` (default: ``"error"``)
        image_key (str, optional): The field in the multi-modal dataset where the image is located
            (default: ``"images"``)
        video_key (str, optional): The field in the multi-modal dataset where the video is located
            (default: ``"videos"``)
        trust_remote_code (bool, optional): If the remote tokenizer has a Python file, this flag
            determines whether to allow using it (default: ``False``)
        custom_cls (CustomClsConfig, optional): Optional: specify a custom dataset class path and name
            if overriding default loading behavior (default: ``CustomClsConfig()``)
        return_multi_modal_inputs (bool, optional): Whether to return multi-modal inputs in the dataset.
            Set to False if rollout generates new multi-modal inputs (default: ``True``)
        sampler (SamplerConfig, optional): Settings related to data sampler (default: ``SamplerConfig()``)
        datagen (DatagenConfig, optional): Data generation configuration for augmenting the dataset
            (default: ``DatagenConfig()``)
    """

    tokenizer: str | None = None
    use_shm: bool = False
    train_files: list[str] = field(default_factory=lambda: ["~/data/rlhf/gsm8k/train.parquet"])
    val_files: list[str] = field(default_factory=lambda: ["~/data/rlhf/gsm8k/test.parquet"])
    prompt_key: str = "prompt"
    reward_fn_key: str = "data_source"
    max_prompt_length: int = 512
    max_response_length: int = 512
    train_batch_size: int = 1024
    val_batch_size: int | None = None
    return_raw_input_ids: bool = False
    return_raw_chat: bool = False
    return_full_prompt: bool = False
    shuffle: bool = True
    dataloader_num_workers: int = 8
    validation_shuffle: bool = False
    filter_overlong_prompts: bool = False
    filter_overlong_prompts_workers: int = 1
    truncation: str = "error"
    image_key: str = "images"
    video_key: str = "videos"
    trust_remote_code: bool = False
    custom_cls: CustomClsConfig = field(default_factory=CustomClsConfig)
    return_multi_modal_inputs: bool = True
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    datagen: DatagenConfig = field(default_factory=DatagenConfig)
