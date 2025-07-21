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
class NPUProfileOptions:
    r"""
    Configuration for NPU profiler options.

    Args:
        save_path (str, optional): Storage path of collected data (default: ``"./profiler_data"``)
        level (str, optional): Collection level, optional values:
            ``"level_none"``, ``"level0"``, ``"level1"``, ``"level2"`` (default: ``"level1"``)
        with_memory (bool, optional): Whether to enable memory analysis (default: ``False``)
        record_shapes (bool, optional): Whether to record tensor shape (default: ``False``)
        with_npu (bool, optional): Whether to record Device-side performance data (default: ``True``)
        with_cpu (bool, optional): Whether to record Host-side performance data (default: ``True``)
        with_module (bool, optional): Whether to record Python call stack information (default: ``False``)
        with_stack (bool, optional): Whether to record operator call stack information (default: ``False``)
        analysis (bool, optional): Whether to automatically parse the data (default: ``True``)
    """

    save_path: str = "./profiler_data"
    level: str = "level1"
    with_memory: bool = False
    record_shapes: bool = False
    with_npu: bool = True
    with_cpu: bool = True
    with_module: bool = False
    with_stack: bool = False
    analysis: bool = True


@dataclass
class NPUProfileConfig:
    r"""
    Configuration for NPU profiler.

    Args:
        options (NPUProfileOptions, optional): NPU profiler options configuration (default: ``NPUProfileOptions()``)
    """

    options: NPUProfileOptions = field(default_factory=NPUProfileOptions)
