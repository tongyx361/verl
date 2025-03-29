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
A Ray logger will receive logging info from different processes.
"""
import logging
import numbers
from typing import Dict


def concat_dict_to_str(dict: Dict, step):
    output = [f'step:{step}']
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f'{k}:{v:.3f}')
    output_str = ' - '.join(output)
    return output_str


class LocalLogger:

    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
        self.print_to_console = print_to_console
        if print_to_console:
            logging.warning('Using LocalLogger is deprecated. The constructor API will change')

        # Set up basic logging configuration if not already configured
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format=
                "[%(levelname)s] [%(asctime)s.%(msecs)d] [pid %(process)d] [%(pathname)s:%(lineno)d:%(funcName)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    def log(self, data, step):
        if self.print_to_console:
            if isinstance(data, dict):
                self.logger.info(concat_dict_to_str(data, step=step))
            else:
                self.logger.info(f"[{step=}] {data}")
