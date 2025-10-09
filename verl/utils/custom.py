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

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.WARN))


@dataclass
class CustomFunctionConfig(BaseConfig):
    """
    Configuration for custom function.

    Attributes:
        path (str): File path / Full name of the module containing the custom function, e.g.,
            - ``"path/to/custom_module.py"``
            - ``"verl.utils.custom"``
        name (str): Name of the function to import from the module.
        kwargs (dict[str, Any]): Additional keyword arguments to pass to the function.
    """

    path: str | None = None
    name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


def get_custom_fn(config: CustomFunctionConfig) -> Callable | None:
    """Load and return a custom function from external file.

    Dynamically imports a function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (CustomFunctionConfig): Configuration object containing function path, name, and kwargs.

    Returns:
        Callable | None: Wrapped function with merged kwargs, or ``None``
                         if no custom function is configured.

    Raises:
        FileNotFoundError: If the specified function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """
    module_path = config.path
    fn_name = config.name
    kwargs = config.kwargs

    if not module_path:
        return None
    assert fn_name is not None, "`fn_name` must be specified."

    logger.info(f"Loading custom function with `{fn_name=}` from `{module_path=}`")
    if os.path.exists(module_path):
        # TODO: What if there are multiple custom modules?
        module = sys.modules.get("custom_module", None)
        if module is None:
            spec = importlib.util.spec_from_file_location("custom_module", module_path)
            assert spec is not None, f"Failed to create spec from `{module_path=}`."
            module = importlib.util.module_from_spec(spec)
            try:
                sys.modules["custom_module"] = module
                assert spec.loader is not None
                spec.loader.exec_module(module)
            except Exception as e:
                raise RuntimeError(f"Error loading module from `{module_path=}`: {e}") from e
    else:
        module = importlib.import_module(module_path)
    assert module, f"Failed to import module from `{module_path=}`."

    raw_fn = getattr(module, fn_name)
    fn = partial(raw_fn, **kwargs)

    return fn
