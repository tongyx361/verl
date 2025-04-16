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
# from . import gsm8k, math, prime_math, prime_code
import logging
from concurrent.futures import TimeoutError

from pebble.concurrent import process

TIMEOUT = 5

logger = logging.getLogger(__name__)


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source.startswith("MATH##") or data_source.startswith("aime"):
        from . import math_verify
        verify_w_timeout = process(timeout=TIMEOUT)(math_verify.compute_score)
        verify_result = verify_w_timeout(solution_str, ground_truth, return_dict=True).result()
        fail_score = -1.0
        res = {"score": fail_score, "acc": False, "pred": ""}
        try:
            res = {
                "score": 1.0 if verify_result["acc"] else fail_score,
                "acc": verify_result["acc"],
                "pred": verify_result["pred"],
            }
        except TimeoutError as e:
            logger.warning(f"Timeout for {data_source=}\n{solution_str=}\n{ground_truth=}\n{extra_info=}\n{e}")
        except Exception as e:
            logger.warning(f"Error for {data_source=}\n{solution_str=}\n{ground_truth=}\n{extra_info=}\n{e}")
    elif data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:
        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == 'math_dapo' or data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
