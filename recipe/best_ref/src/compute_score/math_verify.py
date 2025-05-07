from verl.utils.reward_score import math_verify


def compute_score_math_verify(solution_str, ground_truth, data_source):
    non_correct_score = 0.0 if data_source.startswith("simplelr_") else -1.0

    verify_result = math_verify.compute_score(solution_str, ground_truth, return_dict=True)
    res = {
        "score": 1.0 if verify_result["acc"] else non_correct_score,
        **verify_result,
    }

    return res
