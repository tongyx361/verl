#!/usr/bin/env bash
set -xueo pipefail
# c.f. https://github.com/hkust-nlp/simpleRL-reason?tab=readme-ov-file#training
# bash train_grpo_math_tune_ray.sh --model_name Qwen-2.5-7B \
# --max_response_length 8192  --train_batch_size 1024 --rollout_n 8 \
# --kl_loss_coef 0.0001 --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.75 --rollout_tp 2 --save_freq 5
# TODO