#!/usr/bin/env bash
set -euxo pipefail

TEST=${TEST:-"0"}
NNODES=${NNODES:-16}
ACTOR_LR=${ACTOR_LR:-"1e-6"}

project_name='best-ref'

n_procs_per_node=8
num_procs=$((NNODES * n_procs_per_node))

repeat_factor=1

exp_name="dapo-dmlr-qwen2p5-32b-lr${ACTOR_LR}-${num_procs}gpus"

adv_estimator=grpo
# Clip epsilons
clip_eps_down=0.2
clip_eps_up=0.28
# KL regularization
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0
# Entropy regularization
entropy_coeff=0
# Other tricks
weight_decay=0.1
grad_clip=1.0

temperature=1.0
top_p=1
val_top_p=0.7

sp_size=8
fsdp_size=64

gen_tp=2

reward_manager=dapo
enable_overlong_buf=False

enable_filter_groups=True
filter_metric=acc
max_num_gen_batches=10

if [ "${TEST}" != "1" ]; then
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 20))
    train_batch_size=512
    num_updates_per_batch=16
    n_trajs_per_prompt=16
    ppo_mini_batch_size=$((train_batch_size / num_updates_per_batch))
    val_n=32
else
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 2))
    n_trajs_per_prompt=2
    ppo_mini_batch_size=$((num_procs / n_trajs_per_prompt))
    num_updates_per_batch=2
    train_batch_size=$((ppo_mini_batch_size * num_updates_per_batch))
    exp_name="${exp_name}-test"
    val_n=1
fi



# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-${VERL_HOME:-"${HOME}/verl"}}
MODEL_HOME=${MODEL_HOME:-"${RAY_DATA_HOME}/models"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_HOME}/Qwen2.5-32B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-unique-clean-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024-clean.parquet"}


ppo_epochs=1
total_epochs=100

lr_warmup_steps=10

test_freq=5
save_freq=5

offload=False

use_dynamic_bsz=True
actor_ppo_max_token_len=$((512 * num_procs))
infer_ppo_max_token_len=$((2048 * num_procs))

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.dapo.src.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.repeat.factor=${repeat_factor} \
    +data=cot_boxed_sfx_template \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    reward_model.reward_manager=${reward_manager} \
    reward_model.overlong_buffer.enable=${enable_overlong_buf} \
    reward_model.overlong_buffer.log=True \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.rollout.n=${n_trajs_per_prompt} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_eps_down} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_eps_up} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name="${exp_name}-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)" \
    trainer.n_gpus_per_node=${n_procs_per_node} \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    data.return_raw_chat=True