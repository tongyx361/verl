#!/usr/bin/env bash
set -euxo pipefail

# Config
TEST=${TEST:-"0"}
# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-4}

sp_size=8
n_procs_per_node=8
num_procs=$((NNODES * n_procs_per_node))

project_name='demystify-long-cot'
exp_name="qwen2.5-32b-dlc-cosine-reward-${num_procs}gpus"

enable_cosine_length_scaling_reward=True
correct_cosine_max_reward=2.0
correct_cosine_min_reward=1.0
wrong_cosine_max_reward=0.0
wrong_cosine_min_reward=-10.0

enable_repetition_penalty=True
repetition_ngram=40
repetition_reward=-0.05
repetition_gamma=0.99

exceeding_reward=-10.0

adv_estimator=gae
gae_lambda=1.0
gae_gamma=1.0

clip_eps_down=0.2
clip_eps_up=0.2
value_clip_eps=0.2

kl_coef=0.01
use_kl_loss=False
kl_loss_coef=0
entropy_coeff=0
weight_decay=0.0
grad_clip=1.0

temperature=1.0
top_p=1

fsdp_size=-1
gen_tp=8
gen_dp_size=$((num_procs / gen_tp))

if [ "${TEST}" != "1" ]; then
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 14))
    n_trajs_per_prompt=16
    ppo_mini_batch_size=32
    num_updates_per_batch=16
    train_batch_size=$((ppo_mini_batch_size * num_updates_per_batch))
    val_n=16
    resume_mode=auto
else
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 2))
    n_trajs_per_prompt=2
    ppo_mini_batch_size=$((num_procs / n_trajs_per_prompt))
    num_updates_per_batch=2
    train_batch_size=$((ppo_mini_batch_size * num_updates_per_batch))
    if [ $train_batch_size -lt $gen_dp_size ]; then
        train_batch_size=$gen_dp_size
    fi
    exp_name="${exp_name}-test"
    val_n=1
    resume_mode=disable
fi

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_HOME=${MODEL_HOME:-"${RAY_DATA_HOME}/models"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_HOME}/Qwen2.5-32B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/math/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

shuffle=True
mini_batch_mode=random
ppo_epochs=1
total_epochs=5

actor_lr=5e-7
critic_lr=1e-6
lr_scheduler=cosine
min_lr_ratio=0.1
lr_warmup_steps_ratio=0.03

test_freq=5
save_freq=5

offload=False
gradient_checkpointing=True

use_dynamic_bsz=True
actor_ppo_max_token_len=$((512 * num_procs))
infer_ppo_max_token_len=$((2048 * num_procs))


ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.truncation='left' \
    actor_rollout_ref.rollout.n=${n_trajs_per_prompt} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_eps_down} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_eps_up} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.lam=${gae_lambda} \
    algorithm.gamma=${gae_gamma} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.mini_batch.mode=${mini_batch_mode} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=${gradient_checkpointing} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.optim.warmup_style=${lr_scheduler} \
    actor_rollout_ref.actor.optim.min_lr_ratio=${min_lr_ratio} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.shuffle=${shuffle} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    critic.model.path="${MODEL_PATH}" \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    critic.model.enable_gradient_checkpointing=${gradient_checkpointing} \
    critic.model.use_remove_padding=True \
    critic.model.fsdp_config.param_offload=${offload} \
    critic.model.fsdp_config.optimizer_offload=${offload} \
    critic.model.fsdp_config.fsdp_size=${fsdp_size} \
    critic.ppo_mini_batch_size=${ppo_mini_batch_size} \
    critic.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    critic.forward_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    critic.ulysses_sequence_parallel_size=${sp_size} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    critic.optim.warmup_style=${lr_scheduler} \
    critic.optim.min_lr_ratio=${min_lr_ratio} \
    critic.optim.weight_decay=${weight_decay} \
    critic.ppo_epochs=${ppo_epochs} \
    critic.shuffle=${shuffle} \
    critic.grad_clip=${grad_clip} \
    critic.cliprange_value=${value_clip_eps} \
    reward_model.cosine_length_scaling.enabled=${enable_cosine_length_scaling_reward} \
    reward_model.cosine_length_scaling.correct_reward.max=${correct_cosine_max_reward} \
    reward_model.cosine_length_scaling.correct_reward.min=${correct_cosine_min_reward} \
    reward_model.cosine_length_scaling.wrong_reward.max=${wrong_cosine_max_reward} \
    reward_model.cosine_length_scaling.wrong_reward.min=${wrong_cosine_min_reward} \
    reward_model.exceeding_reward=${exceeding_reward} \
    reward_model.repetition.enabled=${enable_repetition_penalty} \
    reward_model.repetition.ngram_size=${repetition_ngram} \
    reward_model.repetition.reward=${repetition_reward} \
    reward_model.repetition.gamma=${repetition_gamma} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${n_procs_per_node} \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${resume_mode} \
    data.return_raw_chat=True