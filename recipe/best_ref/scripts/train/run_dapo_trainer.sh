#!/usr/bin/env bash
set -uxo pipefail # `read` is incompatible with `set -e`

# Config to infer from
NNODES=${NNODES:-16}
# Inferred
n_procs_per_node=8
num_procs=$((NNODES * n_procs_per_node))

# Model
MODEL_ID=${MODEL_ID:-"qwen2p5-32b"}
if [ "${MODEL_ID}" == "qwen2p5-32b" ]; then
    model_name="Qwen2.5-32B"
    fsdp_size=64
    gen_tp=2
    gpu_mem_util=0.7
elif [ "${MODEL_ID}" == "qwen2p5-7b" ]; then
    model_name="Qwen2.5-7B"
    fsdp_size=8 # In-node
    gen_tp=1
    gpu_mem_util=0.9
else
    echo "Invalid model ID: ${MODEL_ID}"
    exit 1
fi

# Recipe
RECIPE=${RECIPE:-"darloo"}
repeat_factor=1
shuffle_in_batch=True
if [ "${RECIPE}" == "darloo" ]; then
    adv_estimator=rloo
    actor_lr=2e-6
elif [[ "${RECIPE}" =~ "dapo" ]]; then
    adv_estimator=grpo
    actor_lr=1e-6
    if [[ "${RECIPE}" == "dapo-orig-data" ]]; then
        repeat_factor=100
        shuffle_in_batch=False
    fi
else
    echo "Invalid recipe: ${RECIPE}"
    exit 1
fi

# Test
TEST=${TEST:-"0"}
if [ "${TEST}" != "1" ]; then
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 20))
    n_trajs_per_prompt=16
    train_bs=512
    n_updates_per_batch=16
    val_n=32
    val_before_train=True
    resume_mode=auto
    save_freq=5
else
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 2))
    n_trajs_per_prompt=2
    ppo_mini_batch_size=$((num_procs / n_trajs_per_prompt))
    n_updates_per_batch=2
    train_bs=$((ppo_mini_batch_size * n_updates_per_batch))
    val_n=1
    val_before_train=False
    resume_mode=disable
    save_freq=-1
fi

# Customization
TRAIN_BS=${TRAIN_BS:-"${train_bs}"}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-"${val_before_train}"}
ACTOR_LR=${ACTOR_LR:-${actor_lr}}
TRAIN_BS=${TRAIN_BS:-"${train_bs}"}
N_UPDATES_PER_BATCH=${N_UPDATES_PER_BATCH:-${n_updates_per_batch}}
# Ray
RAY_JOB_SUBMIT=${RAY_JOB_SUBMIT:-"1"}
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-${VERL_HOME:-"${HOME}/verl"}}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-unique-clean-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024-clean.parquet"}
MODEL_HOME=${MODEL_HOME:-"${RAY_DATA_HOME}/models"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_HOME}/${model_name}"}
# Info
DEVICE=${DEVICE:-"h800"}

# Common settings
project_name='best-ref'
ppo_mini_batch_size=$((TRAIN_BS / N_UPDATES_PER_BATCH))
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
# Other settings
temperature=1.0
sp_size=8 # In-node
offload=False

use_dynamic_bsz=True
actor_ppo_max_token_len=$((512 * num_procs))
infer_ppo_max_token_len=$((2048 * num_procs))

test_freq=${save_freq}

ckpt_name="${RECIPE}-dmrl-${MODEL_ID}-lr${ACTOR_LR}-bs${TRAIN_BS}-nup${N_UPDATES_PER_BATCH}-${num_procs}gpus"
if [ "${TEST}" == "1" ]; then
    ckpt_name="${ckpt_name}-test"
fi
exp_name="${ckpt_name}-${DEVICE}-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)"

log_home="${RAY_DATA_HOME}/${project_name}/logs/train"
mkdir -p "${log_home}"
log_path="${log_home}/${exp_name}.log"

read -r -d '' py_cmd <<EOF
python3 -m recipe.dapo.src.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    +data=cot_boxed_sfx_template \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size="${TRAIN_BS}" \
    data.shuffle_in_batch=${shuffle_in_batch} \
    data.repeat.factor=${repeat_factor} \
    data.val_repeat_factor=${val_n} \
    data.return_raw_chat=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.clip_ratio_low=${clip_eps_down} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_eps_up} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.n=${n_trajs_per_prompt} \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_procs_per_node} \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=${save_freq} \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${RAY_DATA_HOME}/ckpts/${project_name}/${ckpt_name}" \
    trainer.resume_mode="${resume_mode}" \
    2>&1 | tee "${log_path}"
EOF

if [ "${RAY_JOB_SUBMIT}" == "1" ]; then
    ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
        --working-dir "${WORKING_DIR}" \
        -- "${py_cmd}"
else
    eval "${py_cmd}"
fi