#!/usr/bin/env bash
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-unique-clean-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/aime-2024.parquet"}

mkdir -p "${VERL_HOME}/data"

wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/tongyx361/DAPO-Math-Unique-Clean-17k/resolve/main/data/dapo-math-unique-clean-17k.parquet?download=true"

wget -O "${TEST_FILE}" "https://huggingface.co/datasets/tongyx361/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"