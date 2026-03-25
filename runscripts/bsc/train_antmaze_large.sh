#!/bin/bash

source /home/mananaga/miniconda/etc/profile.d/conda.sh
conda activate floqcritic
export MUJOCO_GL=egl

cd "$(dirname "$0")/../.."

# === Defaults ===
env_name="antmaze-large-navigate-singletask-v0"
alpha=10
beta=1.0
save_dir="./exp"
entity="RL-experiments"
project_name="BSC"
run_name="BSC"

# === Parse CLI overrides (--key=value) ===
for arg in "$@"; do
  case $arg in
    --*=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      key="${key#--}"
      eval "$key=\"$value\""
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

python main.py \
    --env_name=$env_name \
    --agent=agents/bsc.py \
    --agent.alpha=$alpha \
    --agent.beta=$beta \
    --save_dir=$save_dir \
    --entity=$entity \
    --project_name=$project_name \
    --run_name=$run_name
