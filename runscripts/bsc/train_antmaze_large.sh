#!/bin/bash

source /home/mananaga/miniconda/etc/profile.d/conda.sh
conda activate floqcritic
export MUJOCO_GL=egl

cd "$(dirname "$0")/../.."

python main.py \
    --env_name=antmaze-large-navigate-singletask-v0 \
    --agent=agents/bsc.py \
    --agent.alpha=10 \
    --agent.beta=1.0 \
    --save_dir=./exp \
    --entity=RL-experiments \
    --project_name=BSC \
    --run_name=BSC
