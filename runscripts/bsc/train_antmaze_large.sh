#!/bin/bash

source /home/mananaga/miniconda/etc/profile.d/conda.sh
conda activate floqcritic
export MUJOCO_GL=egl

cd /home/mananaga/BSC_branches/manan_bsc/Binary-Success-Critic

python main.py \
    --env_name=antmaze-large-navigate-singletask-v0 \
    --agent=agents/bsc.py \
    --agent.alpha=3 \
    --save_dir=./exp
