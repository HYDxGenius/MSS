#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:12:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
ROOT=/dcs/large/u5535699
export LA_ROOT=$ROOT/datasets/ssmis/LA
export WANDB_PROJECT=ssmis-bcp-official WANDB_ENTITY=yundi-university-of-warwick
export WANDB_RUN_NAME=launchcheck WANDB_RUN_ID=launchcheck WANDB_DIR=$ROOT/bcp-official/wandb
export WANDB_MODE=online
cd $ROOT/bcp-official/code
$ROOT/envs/ssmis/bin/python ../wandb_launch.py ACDC_BCP_train.py \
   --root_path "$ROOT/datasets/ssmis/ACDC" --labelnum 7 --seed 1337 --exp chk \
   --pre_iterations 200 --max_iterations 200 --gpu 0 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -12
echo "rc=${PIPESTATUS[0]}"
