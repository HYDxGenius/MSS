#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
#
# Pancreas-NIH, official BCP code. The script has no argparse, so everything is env-driven.
# It also wraps main() in `except Exception: logger.exception("BUG FOUNDED ! ! !")`, i.e. it
# exits 0 on failure -- the log must be grepped for that string, not just the return code.
set -uo pipefail
ROOT=/dcs/large/u5535699
export PANCREAS_DATA=$ROOT/datasets/ssmis/Pancreas-NIH/h5
export PANCREAS_LISTS=$ROOT/datasets/ssmis/Pancreas-NIH/data_lists
export BCP_DS="${2:-0}"
export PANCREAS_TAG="${3:-}"
export PANCREAS_PCT="${1:-20}"
export WANDB_MODE=disabled
cd $ROOT/bcp-official/code/pancreas
echo "host=$(hostname) job=$SLURM_JOB_ID pct=$PANCREAS_PCT start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
T0=$(date +%s)
$ROOT/envs/ssmis/bin/python -u train_pancreas.py 2>&1 | grep -vE "^ *[0-9]+%\|"
echo "--- elapsed $(( $(date +%s) - T0 ))s  end=$(date -Is) ---"
