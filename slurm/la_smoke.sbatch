#!/bin/bash
#SBATCH --job-name=LAsmoke
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:28:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
ROOT=/dcs/large/u5535699
export LA_ROOT=$ROOT/datasets/ssmis/LA
PY=$ROOT/envs/ssmis/bin/python
cd $ROOT/bcp-official/code
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# sample GPU memory every 2s in the background -> peak is the number we need
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > /tmp/mem_$SLURM_JOB_ID.txt &
SAMPLER=$!

$PY LA_BCP_train.py --root_path "$LA_ROOT" --labelnum 8 \
    --pre_max_iteration 20 --self_max_iteration 0 --gpu 0 2>&1 | tail -40
RC=$?
kill $SAMPLER 2>/dev/null
echo "--- peak GPU memory (MiB) ---"
sort -n /tmp/mem_$SLURM_JOB_ID.txt | tail -1
rm -f /tmp/mem_$SLURM_JOB_ID.txt
echo "rc=$RC end=$(date -Is)"
