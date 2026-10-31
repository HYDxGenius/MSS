#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:28:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
ROOT=/dcs/large/u5535699
export LA_ROOT=$ROOT/datasets/ssmis/LA WANDB_MODE=offline
export BCP_DS="${1:-0}" BCP_W2S="${2:-0}" BCP_EMAEVAL="${3:-0}"
cd $ROOT/bcp-official/code
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 3; done ) > /tmp/m_$SLURM_JOB_ID.txt &
S=$!
T0=$(date +%s)
$ROOT/envs/ssmis/bin/python LA_BCP_train.py --root_path "$LA_ROOT" --labelnum 8 \
  --exp lasmk_d${BCP_DS}_w${BCP_W2S}_e${BCP_EMAEVAL} \
  --pre_max_iteration 200 --self_max_iteration 200 --gpu 0 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -8
RC=${PIPESTATUS[0]}
kill $S 2>/dev/null
echo "DS=$BCP_DS W2S=$BCP_W2S EMA=$BCP_EMAEVAL peakMiB=$(sort -n /tmp/m_$SLURM_JOB_ID.txt|tail -1) elapsed=$(( $(date +%s)-T0 ))s rc=$RC"
rm -f /tmp/m_$SLURM_JOB_ID.txt
exit $RC
