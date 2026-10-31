#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:28:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
DS="${1:?usage: sbatch --job-name=<n> smoke.sbatch <LA|ACDC> <pre> <self>}"
PRE="${2:-200}"; SELF="${3:-200}"
ROOT=/dcs/large/u5535699
export LA_ROOT=$ROOT/datasets/ssmis/LA
PY=$ROOT/envs/ssmis/bin/python
cd $ROOT/bcp-official/code
echo "host=$(hostname) job=$SLURM_JOB_ID ds=$DS pre=$PRE self=$SELF start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > /tmp/mem_$SLURM_JOB_ID.txt &
SAMPLER=$!
T0=$(date +%s)
if [ "$DS" = "LA" ]; then
  $PY LA_BCP_train.py --root_path "$LA_ROOT" --labelnum 8 \
      --pre_max_iteration $PRE --self_max_iteration $SELF --gpu 0 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -30
else
  $PY ACDC_BCP_train.py --root_path "$ROOT/datasets/ssmis/ACDC" --labelnum 7 \
      --pre_iterations $PRE --max_iterations $SELF --gpu 0 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -30
fi
RC=${PIPESTATUS[0]}
kill $SAMPLER 2>/dev/null
echo "--- peak GPU MiB: $(sort -n /tmp/mem_$SLURM_JOB_ID.txt | tail -1)  elapsed: $(( $(date +%s) - T0 ))s ---"
rm -f /tmp/mem_$SLURM_JOB_ID.txt
echo "rc=$RC end=$(date -Is)"
