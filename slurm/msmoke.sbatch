#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:28:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
ROOT=/dcs/large/u5535699
export WANDB_MODE=offline
export BCP_SIB="${1:-0}" BCP_DS="${2:-0}" BCP_SIBN="${3:-surface}" BCP_DSMODE="${4:-vanilla}" BCP_W2S="${5:-0}" BCP_EMAEVAL="${6:-0}" BCP_TRANS="${7:-0}"
cd $ROOT/bcp-official/code
T0=$(date +%s)
$ROOT/envs/ssmis/bin/python ACDC_BCP_train.py --root_path "$ROOT/datasets/ssmis/ACDC" \
  --labelnum 7 --seed 1337 --exp smk_s${BCP_SIB}_d${BCP_DS}_w${BCP_W2S}_e${BCP_EMAEVAL}_t${BCP_TRANS} \
  --pre_iterations 200 --max_iterations 200 --gpu 0 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -8
RC=${PIPESTATUS[0]}
echo "SIB=$BCP_SIB DS=$BCP_DS  elapsed=$(( $(date +%s) - T0 ))s  rc=$RC"
exit $RC
