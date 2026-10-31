#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
ROOT=/dcs/large/u5535699
PY=$ROOT/envs/ssmis/bin/python
cd $ROOT/bcp-official/code
export WANDB_MODE=disabled
run () {  # name  extra-env
  echo "########## $1"
  env $2 $PY ACDC_BCP_train.py --root_path "$ROOT/datasets/ssmis/ACDC" \
      --labelnum 7 --seed 1337 --exp "smoke_$1" --gpu 0 \
      --pre_iterations 210 --max_iterations 420 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -6
  echo "rc=${PIPESTATUS[0]}"
}
run probe   "BCP_DS=1 BCP_DSDETACH=1 BCP_PROBE=100"
run tapw    "BCP_DS=1 BCP_DSW=0,0.7,0"
run carrier "BCP_DS=1 BCP_DSFORM=carrier"
echo "########## probe output"
P=$ROOT/bcp-official/code/model/BCP/ACDC_smoke_probe_7_labeled/self_train/gradgeom.jsonl
wc -l "$P"; head -1 "$P"; grep '"kind": "geom"' "$P" | tail -1; grep '"kind": "sens"' "$P" | tail -2
