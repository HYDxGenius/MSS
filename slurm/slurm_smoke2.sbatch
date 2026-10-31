#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:28:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
ROOT=/dcs/large/u5535699
PY=$ROOT/envs/ssmis/bin/python
cd $ROOT/bcp-official/code
export WANDB_MODE=disabled
run () {
  echo "########## $1"
  env $2 $PY ACDC_BCP_train.py --root_path "$ROOT/datasets/ssmis/ACDC" \
      --labelnum 7 --seed 1337 --exp "sm2_$1" --gpu 0 \
      --pre_iterations 210 --max_iterations 420 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -4
  echo "rc=${PIPESTATUS[0]}"
}
run region1 "BCP_DS=1 BCP_DSFORM=region1"
run probe2  "BCP_DS=1 BCP_PROBE=100"
P=$ROOT/bcp-official/code/model/BCP/ACDC_sm2_probe2_7_labeled/self_train/gradgeom.jsonl
echo "########## record kinds"; python3 -c "
import json,collections,sys
c=collections.Counter(); last={}
for l in open('$P'):
    r=json.loads(l); c[r['kind']]+=1; last[r['kind']]=r
print(dict(c))
for k in ('carrier','plabel'):
    if k in last: print(k, json.dumps(last[k])[:600])
"
