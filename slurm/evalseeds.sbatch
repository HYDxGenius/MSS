#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
# Official-path four-metric evaluation of all three seeds, both arms.
set -uo pipefail
PY=/dcs/large/u5535699/envs/ssmis/bin/python
cd /dcs/large/u5535699/bcp-official/code
for s in 1337 1338 1339; do
  echo "### baseline seed $s"
  $PY eval_any.py --dataset ACDC --exp s$s --labelnum 7 2>&1 | grep -E "Dice |ABORT" | tail -2
  echo "### +MSS seed $s"
  $PY eval_any.py --dataset ACDC --exp s${s}_ds1 --labelnum 7 --ds 2>&1 | grep -E "Dice |ABORT" | tail -2
done
echo ALLDONE
