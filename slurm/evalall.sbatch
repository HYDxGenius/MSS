#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
#
# Full four-metric evaluation (Dice / Jaccard / 95HD / ASD) of every finished checkpoint, using
# BCP's own test scripts. The training loops log Dice only, but both reference papers report all
# four, so the comparison table needs them.
set -uo pipefail
ROOT=/dcs/large/u5535699
export LA_ROOT=$ROOT/datasets/ssmis/LA
PY=$ROOT/envs/ssmis/bin/python
cd $ROOT/bcp-official/code

run_acdc () {   # $1 exp  $2 labelnum
  echo "##### ACDC exp=$1 labelnum=$2"
  $PY test_ACDC.py --root_path "$ROOT/datasets/ssmis/ACDC" --exp "$1" --labelnum "$2" \
      2>&1 | grep -vE "^ *[0-9]+%\|" | tail -5
}
run_la () {     # $1 exp  $2 labelnum
  echo "##### LA exp=$1 labelnum=$2"
  $PY test_LA.py --root_path "$LA_ROOT/" --exp "$1" --labelnum "$2" \
      2>&1 | grep -vE "^ *[0-9]+%\|" | tail -5
}

run_acdc s1337        7
run_acdc s1337        3
run_acdc s1337_ds1  7
run_la   s1337        8
run_la   s1337        16
run_la   s1337_ds1  8
run_la   s1337_ds1  16
echo "ALL DONE"
