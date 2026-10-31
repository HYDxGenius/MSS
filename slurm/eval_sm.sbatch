#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -u
cd /dcs/large/u5535699/bcp-official/code
PY=/dcs/large/u5535699/envs/ssmis/bin/python
E(){ echo "@@@ $1"; shift; $PY eval_any.py --dataset ACDC --labelnum 7 "$@" 2>&1 | tail -2; }
for s in 1337 1338 1339; do E "SM_reg_$s" --exp s${s}_ds1_w0201005_fregion1 --ds; done
for s in 1337 1338 1339; do E "X2_unm_$s" --exp s${s}_ds1_w080402 --ds; done
echo "@@@ done"
