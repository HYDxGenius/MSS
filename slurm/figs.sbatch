#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -u
PY=/dcs/large/u5535699/envs/ssmis/bin/python
cd /dcs/large/u5535699/bcp-official/code
export LA_ROOT=/dcs/large/u5535699/datasets/ssmis/LA
$PY make_figs.py --dataset ACDC --exp_a s1337 --exp_b s1337_ds1     --labelnum 7  --n 2 --out ../figs
$PY make_figs.py --dataset LA   --exp_a s1337 --exp_b s1337_ds1_v2  --labelnum 16 --n 2 --out ../figs
echo "@@@ figs done"; ls -l ../figs
