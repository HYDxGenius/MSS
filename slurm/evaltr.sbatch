#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
PY=/dcs/large/u5535699/envs/ssmis/bin/python
cd /dcs/large/u5535699/bcp-official/code
echo "--- Transformer only ---"
$PY eval_any.py --dataset ACDC --exp s1337_tr4 --labelnum 7 --trans 4 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -6
echo "--- Transformer + MSS ---"
$PY eval_any.py --dataset ACDC --exp s1337_ds1_tr4 --labelnum 7 --ds --trans 4 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -6
echo ALLDONE
