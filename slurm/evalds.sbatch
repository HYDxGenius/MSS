#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
PY=/dcs/large/u5535699/envs/ssmis/bin/python
cd /dcs/large/u5535699/bcp-official/code
$PY eval_any.py --dataset ACDC --exp s1337_ds1 --labelnum 7  --ds 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -4
$PY eval_any.py --dataset LA   --exp s1337_ds1 --labelnum 8  --ds 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -4
$PY eval_any.py --dataset LA   --exp s1337_ds1 --labelnum 16 --ds 2>&1 | grep -vE "^ *[0-9]+%\|" | tail -4
echo "ALL DONE"
