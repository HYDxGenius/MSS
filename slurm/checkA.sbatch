#!/bin/bash
#SBATCH --job-name=checkA
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:28:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -uo pipefail
cd /dcs/large/u5535699/bcp-official/code
/dcs/large/u5535699/envs/ssmis/bin/python checkA.py 2>&1 | grep -vE "^ *[0-9]+%\|"
echo "rc=${PIPESTATUS[0]}"
