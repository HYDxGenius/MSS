#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:28:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -u
cd /dcs/large/u5535699/bcp-official/code
/dcs/large/u5535699/envs/ssmis/bin/python make_figs.py \
    --dataset ACDC --exp_a s1337 --exp_b s1337_ds1 --labelnum 7 --n 2 --out ../figs
echo "@@@ acdc figs done"
