#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
cd /dcs/large/u5535699/bcp-official/code
export LA_ROOT=/dcs/large/u5535699/datasets/ssmis/LA
/dcs/large/u5535699/envs/ssmis/bin/python make_fig_combined.py --out ../figs
