#!/bin/bash
#SBATCH --partition=gecko-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
cd /dcs/large/u5535699/bcp-official/code
/dcs/large/u5535699/envs/ssmis/bin/python loss_form_diag.py --exp s1337_ds1 --stage pre_train --batches 24
