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
echo "########## shared pre-trained initialisation (where the three arms diverge)"
$PY loss_form_diag.py --exp s1337_ds1 --stage pre_train  --batches 16
echo
echo "########## converged MSS checkpoint"
$PY loss_form_diag.py --exp s1337_ds1 --stage self_train --batches 16
echo
echo "########## converged region-uniform checkpoint"
$PY loss_form_diag.py --exp s1337_ds1_fregion1 --stage self_train --batches 16
