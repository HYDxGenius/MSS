#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
set -u
PY=/dcs/large/u5535699/envs/ssmis/bin/python
cd /dcs/large/u5535699/bcp-official/code
E() { echo "@@@ $1"; shift; $PY eval_any.py --dataset ACDC "$@" 2>&1 | tail -2; }
# ACDC 20%: the column DiCoP-SAM and EUTNet both report and we did not have.
E A20_bcp  --labelnum 14 --exp s1337
E A20_mss  --labelnum 14 --exp s1337_ds1 --ds
# Carrier-weighted auxiliary target at two more seeds, so the -2.47 result is not single-seed.
E Q2_carr38 --labelnum 7 --exp s1338_ds1_fcarrier --ds
E Q3_carr39 --labelnum 7 --exp s1339_ds1_fcarrier --ds
# Replication of the baseline-trajectory probe (P1 = 88.10 against its control R3 = 89.44).
E Q1_probeB2 --labelnum 7 --exp s1337_ds1_det_pr500_rep2 --ds
echo "@@@ done"
