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
E() { echo "@@@ $1"; shift; $PY eval_any.py --dataset ACDC "$@" 2>&1 | tail -2; }
# ACDC 20%, the two extra seeds for both arms
for s in 1338 1339; do E "A20_bcp_$s"  --labelnum 14 --exp s$s; done
for s in 1338 1339; do E "A20_mss_$s"  --labelnum 14 --exp s${s}_ds1 --ds; done
# Region-uniform auxiliary target, three seeds
for s in 1337 1338 1339; do E "REG_$s" --labelnum 7 --exp s${s}_ds1_fregion1 --ds; done
# Placement at extra seeds
for s in 1338 1339; do E "T1_$s" --labelnum 7 --exp s${s}_ds1_w0700 --ds; done
for s in 1338 1339; do E "T2_$s" --labelnum 7 --exp s${s}_ds1_w0070 --ds; done
# Probe v2 runs: these two are correctness checks, not results (must sit with 91.02 / 89.4x)
E P3_probeM  --labelnum 7 --exp s1337_ds1_pr500_p2 --ds
E P4_probeB  --labelnum 7 --exp s1337_ds1_det_pr500_p2 --ds
echo "@@@ done"
