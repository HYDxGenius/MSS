#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
#
# Four-metric evaluation of every v3 arm on the official path (nms=1). eval_any.py aborts if the
# checkpoint carries keys the constructed network has nowhere to put, so a silent partial load
# cannot turn into a reported number.
set -u
ROOT=/dcs/large/u5535699
PY=$ROOT/envs/ssmis/bin/python
cd $ROOT/bcp-official/code
E() {  # label dataset labelnum exp [extra flags]
  lbl=$1; shift
  echo "@@@ $lbl | $*"
  $PY eval_any.py --dataset "$1" --labelnum "$2" --exp "$3" ${4:-} 2>&1 | tail -4
}
# --- LA, corrected auxiliary form
E L1_la10ds  LA  8 s1337_ds1_v2 --ds
E L2_la20ds  LA 16 s1337_ds1_v2 --ds
E L5_la10ds8 LA  8 s1338_ds1_v2 --ds
E L6_la10ds9 LA  8 s1339_ds1_v2 --ds
E L3_la10b8  LA  8 s1338
E L4_la10b9  LA  8 s1339
# --- ACDC 5%
E A5_acdc5ds ACDC 3 s1337_ds1 --ds
# --- tap ablation
E T1_tap1    ACDC 7 s1337_ds1_w0700 --ds
E T2_tap2    ACDC 7 s1337_ds1_w0070 --ds
E T3_tap3    ACDC 7 s1337_ds1_w0007 --ds
E T4_tap12   ACDC 7 s1337_ds1_w04667023330 --ds
E T5_tap23   ACDC 7 s1337_ds1_w00466702333 --ds
E T6_tapunif ACDC 7 s1337_ds1_w023330233302333 --ds
# --- auxiliary target-weighting control
E F1_carrier ACDC 7 s1337_ds1_fcarrier --ds
# --- probe runs: these two numbers are the probe's correctness test, not results.
#     P1 must reproduce the BCP baseline (89.27) and P2 the MSS arm (91.02).
E P1_probeB  ACDC 7 s1337_ds1_det_pr500 --ds
E P2_probeM  ACDC 7 s1337_ds1_pr500 --ds
echo "@@@ done"
