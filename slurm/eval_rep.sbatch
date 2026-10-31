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
# Same-seed reproducibility: each of these repeats an arm already measured, under today's code,
# with no switch changed. Any gap is run-to-run variation at a fixed seed.
echo "@@@ R1_repMSS (repeat of 62588 = 91.02)"; $PY eval_any.py --dataset ACDC --labelnum 7 --exp s1337_ds1_rep --ds 2>&1 | tail -2
echo "@@@ R2_repBCP (repeat of 62401 = 89.27)"; $PY eval_any.py --dataset ACDC --labelnum 7 --exp s1337_rep 2>&1 | tail -2
echo "@@@ R3_repDET (control for P1 = 88.10)";  $PY eval_any.py --dataset ACDC --labelnum 7 --exp s1337_ds1_det_rep --ds 2>&1 | tail -2
echo "@@@ done"
