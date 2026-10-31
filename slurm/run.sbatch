#!/bin/bash
#SBATCH --partition=gecko
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/dcs/large/u5535699/bcp-official/logs/%x-%j.out
#SBATCH --error=/dcs/large/u5535699/bcp-official/logs/%x-%j.err
#
# Official BCP reproduction, stock hyper-parameters, one dataset+label-budget per job.
#   sbatch --job-name=LA10   run.sbatch LA   8 1337
#   sbatch --job-name=ACDC10 run.sbatch ACDC 7 1337
#
# Upstream has no resume, so jobs are sized to finish inside --time. The snapshot dir is
# ./model/BCP/<DS>_<exp>_<labelnum>_labeled and exp carries the seed, so two jobs at the same
# label budget with different seeds cannot overwrite each other's checkpoints.
set -uo pipefail
DS="${1:?usage: run.sbatch <LA|ACDC> <labelnum> <seed>}"
LN="${2:?}"; SEED="${3:-1337}"; RED="${4:-native}"
SIB="${5:-0}"; DSW="${6:-0}"; SIBN="${7:-surface}"; DSMODE="${8:-vanilla}"; W2S="${9:-0}"; EMAE="${10:-0}"; TRANS="${11:-0}"   # DSW, never DS: $DS is the dataset
# v3 switches. TAPW is the per-tap weight vector, coarse -> fine; every tap-ablation arm holds
# its sum at 0.7, so the arms differ in placement and not in total auxiliary mass.
TAPW="${12:-0.4,0.2,0.1}"; DSFORM="${13:-uniform}"; DSDET="${14:-0}"; PROBE="${15:-0}"
# TAGX is appended verbatim to EXP. Its only job is to keep a corrected run from overwriting
# the snapshot of the run it corrects, so both stay auditable.
TAGX="${16:-}"
ROOT=/dcs/large/u5535699
export LA_ROOT=$ROOT/datasets/ssmis/LA
export WANDB_PROJECT=ssmis-bcp-official
export WANDB_ENTITY=yundi-university-of-warwick
export WANDB_DIR=$ROOT/bcp-official/wandb
# BCP writes nine image summaries per logging step on ACDC; sync_tensorboard mirrors every one
# of them into wandb media (200-400 MB per run). The add_image calls are now disabled at source;
# this is the second line of defence so a reverted patch cannot quietly refill the quota.
export WANDB_IGNORE_GLOBS="media/**,*.png,*.jpg,*.jpeg,*.gif,*.mp4"
mkdir -p "$WANDB_DIR"
PY=$ROOT/envs/ssmis/bin/python
cd $ROOT/bcp-official/code
# BCP_DICE_REDUCE selects how the unlabeled Dice term is reduced: "native" is bit-identical to
# upstream (verified against losses.py.orig), "sample" normalises per sample, "batch" pools over
# the batch. It goes into EXP so a reduction variant can never overwrite the native run.
export BCP_DICE_REDUCE="$RED"
export BCP_SIB="$SIB" BCP_DS="$DSW" BCP_SIBN="$SIBN" BCP_DSMODE="$DSMODE" BCP_W2S="$W2S" BCP_EMAEVAL="$EMAE" BCP_TRANS="$TRANS"
export BCP_DSW="$TAPW" BCP_DSFORM="$DSFORM" BCP_DSDETACH="$DSDET" BCP_PROBE="$PROBE"
# EXP encodes every method switch, so no two arms can share a snapshot directory.
TAG=""
[ "$RED"  != "native"  ] && TAG="${TAG}_r${RED}"
[ "$SIB"  != "0"       ] && TAG="${TAG}_sib${SIB}"
[ "$SIBN" != "surface" ] && TAG="${TAG}_n${SIBN}"
[ "$DSW"  != "0"        ] && TAG="${TAG}_ds${DSW}"
[ "$DSMODE" != "vanilla" ] && TAG="${TAG}_${DSMODE}"
[ "$W2S"  != "0" ] && TAG="${TAG}_w2s${W2S}"
[ "$EMAE" != "0" ] && TAG="${TAG}_ema"
[ "$TRANS" != "0" ] && TAG="${TAG}_tr${TRANS}"
[ "$TAPW"   != "0.4,0.2,0.1" ] && TAG="${TAG}_w$(echo "$TAPW" | tr -d '.,')"
[ "$DSFORM" != "uniform"     ] && TAG="${TAG}_f${DSFORM}"
[ "$DSDET"  != "0"           ] && TAG="${TAG}_det"
[ "$PROBE"  != "0"           ] && TAG="${TAG}_pr${PROBE}"
EXP="s${SEED}${TAG}${TAGX:+_$TAGX}"
# wandb identity has to be derived AFTER TAG exists, otherwise every ablation arm writes to the
# same run id and they clobber each other -- which is exactly what happened to 62587-62590.
export WANDB_RUN_NAME="${DS}_ln${LN}_${EXP}"
# The job id keeps the run id unique. wandb permanently reserves a deleted run's id, so a
# deterministic id makes any resubmission after a cleanup fail with "previously created and
# deleted" -- which is exactly how X_C1 (63506) died.
export WANDB_RUN_ID="${DS}-ln${LN}-${EXP}-${SLURM_JOB_ID}"
echo "dice_reduce=$RED sib=$SIB($SIBN) ds=$DSW($DSMODE) tapw=$TAPW form=$DSFORM detach=$DSDET probe=$PROBE w2s=$W2S ema=$EMAE exp=$EXP"
echo "host=$(hostname) job=$SLURM_JOB_ID ds=$DS labelnum=$LN seed=$SEED exp=$EXP start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
T0=$(date +%s)
if [ "$DS" = "LA" ]; then
  $PY ../wandb_launch.py LA_BCP_train.py --root_path "$LA_ROOT" \
      --labelnum "$LN" --seed "$SEED" --exp "$EXP" --gpu 0 2>&1 | grep -vE "^ *[0-9]+%\|"
else
  $PY ../wandb_launch.py ACDC_BCP_train.py --root_path "$ROOT/datasets/ssmis/ACDC" \
      --labelnum "$LN" --seed "$SEED" --exp "$EXP" --gpu 0 2>&1 | grep -vE "^ *[0-9]+%\|"
fi
RC=${PIPESTATUS[0]}
echo "--- elapsed $(( $(date +%s) - T0 ))s  rc=$RC  end=$(date -Is) ---"
# The job's exit status must reflect TRAINING, not this reporting tail. A trailing command that
# returns non-zero made Slurm mark two fully successful 30k-iteration runs as FAILED, which in
# turn fooled the guard chain. Report, then exit with the training return code explicitly.
echo "--- best mean_dice seen ---"
grep -oE "mean_dice : [0-9.]+" "/dcs/large/u5535699/bcp-official/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null \
  | awk '{print $3}' | sort -g | tail -1
exit $RC
