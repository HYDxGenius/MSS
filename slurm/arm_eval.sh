#!/bin/bash
# The submit limit (16) blocked the dependent evaluation job. Retry until a slot frees, then
# submit it held behind all 16 training jobs with afterany, so it runs even if one fails and
# the rest of the arms still get evaluated.
D=afterany:65058:65059:65060:65061:65062:65063:65064:65065:65066:65067:65068:65069:65070:65071:65072:65073
for i in $(seq 1 400); do
  out=$(sbatch --job-name=evalv3 --dependency=$D /dcs/large/u5535699/bcp-official/eval_v3.sbatch 2>&1)
  case "$out" in
    *"Submitted batch job"*) echo "EVAL_ARMED $out"; exit 0 ;;
  esac
  sleep 300
done
echo "EVAL_ARM_FAILED: $out"; exit 1
