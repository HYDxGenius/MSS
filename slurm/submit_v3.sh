#!/bin/bash
# v3 experiment set. Args to run.sbatch, in order:
#   DS LN SEED RED SIB DSW SIBN DSMODE W2S EMAE TRANS TAPW DSFORM DSDET PROBE TAGX
# The QOS caps this account at 3 concurrent GPUs, so all 16 are submitted at once and Slurm
# runs them three at a time. Longest first, so the 6.5 h LA jobs are not left for the tail.
set -u
cd /dcs/large/u5535699/bcp-official
S() { n=$1; shift; echo -n "$n "; sbatch --job-name="$n" run.sbatch "$@" | awk '{print $4}'; }

D="native 0"; W="0.4,0.2,0.1"
# --- LA: MSS re-run under the single corrected auxiliary form (TAGX=v2 keeps the superseded
#     snapshots of 63549/63550 intact), plus the two extra seeds at 10%.
S L1_la10ds  LA  8 1337 $D 1 surface vanilla 0 0 0 $W uniform 0 0 v2
S L2_la20ds  LA 16 1337 $D 1 surface vanilla 0 0 0 $W uniform 0 0 v2
S L5_la10ds8 LA  8 1338 $D 1 surface vanilla 0 0 0 $W uniform 0 0 v2
S L6_la10ds9 LA  8 1339 $D 1 surface vanilla 0 0 0 $W uniform 0 0 v2
S L3_la10b8  LA  8 1338 $D 0
S L4_la10b9  LA  8 1339 $D 0
# --- ACDC: the missing 5% cell
S A5_acdc5ds ACDC 3 1337 $D 1
# --- tap ablation, ACDC 10%, sum(w) held at 0.7 in every arm
S T1_tap1    ACDC 7 1337 $D 1 surface vanilla 0 0 0 0.7,0,0
S T2_tap2    ACDC 7 1337 $D 1 surface vanilla 0 0 0 0,0.7,0
S T3_tap3    ACDC 7 1337 $D 1 surface vanilla 0 0 0 0,0,0.7
S T4_tap12   ACDC 7 1337 $D 1 surface vanilla 0 0 0 0.4667,0.2333,0
S T5_tap23   ACDC 7 1337 $D 1 surface vanilla 0 0 0 0,0.4667,0.2333
S T6_tapunif ACDC 7 1337 $D 1 surface vanilla 0 0 0 0.2333,0.2333,0.2333
# --- auxiliary target-weighting control
S F1_carrier ACDC 7 1337 $D 1 surface vanilla 0 0 0 $W carrier
# --- gradient-geometry probes. P1 detaches the taps, so its trunk follows the BCP baseline
#     trajectory exactly and must reproduce 89.27; P2 is the MSS trajectory and must
#     reproduce 91.02. Those two equalities are the probe's correctness test.
S P1_probeB  ACDC 7 1337 $D 1 surface vanilla 0 0 0 $W uniform 1 500
S P2_probeM  ACDC 7 1337 $D 1 surface vanilla 0 0 0 $W uniform 0 500
