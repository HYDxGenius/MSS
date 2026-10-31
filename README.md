# MSS — multi-scale auxiliary supervision for semi-supervised medical image segmentation

Experiment code for a study of auxiliary ("deep") supervision added to **BCP**
(Bidirectional Copy-Paste, CVPR 2023) on ACDC, LA and Pancreas-NIH.

This repository contains source only. Checkpoints, logs, figures and result tables are
excluded.

## Relation to upstream

Built on the official BCP release (`DeepMed-Lab-ECNU/BCP`), MIT licensed; the upstream
`LICENSE` is preserved at the root and applies to the files derived from it. Most files under
`code/` are upstream's, modified in place. The files below are new in this work:

| path | what it is |
|---|---|
| `code/networks/unet_ds.py` | U-Net with three 1×1 auxiliary heads on decoder stages (strides 8/4/2); `forward_ds(detach_taps=)` optionally cuts the auxiliary gradient off the trunk |
| `code/networks/vnet_ds.py` | the same for the 3D V-Net (LA), taps on `x6/x7/x8` |
| `code/pancreas/vnet_ds.py` | the same for the Pancreas pipeline, whose V-Net has a different decoder shape |
| `code/networks/unet_trans_ds.py` | Transformer bottleneck on the 1/16 encoder stage, used as a capacity control |
| `code/utils/smds.py` | the auxiliary loss, with per-tap weights and an optional survival-weighted variant |
| `code/utils/dsprobe.py` | gradient-geometry probe: alignment, norm ratio, conflict rate and tap redundancy on the parameters the taps share with the main head; the descent-lemma ratio on the live training batch; target-corruption and real-pseudo-label sensitivity |
| `code/loss_form_diag.py` | no-training diagnostic comparing auxiliary loss formulations by value, gradient norm and cosine on one checkpoint |
| `code/eval_any.py` | four-metric evaluation that builds whichever network a checkpoint actually is, and aborts rather than silently dropping tensors |
| `code/make_figs.py`, `code/make_fig_combined.py` | qualitative figures with signed error maps |
| `code/utils/sib_loss.py`, `code/utils/strongaug.py` | components that were tested and not kept |
| `code/panc_prep.py` | Pancreas-NIH preprocessing via SimpleITK's GDCM series reader |
| `analysis/` | result extraction and aggregation scripts |
| `patches/` | the scripts that applied each change to the upstream tree, kept as a record of what was modified and where |

## Running

`slurm/run.sbatch` runs one arm per job:

```
sbatch run.sbatch <LA|ACDC> <labelnum> <seed> <reduce> <SIB> <DSW> <SIBN> <DSMODE> \
                  <W2S> <EMAE> <TRANS> <TAPW> <DSFORM> <DSDET> <PROBE> <TAGX>
```

Everything defaults to stock BCP. The switches reach the trainers as environment variables:

| variable | effect |
|---|---|
| `BCP_DS` | weight λ of the auxiliary loss (0 = off) |
| `BCP_DSW` | per-tap weights, coarse → fine, comma separated |
| `BCP_DSFORM` | `uniform` (one unmasked loss over the whole mixed target), `carrier` (the two-region form with the pseudo half at `u_weight`), or `region1` (two-region with both halves at 1) |
| `BCP_DSDETACH` | train the auxiliary heads without letting their gradient reach the shared trunk |
| `BCP_PROBE` | gradient-probe interval in iterations (0 = off) |
| `BCP_TRANS` | number of Transformer blocks in the bottleneck control |
| `BCP_SIB`, `BCP_SIBN`, `BCP_DSMODE`, `BCP_W2S`, `BCP_EMAEVAL`, `BCP_DICE_REDUCE` | components that were tested and not kept |

`EXP` is derived from every switch, so two arms cannot share a snapshot directory.

The scripts under `slurm/` carry absolute paths for the cluster they were run on
(`/dcs/large/u5535699/...`) and a Slurm partition name; they are included as a record of what
was run and need editing before use elsewhere.

## Evaluation

The two datasets' official test paths differ and `eval_any.py` reproduces each:
`test_ACDC.py` is a plain arg-max with no post-processing, while `test_LA.py` defaults to
`nms=1`, largest-connected-component.
