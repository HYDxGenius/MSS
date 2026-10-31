#!/usr/bin/env python
"""J12 -- is a frozen backbone's global-token alignment localising or ANTI-localising on
medical frames, and does that depend on the architecture or on the readout grid?

BACKGROUND. LaStViT defines Patch Score = cos(patch feature, CLS) and Point-in-Box (PiB) = does
the single highest-scoring patch fall inside the annotated foreground. On ImageNet it reports
ViT PiB 42.7 vs ResNet 68.4, i.e. ViTs route global semantics through background patches
("lazy aggregation"). This script ports that measurement to three 2D medical datasets.

WHAT THE FIRST PASS FOUND (2026-10-09) AND WHY THIS VERSION EXISTS. Both ViTs scored BELOW their
own chance baseline while ResNet-50 scored well above it. But the models were read out on
different grids (ViT 16x16, ResNet 7x7), and a coarser grid inflates PiB mechanically. A
conclusion drawn from that comparison would have been a measurement artifact.

So every model's score map is now resampled to a COMMON grid before PiB and AUROC are computed,
and the whole table is reported at several common grids. If the ViT/ConvNet ordering survives
grid matching in both directions, it is a property of the models; if it moves with the grid, it
was never there.

CONTROLS
  chance PiB  = the foreground patch fraction of that image. A random top patch hits the
                foreground with exactly this probability, so PiB alone means nothing.
  resnet50    = ConvNet control with GAP as the implicit CLS, per LaStViT.
  vit_sup     = label-supervised ViT, so the finding cannot be pinned on one pretraining recipe.

SCOPE OF ANY CLAIM. Patch Score measures the GLOBAL-TOKEN ALIGNMENT pathway. It does not measure
whether a backbone's patch features support dense prediction through a trained decoder -- DINOv2
is strong at that. Nothing here licenses "DINOv2 is bad for medical segmentation".

Usage: python j12_foundation_artifact.py [--models dinov2,vit_sup,resnet50] [--n_images 120]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

FRAMEWORK = "/home/u5535699/Desktop/dcslarge/ssmis-stability"
sys.path.insert(0, FRAMEWORK)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/home/u5535699/Desktop/dcslarge/hf_cache_full/huggingface")

from ssmis.data.build import build  # noqa: E402

DATASETS = [("ACDC", "7cases"), ("Kvasir-SEG", "10pct"), ("ISIC2018", "10pct")]
IMNET = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
COMMON_GRIDS = [14, 7]  # every model is also reported on these shared grids


class ResNetTokens(torch.nn.Module):
    """ConvNet control: patch tokens are the spatial feature map, the global token is GAP.

    `stage` picks the output stride so the ConvNet can be read out at a ViT-like resolution:
    layer2 -> stride 8, layer3 -> stride 16, layer4 -> stride 32.
    """

    def __init__(self, stage="layer4"):
        super().__init__()
        import timm
        # timm features_only indices: 0 stem/s2, 1 layer1/s4, 2 layer2/s8, 3 layer3/s16,
        # 4 layer4/s32. Using idx-1 here silently returned stride 16 while the caller believed
        # it had stride 32 -- caught only because the token count did not match.
        idx = {"layer2": 2, "layer3": 3, "layer4": 4}[stage]
        self.net = timm.create_model("resnet50", pretrained=True, features_only=True,
                                     out_indices=(idx,))

    def forward(self, pixel_values):
        f = self.net(pixel_values)[0]
        patches = f.flatten(2).transpose(1, 2)
        cls = patches.mean(1)
        return torch.cat([cls[:, None], patches], 1)


class HFTokens(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state


def load_model(name):
    """Returns (module, output_stride, normalisation)."""
    from transformers import AutoModel
    if name == "dinov2":
        return HFTokens(AutoModel.from_pretrained("facebook/dinov2-base")).cuda().eval(), 14, IMNET
    if name == "vit_sup":
        return HFTokens(AutoModel.from_pretrained("google/vit-base-patch16-224")).cuda().eval(), 16, IMNET
    if name == "resnet50":
        return ResNetTokens("layer4").cuda().eval(), 32, IMNET
    if name == "resnet50_s16":
        return ResNetTokens("layer3").cuda().eval(), 16, IMNET
    if name == "resnet50_s8":
        return ResNetTokens("layer2").cuda().eval(), 8, IMNET
    raise ValueError(name)


def auroc(score, positive):
    score = np.asarray(score, float)
    positive = np.asarray(positive, bool)
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[positive].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def score_at_grid(ps, gt_full, grid, patch_for_native):
    """Resample a score map and the GT to a common grid.

    ps       : (g,g) patch-score map at the model's native grid
    gt_full  : (H,W) binary ground truth at full resolution
    grid     : target grid, or None to use the model's native grid
    Scores are area-averaged (a coarse cell's score is the mean of the fine cells it covers);
    GT uses max-pooling, so a cell counts as foreground if any of it is foreground.
    """
    t = torch.from_numpy(ps)[None, None].float()
    g_full = torch.from_numpy(gt_full)[None, None].float()
    if grid is None:
        grid = ps.shape[0]
        gtg = F.max_pool2d(g_full, kernel_size=patch_for_native,
                           stride=patch_for_native)[0, 0].numpy() > 0
        return ps, gtg
    psg = F.adaptive_avg_pool2d(t, grid)[0, 0].numpy()
    gtg = F.adaptive_max_pool2d(g_full, grid)[0, 0].numpy() > 0
    return psg, gtg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dinov2,vit_sup,resnet50,resnet50_s16")
    ap.add_argument("--n_images", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for mname in args.models.split(","):
        try:
            model, stride, norm = load_model(mname)
        except Exception as e:
            print(f"\n=== {mname}: SKIPPED ({type(e).__name__}: {str(e)[:110]})")
            continue
        mean = torch.tensor(norm["mean"]).view(1, 3, 1, 1).cuda()
        std = torch.tensor(norm["std"]).view(1, 3, 1, 1).cuda()
        # Infer the native grid from the actual token count rather than trusting the declared
        # stride: a wrong stride assumption is silent, a wrong token count is not.
        with torch.no_grad():
            n_tok = model(torch.zeros(1, 3, 224, 224).cuda()).shape[1] - 1
        native = int(round(n_tok ** 0.5))
        if native * native != n_tok:
            print(f"  SKIP {mname}: {n_tok} patch tokens is not a square grid")
            del model
            continue
        stride = 224 // native
        print(f"\n=== {mname}: {n_tok} tokens -> native grid {native}x{native} (stride {stride})")

        for ds, tag in DATASETS:
            lab, unlab, val, test, meta = build(ds, tag, size=(224, 224))
            unlab.transform = None
            rng = np.random.default_rng(args.seed)
            ids = list(unlab.ids)
            rng.shuffle(ids)
            unlab.ids = ids[:args.n_images]

            for i in range(len(unlab)):
                s = unlab[i]
                img = s["image"]
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                with torch.no_grad():
                    h = model((img[None].cuda() - mean) / std)
                cls, pt = h[:, 0], h[:, 1:]
                ps = F.cosine_similarity(pt, cls[:, None], dim=-1)[0]
                ps = ps.reshape(native, native).float().cpu().numpy()
                gt = (s["label"].numpy() > 0).astype(np.float32)

                for grid in [None] + COMMON_GRIDS:
                    psg, gtg = score_at_grid(ps, gt, grid, stride)
                    if gtg.sum() == 0 or gtg.all():
                        continue
                    top = np.unravel_index(psg.argmax(), psg.shape)
                    rows.append(dict(
                        model=mname, stride=stride, grid=("native" if grid is None else grid),
                        dataset=ds, image=s["id"],
                        fg_fraction=round(float(gt.mean()), 6),
                        chance=round(float(gtg.mean()), 6),
                        pib=int(bool(gtg[top])),
                        auroc_fg=round(auroc(psg.ravel(), gtg.ravel()), 4)))
        del model
        torch.cuda.empty_cache()

    out = os.path.join(HERE, "outputs", "tables")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "j12_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'model':14s} {'grid':>7s} {'scope':12s} {'n':>5s} {'PiB':>7s} {'chance':>7s} "
          f"{'PiB-ch':>8s} {'AUROC_fg':>9s}")
    summ = []
    for mname in args.models.split(","):
        for grid in ["native"] + COMMON_GRIDS:
            for scope in ["ALL"] + [d for d, _ in DATASETS]:
                sel = [r for r in rows if r["model"] == mname and r["grid"] == grid
                       and (scope == "ALL" or r["dataset"] == scope)]
                if len(sel) < 10:
                    continue
                pb = np.array([r["pib"] for r in sel], float)
                ch = np.array([r["chance"] for r in sel], float)
                au = np.array([r["auroc_fg"] for r in sel], float)
                rec = dict(model=mname, grid=grid, scope=scope, n=len(sel),
                           pib=round(float(pb.mean()), 4),
                           chance=round(float(ch.mean()), 4),
                           pib_minus_chance=round(float(pb.mean() - ch.mean()), 4),
                           auroc_fg=round(float(np.nanmean(au)), 4))
                summ.append(rec)
                if scope == "ALL":
                    print(f"{mname:14s} {str(grid):>7s} {scope:12s} {len(sel):5d} "
                          f"{rec['pib']:7.3f} {rec['chance']:7.3f} "
                          f"{rec['pib_minus_chance']:+8.3f} {rec['auroc_fg']:9.4f}")
    with open(os.path.join(out, "j12_foundation_artifact.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"\nwrote {out}/j12_foundation_artifact.csv  ({len(summ)} rows incl. per-dataset)")
    print("The finding is real only if the ViT/ConvNet ordering SURVIVES at a matched grid.")


if __name__ == "__main__":
    main()
