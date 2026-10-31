#!/usr/bin/env python
"""Qualitative comparison: ground truth, BCP, BCP+MSS, and signed error maps.

Two choices here exist to stop the figure from being an argument from cherry-picking.

**Cases are stratified, not ranked.** Per-case Dice and 95HD are computed for both methods, and
the figure shows the `--n` cases where MSS improves 95HD most *and* the `--n` cases where it
improves least -- which on some cases means not at all, or worse. Both blocks are labelled, and
every row carries its own numbers. A reader who only believes the bottom block still sees it.

**The slice within a case is chosen by a stated rule**, not by eye: the slice with the largest
symmetric difference between the two methods' error sets, i.e. where they disagree most about
what they get wrong.

Error maps use one colour convention everywhere, for both datasets:

    false negative   ground truth is foreground here, the method is not
    false positive   the method claims foreground here, ground truth is background
    confusion        both call it foreground, but a different class (multi-class only)

    python make_figs.py --dataset ACDC --exp_a s1337 --exp_b s1337_ds1 --labelnum 7 --n 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = "/dcs/large/u5535699"
# Wong's colour-blind-safe palette; index 0 is transparent so the greyscale shows through
SEG = ListedColormap([(0, 0, 0, 0), (0.00, 0.45, 0.70), (0.90, 0.62, 0.00), (0.00, 0.62, 0.45)])
C_FN, C_FP, C_CF = (0.84, 0.19, 0.15), (0.12, 0.47, 0.71), (0.58, 0.40, 0.74)
ERR = ListedColormap([(0, 0, 0, 0), C_FN, C_FP, C_CF])
ERR_LEGEND = [Patch(facecolor=C_FN, label="false negative (missed)"),
              Patch(facecolor=C_FP, label="false positive (spurious)"),
              Patch(facecolor=C_CF, label="class confusion")]


def signed_error(pred, gt):
    """0 correct, 1 false negative, 2 false positive, 3 class confusion."""
    pf, gf = pred > 0, gt > 0
    e = np.zeros(gt.shape, dtype=np.uint8)
    e[gf & ~pf] = 1
    e[~gf & pf] = 2
    e[gf & pf & (pred != gt)] = 3
    return e


def build(dataset, ds):
    if dataset == "ACDC":
        from networks.unet import UNet_2d
        from networks.unet_ds import UNetDS
        return (UNetDS(1, 4) if ds else UNet_2d(in_chns=1, class_num=4)).cuda()
    from networks.VNet import VNet
    from networks.vnet_ds import VNetDS
    return (VNetDS(1, 2, normalization="batchnorm", has_dropout=False) if ds
            else VNet(n_channels=1, n_classes=2, normalization="batchnorm",
                      has_dropout=False)).cuda()


def load(dataset, exp, labelnum, ds):
    net = build(dataset, ds)
    ck = (f"model/BCP/{dataset}_{exp}_{labelnum}_labeled/self_train/"
          f"{'unet' if dataset == 'ACDC' else 'VNet'}_best_model.pth")
    st = torch.load(ck, map_location="cuda")
    st = st.get("net", st) if isinstance(st, dict) else st
    _, unexpected = net.load_state_dict(st, strict=False)
    if unexpected:
        raise SystemExit(f"ABORT {ck}: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    net.eval()
    return net


@torch.no_grad()
def predict_acdc(net, img):
    from scipy.ndimage import zoom
    pred = np.zeros(img.shape, dtype=np.uint8)
    for s in range(img.shape[0]):
        x, y = img[s].shape
        t = torch.from_numpy(zoom(img[s], (256 / x, 256 / y), order=0))
        o = net(t.unsqueeze(0).unsqueeze(0).float().cuda())
        o = o[0] if isinstance(o, (tuple, list)) else o
        o = torch.argmax(torch.softmax(o, 1), 1).squeeze(0).cpu().numpy()
        pred[s] = zoom(o, (x / 256, y / 256), order=0)
    return pred


def case_metrics(pred, gt, classes):
    from medpy import metric
    out = []
    for k in classes:
        p, g = pred == k, gt == k
        if g.sum() == 0:
            continue
        if p.sum() == 0:                       # empty prediction: hd95 is undefined
            out.append([0.0, np.nan])
            continue
        out.append([metric.binary.dc(p, g), metric.binary.hd95(p, g)])
    return np.nanmean(np.asarray(out, dtype=float), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ACDC", "LA"], required=True)
    ap.add_argument("--exp_a", required=True); ap.add_argument("--exp_b", required=True)
    ap.add_argument("--labelnum", type=int, required=True)
    ap.add_argument("--n", type=int, default=2, help="cases per block (2 blocks)")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    net_a = load(a.dataset, a.exp_a, a.labelnum, ds=False)
    net_b = load(a.dataset, a.exp_b, a.labelnum, ds=True)
    classes = (1, 2, 3) if a.dataset == "ACDC" else (1,)

    cases = []   # (name, img, gt, pa, pb, dice_a, dice_b, hd_a, hd_b)
    import h5py
    if a.dataset == "ACDC":
        names = [l.strip() for l in open(f"{ROOT}/datasets/ssmis/ACDC/test.list") if l.strip()]
        for c in names:
            with h5py.File(f"{ROOT}/datasets/ssmis/ACDC/data/{c}.h5", "r") as h:
                img, gt = h["image"][:], h["label"][:]
            pa, pb = predict_acdc(net_a, img), predict_acdc(net_b, img)
            cases.append((c, img, gt, pa, pb))
    else:
        from utils.test_3d_patch import test_single_case
        names = [l.strip() for l in open(f"{ROOT}/datasets/ssmis/LA/test.list") if l.strip()]
        for i in names:
            with h5py.File(f"{ROOT}/datasets/ssmis/LA/2018LA_Seg_Training Set/{i}/mri_norm2.h5",
                           "r") as h:
                img, gt = h["image"][:], h["label"][:]
            pa = test_single_case(net_a, img, 18, 4, (112, 112, 80), num_classes=2)
            pb = test_single_case(net_b, img, 18, 4, (112, 112, 80), num_classes=2)
            pa = (pa[0] if isinstance(pa, tuple) else pa).astype(np.uint8)
            pb = (pb[0] if isinstance(pb, tuple) else pb).astype(np.uint8)
            # LA volumes are (H, W, D); move depth first so slicing is uniform with ACDC
            cases.append((i[:8], img.transpose(2, 0, 1), gt.transpose(2, 0, 1),
                          pa.transpose(2, 0, 1), pb.transpose(2, 0, 1)))

    rows = []
    for name, img, gt, pa, pb in cases:
        da, ha = case_metrics(pa, gt, classes)
        db, hb = case_metrics(pb, gt, classes)
        rows.append(dict(name=name, img=img, gt=gt, pa=pa, pb=pb,
                         d_a=da * 100, d_b=db * 100, h_a=ha, h_b=hb,
                         d_gain=(db - da) * 100, h_gain=ha - hb))   # positive = MSS better
    # Stratify on the boundary metric, which is where the method's effect is largest.
    rows.sort(key=lambda r: -(r["h_gain"] if np.isfinite(r["h_gain"]) else -1e9))
    picked = [("marked improvement", r) for r in rows[:a.n]] + \
             [("limited improvement", r) for r in rows[-a.n:]]

    sel = []
    for tag, r in picked:
        scores = []
        for z in range(r["gt"].shape[0]):
            g = r["gt"][z]
            if g.max() == 0:
                continue
            scores.append((int(((r["pa"][z] != g) ^ (r["pb"][z] != g)).sum()), z))
        if not scores:
            continue
        sel.append((tag, r, max(scores)[1]))

    cols = ["Input", "Ground truth", "BCP", "BCP + MSS", "BCP error", "BCP + MSS error"]
    fig, ax = plt.subplots(len(sel), 6, figsize=(2.15 * 6, 2.35 * len(sel)))
    ax = np.atleast_2d(ax)
    log = []
    for i, (tag, r, z) in enumerate(sel):
        img, g, pa, pb = r["img"][z], r["gt"][z], r["pa"][z], r["pb"][z]
        panels = [(None, None), (g, SEG), (pa, SEG), (pb, SEG),
                  (signed_error(pa, g), ERR), (signed_error(pb, g), ERR)]
        for c, (ov, cm) in enumerate(panels):
            A = ax[i, c]
            A.imshow(img, cmap="gray", interpolation="nearest")
            if ov is not None:
                A.imshow(ov, cmap=cm, vmin=0, vmax=3, interpolation="nearest", alpha=0.85)
            A.set_xticks([]); A.set_yticks([])
            if i == 0:
                A.set_title(cols[c], fontsize=10)
        ax[i, 0].set_ylabel(f"{tag}\n{r['name']} z={z}\n"
                            f"Dice {r['d_a']:.1f}→{r['d_b']:.1f}\n"
                            f"95HD {r['h_a']:.1f}→{r['h_b']:.1f}", fontsize=7.5)
        log.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in r.items() if k not in ("img", "gt", "pa", "pb")}
                   | {"block": tag, "slice": int(z)})
        print(f"{tag:22s} {r['name']:12s} z={z:3d}  Dice {r['d_a']:.2f}->{r['d_b']:.2f}  "
              f"95HD {r['h_a']:.2f}->{r['h_b']:.2f}")

    fig.legend(handles=ERR_LEGEND, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{a.dataset}, {a.labelnum} labeled cases. Top block: the {a.n} cases where MSS "
                 f"improves 95HD most. Bottom block: the {a.n} where it improves least.\n"
                 f"Slice chosen per case as the one where the two methods' error sets differ most.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    for ext in ("pdf", "png"):
        p = os.path.join(a.out, f"qual_{a.dataset}_{a.labelnum}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight"); print("wrote", p)
    with open(os.path.join(a.out, f"qual_{a.dataset}_{a.labelnum}_cases.json"), "w") as fh:
        json.dump({"selected": log,
                   "all_cases": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                  for k, v in r.items() if k not in ("img", "gt", "pa", "pb")}
                                 for r in rows]}, fh, indent=1)


main()
