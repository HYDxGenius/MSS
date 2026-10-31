#!/usr/bin/env python
"""One 4x6 panel: the two ACDC rows and the two LA rows of the marked-improvement blocks.

Cases and slices are not re-selected here -- they are the ones `make_figs.py` picked by its
stated rule, hard-coded so this figure is reproducible without re-running the search.

    python make_fig_combined.py --out ../figs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = "/dcs/large/u5535699"
SEG = ListedColormap([(0, 0, 0, 0), (0.00, 0.45, 0.70), (0.90, 0.62, 0.00), (0.00, 0.62, 0.45)])
C_FN, C_FP, C_CF = (0.84, 0.19, 0.15), (0.12, 0.47, 0.71), (0.58, 0.40, 0.74)
ERR = ListedColormap([(0, 0, 0, 0), C_FN, C_FP, C_CF])

ROWS = [("ACDC", "patient007_frame01", 2), ("ACDC", "patient007_frame02", 2),
        ("LA", "ZIJLJAVQ", 40), ("LA", "XYDLYJ5C", 45)]
EXPS = {"ACDC": ("s1337", "s1337_ds1", 7), "LA": ("s1337", "s1337_ds1_v2", 16)}
BENIGN = ("projection_head.", "prediction_head.", "contrastive_class_selector")


def load(dataset, exp, labelnum, ds):
    if dataset == "ACDC":
        from networks.unet import UNet_2d
        from networks.unet_ds import UNetDS
        net = (UNetDS(1, 4) if ds else UNet_2d(in_chns=1, class_num=4)).cuda()
        f = "unet"
    else:
        from networks.VNet import VNet
        from networks.vnet_ds import VNetDS
        net = (VNetDS(1, 2, normalization="batchnorm", has_dropout=False) if ds
               else VNet(n_channels=1, n_classes=2, normalization="batchnorm",
                         has_dropout=False)).cuda()
        f = "VNet"
    ck = f"model/BCP/{dataset}_{exp}_{labelnum}_labeled/self_train/{f}_best_model.pth"
    st = torch.load(ck, map_location="cuda")
    st = st.get("net", st) if isinstance(st, dict) else st
    _, unexpected = net.load_state_dict(st, strict=False)
    bad = [k for k in unexpected if not k.startswith(BENIGN)]
    if bad:
        raise SystemExit(f"ABORT {ck}: {len(bad)} unexpected keys, e.g. {bad[:3]}")
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


def signed_error(pred, gt):
    pf, gf = pred > 0, gt > 0
    e = np.zeros(gt.shape, dtype=np.uint8)
    e[gf & ~pf] = 1
    e[~gf & pf] = 2
    e[gf & pf & (pred != gt)] = 3
    return e


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="../figs")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

nets = {d: (load(d, EXPS[d][0], EXPS[d][2], False), load(d, EXPS[d][1], EXPS[d][2], True))
        for d in ("ACDC", "LA")}

panels = []
for dataset, case, z in ROWS:
    if dataset == "ACDC":
        with h5py.File(f"{ROOT}/datasets/ssmis/ACDC/data/{case}.h5", "r") as h:
            img, gt = h["image"][:], h["label"][:]
        pa = predict_acdc(nets["ACDC"][0], img)
        pb = predict_acdc(nets["ACDC"][1], img)
    else:
        from utils.test_3d_patch import test_single_case
        ids = [l.strip() for l in open(f"{ROOT}/datasets/ssmis/LA/test.list") if l.strip()]
        full = next(i for i in ids if i.startswith(case))
        with h5py.File(f"{ROOT}/datasets/ssmis/LA/2018LA_Seg_Training Set/{full}/mri_norm2.h5",
                       "r") as h:
            img, gt = h["image"][:], h["label"][:]
        pa = test_single_case(nets["LA"][0], img, 18, 4, (112, 112, 80), num_classes=2)
        pb = test_single_case(nets["LA"][1], img, 18, 4, (112, 112, 80), num_classes=2)
        pa = (pa[0] if isinstance(pa, tuple) else pa).astype(np.uint8).transpose(2, 0, 1)
        pb = (pb[0] if isinstance(pb, tuple) else pb).astype(np.uint8).transpose(2, 0, 1)
        img, gt = img.transpose(2, 0, 1), gt.transpose(2, 0, 1)
    panels.append((dataset, img[z], gt[z], pa[z], pb[z]))
    print(f"{dataset:5s} {case:20s} z={z}")

cols = ["Input", "Ground truth", "BCP", "BCP + MSS", "BCP error", "BCP + MSS error"]
fig, ax = plt.subplots(4, 6, figsize=(2.15 * 6, 2.25 * 4))
for r, (dataset, img, gt, pa, pb) in enumerate(panels):
    ov = [(None, None), (gt, SEG), (pa, SEG), (pb, SEG),
          (signed_error(pa, gt), ERR), (signed_error(pb, gt), ERR)]
    for c, (o, cm) in enumerate(ov):
        A = ax[r, c]
        A.imshow(img, cmap="gray", interpolation="nearest")
        if o is not None:
            A.imshow(o, cmap=cm, vmin=0, vmax=3, interpolation="nearest", alpha=0.85)
        A.set_xticks([]); A.set_yticks([])
        for sp in A.spines.values():
            sp.set_visible(False)
        if r == 0:
            A.set_title(cols[c], fontsize=11)
fig.tight_layout(rect=(0.035, 0.055, 1, 1))

# One dataset label per pair of rows, centred on the pair rather than on either row.
for name, pair in (("ACDC", (0, 1)), ("LA", (2, 3))):
    boxes = [ax[r, 0].get_position() for r in pair]
    y = (boxes[0].y1 + boxes[-1].y0) / 2
    fig.text(0.022, y, name, rotation=90, va="center", ha="center",
             fontsize=14, fontweight="bold")

fig.legend(handles=[Patch(facecolor=C_FN, label="false negative (missed)"),
                    Patch(facecolor=C_FP, label="false positive (spurious)"),
                    Patch(facecolor=C_CF, label="class confusion")],
           loc="lower center", ncol=3, frameon=False, fontsize=10)
for ext in ("pdf", "png"):
    p = os.path.join(a.out, f"qual_combined.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print("wrote", p)
