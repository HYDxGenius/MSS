"""Check A - does object size predict segmentation quality, on the benchmark we actually use?

Runs the reproduced official-BCP checkpoints over the official test splits and emits one row per
(case, class) with the GT foreground size and the Dice the model achieved. The premise under
test: small structures are harder AND carry little of the unlabeled loss, which in SSL is
self-reinforcing because the pseudo-labels themselves keep improving only where the gradient is.

Decisive column: LA. ACDC already showed rho ~ +0.44 in-house. LA is a single large structure,
so if the sign holds there too the effect is not an artefact of ACDC's multi-class anatomy.
A sign flip on LA stops this line.

Run on the cluster from bcp-official/code:
    python checkA.py
"""
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/dcs/large/u5535699"
BCP = f"{ROOT}/bcp-official/code"
OUT = f"{ROOT}/bcp-official/checkA_rows_variant.csv"


def dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    s = p.sum() + g.sum()
    return float("nan") if s == 0 else 2.0 * np.logical_and(p, g).sum() / s


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), 0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return (float((rx * ry).sum() / d) if d else float("nan")), len(x)


rows = []

# ---------------------------------------------------------------- ACDC (2D U-Net, 4 classes)
from networks.net_factory import BCP_net
from scipy.ndimage import zoom

for ln in (7,):
    ck = f"{BCP}/model/BCP/ACDC_s1337_rsample_{ln}_labeled/self_train/unet_best_model.pth"
    if not os.path.exists(ck):
        print(f"[skip] {ck}")
        continue
    net = BCP_net(in_chns=1, class_num=4)
    net.load_state_dict(torch.load(ck, map_location="cuda"))
    net.eval()
    ids = [l.strip() for l in open(f"{ROOT}/datasets/ssmis/ACDC/test.list") if l.strip()]
    for cid in ids:
        with h5py.File(f"{ROOT}/datasets/ssmis/ACDC/data/{cid}.h5", "r") as h:
            img, lab = np.asarray(h["image"]), np.asarray(h["label"])
        pred = np.zeros_like(lab)
        for s in range(img.shape[0]):
            sl = img[s]
            x, y = sl.shape
            inp = zoom(sl, (256 / x, 256 / y), order=0)
            t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                o = torch.argmax(torch.softmax(net(t), dim=1), dim=1).squeeze(0)
            pred[s] = zoom(o.cpu().numpy(), (x / 256, y / 256), order=0)
        for c in (1, 2, 3):
            g = (lab == c)
            if g.sum() == 0:
                continue
            rows.append(("ACDC", f"{ln}cases", cid, c, int(g.sum()), dice(pred == c, g)))
    del net
    torch.cuda.empty_cache()
    print(f"[done] ACDC labelnum={ln}")

# ---------------------------------------------------------------- LA (3D V-Net, binary)
from networks.net_factory import net_factory
from utils.test_3d_patch import test_single_case

for ln in (8,):
    ck = f"{BCP}/model/BCP/LA_s1337_rbatch_{ln}_labeled/self_train/VNet_best_model.pth"
    if not os.path.exists(ck):
        print(f"[skip] {ck}")
        continue
    net = net_factory(net_type="VNet", in_chns=1, class_num=2, mode="test")
    st = torch.load(ck, map_location="cuda")
    net.load_state_dict(st["net"] if isinstance(st, dict) and "net" in st else st)
    net.eval()
    ids = [l.strip() for l in open(f"{ROOT}/datasets/ssmis/LA/test.list") if l.strip()]
    for cid in ids:
        with h5py.File(f"{ROOT}/datasets/ssmis/LA/2018LA_Seg_Training Set/{cid}/mri_norm2.h5", "r") as h:
            img, lab = np.asarray(h["image"]), np.asarray(h["label"])
        pred, _ = test_single_case(net, img, 18, 4, (112, 112, 80), num_classes=2)
        rows.append(("LA", f"{ln}lab", cid, 1, int((lab > 0).sum()), dice(pred > 0, lab > 0)))
    del net
    torch.cuda.empty_cache()
    print(f"[done] LA labelnum={ln}")

with open(OUT, "w") as f:
    f.write("dataset,budget,case,cls,gt_size,dice\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")

print(f"\nwrote {len(rows)} rows -> {OUT}\n")
print(f"{'dataset':8}{'budget':9}{'cls':>4}{'n':>5}{'rho(size,Dice)':>16}"
      f"{'Dice Q1(small)':>16}{'Q4(large)':>11}{'gap':>9}{'massQ1%':>9}")
print("-" * 87)
import itertools
key = lambda r: (r[0], r[1], r[3])
for k, grp in itertools.groupby(sorted(rows, key=key), key=key):
    g = list(grp)
    sz = [r[4] for r in g]
    dc = [r[5] for r in g]
    rho, n = spearman(sz, dc)
    order = np.argsort(sz)
    q = max(1, len(order) // 4)
    q1 = np.nanmean([dc[i] for i in order[:q]])
    q4 = np.nanmean([dc[i] for i in order[-q:]])
    mass = sum(sz[i] for i in order[:q]) / max(1, sum(sz))
    print(f"{k[0]:8}{k[1]:9}{k[2]:>4}{n:>5}{rho:16.3f}{q1:16.4f}{q4:11.4f}"
          f"{q4-q1:+9.4f}{100*mass:8.2f}%")

# pooled over ACDC classes, which is how the loss actually sees them
for ds in ("ACDC", "LA"):
    for b in sorted({r[1] for r in rows if r[0] == ds}):
        g = [r for r in rows if r[0] == ds and r[1] == b]
        if not g:
            continue
        rho, n = spearman([r[4] for r in g], [r[5] for r in g])
        print(f"  pooled {ds:5} {b:9} rho={rho:+.3f}  n={n}")
