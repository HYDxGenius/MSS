"""J20 - is the size-Dice effect a model deficiency or a geometric property of Dice?

Dice loss is approximately |A triangle B| / (|A| + |B|). If the predicted boundary sits a mean
distance d from the true boundary, the symmetric difference is about d * S with S the surface
area, so

    1 - Dice  ~=  d * S / (2V)  =  d * (S/V) / 2   and   S/V ~ 1/r

for a compact object of radius r. **A constant d therefore produces a Dice that falls with
shrinking size, with no model deficiency at all.** The quantity to regress on size is d, not
Dice.

This script emits, per (case, class):
    gt_size      V      GT foreground voxels
    gt_surface   S      GT boundary voxels (6/4-connectivity erosion)
    dice
    band         d_hat  |A triangle B| / S   -- mean boundary displacement in voxels
    asd, hd95           independent boundary metrics, for cross-checking d_hat

Decision rule, fixed before looking:
  * rho(size, band) ~ 0  and  rho(size, dice) > 0   -> the Dice effect is a METRIC ARTEFACT;
    small structures are not segmented worse, they are only scored worse. The "small structures
    are harder" framing is then wrong and must be dropped.
  * rho(size, band) clearly positive                 -> small structures really do get a wider
    error band; a genuine deficiency survives and is worth attacking.
  * rho(size, band) clearly negative                 -> small structures get a TIGHTER absolute
    band but still lose Dice: artefact, even more strongly.

Run from bcp-official/code.
"""
import itertools
import os
import sys

import h5py
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/dcs/large/u5535699"
BCP = f"{ROOT}/bcp-official/code"
OUT = f"{ROOT}/bcp-official/j20_rows.csv"


def surface(mask):
    """Boundary voxels of a binary mask: those that erosion removes."""
    if mask.sum() == 0:
        return np.zeros_like(mask)
    return mask & ~binary_erosion(mask)


def metrics(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    if g.sum() == 0:
        return None
    inter = np.logical_and(p, g).sum()
    dice = 2.0 * inter / (p.sum() + g.sum())
    sym = np.logical_xor(p, g).sum()          # |A triangle B|
    gs = surface(g)
    S = int(gs.sum())
    band = sym / S if S else np.nan           # mean boundary displacement, in voxels
    if p.sum() == 0:
        return dict(dice=dice, size=int(g.sum()), surf=S, band=band, asd=np.nan, hd95=np.nan)
    ps = surface(p)
    dt_g = distance_transform_edt(~gs)
    dt_p = distance_transform_edt(~ps)
    d_pg = dt_g[ps]                           # pred surface -> gt surface
    d_gp = dt_p[gs]
    both = np.concatenate([d_pg, d_gp])
    return dict(dice=dice, size=int(g.sum()), surf=S, band=band,
                asd=float(both.mean()), hd95=float(np.percentile(both, 95)))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return (float((rx * ry).sum() / d) if d else float("nan")), len(x)


rows = []

# ------------------------------------------------------------------ ACDC (2D U-Net, 4 classes)
from networks.net_factory import BCP_net

for tag, exp, ln in (("native", "s1337", 7), ("per-sample", "s1337_rsample", 7),
                     ("native", "s1337", 3)):
    ck = f"{BCP}/model/BCP/ACDC_{exp}_{ln}_labeled/self_train/unet_best_model.pth"
    if not os.path.exists(ck):
        print(f"[skip] {ck}"); continue
    net = BCP_net(in_chns=1, class_num=4)
    net.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
    net.eval()
    for cid in [l.strip() for l in open(f"{ROOT}/datasets/ssmis/ACDC/test.list") if l.strip()]:
        with h5py.File(f"{ROOT}/datasets/ssmis/ACDC/data/{cid}.h5", "r") as h:
            img, lab = np.asarray(h["image"]), np.asarray(h["label"])
        pred = np.zeros_like(lab)
        for s in range(img.shape[0]):
            x, y = img[s].shape
            t = torch.from_numpy(zoom(img[s], (256 / x, 256 / y), order=0)) \
                     .unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                o = torch.argmax(torch.softmax(net(t), 1), 1).squeeze(0).cpu().numpy()
            pred[s] = zoom(o, (x / 256, y / 256), order=0)
        for c in (1, 2, 3):
            m = metrics(pred == c, lab == c)
            if m:
                rows.append(dict(dataset="ACDC", arm=tag, budget=f"{ln}cases", case=cid,
                                 cls=c, **m))
    del net; torch.cuda.empty_cache()
    print(f"[done] ACDC {tag} ln={ln}", flush=True)

# ------------------------------------------------------------------ LA (3D V-Net, binary)
from networks.net_factory import net_factory
from utils.test_3d_patch import test_single_case

for tag, exp, ln in (("native", "s1337", 8), ("batch-pooled", "s1337_rbatch", 8),
                     ("native", "s1337", 16)):
    ck = f"{BCP}/model/BCP/LA_{exp}_{ln}_labeled/self_train/VNet_best_model.pth"
    if not os.path.exists(ck):
        print(f"[skip] {ck}"); continue
    net = net_factory(net_type="VNet", in_chns=1, class_num=2, mode="test")
    st = torch.load(ck, map_location="cuda", weights_only=False)
    net.load_state_dict(st["net"] if isinstance(st, dict) and "net" in st else st)
    net.eval()
    for cid in [l.strip() for l in open(f"{ROOT}/datasets/ssmis/LA/test.list") if l.strip()]:
        with h5py.File(f"{ROOT}/datasets/ssmis/LA/2018LA_Seg_Training Set/{cid}/mri_norm2.h5",
                       "r") as h:
            img, lab = np.asarray(h["image"]), np.asarray(h["label"])
        pred, _ = test_single_case(net, img, 18, 4, (112, 112, 80), num_classes=2)
        m = metrics(pred > 0, lab > 0)
        if m:
            rows.append(dict(dataset="LA", arm=tag, budget=f"{ln}lab", case=cid, cls=1, **m))
    del net; torch.cuda.empty_cache()
    print(f"[done] LA {tag} ln={ln}", flush=True)

keys = ["dataset", "arm", "budget", "case", "cls", "size", "surf", "dice", "band", "asd", "hd95"]
with open(OUT, "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(str(r[k]) for k in keys) + "\n")
print(f"\nwrote {len(rows)} rows -> {OUT}\n")

hdr = ("%-6s %-13s %-8s %4s %5s | %11s %11s %11s %11s" %
       ("ds", "arm", "budget", "cls", "n", "rho(sz,dice)", "rho(sz,band)",
        "rho(sz,asd)", "rho(sz,S/V)"))
print(hdr); print("-" * len(hdr))
key = lambda r: (r["dataset"], r["arm"], r["budget"], r["cls"])
for k, grp in itertools.groupby(sorted(rows, key=key), key=key):
    g = list(grp)
    sz = [r["size"] for r in g]
    sv = [r["surf"] / r["size"] for r in g]
    rd, n = spearman(sz, [r["dice"] for r in g])
    rb, _ = spearman(sz, [r["band"] for r in g])
    ra, _ = spearman(sz, [r["asd"] for r in g])
    rs, _ = spearman(sz, sv)
    print("%-6s %-13s %-8s %4d %5d | %11.3f %11.3f %11.3f %11.3f"
          % (k[0], k[1], k[2], k[3], n, rd, rb, ra, rs))

print("\n--- mean boundary displacement by size quartile (voxels) ---")
print("%-6s %-13s %-8s %4s | %8s %8s %8s %8s | %8s %8s"
      % ("ds", "arm", "budget", "cls", "bandQ1", "bandQ2", "bandQ3", "bandQ4", "diceQ1", "diceQ4"))
for k, grp in itertools.groupby(sorted(rows, key=key), key=key):
    g = sorted(grp, key=lambda r: r["size"])
    n = len(g); q = max(1, n // 4)
    qs = [g[:q], g[q:2*q], g[2*q:3*q], g[-q:]]
    b = [np.nanmean([r["band"] for r in x]) for x in qs]
    print("%-6s %-13s %-8s %4d | %8.3f %8.3f %8.3f %8.3f | %8.4f %8.4f"
          % (k[0], k[1], k[2], k[3], b[0], b[1], b[2], b[3],
             np.nanmean([r["dice"] for r in qs[0]]), np.nanmean([r["dice"] for r in qs[3]])))

print("\n--- predicted-vs-observed Dice from a CONSTANT band ---")
print("If the band is size-independent, Dice is fully determined by geometry:")
print("    1 - Dice ~= band * S / (2V).  Using each group's MEDIAN band for every case:")
print("%-6s %-13s %-8s %4s | %10s %10s %10s"
      % ("ds", "arm", "budget", "cls", "diceQ1_obs", "diceQ1_pred", "diceQ4_obs"))
for k, grp in itertools.groupby(sorted(rows, key=key), key=key):
    g = sorted(grp, key=lambda r: r["size"])
    n = len(g); q = max(1, n // 4)
    med = float(np.nanmedian([r["band"] for r in g]))
    q1 = g[:q]
    pred = np.nanmean([1 - med * r["surf"] / (2.0 * r["size"]) for r in q1])
    print("%-6s %-13s %-8s %4d | %10.4f %10.4f %10.4f"
          % (k[0], k[1], k[2], k[3], np.nanmean([r["dice"] for r in q1]), pred,
             np.nanmean([r["dice"] for r in g[-q:]])))
