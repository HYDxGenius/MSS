#!/usr/bin/env python
"""J11 - foreground-fraction profile across the three 2D datasets.

This is the mechanism axis of the paper. The lazy-aggregation hypothesis (LaStViT) says ViTs
route global semantics through background patches; if that is what hurts pseudo-labels, artifact
severity should track how much of the frame is background. These three datasets differ by close
to an order of magnitude in foreground fraction, which is what makes that testable.

No checkpoint required. Reads ground-truth masks only, over the PINNED train splits.

Outputs -> 04_experiments/outputs/
  tables/j11_foreground_profile.csv     per-dataset summary
  tables/j11_per_image.csv              per-image foreground fraction (feeds J03 scatter)
  figures/j11_foreground_profile.png
"""
import csv, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(HERE, "outputs", "splits")
TAB = os.path.join(HERE, "outputs", "tables")
FIG = os.path.join(HERE, "outputs", "figures")
ROOT = "/home/u5535699/Desktop/dcslarge/datasets/ssmis"

def read_list(ds, name):
    p = os.path.join(SPLITS, ds, name)
    return [l.strip() for l in open(p) if l.strip()]

def acdc_masks():
    """ACDC labels are multi-class: 0 bg, 1 RV, 2 Myo, 3 LV. Iterate the pinned train slices."""
    import h5py
    base = os.path.join(ROOT, "ACDC", "data", "slices")
    for sid in read_list("ACDC", "train_slices.list"):
        f = os.path.join(base, sid + ".h5")
        if not os.path.exists(f):
            continue
        with h5py.File(f, "r") as h:
            lab = np.asarray(h["label"])
        yield sid, lab

def png_masks(ds, listname, path_fn):
    from PIL import Image
    for i in read_list(ds, listname):
        p = path_fn(i)
        if not os.path.exists(p):
            continue
        a = np.asarray(Image.open(p).convert("L"))
        yield i, (a > 127).astype(np.uint8)

def profile(name, it, n_classes=1):
    rows, fracs = [], []
    per_class = {c: [] for c in range(1, n_classes + 1)}
    empty = 0
    for ident, m in it:
        fg = (m > 0)
        fr = float(fg.mean())
        fracs.append(fr)
        if fr == 0.0:
            empty += 1
        for c in per_class:
            per_class[c].append(float((m == c).mean()))
        rows.append(dict(dataset=name, image=ident, fg_fraction=round(fr, 6),
                         height=m.shape[0], width=m.shape[1]))
    a = np.array(fracs)
    if a.size == 0:
        return rows, None
    summ = dict(
        dataset=name, n_images=int(a.size),
        fg_mean=round(float(a.mean()), 5), fg_std=round(float(a.std()), 5),
        fg_p5=round(float(np.percentile(a, 5)), 5),
        fg_median=round(float(np.median(a)), 5),
        fg_p95=round(float(np.percentile(a, 95)), 5),
        fg_min=round(float(a.min()), 6), fg_max=round(float(a.max()), 5),
        empty_mask_images=empty,
        background_fraction_mean=round(float(1 - a.mean()), 5),
    )
    if n_classes > 1:
        for c, v in per_class.items():
            summ[f"class{c}_mean"] = round(float(np.mean(v)), 5)
    return rows, summ

def main():
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    all_rows, summaries = [], []

    r, s = profile("ACDC", acdc_masks(), n_classes=3)
    all_rows += r; summaries.append(s)

    kv = os.path.join(ROOT, "Kvasir-SEG", "masks")
    r, s = profile("Kvasir-SEG",
                   png_masks("Kvasir-SEG", "train.list", lambda i: os.path.join(kv, i + ".png")))
    all_rows += r; summaries.append(s)

    gtdirs = glob.glob(os.path.join(ROOT, "ISIC2018", "*GroundTruth*"))
    if gtdirs:
        gt = gtdirs[0]
        r, s = profile("ISIC2018",
                       png_masks("ISIC2018", "train.list",
                                 lambda i: os.path.join(gt, i + "_segmentation.png")))
        all_rows += r; summaries.append(s)
    else:
        print("WARNING: ISIC ground-truth directory not found; ISIC skipped", file=sys.stderr)

    summaries = [s for s in summaries if s]
    keys = sorted({k for s in summaries for k in s}, key=lambda k: (k != "dataset", k))
    with open(os.path.join(TAB, "j11_foreground_profile.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for s in summaries: w.writerow(s)
    with open(os.path.join(TAB, "j11_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "image", "fg_fraction", "height", "width"])
        w.writeheader(); w.writerows(all_rows)

    # figure: distribution of foreground fraction per dataset, log x
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [s["dataset"] for s in summaries]
    data = [[r["fg_fraction"] for r in all_rows if r["dataset"] == d and r["fg_fraction"] > 0]
            for d in order]
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=200)
    parts = ax.violinplot(data, vert=False, showextrema=False, widths=0.85)
    for b in parts["bodies"]:
        b.set_alpha(0.35)
    for i, d in enumerate(data, start=1):
        ax.plot(np.median(d), i, "o", ms=5, color="black", zorder=3)
    ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order)
    ax.set_xscale("log")
    ax.set_xlabel("foreground fraction of the frame (log scale, non-empty masks)")
    ax.set_title("J11 — foreground fraction, pinned train splits", fontsize=10)
    ax.grid(axis="x", alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "j11_foreground_profile.png"))

    print(json.dumps(summaries, indent=2))
    print("\nwrote:", os.path.join(TAB, "j11_foreground_profile.csv"))
    print("wrote:", os.path.join(FIG, "j11_foreground_profile.png"))

if __name__ == "__main__":
    main()
