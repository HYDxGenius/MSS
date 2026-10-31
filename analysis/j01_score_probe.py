#!/usr/bin/env python
"""J01 + J02 -- the GO/NO-GO probe for STABLE-SSMIS.

QUESTION: does a reliability score rank CORRECT model predictions above INCORRECT ones, at the
pixel level? If the Stability Score cannot do that on dense decoder features, the method has no
basis and dies here, before any GPU time goes into the matrix.

POPULATION: the **unlabeled** training subset. Not val -- val is what the checkpoint was
selected on, so its error distribution has already been optimised against. The unlabeled train
pixels are exactly the ones the method would admit pseudo-labels for, and they were never used
for supervision or for selection. Their ground truth is read HERE ONLY, as probe truth.

WHAT IS REPORTED
  J01  AUROC of each score for predicting per-pixel correctness
         - all pixels, and restricted to the decision-relevant region (pred>0 or gt>0),
           because an all-pixel AUROC is dominated by easy background
  J02  degeneracy checks: constant-score maps, score spread, and the admitted-error rate at the
       admission quantiles the method would actually use

Scores are compared ACROSS FEATURE TAPS, because 'which feature map' was an open [measure] item
and the channel depth differs by 16x between taps (enc4: 129 rFFT bins, dec4: 9).

Usage:  python j01_score_probe.py --run_id R019 [--max_images 120] [--pixels_per_image 4000]
"""
import argparse, csv, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

FRAMEWORK = "/home/u5535699/Desktop/dcslarge/ssmis-stability"
sys.path.insert(0, FRAMEWORK)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ssmis import scores as S                                    # noqa: E402
from ssmis.data.build import SPLITS_ROOT, build                  # noqa: E402
from ssmis.data.datasets import read_list                        # noqa: E402
from ssmis.nets.build import build as build_net                  # noqa: E402
from ssmis.nets.tap import Tapped, UNET_TAPS                     # noqa: E402

TAPS = ["enc4", "dec1", "dec2", "dec3", "dec4"]
PRED_SCORES = ["softmax_max", "teacher_entropy"]
FEAT_SCORES = ["feature_cosine", "stability"]
# The proposal listed reduce and bandwidth as open [measure] items, so the probe must cover
# them before the score can be declared dead -- they were written down as open BEFORE any run.
STAB_VARIANTS = [dict(reduce="vote", K=32, bandwidth=0.25),
                 dict(reduce="vote", K=8,  bandwidth=0.25),
                 dict(reduce="mean", bandwidth=0.25),
                 dict(reduce="mean", bandwidth=0.10),
                 dict(reduce="mean", bandwidth=0.50)]


def variant_tag(v):
    return "stability[" + ",".join(f"{k}={v[k]}" for k in sorted(v)) + "]"


class ACDCTestSlices:
    """Flattens ACDC test VOLUMES into individual slices, resized like training slices.

    Needed so ACDC can be probed out of sample on the same per-slice footing as ISIC and
    Kvasir. Exposes the `.ids` / `__getitem__` surface the probe expects.
    """

    def __init__(self, vol_ds, size=(224, 224)):
        from ssmis.data.transforms import resize_to, to_tensor
        self._resize, self._to_tensor = resize_to, to_tensor
        self._vol = vol_ds
        self._size = size
        # id -> (volume index, slice index). A dict, not a list, because the caller shuffles
        # and truncates `.ids`; positional lookup would silently return the wrong slice.
        self._where = {}
        for vi in range(len(vol_ds)):
            n = vol_ds[vi]["image"].shape[0]
            for si in range(n):
                self._where[f"{vol_ds.ids[vi]}_s{si}"] = (vi, si)
        self.ids = list(self._where)
        self.transform = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        key = self.ids[i]
        vi, si = self._where[key]
        s = self._vol[vi]
        img = s["image"][si].numpy()[None].astype("float32")
        lab = s["label"][si].numpy().astype("int64")
        img, lab = self._resize(img, lab, self._size)
        img, lab = self._to_tensor(img, lab)
        return dict(image=img, label=lab, id=key)


def auroc(score, positive):
    """Rank-based AUROC. positive = 1 where the prediction is CORRECT."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties, otherwise ties inflate or deflate the statistic
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def risk_at_coverage(score, correct, quantiles=(0.5, 0.8, 0.95)):
    """Error rate among ADMITTED pixels at each admission quantile -- what the method controls."""
    out = {}
    for q in quantiles:
        tau = np.quantile(score, q)
        adm = score > tau
        out[f"err@q{q}"] = float(1.0 - correct[adm].mean()) if adm.any() else float("nan")
        out[f"cov@q{q}"] = float(adm.mean())
    out["err_all"] = float(1.0 - correct.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--max_images", type=int, default=120)
    ap.add_argument("--pixels_per_image", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="unlabeled",
                    choices=["unlabeled", "test", "labeled", "val"],
                    help="'unlabeled' is the right population for a semi-supervised run. For a "
                         "sup_ref_budget_matched run every training image WAS labeled, so the "
                         "unlabeled pool is in-sample and 'test' is the only honest choice.")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "tables"))
    args = ap.parse_args()

    manifest = os.path.join(HERE, "run_manifest.csv")
    run = next(r for r in csv.DictReader(open(manifest)) if r["run_id"] == args.run_id)
    ds = run["dataset"]
    tag = ("7cases" if ds == "ACDC" else "10pct")

    ck_path = os.path.join(FRAMEWORK, "outputs", args.run_id, "best.pth")
    if not os.path.exists(ck_path):
        raise SystemExit(f"no checkpoint at {ck_path} -- train {args.run_id} first")

    lab, unlab, val, test, meta = build(ds, tag)
    if args.split == "test":
        # ACDC test items are whole volumes; flatten them to slices so all three datasets are
        # probed on the same per-slice footing, out of sample.
        unlab = ACDCTestSlices(test) if ds == "ACDC" else test
    elif args.split == "val":
        # The repair for the falsified labeled-batch estimator: validation is labelled AND out
        # of sample, so its error rate is comparable to the unlabeled pool. Using it for the
        # gate means val influences training, not just checkpoint selection -- a protocol
        # change that has to be declared, never slipped in.
        unlab = ACDCTestSlices(val) if ds == "ACDC" else val
    elif args.split == "labeled":
        # The ValidityGate assumption under test: does validity measured where labels exist
        # predict validity where they do not? This split is IN-SAMPLE by construction -- that
        # is the point, since the gate would also have to estimate from in-sample pixels.
        unlab = lab
    rng = np.random.default_rng(args.seed)
    ids = list(unlab.ids)
    rng.shuffle(ids)
    ids = ids[:args.max_images]
    unlab.ids = ids
    unlab.transform = None                      # probe the model, not the augmentation

    base = build_net("unet", meta["in_ch"], meta["n_classes"])
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    base.load_state_dict(ck["models"]["model"])
    base.cuda().eval()
    print(f"[{args.run_id}] {ds}: checkpoint from iter {ck['iter']}, "
          f"best val dice {ck['extra']['best']['dice_mean']:.4f}")
    print(f"probing {len(ids)} {args.split.upper()} images, "
          f"{args.pixels_per_image} px each, {meta['n_classes']} classes")

    # one tapped wrapper per tap, all sharing the same weights
    tapped = {t: Tapped(build_net("unet", meta["in_ch"], meta["n_classes"]), UNET_TAPS[t][0])
              for t in TAPS}
    for t in TAPS:
        tapped[t].net.load_state_dict(ck["models"]["model"])
        tapped[t].cuda().eval()

    cols = {f"{s}": [] for s in PRED_SCORES}
    cols.update({f"feature_cosine@{t}": [] for t in TAPS})
    cols.update({f"{variant_tag(v)}@{t}": [] for v in STAB_VARIANTS for t in TAPS})
    correct, fg_region, degenerate = [], [], {k: 0 for k in cols}

    with torch.no_grad():
        for n, i in enumerate(range(len(unlab))):
            smp = unlab[i]
            x = smp["image"][None].cuda()
            gt = smp["label"].numpy()
            logits, _ = tapped[TAPS[0]](x)
            pred = logits.argmax(1)[0].cpu().numpy()
            hw = logits.shape[-2:]

            per_img = {}
            for s in PRED_SCORES:
                per_img[s] = S.compute(s, logits=logits, out_hw=hw)[0].cpu().numpy()
            for t in TAPS:
                _, f = tapped[t](x)
                per_img[f"feature_cosine@{t}"] = S.compute(
                    "feature_cosine", logits=logits, feats=f, out_hw=hw)[0].cpu().numpy()
                for var in STAB_VARIANTS:
                    per_img[f"{variant_tag(var)}@{t}"] = S.compute(
                        "stability", logits=logits, feats=f, out_hw=hw,
                        **var)[0].cpu().numpy()

            ok = (pred == gt).ravel()
            fg = ((pred > 0) | (gt > 0)).ravel()
            k = min(args.pixels_per_image, ok.size)
            sel = rng.choice(ok.size, size=k, replace=False)
            correct.append(ok[sel]); fg_region.append(fg[sel])
            for key, v in per_img.items():
                vv = v.ravel()
                if np.ptp(vv) < 1e-12:
                    degenerate[key] += 1
                cols[key].append(vv[sel])
            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(unlab)}", flush=True)

    correct = np.concatenate(correct); fg_region = np.concatenate(fg_region)
    rows = []
    for key, chunks in cols.items():
        v = np.concatenate(chunks)
        r = dict(run_id=args.run_id, dataset=ds, score=key,
                 n_pixels=int(v.size),
                 auroc_all=round(auroc(v, correct), 4),
                 auroc_fg=round(auroc(v[fg_region], correct[fg_region]), 4),
                 degenerate_images=degenerate[key],
                 score_std=round(float(v.std()), 6))
        r.update({k: round(x, 4) for k, x in
                  risk_at_coverage(v, correct.astype(float)).items()})
        rows.append(r)

    rows.sort(key=lambda r: -(r["auroc_fg"] if np.isfinite(r["auroc_fg"]) else -1))
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, f"j01_score_probe_{args.run_id}.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n  overall pixel error rate: {1 - correct.mean():.4f}"
          f"   (foreground region: {1 - correct[fg_region].mean():.4f})")
    print(f"\n{'score':46s} {'AUROC all':>10s} {'AUROC fg':>10s} {'err@q0.8':>9s} {'degen':>6s}")
    for r in rows:
        print(f"{r['score']:46s} {r['auroc_all']:10.4f} {r['auroc_fg']:10.4f} "
              f"{r['err@q0.8']:9.4f} {r['degenerate_images']:6d}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
