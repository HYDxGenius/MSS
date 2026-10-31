#!/usr/bin/env python
"""J14 -- can the ValidityGate's v_t be estimated at training cadence?

J13 established two things: confidence validity varies a lot across datasets (fg-AUROC 0.28 to
0.85), and it can only be read off a HELD-OUT labelled split -- the training-labeled pixels
return 0.97-0.99 regardless of the truth.

That leaves one blocker before the gate can be built. At training cadence the gate cannot afford
a full pass over validation every step; it would see a handful of validation images, i.e. a few
thousand pixels. So:

    Is fg-AUROC estimated from N subsampled validation pixels stable enough that the gate's
    DECISION (admit-high / uniform / suppress) is stable?

The decision, not the number, is what has to be stable. A gate that oscillates between
"this score is great" and "this score is inverted" from step to step is unusable even if its
long-run average is right.

Reported per checkpoint and per N:
  mean, sd, and the 5th-95th percentile of the bootstrap AUROC
  decision_flip_rate -- the fraction of subsamples whose decision differs from the full-split
                        decision, at margin m
Falsifier: if the flip rate stays high at realistic N, the mechanism is unusable and the
direction stops here.

Usage: python j14_gate_estimability.py --run_id R025 [--margin 0.05]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

FRAMEWORK = "/home/u5535699/Desktop/dcslarge/ssmis-stability"
sys.path.insert(0, FRAMEWORK)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from j01_score_probe import ACDCTestSlices, auroc  # noqa: E402
from ssmis.data.build import build  # noqa: E402
from ssmis.nets.build import build as build_net  # noqa: E402

N_GRID = [500, 1000, 2000, 5000, 10000, 20000]
N_BOOT = 400


def decision(v, margin):
    if v > 0.5 + margin:
        return "admit_high"
    if v < 0.5 - margin:
        return "suppress"
    return "uniform"


def collect(run_id, max_images, rng):
    """Run the checkpoint over its validation split; return (score, correct) in the fg region."""
    manifest = os.path.join(HERE, "run_manifest.csv")
    run = next(r for r in csv.DictReader(open(manifest)) if r["run_id"] == run_id)
    ds = run["dataset"]
    tag = "7cases" if ds == "ACDC" else "10pct"

    ck_path = os.path.join(FRAMEWORK, "outputs", run_id, "best.pth")
    lab, unlab, val, test, meta = build(ds, tag)
    pop = ACDCTestSlices(val) if ds == "ACDC" else val
    ids = list(pop.ids)
    rng.shuffle(ids)
    pop.ids = ids[:max_images]
    pop.transform = None

    net = build_net("unet", meta["in_ch"], meta["n_classes"])
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["models"]["model"])
    net.cuda().eval()

    S, C = [], []
    with torch.no_grad():
        for i in range(len(pop)):
            s = pop[i]
            logits = net(s["image"][None].cuda())
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            conf = torch.softmax(logits, 1).max(1).values[0].cpu().numpy().ravel()
            pred = logits.argmax(1)[0].cpu().numpy().ravel()
            gt = s["label"].numpy().ravel()
            fg = (pred > 0) | (gt > 0)
            S.append(conf[fg])
            C.append((pred == gt)[fg])
    return ds, run, np.concatenate(S), np.concatenate(C), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_ids", default="R019,R025,R022")
    ap.add_argument("--max_images", type=int, default=100)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for run_id in args.run_ids.split(","):
        rng = np.random.default_rng(args.seed)
        ds, run, S, C, ck = collect(run_id, args.max_images, rng)
        full = auroc(S, C)
        full_dec = decision(full, args.margin)
        print(f"\n=== {run_id} {ds} ({run['labels']}) ckpt iter {ck['iter']}")
        print(f"    full validation split: {len(S):,} fg pixels, fg-AUROC {full:.4f} "
              f"-> decision '{full_dec}'")
        print(f"    {'N':>7s} {'mean':>7s} {'sd':>7s} {'p5':>7s} {'p95':>7s} "
              f"{'flip rate':>10s}")
        for n in N_GRID:
            if n > len(S):
                continue
            vals = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.choice(len(S), size=n, replace=False)
                vals[b] = auroc(S[idx], C[idx])
            flips = float(np.mean([decision(v, args.margin) != full_dec for v in vals]))
            print(f"    {n:7d} {vals.mean():7.4f} {vals.std():7.4f} "
                  f"{np.percentile(vals, 5):7.4f} {np.percentile(vals, 95):7.4f} "
                  f"{flips:10.3f}")
            rows.append(dict(run_id=run_id, dataset=ds, labels=run["labels"],
                             full_auroc=round(full, 4), full_decision=full_dec,
                             n_pixels=n, mean=round(float(vals.mean()), 4),
                             sd=round(float(vals.std()), 4),
                             p5=round(float(np.percentile(vals, 5)), 4),
                             p95=round(float(np.percentile(vals, 95)), 4),
                             decision_flip_rate=round(flips, 4)))

    out = os.path.join(HERE, "outputs", "tables", "j14_gate_estimability.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    print("Gate is usable iff the flip rate is near zero at an N the training loop can afford.")


if __name__ == "__main__":
    main()
