#!/usr/bin/env python3
"""Aggregate the gradient-geometry probe into the tables of the theory section.

Input: one or more `gradgeom.jsonl` files written by `utils/dsprobe.py` during self-training.
Each file holds a header, then per-probe-step records of two kinds:

  geom  alignment a_l, norm ratio r_l, conflict rate q_l, pairwise c_ij, descent ratio rho
  sens  relative gradient displacement S_l under a corrupted target, per corruption level

Output: the per-tap table over the whole trajectory (median and inter-quartile range, because
the early iterations are not comparable to the converged regime), the same split into the first
and last thirds of training, and the corruption table.

    python3 j27_gradgeom.py probeM.jsonl [probeB.jsonl ...]
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

TAPS = ("main", "l=1 (s=8)", "l=2 (s=4)", "l=3 (s=2)")


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def iqr(xs):
    return f"{q(xs, .5):.3f} [{q(xs, .25):.3f}, {q(xs, .75):.3f}]"


def report(path):
    hdr, geom, sens = None, [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            {"header": lambda: None, "geom": lambda: geom.append(r),
             "sens": lambda: sens.append(r)}[r["kind"]]()
            if r["kind"] == "header":
                hdr = r
    if not geom:
        print(f"{path}: no probe records"); return
    its = [r["iter"] for r in geom]
    print(f"\n{'=' * 78}\n{path}")
    print(f"weights w = {hdr['weights']}   shared parameters: "
          f"{hdr['n_shared_scalars']:,} scalars in {hdr['n_shared_tensors']} tensors "
          f"({', '.join(hdr['shared_prefixes'])})")
    print(f"{len(geom)} probe steps, iterations {min(its)}-{max(its)}")

    def block(sub, label):
        print(f"\n-- {label}  (n={len(sub)})")
        print(f"{'tap':<12}{'alignment a_l':<26}{'norm ratio r_l':<26}{'conflict q_l':<26}")
        for i, name in enumerate(TAPS[1:]):
            print(f"{name:<12}{iqr([r['a'][i] for r in sub]):<26}"
                  f"{iqr([r['r'][i] for r in sub]):<26}"
                  f"{iqr([r['q'][i] for r in sub]):<26}")
        print(f"{'rho':<12}{iqr([r['rho'] for r in sub]):<26}"
              f"{'|g0+a|/|g0|':<14}{iqr([r['step_norm_ratio'] for r in sub])}")
        keys = sorted(sub[0]["c"])
        print("pairwise c_ij  " + "   ".join(
            f"c{k}={q([r['c'][k] for r in sub], .5):.3f}" for k in keys))

    block(geom, "whole trajectory")
    t = len(geom) // 3
    if t:
        block(geom[:t], "first third")
        block(geom[-t:], "last third")

    if sens:
        print("\n-- target-corruption sensitivity  S_l = |grad L_l(Y~) - grad L_l(Y)| / "
              "|grad L_l(Y)|, median over the trajectory")
        by = defaultdict(list)
        for r in sens:
            by[(r["mode"], r["level"])].append(r["S_rel"])
        print(f"{'corruption':<22}" + "".join(f"{n:<14}" for n in TAPS)
              + "ratio l=1 / main")
        for k in sorted(by):
            rows = by[k]
            med = [st.median([row[i] for row in rows]) for i in range(len(rows[0]))]
            print(f"{k[0] + ' ' + str(k[1]):<22}" + "".join(f"{v:<14.3f}" for v in med)
                  + f"{med[1] / med[0]:.3f}")


if not os.environ.get("J27_V2"):
 for p in sys.argv[1:]:
    report(p)


def report2(path):
    """The carrier-consistent descent-lemma record and the real-pseudo-label record."""
    car, pl = [], []
    for line in open(path):
        r = json.loads(line)
        if r["kind"] == "carrier":
            car.append(r)
        elif r["kind"] == "plabel":
            pl.append(r)
    print(f"\n{'=' * 78}\n{path}")
    if car:
        print(f"\n-- descent lemma on the LIVE training batch (n={len(car)}): "
              f"G_0 = grad L_BCP, D = G_0 + sum_l w_l G_l")
        print(f"{'tap':<12}{'alignment a_l':<26}{'norm ratio r_l':<26}{'conflict q_l':<26}")
        for i, name in enumerate(TAPS[1:]):
            print(f"{name:<12}{iqr([r['a'][i] for r in car]):<26}"
                  f"{iqr([r['r'][i] for r in car]):<26}"
                  f"{iqr([r['q'][i] for r in car]):<26}")
        print(f"{'rho':<12}{iqr([r['rho'] for r in car]):<26}"
              f"{'|D|/|G0|':<14}{iqr([r['step_norm_ratio'] for r in car])}")
        neg = sum(1 for r in car if r["rho"] <= 0)
        print(f"probe steps with rho <= 0 (D not a descent direction for L_BCP): {neg}/{len(car)}")
    if pl:
        print(f"\n-- teacher pseudo-label error on the labeled probe batch (n={len(pl)}), "
              f"band b={pl[0]['band']}")
        for k, lbl in (("err_rate", "error rate, all voxels"),
                       ("band_frac", "share of voxels in the boundary band"),
                       ("err_rate_band", "error rate inside the band"),
                       ("err_rate_interior", "error rate in the interior"),
                       ("err_share_in_band", "share of all errors inside the band"),
                       ("enrichment", "enrichment (P[err|band]/P[err])")):
            print(f"  {lbl:<42}{iqr([r[k] for r in pl])}")
        print(f"\n-- gradient response to the teacher's REAL errors, median over the trajectory")
        print(f"{'target':<26}" + "".join(f"{n:<14}" for n in TAPS) + "ratio l=1 / main")
        for tag, lbl in (("all", "all errors"), ("band", "boundary-band errors only"),
                         ("interior", "interior errors only")):
            med = [st.median([r[f"S_{tag}"][i] for r in pl]) for i in range(4)]
            print(f"{lbl:<26}" + "".join(f"{v:<14.3f}" for v in med)
                  + f"{med[1] / med[0]:.3f}")


if __name__ == "__main__" and os.environ.get("J27_V2"):
    for p in sys.argv[1:]:
        report2(p)
