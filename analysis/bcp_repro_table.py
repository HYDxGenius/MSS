#!/usr/bin/env python
"""Compare our BCP official-code reproductions against the published numbers.

This table is the gate for the whole project: until our run of the authors' own code on the
authors' own splits lands on the authors' own numbers, no comparison we make is trustworthy.
Our previous in-house BCP sat ~7 Dice points below published on ACDC (deviation D1).

Reads the self_train log.txt that BCP writes under
    <bcp-official>/code/model/BCP/<DS>_<exp>_<labelnum>_labeled/self_train/log.txt
and reports the best validation/test Dice it recorded.

    python bcp_repro_table.py [--root /dcs/large/u5535699/bcp-official]
"""
import argparse
import glob
import os
import re

# Published values, BCP CVPR 2023 (arXiv:2305.00673v1), Tables 1 and 3.
# ACDC: Table 3, "Ours" row, Dice. LA: Table 1, "Ours" row, Dice.
PUBLISHED = {
    ("ACDC", 3): 87.59,
    ("ACDC", 7): 88.84,
    ("LA", 8): 89.62,
    ("LA", 16): 90.38,
}
# What the authors report for the supervised lower bound, for context on the gap we have to clear.
SUP_LOWER = {("ACDC", 3): 47.83, ("ACDC", 7): 79.41, ("LA", 8): 84.93, ("LA", 16): 86.03}

NUM = r"([0-9]*\.?[0-9]+)"


def best_from_log(path):
    """BCP logs a line per evaluation; pull every dice-like number and keep the max."""
    if not os.path.exists(path):
        return None, 0
    best, n = None, 0
    pat = re.compile(r"(?:best[_ ]?dice|dice_sample|mean_dice|performance)\D{0,12}" + NUM, re.I)
    for line in open(path, errors="replace"):
        for m in pat.finditer(line):
            v = float(m.group(1))
            if v > 1.0:            # some logs are in %, some in [0,1]
                v /= 100.0
            if 0.0 < v <= 1.0:
                n += 1
                best = v if best is None else max(best, v)
    return best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/dcs/large/u5535699/bcp-official")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.root, "code", "model", "BCP", "*_labeled"))):
        m = re.match(r"(ACDC|LA)_(.+)_(\d+)_labeled$", os.path.basename(d))
        if not m:
            continue
        ds, exp, ln = m.group(1), m.group(2), int(m.group(3))
        best, n = best_from_log(os.path.join(d, "self_train", "log.txt"))
        rows.append((ds, ln, exp, best, n))

    if not rows:
        print(f"no BCP snapshot dirs under {args.root}/code/model/BCP - nothing has finished yet")
        return

    print(f"{'dataset':8}{'labels':>7}{'exp':>10}{'ours':>9}{'published':>11}{'gap':>8}"
          f"{'sup lower':>11}{'evals':>7}")
    print("-" * 71)
    for ds, ln, exp, best, n in sorted(rows):
        pub = PUBLISHED.get((ds, ln))
        sup = SUP_LOWER.get((ds, ln))
        o = f"{best*100:9.2f}" if best is not None else "      n/a"
        p = f"{pub:11.2f}" if pub else "        n/a"
        g = f"{best*100 - pub:+8.2f}" if (best is not None and pub) else "     n/a"
        s = f"{sup:11.2f}" if sup else "        n/a"
        print(f"{ds:8}{ln:>7}{exp:>10}{o}{p}{g}{s}{n:>7}")
    print("\nPublished values: BCP CVPR 2023 Table 1 (LA) and Table 3 (ACDC), 'Ours' row.")
    print("A gap inside roughly +-0.5 Dice means the pipeline reproduces; a systematic deficit")
    print("means the problem is on our side and must be fixed before any comparison is run.")


if __name__ == "__main__":
    main()
