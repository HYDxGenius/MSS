#!/usr/bin/env python
"""Stage-aware re-extraction of every reported ACDC/LA number.

BCP runs pre-training and self-training in one process and the iteration counter restarts, so
`grep mean_dice | sort -g | tail -1` returns the max over BOTH stages. For the plain-boundary
arm that silently reported a pre-training value (0.8303) for a self-training run that had in
fact collapsed to 0.0. Every number extracted that way has to be re-checked.

Prints, per log: the best over pre-training, the best over self-training, and the final
self-training value. The reported number must be the self-training one.
"""
import glob
import os
import re
import sys

R = "/dcs/large/u5535699/bcp-official/logs"
VAL = re.compile(r"iteration (\d+) : mean_dice : ([0-9.eE+-]+)")
LA_CKPT = re.compile(r"(pre_train|self_train)/iter_(\d+)_dice_([0-9.]+)\.pth")
STAGE2 = re.compile(r"Start self_training|Start self-training|self_train", re.I)


def scan(path):
    pre, self_, stage = {}, {}, 0
    for line in open(path, errors="replace"):
        m = LA_CKPT.search(line)
        if m:
            (self_ if m.group(1) == "self_train" else pre)[int(m.group(2))] = float(m.group(3))
            continue
        if "Start self_training" in line or "Start self-training" in line:
            stage = 1
            continue
        m = VAL.search(line)
        if m:
            (self_ if stage else pre)[int(m.group(1))] = float(m.group(2))
    return pre, self_


names = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{R}/*.out"))
print(f"{'log':22} {'pre best':>10} {'self best':>11} {'self final':>11} {'n_self':>7}  flag")
print("-" * 76)
for n in names:
    p = f"{R}/{n}.out"
    if not os.path.exists(p):
        continue
    pre, self_ = scan(p)
    if not pre and not self_:
        continue
    pb = max(pre.values()) if pre else float("nan")
    sb = max(self_.values()) if self_ else float("nan")
    sf = self_[max(self_)] if self_ else float("nan")
    flag = ""
    if self_ and pre and pb > sb:
        flag = "<-- pre-train beats self-train: a naive max would MISREPORT this run"
    if self_ and sb == 0.0:
        flag = "<-- self-training COLLAPSED"
    print(f"{n:22} {pb:10.4f} {sb:11.4f} {sf:11.4f} {len(self_):7d}  {flag}")
