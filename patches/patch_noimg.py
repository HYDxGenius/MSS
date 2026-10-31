#!/usr/bin/env python
"""Stop BCP's training scripts from writing images to TensorBoard.

`wandb.init(sync_tensorboard=True)` mirrors every `add_image` into wandb media. BCP writes nine
image summaries per logging step on ACDC and two on LA, which came to 200-400 MB of media per
run and 3.3 GB on disk. The images are debug visualisations; no reported number depends on them.

Comments the calls out rather than deleting them, so the diff against upstream stays obvious,
and leaves every scalar summary untouched.
"""
import re
import sys

CHANGED = 0
for P in ("/dcs/large/u5535699/bcp-official/code/ACDC_BCP_train.py",
          "/dcs/large/u5535699/bcp-official/code/LA_BCP_train.py"):
    lines = open(P).read().split("\n")
    out, n = [], 0
    for ln in lines:
        if re.match(r"\s*writer\.add_images?\(", ln) and not ln.lstrip().startswith("#"):
            indent = ln[:len(ln) - len(ln.lstrip())]
            out.append(f"{indent}# [media disabled] {ln.strip()}")
            n += 1
        else:
            out.append(ln)
    if n:
        open(P, "w").write("\n".join(out))
        CHANGED += n
    print(f"{P}: {n} image summaries disabled")

# Lines that only built the tensors for those summaries are harmless (they just compute a
# snapshot and are never used again), so they are left alone -- removing them would risk
# touching variables the training loop still needs.
print(f"total {CHANGED}")
