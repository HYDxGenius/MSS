#!/usr/bin/env python
"""Make BCP's Pancreas training script runnable here.

Unlike the LA and ACDC scripts it has no argparse: data root, GPU selection and the label
percentage are module-level constants. Four changes, none touching the method:

1. `data_root` -> PANCREAS_DATA (the h5 directory), default unchanged.
2. `os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'` is hardcoded for the author's two-GPU box and
   would select the wrong (or no) device under Slurm, which allocates one. Respect an existing
   CUDA_VISIBLE_DEVICES instead.
3. `label_percent` -> PANCREAS_PCT. Only 20 is usable here: CoraNet publishes a single pancreas
   split (12/50/18, the 20% setting) and no 10% split, so the 10% row cannot be reproduced
   without inventing one.
4. Dead stdlib imports (`turtle`, `cgi`, `audioop`, `asyncore`, `imp`) that nothing in the file
   uses. `turtle` needs tkinter, and the rest are removed in Python 3.12/3.13; none of their
   names are referenced.
"""
import re
import sys

P = "/dcs/large/u5535699/bcp-official/code/pancreas/train_pancreas.py"
src = open(P).read()
if "_PANC_DATA" in src:
    print("already patched")
    sys.exit(0)

lines = src.split("\n")
out, dropped = [], []
DEAD = ("from turtle import", "from cgi import", "from audioop import",
        "from asyncore import", "^import imp$", "from unittest import",
        "from multiprocessing import reduction", "from yaml import load")
for ln in lines:
    if any(re.match(p, ln) if p.startswith("^") else ln.startswith(p) for p in DEAD):
        dropped.append(ln.strip())
        out.append(f"# [removed: unused, and absent in newer Python] {ln.strip()}")
        continue
    out.append(ln)
s = "\n".join(out)

# GPU: Slurm already exposes exactly one device
old_gpu = "os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'"
assert old_gpu in s, "CUDA_VISIBLE_DEVICES line not found"
s = s.replace(old_gpu,
              "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # was '2,3' (author's box)")

# data root + label percentage
old_root = ("data_root, split_name = '/home/ubuntu/byh/code/CoraNet-master/preprocess/data', "
            "'pancreas'")
assert old_root in s, "data_root line not found"
s = s.replace(old_root,
              "_PANC_DATA = os.environ.get('PANCREAS_DATA',\n"
              "                            '/home/ubuntu/byh/code/CoraNet-master/preprocess/data')\n"
              "data_root, split_name = _PANC_DATA, 'pancreas'")

s = re.sub(r"^label_percent = 20$",
           "label_percent = int(os.environ.get('PANCREAS_PCT', '20'))",
           s, flags=re.M)

open(P, "w").write(s)
print(f"patched train_pancreas.py; removed {len(dropped)} dead imports:")
for d in dropped:
    print("   ", d)
