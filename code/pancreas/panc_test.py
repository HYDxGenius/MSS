#!/usr/bin/env python
"""Evaluate a finished Pancreas run with BCP's own test settings.

`test_model` in train_pancreas.py loads `best_2.pth`, but self-training saves
`best_ema_{pct}_self.pth` -- a filename mismatch that raises FileNotFoundError after the full
260 epochs have already run, and is then swallowed by the script's bare `except Exception`.

It also matters which evaluation is being compared: the in-training evaluations use
`test_calculate_metric`'s default stride (s_xy=18), while `test_model` passes s_xy=16. The
published numbers presumably come from the finer stride, so this re-runs it properly instead of
quoting the in-training value.

    python panc_test.py [checkpoint] [--sxy 16] [--sz 4]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PANCREAS_DATA",
                      "/dcs/large/u5535699/datasets/ssmis/Pancreas-NIH/h5")
os.environ.setdefault("PANCREAS_LISTS",
                      "/dcs/large/u5535699/datasets/ssmis/Pancreas-NIH/data_lists")

import torch

import torch.nn as nn

from dataloaders import Pancreas
from test_util import test_calculate_metric
from Vnet import VNet

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?",
                default="result/cutmix/self_train/best_ema_20_self.pth")
ap.add_argument("--sxy", type=int, default=16)
ap.add_argument("--sz", type=int, default=4)
ap.add_argument("--pct", type=int, default=20)
a = ap.parse_args()

# Build exactly as dataloaders.create_Vnet does: bare VNet() defaults (no normalization), then
# wrapped in DataParallel -- the checkpoint's keys carry the resulting "module." prefix, and it
# has no BatchNorm buffers at all.
net = nn.DataParallel(VNet()).cuda()
net.load_state_dict(torch.load(a.ckpt)["net"])
net.eval()
print(f"loaded {a.ckpt}")

testset = Pancreas(os.environ["PANCREAS_DATA"], "pancreas", split="test", labelp=a.pct)
avg, per_case = test_calculate_metric(net, testset, s_xy=a.sxy, s_z=a.sz)
print(f"\nstride s_xy={a.sxy} s_z={a.sz}, {len(testset.image_list)} test cases")
print(f"  Dice    {avg[0]:.4f}   (published 0.8291)")
print(f"  Jaccard {avg[1]:.4f}   (published 0.7097)")
print(f"  95HD    {avg[2]:.4f}   (published 6.43)")
print(f"  ASD     {avg[3]:.4f}   (published 2.25)")
