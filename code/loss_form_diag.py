#!/usr/bin/env python
"""Why does dropping the region split buy +1.19 Dice? Scale, or direction?

No training. On one checkpoint and a fixed set of training batches, built exactly as the
self-training loop builds them, this computes for each auxiliary form and each tap:

  * the loss value;
  * the gradient norm on the parameters the taps share with the main head
    (encoder.* + decoder.up1.*, the same set the gradient probe uses);
  * the cosine similarity between that gradient and the unmasked form's gradient.

If the two forms produce gradients of similar size pointing in different directions, the
performance difference is about *what* is being optimised. If the sizes differ a lot, the
difference may be nothing more than an effective change of the auxiliary weight, and a
scale-matched control is needed before the formulation claim can stand.

    python loss_form_diag.py --exp s1337_ds1 --stage pre_train --batches 12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.unet_ds import UNetDS
from utils import losses

ROOT = "/dcs/large/u5535699"
SHARED = ("encoder.", "decoder.up1.")
NC, W = 4, (0.4, 0.2, 0.1)

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="s1337_ds1")
ap.add_argument("--stage", default="pre_train", choices=["pre_train", "self_train"])
ap.add_argument("--labelnum", type=int, default=7)
ap.add_argument("--batches", type=int, default=12)
ap.add_argument("--seed", type=int, default=1337)
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dice_loss = losses.DiceLoss(n_classes=NC)
ce_none = nn.CrossEntropyLoss(reduction="none")
ce_mean = nn.CrossEntropyLoss()


def mix_loss(out, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    """Verbatim from ACDC_BCP_train.py::mix_loss."""
    img_l, patch_l = img_l.long(), patch_l.long()
    soft = F.softmax(out, dim=1)
    iw, pw = (u_weight, l_weight) if unlab else (l_weight, u_weight)
    pm = 1 - mask
    d = dice_loss(soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * iw
    d += dice_loss(soft, patch_l.unsqueeze(1), pm.unsqueeze(1)) * pw
    c = iw * (ce_none(out, img_l) * mask).sum() / (mask.sum() + 1e-16)
    c += pw * (ce_none(out, patch_l) * pm).sum() / (pm.sum() + 1e-16)
    return (d + c) / 2


def unmasked(out, tgt):
    return 0.5 * (ce_mean(out, tgt.long()) + dice_loss(F.softmax(out, 1), tgt.unsqueeze(1)))


def generate_mask(img):
    b, _, x, y = img.shape
    lm = torch.ones(b, x, y).cuda(); m = torch.ones(x, y).cuda()
    px, py = int(x * 2 / 3), int(y * 2 / 3)
    w = np.random.randint(0, x - px); h = np.random.randint(0, y - py)
    m[w:w + px, h:h + py] = 0; lm[:, w:w + px, h:h + py] = 0
    return m.long(), lm.long()


net = UNetDS(1, NC).cuda()
ck = f"model/BCP/ACDC_{a.exp}_{a.labelnum}_labeled/{a.stage}/unet_best_model.pth"
st = torch.load(ck, map_location="cuda"); st = st.get("net", st)
# BCP's pre-training builds the stock UNet_2d, whose projection and contrastive heads the
# ACDC path never calls and UNetDS does not declare. Those keys are expected to have nowhere
# to go; anything else means the wrong network is being built, which is what the abort is for.
BENIGN_UNUSED = ("projection_head.", "prediction_head.", "contrastive_class_selector")
missing, unexpected = net.load_state_dict(st, strict=False)
bad = [k for k in unexpected if not k.startswith(BENIGN_UNUSED)]
if bad:
    raise SystemExit(f"ABORT: {len(bad)} unexpected keys, e.g. {bad[:3]}")
if unexpected:
    print(f"  {len(unexpected)} unused upstream head keys dropped (projection/contrastive)")
net.eval()
params = [p for n, p in net.named_parameters() if n.startswith(SHARED)]
print(f"checkpoint {ck}\nshared params: {sum(p.numel() for p in params):,} in {len(params)} tensors")
if missing:
    print(f"  {len(missing)} params left at init, e.g. {missing[:2]}")

db = BaseDataSets(base_dir=f"{ROOT}/datasets/ssmis/ACDC", split="train", num=None,
                  transform=transforms.Compose([RandomGenerator([256, 256])]))
n_lab = {3: 68, 7: 136}[a.labelnum]
loader = DataLoader(db, batch_sampler=TwoStreamBatchSampler(
    list(range(n_lab)), list(range(n_lab, len(db))), 24, 12), num_workers=4)


def flat(gs):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(gs, params)])


FORMS = ("unmasked", "region1", "carrier")
val = {f: [[] for _ in range(3)] for f in FORMS}
nrm = {f: [[] for _ in range(3)] for f in FORMS}
cos = {f: [[] for _ in range(3)] for f in FORMS}
agg_n = {f: [] for f in FORMS}; agg_c = {f: [] for f in FORMS}

for bi, sb in enumerate(loader):
    if bi >= a.batches:
        break
    vb, lb = sb["image"].cuda(), sb["label"].cuda()
    img_a, img_b = vb[:6], vb[6:12]
    uimg_a, uimg_b = vb[12:18], vb[18:]
    lab_a, lab_b = lb[:6], lb[6:12]
    with torch.no_grad():
        plab_a = torch.argmax(torch.softmax(net(uimg_a), 1), 1)
        plab_b = torch.argmax(torch.softmax(net(uimg_b), 1), 1)
    img_mask, loss_mask = generate_mask(img_a)
    unl_label = plab_a * img_mask + lab_a * (1 - img_mask)
    l_label = lab_b * img_mask + plab_b * (1 - img_mask)
    x_unl = uimg_a * img_mask + img_a * (1 - img_mask)
    x_l = img_b * img_mask + uimg_b * (1 - img_mask)

    _, aux_unl = net.forward_ds(x_unl)
    _, aux_l = net.forward_ds(x_l)

    for f in FORMS:
        per_tap = []
        for i in range(3):
            if f == "unmasked":
                L = unmasked(aux_unl[i], unl_label) + unmasked(aux_l[i], l_label)
            else:
                uw = 0.5 if f == "carrier" else 1.0
                L = (mix_loss(aux_unl[i], plab_a, lab_a, loss_mask, u_weight=uw, unlab=True)
                     + mix_loss(aux_l[i], lab_b, plab_b, loss_mask, u_weight=uw))
            g = flat(torch.autograd.grad(L, params, retain_graph=True, allow_unused=True))
            per_tap.append(g)
            val[f][i].append(float(L)); nrm[f][i].append(float(g.norm()))
        agg = sum(w * g for w, g in zip(W, per_tap))
        agg_n[f].append(float(agg.norm()))
        if f == "unmasked":
            ref_tap, ref_agg = [g.clone() for g in per_tap], agg.clone()
        else:
            for i in range(3):
                cos[f][i].append(float(torch.dot(ref_tap[i], per_tap[i])
                                       / (ref_tap[i].norm() * per_tap[i].norm() + 1e-12)))
            agg_c[f].append(float(torch.dot(ref_agg, agg) / (ref_agg.norm() * agg.norm() + 1e-12)))

m = lambda v: float(np.mean(v)); s = lambda v: float(np.std(v))
print(f"\n{a.batches} batches, checkpoint stage = {a.stage}\n")
print(f"{'form':<10}{'tap':<6}{'loss value':<20}{'|grad| on shared':<22}{'cos vs unmasked':<20}")
for f in FORMS:
    for i in range(3):
        c = f"{m(cos[f][i]):.4f} ± {s(cos[f][i]):.4f}" if f != "unmasked" else "1 (reference)"
        print(f"{f:<10}l={i+1:<4}{m(val[f][i]):.4f} ± {s(val[f][i]):<9.4f}"
              f"{m(nrm[f][i]):.4f} ± {s(nrm[f][i]):<11.4f}{c:<20}")
print(f"\naggregate a = sum_l w_l grad L_l")
print(f"{'form':<12}{'|a|':<24}{'|a| / |a_unmasked|':<22}{'cos vs unmasked'}")
ref = m(agg_n["unmasked"])
for f in FORMS:
    c = f"{m(agg_c[f]):.4f} ± {s(agg_c[f]):.4f}" if f != "unmasked" else "1 (reference)"
    print(f"{f:<12}{m(agg_n[f]):.4f} ± {s(agg_n[f]):<11.4f}{m(agg_n[f])/ref:<22.3f}{c}")
print(f"\nratio of gradient scale, region1 / unmasked, per tap: "
      + ", ".join(f"{m(nrm['region1'][i])/m(nrm['unmasked'][i]):.3f}" for i in range(3)))
