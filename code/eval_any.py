#!/usr/bin/env python
"""Four-metric evaluation that also handles the deep-supervision checkpoints.

BCP's `test_ACDC.py` / `test_LA.py` construct the stock UNet / VNet, so they cannot load a
UNetDS / VNetDS checkpoint (extra aux heads, and the stock nets carry projection/contrastive
heads the DS variants do not). This builds whichever network the checkpoint actually is.

It keeps the official evaluation settings, which matter: both test scripts default to `nms=1`,
i.e. largest-connected-component post-processing, and the published numbers come from that path
rather than from the training loop's evaluation.

    python eval_any.py --dataset ACDC --exp s1337_ds1 --labelnum 7 --ds
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

ROOT = "/dcs/large/u5535699"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["ACDC", "LA"], required=True)
ap.add_argument("--exp", required=True)
ap.add_argument("--labelnum", type=int, required=True)
ap.add_argument("--ds", action="store_true", help="checkpoint is a deep-supervision variant")
ap.add_argument("--trans", type=int, default=0,
                help="checkpoint has a Transformer bottleneck with this many blocks")
ap.add_argument("--stage", default="self_train")
a = ap.parse_args()

snap = f"model/BCP/{a.dataset}_{a.exp}_{a.labelnum}_labeled/{a.stage}"

if a.dataset == "ACDC":
    from networks.unet import UNet_2d
    from networks.unet_ds import UNetDS
    from networks.unet_trans_ds import UNetTransDS
    if a.trans > 0:
        net = UNetTransDS(1, 4, n_trans=a.trans).cuda()
    elif a.ds:
        net = UNetDS(1, 4).cuda()
    else:
        net = UNet_2d(in_chns=1, class_num=4).cuda()
    ck = os.path.join(snap, "unet_best_model.pth")
else:
    from networks.VNet import VNet
    from networks.vnet_ds import VNetDS
    net = (VNetDS(1, 2, normalization="batchnorm", has_dropout=False) if a.ds
           else VNet(n_channels=1, n_classes=2, normalization="batchnorm",
                     has_dropout=False)).cuda()
    ck = os.path.join(snap, "VNet_best_model.pth")

state = torch.load(ck, map_location="cuda")
state = state.get("net", state) if isinstance(state, dict) else state
missing, unexpected = net.load_state_dict(state, strict=False)
net.eval()
print(f"loaded {ck}")
if unexpected:
    # Dropping checkpoint tensors means evaluating a different network from the one trained --
    # it produced a 7.7-point phantom regression once. Refuse rather than report a number.
    raise SystemExit(f"ABORT: {len(unexpected)} checkpoint keys have nowhere to go, e.g. "
                     f"{unexpected[:4]}. The network built here does not match the checkpoint.")
if missing:
    # Missing keys are benign here: the stock nets carry projection/contrastive heads that
    # training never touches.
    print(f"  note: {len(missing)} params left at init (unused heads), e.g. {missing[:2]}")

if a.dataset == "ACDC":
    import h5py
    from medpy import metric
    from scipy.ndimage import zoom

    def one(case):
        with h5py.File(f"{ROOT}/datasets/ssmis/ACDC/data/{case}.h5", "r") as h:
            img, lab = h["image"][:], h["label"][:]
        pred = np.zeros_like(lab)
        for s in range(img.shape[0]):
            x, y = img[s].shape
            t = torch.from_numpy(zoom(img[s], (256 / x, 256 / y), order=0))
            t = t.unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                o = net(t)
                o = o[0] if isinstance(o, (tuple, list)) else o
                o = torch.argmax(torch.softmax(o, 1), 1).squeeze(0).cpu().numpy()
            pred[s] = zoom(o, (x / 256, y / 256), order=0)
        out = []
        for c in (1, 2, 3):
            p, g = pred == c, lab == c
            if g.sum() == 0:
                continue
            out.append([metric.binary.dc(p, g), metric.binary.jc(p, g),
                        metric.binary.hd95(p, g), metric.binary.asd(p, g)])
        return np.mean(out, axis=0)

    cases = [l.strip() for l in open(f"{ROOT}/datasets/ssmis/ACDC/test.list") if l.strip()]
    avg = np.mean([one(c) for c in cases], axis=0)
else:
    os.environ["LA_ROOT"] = f"{ROOT}/datasets/ssmis/LA"
    from utils.test_3d_patch import test_all_case
    ids = [l.strip() for l in open(f"{ROOT}/datasets/ssmis/LA/test.list") if l.strip()]
    paths = [f"{ROOT}/datasets/ssmis/LA/2018LA_Seg_Training Set/{i}/mri_norm2.h5" for i in ids]
    # test_all_case writes `test_save_path + '../performance.txt'` unconditionally, even with
    # save_result=False, so a real directory has to be supplied or it raises on None + str.
    tmp = f"/tmp/la_eval_{os.getpid()}/sub/"
    os.makedirs(tmp, exist_ok=True)
    avg = test_all_case(net, paths, num_classes=2, patch_size=(112, 112, 80),
                        stride_xy=18, stride_z=4, save_result=False,
                        test_save_path=tmp, metric_detail=0, nms=1)
    avg = np.asarray(avg).ravel()[:4]

print(f"\n{a.dataset} {a.exp} labelnum={a.labelnum}  ds={a.ds}")
print(f"  Dice {avg[0]*100:.2f}   Jaccard {avg[1]*100:.2f}   95HD {avg[2]:.2f}   ASD {avg[3]:.2f}")
