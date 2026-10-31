#!/usr/bin/env python
"""Pancreas-NIH -> h5, following CoraNet's preprocessing.

Steps are CoraNet's `preprocess/pancreas_preprocess.py`: resample to 1 mm isotropic, clip to
[-125, 275] HU, min-max normalise, crop to the label bounding box expanded by 25 voxels with a
96^3 floor, save image+label into <case>.h5 with datasets 'image' and 'label'.

One deviation, and it is the whole reason this script exists. CoraNet concatenates DICOM slices
in sorted-filename order, which suits the nested layout it expects. In this copy the .dcm files
sit flat inside PANCREAS_XXXX, and filename order matches neither ImagePositionPatient nor
InstanceNumber on any case checked. Hand-stacking by z then forces a direction choice that the
headers already encode: over ten cases an HU-plausibility test split 4/6 between ascending and
descending, and two cases were implausible either way.

So the series is read with SimpleITK's GDCM series reader, which applies the direction cosines.
With that, image and label come out on an identical grid (same size, spacing and direction) and
mean HU inside the label lands at 43-103 across the spot-checked cases -- soft tissue, as the
pancreas should be. No manual sorting and no flipping.

    python panc_prep.py --out <dir> [--cases N]
"""
import argparse
import os
import sys

import h5py
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

ROOT = "/dcs/large/u5535699/datasets/ssmis/Pancreas-NIH"
CT, LAB = f"{ROOT}/Pancreas-CT", f"{ROOT}/labels"
CLIP = (-125.0, 275.0)
OFFSET = 25
MIN_CROP = (96, 96, 96)
HU_LO, HU_HI = -20.0, 160.0          # plausible mean HU inside a pancreas label


def read_case(case):
    idx = case[-4:]
    d = os.path.join(CT, case)
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(d)
    if not ids:
        raise RuntimeError("no DICOM series")
    r = sitk.ImageSeriesReader()
    r.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(d, ids[0]))
    img = r.Execute()
    lab = sitk.ReadImage(os.path.join(LAB, f"label{idx}.nii.gz"))
    if img.GetSize() != lab.GetSize():
        raise RuntimeError(f"grid mismatch img{img.GetSize()} lab{lab.GetSize()}")
    # SimpleITK arrays are (z, y, x); spacing is (x, y, z)
    return (idx, sitk.GetArrayFromImage(img).astype(np.float32),
            sitk.GetArrayFromImage(lab).astype(np.uint8),
            tuple(img.GetSpacing()[::-1]))


def bbox_expand(mask, shape):
    i = np.array(np.nonzero(mask))
    lo = np.maximum(i.min(axis=1) - OFFSET, 0)
    hi = np.minimum(i.max(axis=1) + 1 + OFFSET, shape)
    for d in range(3):
        need = MIN_CROP[d] - (hi[d] - lo[d])
        if need > 0:
            lo[d] = max(0, lo[d] - need // 2)
            hi[d] = min(shape[d], lo[d] + MIN_CROP[d])
            lo[d] = max(0, hi[d] - MIN_CROP[d])
    return lo, hi


def process(case, out_dir):
    idx, img, mask, spacing = read_case(case)
    hu = float(img[mask > 0].mean()) if mask.any() else float("nan")
    warn = "" if HU_LO <= hu <= HU_HI else f"  <-- HU {hu:.0f} OUTSIDE [{HU_LO:.0f},{HU_HI:.0f}]"

    f = np.array(spacing, dtype=float) / 1.0              # to 1 mm isotropic
    img = zoom(img, f, order=1)
    mask = zoom(mask, f, order=0)
    img = np.clip(img, *CLIP)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    if mask.sum() == 0:
        return idx, "EMPTY MASK after resample"
    lo, hi = bbox_expand(mask, img.shape)
    sl = tuple(slice(lo[d], hi[d]) for d in range(3))
    ci, cm = img[sl], mask[sl]
    with h5py.File(os.path.join(out_dir, f"{idx}.h5"), "w") as h:
        h.create_dataset("image", data=ci.astype(np.float32))
        h.create_dataset("label", data=cm.astype(np.uint8))
    return idx, f"ok shape{ci.shape} fg{int(cm.sum())} HU{hu:7.1f}{warn}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/h5")
    ap.add_argument("--cases", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cases = sorted(c for c in os.listdir(CT) if c.startswith("PANCREAS_"))
    if a.cases:
        cases = cases[: a.cases]
    bad = warned = 0
    for c in cases:
        try:
            idx, msg = process(c, a.out)
        except Exception as e:
            idx, msg = c[-4:], f"ERROR {type(e).__name__}: {e}"
        if not msg.startswith("ok"):
            bad += 1
        if "OUTSIDE" in msg:
            warned += 1
        print(f"{idx}: {msg}", flush=True)
    print(f"\n{len(cases) - bad}/{len(cases)} written to {a.out}; {warned} HU warnings")


if __name__ == "__main__":
    sys.exit(main())
