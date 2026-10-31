"""Check B - how much foreground does a BCP copy-paste operation actually transfer?

Zero model. Replicates BCP's own mask generators exactly:
  ACDC  generate_mask : random 2/3 x 2/3 window (170x170 in 256x256), uniform position
  LA    context_mask  : random 2/3 cube (74x74x53 in 112x112x80) inside the random 112x112x80 crop

For each training image we draw N random masks as training would, and record how much GT
foreground falls inside the pasted window. A paste that carries no foreground transfers only
background statistics - it still costs an iteration.
"""
import glob, os, sys
import numpy as np

RNG = np.random.default_rng(0)
ROOT = "/home/u5535699/Desktop/dcslarge/datasets/ssmis"
SPL = "/home/u5535699/Desktop/papers/med-ssl-mathematics/04_experiments/outputs/splits"
N_DRAWS = 20


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def report(name, inside, total_fg, n_img):
    inside = np.asarray(inside, dtype=float)
    frac = np.asarray(total_fg, dtype=float)
    captured = np.divide(inside, frac, out=np.zeros_like(inside), where=frac > 0)
    print(f"\n### {name}  ({n_img} images x {N_DRAWS} random masks = {len(inside)} pastes)")
    print(f"  pastes carrying ZERO foreground     : {100*np.mean(inside == 0):6.2f}%")
    print(f"  pastes carrying <1% of the window   : {100*np.mean(inside < 0.01*WIN):6.2f}%")
    print(f"  foreground captured by the window   : mean {100*captured.mean():5.1f}%"
          f"  p5 {100*pct(captured,5):5.1f}%  median {100*pct(captured,50):5.1f}%"
          f"  p95 {100*pct(captured,95):5.1f}%")
    print(f"  window is {100*WIN/IMG:.1f}% of the image volume"
          f"  -> a size-blind paste would capture ~{100*WIN/IMG:.1f}% on average")


# ---------------- ACDC ----------------
import h5py
ids = [l.strip() for l in open(f"{SPL}/ACDC/train_slices.list") if l.strip()]
H = W = 256
PX, PY = int(H * 2 / 3), int(W * 2 / 3)
IMG, WIN = H * W, PX * PY
inside, tot = [], []
for sid in ids:
    f = f"{ROOT}/ACDC/data/slices/{sid}.h5"
    if not os.path.exists(f):
        continue
    with h5py.File(f, "r") as h:
        lab = np.asarray(h["label"])
    if lab.shape != (H, W):                     # RandomGenerator resizes to 256x256
        from scipy.ndimage import zoom
        lab = zoom(lab, (H / lab.shape[0], W / lab.shape[1]), order=0)
    fg = (lab > 0)
    s = fg.sum()
    if s == 0:
        continue
    ii = np.cumsum(np.cumsum(fg.astype(np.int64), 0), 1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    for _ in range(N_DRAWS):
        w = RNG.integers(0, H - PX)
        h_ = RNG.integers(0, W - PY)
        v = ii[w+PX, h_+PY] - ii[w, h_+PY] - ii[w+PX, h_] + ii[w, h_]
        inside.append(v); tot.append(s)
report("ACDC  (2D, 170x170 window in 256x256, fg = any of RV/Myo/LV)", inside, tot, len(tot)//N_DRAWS)

# ---------------- LA ----------------
ids = [l.strip() for l in open(f"{ROOT}/LA/train.list") if l.strip()]
CX = CY, CZ = 112, 80
PX, PY, PZ = int(112*2/3), int(112*2/3), int(80*2/3)
IMG, WIN = 112*112*80, PX*PY*PZ
inside, tot = [], []
for sid in ids:
    f = f"{ROOT}/LA/2018LA_Seg_Training Set/{sid}/mri_norm2.h5"
    if not os.path.exists(f):
        continue
    with h5py.File(f, "r") as h:
        lab = np.asarray(h["label"])
    sx, sy, sz = lab.shape
    for _ in range(N_DRAWS):
        # BCP first RandomCrops a 112x112x80 patch, then masks a 2/3 cube inside it
        if sx < 112 or sy < 112 or sz < 80:
            continue
        ox = RNG.integers(0, sx-112+1); oy = RNG.integers(0, sy-112+1); oz = RNG.integers(0, sz-80+1)
        crop = lab[ox:ox+112, oy:oy+112, oz:oz+80]
        s = int((crop > 0).sum())
        if s == 0:
            inside.append(0); tot.append(0); continue
        w = RNG.integers(0, 112-PX); h_ = RNG.integers(0, 112-PY); z = RNG.integers(0, 80-PZ)
        inside.append(int((crop[w:w+PX, h_:h_+PY, z:z+PZ] > 0).sum())); tot.append(s)
report("LA  (3D, 74x74x53 window in the 112x112x80 crop)", inside, tot, len(tot)//N_DRAWS)
