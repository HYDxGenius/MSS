"""Scale-Invariant Boundary loss (SIB).

Motivation, measured in this project (J20) rather than assumed. Dice behaves like

    1 - Dice  ~=  d * S / (2V)

for a mean boundary displacement d, surface S and volume V. Because S/V ~ 1/r, the SAME
one-voxel boundary error is priced far more heavily on a small structure than on a large one.
On ACDC we measured rho(size, boundary displacement) in [-0.35, +0.08] while rho(size, Dice) is
+0.22..+0.67 -- i.e. the model's actual boundary accuracy is roughly size-independent and the
Dice gap is mostly the S/V factor. On LA the real error even grows with size (rho +0.43) while
Dice stays flat.

Consequence for training: a Dice-driven objective spends its boundary budget unevenly across
structures for purely geometric reasons. SIB adds a boundary term whose price per unit of
displacement is the same for every structure, by normalising the classic distance-map integral
by the target's own surface area:

    L_SIB = sum_k  ( sum_v  phi_k(v) * p_k(v) )  /  ( S_k + eps )

phi_k is the signed distance transform of the target mask for class k (negative inside,
positive outside) -- the standard construction from Kervadec et al.'s boundary loss, which we do
not claim. The contribution is the 1/S_k normalisation, which makes the term scale-invariant.

The distance transform is computed on the detached target (ground truth or pseudo-label), so no
gradient flows through it; only p_k is differentiable.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, distance_transform_edt


def _sdf_and_surface(mask):
    """Signed distance (negative inside) and surface-voxel count for one binary mask."""
    m = mask.astype(bool)
    if not m.any() or m.all():
        return None, 0
    outside = distance_transform_edt(~m)
    inside = distance_transform_edt(m)
    surf = int((m & ~binary_erosion(m)).sum())
    # Normalise by the image diagonal so phi lies in [-1, 1]. Without this the term's VALUE is
    # near zero (inside and outside contributions cancel) while its GRADIENT is d(phi*p)/dp =
    # phi, which reaches hundreds of voxels -- a smoke test diverged to CE 1875 at weight 1.
    # Dividing by the diagonal makes the weight interpretable and the gradient bounded.
    diag = float(np.sqrt(sum(d * d for d in m.shape)))
    return (outside - inside) / diag, surf


class SIBLoss(nn.Module):
    """Scale-invariant boundary loss over the foreground classes.

    Args:
        n_classes: including background, which is skipped.
        normalize: 'surface' (ours), 'none' (plain boundary loss, the ablation control).
    """

    def __init__(self, n_classes, normalize="surface", eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.normalize = normalize
        self.eps = eps

    def forward(self, prob, target, mask=None):
        """prob: (N,C,H,W) softmax. target: (N,H,W) int. mask: (N,H,W) 1 where the loss applies."""
        tgt = target.detach().cpu().numpy().astype(np.int64)
        N = tgt.shape[0]
        dev = prob.device
        total = prob.new_zeros(())
        count = 0
        for n in range(N):
            for k in range(1, self.n_classes):           # background carries no boundary of interest
                sdf, surf = _sdf_and_surface(tgt[n] == k)
                if sdf is None or surf == 0:
                    continue
                phi = torch.from_numpy(sdf).to(dev, dtype=prob.dtype)
                term = phi * prob[n, k]
                if mask is not None:
                    term = term * mask[n]
                denom = (surf + self.eps) if self.normalize == "surface" else 1.0
                total = total + term.sum() / denom
                count += 1
        return total / max(count, 1)
