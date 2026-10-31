"""Photometric strong augmentation for the weak-to-strong branch.

BCP applies the same (weak) augmentation to both streams, so the FixMatch/UniMatch
weak-to-strong consistency that dominates generic SSL is simply absent from it. This module
supplies the strong view.

Only photometric perturbations: the copy-paste mask and the mixed target are defined on the
pixel grid, so any geometric transform would have to be applied to the mask as well and would
change what the loss is comparing. Intensity-only keeps the correspondence exact.

Everything is batched on GPU and parameter-free.
"""
import torch


def strong_augment(x, strength=1.0, generator=None):
    """x: (N,C,...) normalised intensities. Returns an independently perturbed copy.

    Per-sample brightness, contrast, gamma and additive Gaussian noise. Ranges follow the usual
    medical-imaging strong-augmentation settings (nnU-Net's intensity transforms), scaled by
    `strength` so the ablation can sweep it without touching the code.
    """
    n = x.shape[0]
    shape = (n,) + (1,) * (x.dim() - 1)

    def u(lo, hi):
        r = torch.rand(shape, device=x.device, dtype=x.dtype, generator=generator)
        return lo + (hi - lo) * r

    out = x
    # brightness: additive shift
    out = out + strength * u(-0.15, 0.15)
    # contrast: scale around the per-sample mean
    m = out.mean(dim=tuple(range(1, out.dim())), keepdim=True)
    out = m + (out - m) * (1.0 + strength * u(-0.25, 0.25))
    # gamma, applied on a min-max normalised copy so it is defined for negative inputs
    lo = out.amin(dim=tuple(range(1, out.dim())), keepdim=True)
    hi = out.amax(dim=tuple(range(1, out.dim())), keepdim=True)
    rng = (hi - lo).clamp(min=1e-5)
    g = 1.0 + strength * u(-0.3, 0.3)
    out = ((out - lo) / rng).clamp(0, 1).pow(g) * rng + lo
    # additive Gaussian noise
    out = out + strength * 0.05 * torch.randn(x.shape, device=x.device, dtype=x.dtype,
                                              generator=generator)
    return out
