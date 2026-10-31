"""Scale-matched deep supervision.

Vanilla deep supervision applies the same auxiliary loss to every class at every decoder scale.
On ACDC the foreground structures span 4-10x in size within a single dataset (J17), so at 1/8
resolution a small structure has essentially no support left and the auxiliary head is being
asked to predict something the tensor no longer represents. In the semi-supervised setting the
two problems compound: small structures are both the ones that vanish first under downsampling
and the ones whose pseudo-labels are least reliable.

Scale-matched deep supervision weights each class at each auxiliary level by how much of that
class actually survives the corresponding downsampling:

    w[k, l] = |class k after downsampling by f_l, restored| / |class k|   in [0, 1]

A structure that disappears at a coarse level contributes nothing there; a large structure keeps
its full weight. No new parameters, and the weight is computed from the same target tensor the
auxiliary loss already uses, so it works identically for ground-truth and pseudo-labelled
regions.
"""
import torch
import torch.nn.functional as F


def survival_weights(target, n_classes, factor):
    """Per-sample, per-class fraction of a class that survives f-times downsampling.

    target: (N,H,W) integer labels. Returns (N,C) in [0,1].
    Nearest-neighbour down/up is deliberate: it is exactly the information an auxiliary head at
    that resolution can represent.
    """
    n = target.shape[0]
    oh = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2).float()   # N,C,H,W
    small = F.interpolate(oh, scale_factor=1.0 / factor, mode="nearest")
    back = F.interpolate(small, size=oh.shape[2:], mode="nearest")
    kept = (oh * back).sum(dim=(2, 3))
    total = oh.sum(dim=(2, 3))
    return kept / total.clamp(min=1.0)


def scale_matched_ds_loss(aux_logits, target, n_classes, ce_loss, dice_loss,
                          level_weights=(0.4, 0.2, 0.1), factors=(8, 4, 2), matched=True):
    """Auxiliary loss summed over decoder levels.

    matched=False reproduces vanilla deep supervision exactly, so the ablation's two arms share
    one code path and differ only in this flag.
    """
    total = aux_logits[0].new_zeros(())
    for lw, f, a in zip(level_weights, factors, aux_logits):
        if lw == 0:
            continue
        prob = torch.softmax(a, dim=1)
        if not matched:
            term = ce_loss(a, target.long()) + dice_loss(prob, target.unsqueeze(1))
            total = total + lw * 0.5 * term
            continue
        w = survival_weights(target, n_classes, f)            # N,C
        # Per-class Dice weighted by survival. CE is weighted per pixel by the survival of the
        # class that pixel belongs to, which is the same quantity gathered back onto the grid.
        wmap = w.gather(1, target.long().view(w.shape[0], -1)).view_as(target)
        ce_pix = F.cross_entropy(a, target.long(), reduction="none")
        ce = (ce_pix * wmap).sum() / wmap.sum().clamp(min=1.0)
        oh = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2).float()
        inter = (prob * oh).sum(dim=(2, 3))
        denom = (prob * prob).sum(dim=(2, 3)) + (oh * oh).sum(dim=(2, 3))
        dice_k = 1.0 - (2 * inter + 1e-10) / (denom + 1e-10)
        dsc = (w[:, 1:] * dice_k[:, 1:]).sum() / w[:, 1:].sum().clamp(min=1e-6)
        total = total + lw * 0.5 * (ce + dsc)
    return total
