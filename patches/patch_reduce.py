#!/usr/bin/env python
"""Add a switchable Dice reduction to BCP's losses, defaulting to upstream behaviour exactly.

Why: BCP reduces the unlabeled Dice differently on its two datasets. ACDC uses
`DiceLoss._dice_mask_loss`, which sums intersection/union over the WHOLE batch, so a slice
holding a small structure is a rounding error beside one holding a large structure. LA uses
`mask_DiceLoss`, which reduces per sample (N x C) and then averages, so every sample counts the
same regardless of size. J17 measured a strong size-quality correlation on ACDC (rho +0.22..+0.59,
6/6 cells) and none on LA (quartile gap +-0.008). This patch lets us swap only the reduction and
test whether that difference is causal.

Controlled by env BCP_DICE_REDUCE:
    native (default) - byte-equivalent to upstream
    sample           - per-sample normalised
    batch            - batch-pooled
"""
import re
import sys

P = "/dcs/large/u5535699/bcp-official/code/utils/losses.py"
s = open(P).read()

if "_BCP_REDUCE" in s:
    print("already patched")
    sys.exit(0)

header = '''import os as _os

# Reduction mode for the Dice terms; "native" reproduces upstream byte for byte.
_BCP_REDUCE = _os.environ.get("BCP_DICE_REDUCE", "native")
assert _BCP_REDUCE in ("native", "sample", "batch"), _BCP_REDUCE


'''

# put the header after the final top-level import of the file's import block
lines = s.split("\n")
last_import = max(i for i, l in enumerate(lines[:40]) if l.startswith(("import ", "from ")))
lines.insert(last_import + 1, "\n" + header.rstrip("\n"))
s = "\n".join(lines)

# ---- DiceLoss._dice_mask_loss : upstream is batch-pooled -------------------------------
old = '''    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss'''
new = '''    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        if _BCP_REDUCE == "sample":
            # Reduce over spatial dims only, so each sample's Dice is normalised by its own
            # object size and then averaged -- a small structure weighs as much as a large one.
            dims = tuple(range(1, score.dim()))
            intersect = torch.sum(score * target * mask, dim=dims)
            y_sum = torch.sum(target * target * mask, dim=dims)
            z_sum = torch.sum(score * score * mask, dim=dims)
            loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            return 1 - loss.mean()
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss'''
assert old in s, "DiceLoss._dice_mask_loss not found verbatim"
s = s.replace(old, new)

# ---- mask_DiceLoss.forward : upstream is per-sample -------------------------------------
old2 = '''        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()'''
new2 = '''        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        if _BCP_REDUCE == "batch":
            # Pool intersection/union across the batch before the ratio, so a sample's weight
            # becomes proportional to its object size -- what ACDC's DiceLoss does natively.
            inter = inter.sum(0)
            union = union.sum(0)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()'''
n2 = s.count(old2)
assert n2 >= 1, "mask_DiceLoss reduction block not found verbatim"
s = s.replace(old2, new2)

open(P, "w").write(s)
print(f"patched {P}  (mask_DiceLoss blocks replaced: {n2})")
