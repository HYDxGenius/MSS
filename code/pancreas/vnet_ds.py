"""Pancreas V-Net with deep supervision.

Third port of the same component, after `networks/unet_ds.py` (ACDC, 2D) and
`networks/vnet_ds.py` (LA, 3D). The pancreas pipeline carries its own `Vnet.py`, whose decoder
is a method rather than a submodule and whose head is `self.branchs[0]` instead of `out_conv`,
so it needs its own subclass; the tap positions are the same three decoder stages:

    x6 -> 1/8, n_filters*8    x7 -> 1/4, n_filters*4    x8 -> 1/2, n_filters*2

`forward` keeps upstream's contract exactly -- it returns a list, and the training loop indexes
`net(x)[0]`. `forward_ds` additionally returns the auxiliary logits.
"""
import torch.nn as nn
import torch.nn.functional as F

from Vnet import VNet


class VNetDS(VNet):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none',
                 has_dropout=False):
        super().__init__(n_channels, n_classes, n_filters, normalization, has_dropout)
        self.aux1 = nn.Conv3d(n_filters * 8, n_classes, 1)
        self.aux2 = nn.Conv3d(n_filters * 4, n_classes, 1)
        self.aux3 = nn.Conv3d(n_filters * 2, n_classes, 1)

    def _decode_taps(self, features):
        x1, x2, x3, x4, x5 = features
        x5_up = self.block_five_up(x5) + x4
        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6) + x3
        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7) + x2
        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8) + x1
        out = [branch(x8_up) for branch in self.branchs]
        return out, (x6, x7, x8)

    def forward_ds(self, x):
        out, taps = self._decode_taps(self.encoder(x))
        size = out[0].shape[2:]
        aux = [F.interpolate(h(t), size=size, mode='trilinear', align_corners=False)
               for h, t in ((self.aux1, taps[0]), (self.aux2, taps[1]), (self.aux3, taps[2]))]
        return out, aux


def unwrap(net):
    """DataParallel does not expose forward_ds; the pancreas pipeline always wraps its net."""
    return net.module if hasattr(net, "module") else net
