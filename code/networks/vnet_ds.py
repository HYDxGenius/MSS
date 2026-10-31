"""V-Net with deep supervision, drop-in for BCP's LA / Pancreas backbone.

Same encoder/decoder modules and weights as `networks.VNet.VNet`; the only addition is three
1x1x1 auxiliary heads on the decoder's intermediate stages, trilinearly upsampled to the input
resolution. The taps mirror the 2D version exactly:

    x6 -> 1/8 resolution, n_filters*8    x7 -> 1/4, n_filters*4    x8 -> 1/2, n_filters*2

`forward` returns `(out_seg, pooled_features)`, the same two-tuple upstream returns -- both the
training loop (`outputs, _ = model(x)`) and the sliding-window evaluator (`y1, _ = model(patch)`)
unpack it, so evaluation and checkpointing need no changes. `forward_ds` additionally returns
the auxiliary logits.
"""
import torch.nn as nn
import torch.nn.functional as F

from networks.VNet import Decoder, Encoder


class VNetDS(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization="batchnorm",
                 has_dropout=True, has_residual=False):
        super().__init__()
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout,
                               has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout,
                               has_residual)
        self.pool = nn.MaxPool3d(3, stride=2)          # upstream returns this as its 2nd output
        self.aux1 = nn.Conv3d(n_filters * 8, n_classes, 1)
        self.aux2 = nn.Conv3d(n_filters * 4, n_classes, 1)
        self.aux3 = nn.Conv3d(n_filters * 2, n_classes, 1)

    def _decode(self, features):
        d = self.decoder
        x1, x2, x3, x4, x5 = features
        x5_up = d.block_five_up(x5) + x4
        x6 = d.block_six(x5_up)
        x6_up = d.block_six_up(x6) + x3
        x7 = d.block_seven(x6_up)
        x7_up = d.block_seven_up(x7) + x2
        x8 = d.block_eight(x7_up)
        x8_up = d.block_eight_up(x8) + x1
        x9 = d.block_nine(x8_up)
        if d.has_dropout:
            x9 = d.dropout(x9)
        return d.out_conv(x9), (x6, x7, x8)

    def forward(self, x):
        feats = self.encoder(x)
        out, _ = self._decode(feats)
        return out, self.pool(feats[4])

    def forward_ds(self, x):
        feats = self.encoder(x)
        out, taps = self._decode(feats)
        size = out.shape[2:]
        aux = [F.interpolate(h(t), size=size, mode="trilinear", align_corners=False)
               for h, t in ((self.aux1, taps[0]), (self.aux2, taps[1]), (self.aux3, taps[2]))]
        return out, self.pool(feats[4]), aux
