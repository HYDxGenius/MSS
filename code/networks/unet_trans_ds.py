"""U-Net with a Transformer bottleneck and deep supervision, for BCP's ACDC backbone.

Design choice, and why it is this rather than Swin-UNet. Swin's window attention needs the
feature grid to divide by the window size; at ACDC's 256x256 with patch 4 the grid is 64, which
7 does not divide, so a Swin backbone would force either a window change or an input resize --
an extra variable in an ablation whose whole point is to isolate the backbone. A bottleneck
Transformer changes exactly one place: the encoder's deepest stage. Encoder, decoder, skips,
losses and data pipeline stay byte-identical to `networks.unet.UNet_2d`.

It also matches the paper's argument. The Transformer enriches the 1/16 stage, which is the
coarse end of the pyramid; the deep-supervision heads sit at 1/8, 1/4 and 1/2. The claim being
tested is that multi-scale supervision pays off more when the coarse scales carry more
information -- so the two components should interact rather than add.

`n_trans=0` reduces this to UNet_2d plus aux heads (the MSS-only arm), so one class covers three
of the four ablation rows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.unet import Decoder, Encoder


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block over a flattened feature map."""

    def __init__(self, dim, heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class UNetTransDS(nn.Module):
    def __init__(self, in_chns=1, class_num=4, n_trans=4, heads=8, grid=16):
        super().__init__()
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        ft = params['feature_chns']
        self.n_trans = n_trans
        self.grid = grid                          # 256x256 input -> 1/16 stage is 16x16
        if n_trans > 0:
            dim = ft[4]
            # Built here, not lazily in forward: a Parameter created during forward is never
            # seen by the optimizer, so the positional embedding would silently stay at zero.
            self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(n_trans)])
            self.trans_norm = nn.LayerNorm(dim)
        self.aux1 = nn.Conv2d(ft[3], class_num, 1)   # 1/8
        self.aux2 = nn.Conv2d(ft[2], class_num, 1)   # 1/4
        self.aux3 = nn.Conv2d(ft[1], class_num, 1)   # 1/2

    def _bottleneck(self, x4):
        if self.n_trans == 0:
            return x4
        b, c, h, w = x4.shape
        assert h * w == self.pos.shape[1], (
            f"bottleneck grid {h}x{w} does not match the positional embedding "
            f"({self.grid}x{self.grid}); pass grid=... for a different input size")
        t = x4.flatten(2).transpose(1, 2) + self.pos          # B, HW, C
        for blk in self.blocks:
            t = blk(t)
        t = self.trans_norm(t)
        return t.transpose(1, 2).reshape(b, c, h, w)

    def _decode(self, feats):
        d = self.decoder
        x0, x1, x2, x3, x4 = feats
        a = d.up1(x4, x3)
        b = d.up2(a, x2)
        c = d.up3(b, x1)
        last = d.up4(c, x0)
        return d.out_conv(last), (a, b, c)

    def forward(self, x):
        f = self.encoder(x)
        f[4] = self._bottleneck(f[4])
        out, _ = self._decode(f)
        return out

    def forward_ds(self, x):
        f = self.encoder(x)
        f[4] = self._bottleneck(f[4])
        out, taps = self._decode(f)
        size = out.shape[2:]
        aux = [F.interpolate(hd(t), size=size, mode='bilinear', align_corners=False)
               for hd, t in ((self.aux1, taps[0]), (self.aux2, taps[1]), (self.aux3, taps[2]))]
        return out, aux


def BCP_net_trans(in_chns=1, class_num=4, ema=False, n_trans=4):
    net = UNetTransDS(in_chns=in_chns, class_num=class_num, n_trans=n_trans).cuda()
    if ema:
        for p in net.parameters():
            p.detach_()
    return net
