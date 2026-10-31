"""UNet_2d with deep supervision, drop-in for BCP's ACDC backbone.

Identical encoder/decoder weights and topology to `networks.unet.UNet_2d`; the only addition is
three 1x1 auxiliary heads on the decoder's intermediate stages, bilinearly upsampled to the
input resolution. With the auxiliary weight set to 0 this is numerically the same network as
upstream, so the ablation's "no DS" arm needs no separate code path.

forward() returns the main logits, matching upstream. forward_ds() additionally returns the
auxiliary logits, so only the training script needs to know deep supervision exists.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.unet import Decoder, Encoder


class UNetDS(nn.Module):
    def __init__(self, in_chns, class_num):
        super().__init__()
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        ft = params['feature_chns']
        # one head per decoder stage, coarse -> fine
        self.aux1 = nn.Conv2d(ft[3], class_num, 1)
        self.aux2 = nn.Conv2d(ft[2], class_num, 1)
        self.aux3 = nn.Conv2d(ft[1], class_num, 1)

    def _decode(self, feature):
        d = self.decoder
        a = d.up1(feature[4], feature[3])
        b = d.up2(a, feature[2])
        c = d.up3(b, feature[1])
        last = d.up4(c, feature[0])
        return d.out_conv(last), (a, b, c)

    def forward(self, x):
        out, _ = self._decode(self.encoder(x))
        return out

    def forward_ds(self, x, detach_taps=False):
        out, (a, b, c) = self._decode(self.encoder(x))
        if detach_taps:
            # Gradient still reaches the heads, but not the trunk, so a run with the heads
            # attached follows exactly the trajectory it would without them.
            a, b, c = a.detach(), b.detach(), c.detach()
        size = out.shape[2:]
        aux = [F.interpolate(h(t), size=size, mode='bilinear', align_corners=False)
               for h, t in ((self.aux1, a), (self.aux2, b), (self.aux3, c))]
        return out, aux


def BCP_net_ds(in_chns=1, class_num=4, ema=False):
    net = UNetDS(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for p in net.parameters():
            p.detach_()
    return net
