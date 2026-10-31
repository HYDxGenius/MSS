#!/usr/bin/env python
"""Wire deep supervision into BCP's Pancreas pipeline.

    BCP_DS  float, weight of the auxiliary loss (0 = off, reproducing the validated baseline)

The pancreas trainer differs from the ACDC/LA ones: it is epoch-based, wraps its net in
DataParallel, and its net returns a list indexed `[0]`. So the net is built through
`create_Vnet` in dataloaders.py, and the forward sites in `ema_cutmix` take `forward_ds` off the
unwrapped module.

Only self-training is touched; pre-training keeps the stock network, so both arms share one
pre-trained initialisation exactly as on ACDC and LA.
"""
import re
import sys

# ---- 1. dataloaders.create_Vnet: build the DS variant when asked -------------------------
P = "/dcs/large/u5535699/bcp-official/code/pancreas/dataloaders.py"
s = open(P).read()
if "_PANC_DS" not in s:
    old = """    net = VNet()
    net = nn.DataParallel(net)"""
    new = """    import os as _os
    _PANC_DS = float(_os.environ.get("BCP_DS", "0"))
    if _PANC_DS > 0:
        from vnet_ds import VNetDS
        net = VNetDS()
    else:
        net = VNet()
    net = nn.DataParallel(net)"""
    assert old in s, "create_Vnet body not found"
    open(P, "w").write(s.replace(old, new))
    print("patched dataloaders.create_Vnet")
else:
    print("dataloaders already patched")

# ---- 2. train_pancreas: auxiliary loss in self-training ----------------------------------
Q = "/dcs/large/u5535699/bcp-official/code/pancreas/train_pancreas.py"
lines = open(Q).read().split("\n")
if any("_BCP_DS" in l for l in lines):
    print("train_pancreas already patched")
    sys.exit(0)


def find(pred, start=0, what=""):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise SystemExit(f"anchor not found: {what}")


i = find(lambda l: l.startswith("label_percent = "), 0, "label_percent")
lines[i + 1:i + 1] = [
    'from vnet_ds import unwrap',
    '_BCP_DS = float(os.environ.get("BCP_DS", "0"))',
    'print(f"[method] pancreas DS={_BCP_DS}")',
]

a = find(lambda l: l.strip() == "mix_output_l = net(net3_input_l)[0]", 0, "forward l")
b = find(lambda l: l.strip() == "mix_output_2 = net(net3_input_unlab)[0]", a, "forward unlab")
assert "loss_1 = mix_loss" in lines[a + 1] and "loss_2 = mix_loss" in lines[b + 1]

ind = " " * (len(lines[a]) - len(lines[a].lstrip()))

# Replace each forward rather than adding one: calling forward_ds on top of the existing call
# would run the network twice per step and double the cost of the arm being measured.
lines[b:b + 1] = [
    ind + "if _BCP_DS > 0:",
    ind + "    _o2, _aux_u = unwrap(net).forward_ds(net3_input_unlab)",
    ind + "    mix_output_2 = _o2[0]",
    ind + "else:",
    ind + "    mix_output_2 = net(net3_input_unlab)[0]",
]
lines[a:a + 1] = [
    ind + "if _BCP_DS > 0:",
    ind + "    _o1, _aux_l = unwrap(net).forward_ds(net3_input_l)",
    ind + "    mix_output_l = _o1[0]",
    ind + "else:",
    ind + "    mix_output_l = net(net3_input_l)[0]",
]

# both replacements grew the file by 4 lines each; re-find the loss_2 line to append after it
c = find(lambda l: l.strip().startswith("loss_2 = mix_loss("), a, "loss_2")
lines[c + 1:c + 1] = [
    "",
    ind + "# deep supervision: the same mixed targets, applied to three decoder stages",
    ind + "if _BCP_DS > 0:",
    ind + "    for _w, _a1, _a2 in zip((0.4, 0.2, 0.1), _aux_l, _aux_u):",
    ind + "        loss_1 = loss_1 + _BCP_DS * _w * mix_loss(",
    ind + "            _a1, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)",
    ind + "        loss_2 = loss_2 + _BCP_DS * _w * mix_loss(",
    ind + "            _a2, lab_a, uimg_b_plab.long(), loss_mask)",
]

open(Q, "w").write("\n".join(lines))
print(f"patched train_pancreas.py: forwards at {a}/{b}, aux loss after {c}")
