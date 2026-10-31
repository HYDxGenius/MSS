#!/usr/bin/env python
"""Wire SIB + deep supervision into BCP's ACDC self-training stage.

Both default to off, so with no env vars the script stays the reproduction already validated
(ACDC 10% = 89.04). Switches:

    BCP_SIB   float, weight of the scale-invariant boundary term   (0 = off)
    BCP_SIBN  'surface' (ours) | 'none' (plain boundary loss, the ablation control)
    BCP_DS    float, weight of the deep-supervision term           (0 = off)

Applied to self-training only: that is where pseudo-labels exist and where the J20 motivation
lives. Pre-training is untouched, so every ablation arm shares one pre-trained initialisation.

Line-anchored rather than exact-match: the upstream file carries trailing whitespace that makes
verbatim matching brittle.
"""
import re
import sys

Q = "/dcs/large/u5535699/bcp-official/code/ACDC_BCP_train.py"
src = open(Q).read()
if "_BCP_SIB" in src:
    print("already patched")
    sys.exit(0)

lines = src.split("\n")


def find(pred, start=0, what=""):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise SystemExit(f"anchor not found: {what}")


# ---- 1. header: imports, flags, loss object --------------------------------------------
i = find(lambda l: l.strip().startswith("dice_loss = losses.DiceLoss"), 0, "dice_loss")
lines[i + 1:i + 1] = [
    "",
    "import os as _os",
    "from utils.sib_loss import SIBLoss",
    "from networks.unet_ds import BCP_net_ds",
    "",
    '_BCP_SIB = float(_os.environ.get("BCP_SIB", "0"))',
    '_BCP_SIBN = _os.environ.get("BCP_SIBN", "surface")',
    '_BCP_DS = float(_os.environ.get("BCP_DS", "0"))',
    "_sib = SIBLoss(n_classes=4, normalize=_BCP_SIBN) if _BCP_SIB > 0 else None",
    'print(f"[method] SIB={_BCP_SIB} ({_BCP_SIBN})  DS={_BCP_DS}")',
]

# ---- 2. checkpoint loading: the DS heads are not in the pre-trained state ---------------
i = find(lambda l: l.strip() == "net.load_state_dict(state['net'])", 0, "load_net strict")
lines[i] = "    net.load_state_dict(state['net'], strict=(_BCP_DS == 0))"
i = find(lambda l: l.strip() == "net.load_state_dict(state['net'])", i + 1, "load_net_opt strict")
lines[i] = "    net.load_state_dict(state['net'], strict=(_BCP_DS == 0))"
i = find(lambda l: l.strip() == "optimizer.load_state_dict(state['opt'])", 0, "opt load")
lines[i:i + 1] = [
    "    if _BCP_DS == 0:",
    "        optimizer.load_state_dict(state['opt'])",
    "    else:",
    "        # DS adds parameters, so the saved optimizer state no longer matches the param",
    "        # groups. SGD momentum rebuilds in a few iterations; a mismatched load is worse.",
    '        print("[method] DS on: optimizer state not restored (param set differs)")',
]

# ---- 3. self_train: DS-capable network ---------------------------------------------------
i = find(lambda l: l.strip().startswith("def self_train("), 0, "self_train")
j = find(lambda l: l.strip() == "model = BCP_net(in_chns=1, class_num=num_classes)", i, "model")
assert lines[j + 1].strip() == "ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)"
lines[j:j + 2] = [
    "    _maker = BCP_net_ds if _BCP_DS > 0 else BCP_net",
    "    model = _maker(in_chns=1, class_num=num_classes)",
    "    ema_model = _maker(in_chns=1, class_num=num_classes, ema=True)",
]

# ---- 4. self_train: auxiliary forward ----------------------------------------------------
k = find(lambda l: l.strip() == "out_unl = model(net_input_unl)", j, "forward")
assert lines[k + 1].strip() == "out_l = model(net_input_l)"
lines[k:k + 2] = [
    "            if _BCP_DS > 0:",
    "                out_unl, out_unl_aux = model.forward_ds(net_input_unl)",
    "                out_l, out_l_aux = model.forward_ds(net_input_l)",
    "            else:",
    "                out_unl = model(net_input_unl)",
    "                out_l = model(net_input_l)",
]

# ---- 5. self_train: add the two terms ----------------------------------------------------
m = find(lambda l: re.match(r"\s*loss = \(loss_dice \+ loss_ce\) / 2\s*$", l), k, "loss sum")
lines[m + 1:m + 1] = [
    "",
    "            # scale-invariant boundary term (ours): a one-voxel boundary error costs the",
    "            # same on a small structure as on a large one, which plain Dice does not do",
    "            # because its price scales with S/V (measured, J20).",
    "            if _sib is not None:",
    "                loss = loss + _BCP_SIB * 0.5 * (",
    "                    _sib(torch.softmax(out_unl, dim=1), unl_label)",
    "                    + _sib(torch.softmax(out_l, dim=1), l_label))",
    "",
    "            # deep supervision on the decoder's intermediate stages",
    "            if _BCP_DS > 0:",
    "                for _out, _tgt in ((out_unl_aux, unl_label), (out_l_aux, l_label)):",
    "                    for _w, _a in zip((0.4, 0.2, 0.1), _out):",
    "                        loss = loss + _BCP_DS * _w * 0.5 * (",
    "                            ce_loss(_a, _tgt.long())",
    "                            + dice_loss(torch.softmax(_a, dim=1), _tgt.unsqueeze(1)))",
]

open(Q, "w").write("\n".join(lines))
print("patched ACDC_BCP_train.py (SIB + DS, both default off)")
