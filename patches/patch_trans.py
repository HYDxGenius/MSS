#!/usr/bin/env python
"""Wire the Transformer-bottleneck backbone into BCP's ACDC script.

    BCP_TRANS  int, number of Transformer blocks at the encoder's 1/16 stage (0 = plain U-Net)

Combined with the existing BCP_DS this gives the four ablation arms from one code path:

    BCP_DS=0 BCP_TRANS=0   plain BCP            (the validated reproduction, 89.27)
    BCP_DS=1 BCP_TRANS=0   + MSS
    BCP_DS=0 BCP_TRANS=4   + backbone
    BCP_DS=1 BCP_TRANS=4   full

Pre-training is left on the stock network in every arm so all arms share one initialisation and
the ablation isolates the self-training stage -- except when the backbone changes, where the
extra blocks simply have no pre-trained weights to load (non-strict, as for the DS heads).
"""
import re
import sys

P = "/dcs/large/u5535699/bcp-official/code/ACDC_BCP_train.py"
lines = open(P).read().split("\n")
if any("_BCP_TRANS" in l for l in lines):
    print("already patched")
    sys.exit(0)


def find(pred, start=0, what=""):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise SystemExit(f"anchor not found: {what}")


# ---- flag + import ------------------------------------------------------------------------
i = find(lambda l: l.startswith("_BCP_EMAEVAL"), 0, "EMAEVAL flag")
lines[i + 1:i + 1] = [
    "from networks.unet_trans_ds import UNetTransDS",
    '_BCP_TRANS = int(_os.environ.get("BCP_TRANS", "0"))',
]
i = find(lambda l: l.strip().startswith('print(f"[method] SIB='), 0, "method print")
lines[i] = lines[i][:-2] + ' TRANS={_BCP_TRANS}")'

# ---- network construction in self_train ----------------------------------------------------
st = find(lambda l: "def self_train(" in l, 0, "self_train")
j = find(lambda l: l.strip() == "_maker = BCP_net_ds if _BCP_DS > 0 else BCP_net", st, "maker")
lines[j:j + 1] = [
    "    if _BCP_TRANS > 0:",
    "        # One class covers both: n_trans=0 is U-Net plus aux heads, so the backbone and the",
    "        # deep-supervision arms differ only in this argument.",
    "        _maker = lambda in_chns, class_num, ema=False: UNetTransDS(",
    "            in_chns=in_chns, class_num=class_num, n_trans=_BCP_TRANS).cuda()",
    "    else:",
    "        _maker = BCP_net_ds if _BCP_DS > 0 else BCP_net",
]

# ---- the pre-trained checkpoint has neither aux heads nor Transformer blocks ----------------
for _ in range(2):
    k = find(lambda l: "net.load_state_dict(state['net'], strict=(_BCP_DS == 0))" in l,
             0, "strict load")
    lines[k] = lines[k].replace("strict=(_BCP_DS == 0)",
                                "strict=(_BCP_DS == 0 and _BCP_TRANS == 0)")
k = find(lambda l: l.strip() == "if _BCP_DS == 0:", 0, "opt load guard")
lines[k] = "    if _BCP_DS == 0 and _BCP_TRANS == 0:"

# ---- forward: use forward_ds whenever aux outputs are needed --------------------------------
k = find(lambda l: l.strip() == "if _BCP_DS > 0:", st, "self_train ds forward guard")
assert "forward_ds(net_input_unl)" in lines[k + 1], lines[k + 1]

open(P, "w").write("\n".join(lines))
print(f"patched: maker at {j}, flag inserted, strict loads relaxed")
