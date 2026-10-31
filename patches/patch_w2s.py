#!/usr/bin/env python
"""Add the two remaining stack components to BCP's ACDC SELF-TRAINING stage.

C1  BCP_W2S  float  weight of the weak-to-strong consistency branch (0 = off)
    The student gets a second, photometrically perturbed view of each mixed image, supervised by
    the same mixed target. BCP augments both streams identically, so this FixMatch/UniMatch
    asymmetry is absent from it. Costs one extra student forward.

C3  BCP_EMAEVAL  0/1  evaluate with the EMA teacher instead of the student
    BCP tests with the student; Mean Teacher's own prescription is the averaged weights.

Both default off, so an unset environment reproduces the validated baseline exactly.

NOTE: every anchor is searched from inside self_train. The first `loss = (loss_dice + loss_ce)/2`
in the file belongs to pre_train, where plab_a / net_input_unl do not exist; inserting there
yields a NameError at runtime.
"""
import re
import sys

P = "/dcs/large/u5535699/bcp-official/code/ACDC_BCP_train.py"
lines = open(P).read().split("\n")
if any("_BCP_W2S" in l for l in lines):
    print("already patched")
    sys.exit(0)


def find(pred, start=0, what=""):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise SystemExit(f"anchor not found: {what}")


# ---- flags -------------------------------------------------------------------------------
i = find(lambda l: l.startswith("_BCP_DSMODE"), 0, "DSMODE flag")
lines[i + 2:i + 2] = [
    "from utils.strongaug import strong_augment",
    '_BCP_W2S = float(_os.environ.get("BCP_W2S", "0"))',
    '_BCP_W2SSTR = float(_os.environ.get("BCP_W2SSTR", "1.0"))',
    '_BCP_EMAEVAL = int(_os.environ.get("BCP_EMAEVAL", "0"))',
]
i = find(lambda l: l.strip().startswith('print(f"[method] SIB='), 0, "method print")
lines[i] = ('print(f"[method] SIB={_BCP_SIB}({_BCP_SIBN}) DS={_BCP_DS}({_BCP_DSMODE}) '
            'W2S={_BCP_W2S}@{_BCP_W2SSTR} EMAeval={_BCP_EMAEVAL}")')

# ---- locate self_train, then its loss sum ------------------------------------------------
st = find(lambda l: "def self_train(" in l, 0, "self_train")
fw = find(lambda l: "out_unl, out_unl_aux = model.forward_ds(net_input_unl)" in l
          or l.strip() == "out_unl = model(net_input_unl)", st, "self_train forward")
i = find(lambda l: re.match(r"\s*loss = \(loss_dice \+ loss_ce\) / 2\s*$", l), fw,
         "self_train loss sum")

lines[i + 1:i + 1] = [
    "",
    "            # weak-to-strong consistency: the pseudo-label already comes from the EMA",
    "            # teacher on the unperturbed image, so a photometrically perturbed student view",
    "            # supervised by the same mixed target is exactly FixMatch's asymmetry, which BCP",
    "            # does not have (it augments both streams identically).",
    "            if _BCP_W2S > 0:",
    "                _su = strong_augment(net_input_unl, _BCP_W2SSTR)",
    "                _sl = strong_augment(net_input_l, _BCP_W2SSTR)",
    "                _ou = model(_su) if _BCP_DS == 0 else model.forward_ds(_su)[0]",
    "                _ol = model(_sl) if _BCP_DS == 0 else model.forward_ds(_sl)[0]",
    "                _d1, _c1 = mix_loss(_ou, plab_a, lab_a, loss_mask,",
    "                                    u_weight=args.u_weight, unlab=True)",
    "                _d2, _c2 = mix_loss(_ol, lab_b, plab_b, loss_mask, u_weight=args.u_weight)",
    "                loss = loss + _BCP_W2S * ((_d1 + _d2) + (_c1 + _c2)) / 2",
]

# ---- C3: evaluate the EMA teacher (self_train's evaluation call only) ---------------------
k = find(lambda l: "val_2d.test_single_volume" in l, i, "self_train eval call")
assert "model," in lines[k], lines[k]
lines[k] = lines[k].replace("model,", "(ema_model if _BCP_EMAEVAL else model),", 1)

open(P, "w").write("\n".join(lines))
print(f"patched: W2S block at line {i + 2}, EMA eval at line {k + 15}")
