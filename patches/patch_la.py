#!/usr/bin/env python
"""Port the stack components to BCP's LA (3D V-Net) self-training stage.

Mirrors the ACDC patch so the two datasets run the same arms:

    BCP_DS       float  deep supervision weight (0 = off)  -- the only component that helped on ACDC
    BCP_W2S      float  weak-to-strong consistency weight (0 = off)
    BCP_W2SSTR   float  strong-augmentation strength
    BCP_EMAEVAL  0/1    evaluate the EMA teacher instead of the student

All default off, so an unset environment reproduces the validated LA baseline (0.8910 at 10%).
Anchors are searched from inside self_train: LA's pre_train has a structurally similar loss line
and patching there would reference variables that do not exist yet.
"""
import re
import sys

P = "/dcs/large/u5535699/bcp-official/code/LA_BCP_train.py"
lines = open(P).read().split("\n")
if any("_BCP_DS" in l for l in lines):
    print("already patched")
    sys.exit(0)


def find(pred, start=0, what=""):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise SystemExit(f"anchor not found: {what}")


# ---- flags, after the last import ---------------------------------------------------------
i = max(j for j, l in enumerate(lines[:60]) if l.startswith(("import ", "from ")))
lines[i + 1:i + 1] = [
    "",
    "import os as _os",
    "from networks.vnet_ds import VNetDS",
    "from utils.strongaug import strong_augment",
    'from utils.losses import mask_DiceLoss as _MaskDice',
    "",
    '_BCP_DS = float(_os.environ.get("BCP_DS", "0"))',
    '_BCP_W2S = float(_os.environ.get("BCP_W2S", "0"))',
    '_BCP_W2SSTR = float(_os.environ.get("BCP_W2SSTR", "1.0"))',
    '_BCP_EMAEVAL = int(_os.environ.get("BCP_EMAEVAL", "0"))',
    'print(f"[method] DS={_BCP_DS} W2S={_BCP_W2S}@{_BCP_W2SSTR} EMAeval={_BCP_EMAEVAL}")',
]

st = find(lambda l: "def self_train(" in l, 0, "self_train")

# ---- DS-capable networks ------------------------------------------------------------------
j = find(lambda l: l.strip().startswith("model = net_factory("), st, "self_train model")
assert lines[j + 1].strip().startswith("ema_model = net_factory("), lines[j + 1]
orig_model, orig_ema = lines[j].strip(), lines[j + 1].strip()
lines[j:j + 2] = [
    "    if _BCP_DS > 0:",
    "        model = VNetDS(n_channels=1, n_classes=num_classes,",
    "                       normalization='batchnorm', has_dropout=True).cuda()",
    "        ema_model = VNetDS(n_channels=1, n_classes=num_classes,",
    "                           normalization='batchnorm', has_dropout=True).cuda()",
    "        for p in ema_model.parameters():",
    "            p.detach_()",
    "    else:",
    "        " + orig_model,
    "        " + orig_ema,
]

# ---- the pre-trained checkpoint has no auxiliary heads -------------------------------------
for what in ("load_net", "load_net_opt"):
    k = find(lambda l: l.strip() == "net.load_state_dict(state['net'])", 0, what)
    lines[k] = "    net.load_state_dict(state['net'], strict=(_BCP_DS == 0))"
k = find(lambda l: l.strip() == "optimizer.load_state_dict(state['opt'])", 0, "opt load")
lines[k:k + 1] = [
    "    if _BCP_DS == 0:",
    "        optimizer.load_state_dict(state['opt'])",
    "    else:",
    '        print("[method] DS on: optimizer state not restored (param set differs)")',
]

# ---- forward + the two extra terms ---------------------------------------------------------
st = find(lambda l: "def self_train(" in l, 0, "self_train")
f = find(lambda l: l.strip() == "outputs_l, _ = model(mixl_img)", st, "self_train forward")
assert lines[f + 1].strip() == "outputs_u, _ = model(mixu_img)"
lines[f:f + 2] = [
    "            if _BCP_DS > 0:",
    "                outputs_l, _, aux_l = model.forward_ds(mixl_img)",
    "                outputs_u, _, aux_u = model.forward_ds(mixu_img)",
    "            else:",
    "                outputs_l, _ = model(mixl_img)",
    "                outputs_u, _ = model(mixu_img)",
]

g = find(lambda l: re.match(r"\s*loss = loss_l \+ loss_u\s*$", l), f, "self_train loss sum")
lines[g + 1:g + 1] = [
    "",
    "            # deep supervision on the decoder's intermediate stages (the only component",
    "            # that helped on ACDC: +0.0192)",
    "            if _BCP_DS > 0:",
    "                for _aux, _ia, _ib in ((aux_l, lab_a, plab_a), (aux_u, plab_b, lab_b)):",
    "                    for _w, _a in zip((0.4, 0.2, 0.1), _aux):",
    "                        loss = loss + _BCP_DS * _w * mix_loss(",
    "                            _a, _ia, _ib, loss_mask, u_weight=args.u_weight)",
    "",
    "            # weak-to-strong consistency: a photometrically perturbed student view against",
    "            # the same mixed targets",
    "            if _BCP_W2S > 0:",
    "                _sl = strong_augment(mixl_img, _BCP_W2SSTR)",
    "                _su = strong_augment(mixu_img, _BCP_W2SSTR)",
    "                _ol = model.forward_ds(_sl)[0] if _BCP_DS > 0 else model(_sl)[0]",
    "                _ou = model.forward_ds(_su)[0] if _BCP_DS > 0 else model(_su)[0]",
    "                loss = loss + _BCP_W2S * (",
    "                    mix_loss(_ol, lab_a, plab_a, loss_mask, u_weight=args.u_weight)",
    "                    + mix_loss(_ou, plab_b, lab_b, loss_mask, u_weight=args.u_weight,",
    "                               unlab=True))",
]

# ---- EMA-teacher evaluation (self_train's call only) ---------------------------------------
e = find(lambda l: "var_all_case_LA(model," in l, g, "self_train eval")
lines[e] = lines[e].replace("var_all_case_LA(model,",
                            "var_all_case_LA(ema_model if _BCP_EMAEVAL else model,")

open(P, "w").write("\n".join(lines))
print(f"patched LA_BCP_train.py: nets at {j}, forward at {f}, losses after {g}, eval at {e}")
