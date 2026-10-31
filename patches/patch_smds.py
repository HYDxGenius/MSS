#!/usr/bin/env python
"""Switch the deep-supervision block between vanilla and scale-matched.

BCP_DSMODE = vanilla (default, reproduces the J21 arm that scored 0.9096) | matched.
Both go through one code path so the ablation differs only in this flag.
"""
import sys

P = "/dcs/large/u5535699/bcp-official/code/ACDC_BCP_train.py"
s = open(P).read()
if "_BCP_DSMODE" in s:
    print("already patched")
    sys.exit(0)

s = s.replace(
    "from networks.unet_ds import BCP_net_ds",
    "from networks.unet_ds import BCP_net_ds\nfrom utils.smds import scale_matched_ds_loss")

old_flag = '_BCP_DS = float(_os.environ.get("BCP_DS", "0"))'
assert old_flag in s, "DS flag not found"
s = s.replace(old_flag, old_flag + '\n'
              '_BCP_DSMODE = _os.environ.get("BCP_DSMODE", "vanilla")   # vanilla | matched\n'
              'assert _BCP_DSMODE in ("vanilla", "matched"), _BCP_DSMODE')

old_print = 'print(f"[method] SIB={_BCP_SIB} ({_BCP_SIBN})  DS={_BCP_DS}")'
assert old_print in s, "method print not found"
s = s.replace(old_print,
              'print(f"[method] SIB={_BCP_SIB} ({_BCP_SIBN})  DS={_BCP_DS} ({_BCP_DSMODE})")')

old = """            if _BCP_DS > 0:
                for _out, _tgt in ((out_unl_aux, unl_label), (out_l_aux, l_label)):
                    for _w, _a in zip((0.4, 0.2, 0.1), _out):
                        loss = loss + _BCP_DS * _w * 0.5 * (
                            ce_loss(_a, _tgt.long())
                            + dice_loss(torch.softmax(_a, dim=1), _tgt.unsqueeze(1)))"""
new = """            if _BCP_DS > 0:
                for _out, _tgt in ((out_unl_aux, unl_label), (out_l_aux, l_label)):
                    loss = loss + _BCP_DS * scale_matched_ds_loss(
                        _out, _tgt, num_classes, ce_loss, dice_loss,
                        matched=(_BCP_DSMODE == "matched"))"""
assert old in s, "DS block not found verbatim"
s = s.replace(old, new)

open(P, "w").write(s)
print("patched ACDC_BCP_train.py (BCP_DSMODE = vanilla | matched)")
