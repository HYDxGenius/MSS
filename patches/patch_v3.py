#!/usr/bin/env python
"""Three additions needed for the v3 experiment set, and one bug fix.

1. Per-tap auxiliary weights (`BCP_DSW`), so "which taps and in what proportion" is an
   experiment rather than a constant. Every arm holds sum(w) = 0.7 so the arms stay comparable.
2. `BCP_DSFORM` selects the auxiliary target weighting: "uniform" (the mixed target at weight 1
   everywhere) reproduces every validated ACDC number in this project; "carrier" instead reuses
   the carrier's ground-truth-1 / pseudo-u_weight split.
3. `BCP_PROBE` / `BCP_DSDETACH` drive the gradient-geometry probe. Detaching the taps lets the
   auxiliary heads train while the trunk's trajectory stays bit-identical to the baseline, so
   the probe can measure auxiliary gradients along the *baseline* path.
4. LA's auxiliary loss called mix_loss without unlab=True on the outward branch, which priced
   the pseudo-labelled half at 1 and the ground-truth half at u_weight -- the inverse of the
   carrier, and the inverse of what the inward branch did. Replaced by the same uniform form
   ACDC uses, so MSS has one definition on both datasets.
"""
import sys

C = "/dcs/large/u5535699/bcp-official/code"


def edit(path, pairs, marker):
    s = open(path).read()
    if marker in s:
        print(f"  {path}: already patched")
        return
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            sys.exit(f"ABORT {path}: anchor occurs {n} times, expected 1:\n{old[:160]}")
        s = s.replace(old, new)
    open(path, "w").write(s)
    print(f"  {path}: {len(pairs)} edits applied")


# ---- networks/unet_ds.py: optional detached taps --------------------------------------------
edit(f"{C}/networks/unet_ds.py", [(
    """    def forward_ds(self, x):
        out, (a, b, c) = self._decode(self.encoder(x))
        size = out.shape[2:]""",
    """    def forward_ds(self, x, detach_taps=False):
        out, (a, b, c) = self._decode(self.encoder(x))
        if detach_taps:
            # Gradient still reaches the heads, but not the trunk, so a run with the heads
            # attached follows exactly the trajectory it would without them.
            a, b, c = a.detach(), b.detach(), c.detach()
        size = out.shape[2:]""",
)], "detach_taps")

# ---- utils/smds.py: accept a zero weight without doing the work -----------------------------
edit(f"{C}/utils/smds.py", [(
    """    total = aux_logits[0].new_zeros(())
    for lw, f, a in zip(level_weights, factors, aux_logits):""",
    """    total = aux_logits[0].new_zeros(())
    for lw, f, a in zip(level_weights, factors, aux_logits):
        if lw == 0:
            continue""",
)], "if lw == 0")

# ---- ACDC trainer ---------------------------------------------------------------------------
edit(f"{C}/ACDC_BCP_train.py", [
    ("""_BCP_DSMODE = _os.environ.get("BCP_DSMODE", "vanilla")   # vanilla | matched
assert _BCP_DSMODE in ("vanilla", "matched"), _BCP_DSMODE""",
     """_BCP_DSMODE = _os.environ.get("BCP_DSMODE", "vanilla")   # vanilla | matched
assert _BCP_DSMODE in ("vanilla", "matched"), _BCP_DSMODE
# Per-tap weights, coarse -> fine. Every arm in the tap ablation holds sum(w) = 0.7, so the
# arms differ in *where* the auxiliary signal is applied and not in how much of it there is.
_DSW = tuple(float(v) for v in _os.environ.get("BCP_DSW", "0.4,0.2,0.1").split(","))
assert len(_DSW) == 3, _DSW
_BCP_DSFORM = _os.environ.get("BCP_DSFORM", "uniform")   # uniform | carrier
assert _BCP_DSFORM in ("uniform", "carrier"), _BCP_DSFORM
_BCP_DSDETACH = int(_os.environ.get("BCP_DSDETACH", "0"))
_BCP_PROBE = int(_os.environ.get("BCP_PROBE", "0"))
from utils.dsprobe import GradProbe"""),

    ("""def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'], strict=(_BCP_DS == 0 and _BCP_TRANS == 0))""",
     """_STRICT = (_BCP_DS == 0 and _BCP_TRANS == 0 and _BCP_PROBE == 0)


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'], strict=_STRICT)"""),

    ("""    net.load_state_dict(state['net'], strict=(_BCP_DS == 0 and _BCP_TRANS == 0))
    if _BCP_DS == 0 and _BCP_TRANS == 0:""",
     """    net.load_state_dict(state['net'], strict=_STRICT)
    if _STRICT:"""),

    ("""        _maker = BCP_net_ds if _BCP_DS > 0 else BCP_net""",
     """        _maker = BCP_net_ds if (_BCP_DS > 0 or _BCP_PROBE > 0) else BCP_net"""),

    ("""            if _BCP_DS > 0:
                out_unl, out_unl_aux = model.forward_ds(net_input_unl)
                out_l, out_l_aux = model.forward_ds(net_input_l)""",
     """            if _BCP_DS > 0 or _BCP_PROBE > 0:
                out_unl, out_unl_aux = model.forward_ds(net_input_unl,
                                                        detach_taps=bool(_BCP_DSDETACH))
                out_l, out_l_aux = model.forward_ds(net_input_l,
                                                    detach_taps=bool(_BCP_DSDETACH))"""),

    ("""                _ou = model(_su) if _BCP_DS == 0 else model.forward_ds(_su)[0]
                _ol = model(_sl) if _BCP_DS == 0 else model.forward_ds(_sl)[0]""",
     """                _plain = (_BCP_DS == 0 and _BCP_PROBE == 0)
                _ou = model(_su) if _plain else model.forward_ds(_su)[0]
                _ol = model(_sl) if _plain else model.forward_ds(_sl)[0]"""),

    ("""            if _BCP_DS > 0:
                for _out, _tgt in ((out_unl_aux, unl_label), (out_l_aux, l_label)):
                    loss = loss + _BCP_DS * scale_matched_ds_loss(
                        _out, _tgt, num_classes, ce_loss, dice_loss,
                        matched=(_BCP_DSMODE == "matched"))""",
     """            if _BCP_DS > 0:
                for _out, _tgt, _ia, _ib, _ul in (
                        (out_unl_aux, unl_label, plab_a, lab_a, True),
                        (out_l_aux, l_label, lab_b, plab_b, False)):
                    if _BCP_DSFORM == "carrier":
                        for _w, _a in zip(_DSW, _out):
                            if _w == 0:
                                continue
                            _ad, _ac = mix_loss(_a, _ia, _ib, loss_mask,
                                                u_weight=args.u_weight, unlab=_ul)
                            loss = loss + _BCP_DS * _w * (_ad + _ac) / 2
                    else:
                        loss = loss + _BCP_DS * scale_matched_ds_loss(
                            _out, _tgt, num_classes, ce_loss, dice_loss,
                            level_weights=_DSW,
                            matched=(_BCP_DSMODE == "matched"))"""),

    ("""            iter_num += 1
            update_model_ema(model, ema_model, 0.99)""",
     """            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            if _probe is not None:
                if _probe.x is None:
                    _probe.set_batch(volume_batch[:args.labeled_bs],
                                     label_batch[:args.labeled_bs])
                _probe.step(iter_num)"""),

    ("""    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100""",
     """    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    _probe = GradProbe(model, num_classes, _DSW,
                       os.path.join(snapshot_path, "gradgeom.jsonl"),
                       _BCP_PROBE) if _BCP_PROBE > 0 else None
    best_performance = 0.0
    best_hd = 100"""),
], "_BCP_PROBE")

# ---- LA trainer -----------------------------------------------------------------------------
edit(f"{C}/LA_BCP_train.py", [
    ("from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables",
     "from utils.BCP_utils import context_mask, mix_loss, sup_loss, parameter_sharing, update_ema_variables"),

    ("""_BCP_DS = float(_os.environ.get("BCP_DS", "0"))
_BCP_W2S = float(_os.environ.get("BCP_W2S", "0"))""",
     """_BCP_DS = float(_os.environ.get("BCP_DS", "0"))
_DSW = tuple(float(v) for v in _os.environ.get("BCP_DSW", "0.4,0.2,0.1").split(","))
assert len(_DSW) == 3, _DSW
_BCP_W2S = float(_os.environ.get("BCP_W2S", "0"))"""),

    ("""            if _BCP_DS > 0:
                for _aux, _ia, _ib in ((aux_l, lab_a, plab_a), (aux_u, plab_b, lab_b)):
                    for _w, _a in zip((0.4, 0.2, 0.1), _aux):
                        loss = loss + _BCP_DS * _w * mix_loss(
                            _a, _ia, _ib, loss_mask, u_weight=args.u_weight)""",
     """            if _BCP_DS > 0:
                # One definition of MSS on both datasets: the mixed target at weight 1
                # everywhere. The previous form called mix_loss without unlab=True on the
                # outward branch, which priced the pseudo-labelled half at 1 and the
                # ground-truth half at u_weight -- the inverse of the carrier, and the inverse
                # of what the inward branch did.
                for _aux, _lab in ((aux_l, mixl_lab), (aux_u, mixu_lab)):
                    for _w, _a in zip(_DSW, _aux):
                        if _w == 0:
                            continue
                        loss = loss + _BCP_DS * _w * sup_loss(_a, _lab)"""),
], "_DSW")
print("done")
