"""Gradient geometry of multi-scale supervision, its sensitivity to target corruption, and the
reliability of the teacher's pseudo-labels.

Three record kinds are written, one JSONL line each per probe step.

`geom` / `sens` -- the *clean* geometry. Measured on a fixed labeled probe batch with ground
truth and no copy-paste mask, which isolates the geometry of supervision placement from
pseudo-label noise and mask randomness. Quantities, all restricted to the parameters the taps
and the main head share (a gradient on parameters only one of them reaches has no direction to
agree or disagree with):

    a_l  = <g_0, g_l> / (|g_0| |g_l|)          alignment
    r_l  = |g_l| / |g_0|                       relative magnitude
    q_l  = mean 1[sign(g_0) != sign(g_l)]      coordinate-wise conflict rate
    c_ij = <g_i, g_j> / (|g_i| |g_j|)          redundancy between taps

`carrier` -- the quantity the descent lemma actually needs. The `geom` record's rho is built
from probe-batch gradients of the clean supervised loss, which is NOT the carrier objective the
optimiser descends, so it cannot bound the step size for that objective. This record instead
takes both gradients from the live training batch:

    G_0 = grad_theta L_BCP        the carrier objective, on the mixed batch being trained on
    G_l = grad_theta L_l          tap l's own auxiliary term, same batch, unweighted
    D   = G_0 + sum_l w_l G_l     the direction the update actually takes
    rho = <G_0, D> / |D|^2

For an L_f-smooth L_BCP, a step eta along -D decreases L_BCP whenever rho > L_f eta / 2. Every
symbol here is the one in the update, so the condition is testable rather than analogical.

`plabel` -- the teacher's real errors rather than synthetic corruption. On the same labeled
probe batch, the EMA teacher's pseudo-label Yt is compared against ground truth Y, errors are
split into a band around the ground-truth class boundaries and the interior, and the gradient
response of each tap is measured against three targets: Yt (all errors), Yt restricted to the
boundary band, and Yt restricted to the interior. This tests on real pseudo-label error what
`sens` tests on dilations and random flips.

The probe runs the network in eval mode, so dropout draws no random numbers and batch-norm
running statistics are read rather than updated.
"""
import json
import os

import torch
import torch.nn.functional as F

SHARED_PREFIXES = ("encoder.", "decoder.up1.")


def _flat(gs, params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(gs, params)])


def _pool(x, k):
    return (F.max_pool2d if x.dim() == 4 else F.max_pool3d)(x, 2 * k + 1, 1, k)


def _dilate_fg(y, k, n_classes):
    """Dilate every foreground class by k voxels; ties go to the lowest class index."""
    oh = F.one_hot(y.long(), n_classes).movedim(-1, 1).float()
    grown = _pool(oh[:, 1:], k)
    claimed = grown.max(dim=1).values > 0.5
    return torch.where(claimed, grown.argmax(dim=1) + 1, torch.zeros_like(y.long())).to(y.dtype)


def _flip(y, p, n_classes, gen):
    r = torch.rand(y.shape, device=y.device, generator=gen)
    rnd = torch.randint(0, n_classes, y.shape, device=y.device, generator=gen, dtype=y.dtype)
    return torch.where(r < p, rnd, y)


def boundary_band(y, k):
    """Voxels whose (2k+1) neighbourhood is not label-constant: the class-boundary band."""
    f = y.float().unsqueeze(1)
    return (_pool(f, k) != -_pool(-f, k)).squeeze(1)


class GradProbe:
    def __init__(self, model, n_classes, weights, out_path, interval, band=2,
                 corrupt=(("boundary", 1), ("boundary", 2), ("boundary", 4),
                          ("uniform", 0.05), ("uniform", 0.10))):
        self.model, self.n_classes, self.w = model, n_classes, list(weights)
        self.interval, self.corrupt, self.band = interval, list(corrupt), band
        self.x = self.y = self.teacher = None
        self.gen = torch.Generator(device="cuda").manual_seed(20261019)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.fh = open(out_path, "a", buffering=1)
        named = [(n, p) for n, p in model.named_parameters() if n.startswith(SHARED_PREFIXES)]
        self.params = [p for _, p in named]
        self.fh.write(json.dumps({"kind": "header", "weights": self.w, "band": band,
                                  "n_shared_tensors": len(self.params),
                                  "n_shared_scalars": sum(p.numel() for p in self.params),
                                  "shared_prefixes": list(SHARED_PREFIXES),
                                  "first_names": [n for n, _ in named[:3]],
                                  "last_names": [n for n, _ in named[-3:]]}) + "\n")

    def due(self, iter_num):
        return self.interval > 0 and iter_num % self.interval == 0

    def set_batch(self, x, y):
        self.x, self.y = x.detach().clone(), y.detach().clone()

    def set_teacher(self, ema_model):
        self.teacher = ema_model

    def _g(self, loss, retain=True):
        return _flat(torch.autograd.grad(loss, self.params, retain_graph=retain,
                                         allow_unused=True), self.params)

    # ---- the carrier record: both gradients from the live training batch ------------------
    def carrier(self, iter_num, carrier_loss, aux_losses):
        """Called before backward(), with L_BCP and the three unweighted per-tap losses."""
        if not self.due(iter_num):
            return
        G0 = self._g(carrier_loss)
        Gl = [self._g(L) for L in aux_losses]
        n0 = float(G0.norm())
        D = G0 + sum(w * g for w, g in zip(self.w, Gl))
        self.fh.write(json.dumps({
            "kind": "carrier", "iter": int(iter_num), "G0_norm": n0,
            "a": [float(torch.dot(G0, g) / (n0 * float(g.norm()) + 1e-12)) for g in Gl],
            "r": [float(g.norm()) / (n0 + 1e-12) for g in Gl],
            "q": [float((torch.sign(G0) != torch.sign(g)).float().mean()) for g in Gl],
            "rho": float(torch.dot(G0, D) / (D.dot(D) + 1e-12)),
            "step_norm_ratio": float(D.norm() / (n0 + 1e-12)),
        }) + "\n")

    # ---- the clean records: fixed labeled probe batch, ground truth ----------------------
    def _losses(self, out, aux, y):
        yl = y.long()

        def term(logits):
            prob = torch.softmax(logits, dim=1)
            oh = F.one_hot(yl, self.n_classes).movedim(-1, 1).float()
            dims = tuple(range(2, prob.dim()))
            inter = (prob * oh).sum(dim=dims)
            den = (prob * prob).sum(dim=dims) + (oh * oh).sum(dim=dims)
            dice = (1.0 - (2 * inter + 1e-10) / (den + 1e-10))[:, 1:].mean()
            return 0.5 * (F.cross_entropy(logits, yl) + dice)
        return term(out), [term(a) for a in aux]

    def _grads(self, y):
        out, aux = self.model.forward_ds(self.x)
        L0, Ls = self._losses(out, aux, y)
        return [self._g(L) for L in [L0] + Ls]

    @torch.enable_grad()
    def step(self, iter_num):
        if self.x is None or not self.due(iter_num):
            return
        was_training = self.model.training
        self.model.eval()
        try:
            g = self._grads(self.y)
            n = [float(v.norm()) for v in g]
            a = sum(w * g[i + 1] for i, w in enumerate(self.w))
            s = g[0] + a
            self.fh.write(json.dumps({
                "kind": "geom", "iter": int(iter_num), "g0_norm": n[0],
                "a": [float(torch.dot(g[0], g[i]) / (n[0] * n[i] + 1e-12)) for i in range(1, len(g))],
                "r": [n[i] / (n[0] + 1e-12) for i in range(1, len(g))],
                "q": [float((torch.sign(g[0]) != torch.sign(g[i])).float().mean())
                      for i in range(1, len(g))],
                "c": {f"{i}{j}": float(torch.dot(g[i], g[j]) / (n[i] * n[j] + 1e-12))
                      for i in range(1, len(g)) for j in range(i + 1, len(g))},
                "rho_clean": float(torch.dot(g[0], s) / (s.dot(s) + 1e-12)),
                "step_norm_ratio": float(s.norm() / (n[0] + 1e-12)),
            }) + "\n")

            base = [v.clone() for v in g]
            for mode, lvl in self.corrupt:
                yc = (_dilate_fg(self.y, int(lvl), self.n_classes) if mode == "boundary"
                      else _flip(self.y, float(lvl), self.n_classes, self.gen))
                gc = self._grads(yc)
                self.fh.write(json.dumps({
                    "kind": "sens", "iter": int(iter_num), "mode": mode, "level": lvl,
                    "S_rel": [float((gc[i] - base[i]).norm() / (base[i].norm() + 1e-12))
                              for i in range(len(base))]}) + "\n")

            if self.teacher is not None:
                self._plabel(iter_num, base)
        finally:
            if was_training:
                self.model.train()

    def _plabel(self, iter_num, base):
        """The teacher's real errors, split into a boundary band and the interior."""
        was = self.teacher.training
        self.teacher.eval()
        with torch.no_grad():
            t = self.teacher(self.x)
            t = t[0] if isinstance(t, (tuple, list)) else t
            yt = torch.argmax(torch.softmax(t, dim=1), dim=1).to(self.y.dtype)
        if was:
            self.teacher.train()

        B = boundary_band(self.y, self.band)
        err = yt != self.y
        n_tot = float(err.numel())
        rec = {"kind": "plabel", "iter": int(iter_num), "band": self.band,
               "err_rate": float(err.float().mean()),
               "band_frac": float(B.float().mean()),
               "err_rate_band": float(err[B].float().mean()) if B.any() else 0.0,
               "err_rate_interior": float(err[~B].float().mean()) if (~B).any() else 0.0,
               "err_share_in_band": float((err & B).sum() / err.sum().clamp(min=1)),
               "n_err": float(err.sum()) , "n_vox": n_tot}
        # concentration: how much likelier an error is inside the band than at large
        rec["enrichment"] = (rec["err_rate_band"] / rec["err_rate"]) if rec["err_rate"] > 0 else 0.0
        for tag, yy in (("all", yt),
                        ("band", torch.where(B, yt, self.y)),
                        ("interior", torch.where(B, self.y, yt))):
            gc = self._grads(yy)
            rec[f"S_{tag}"] = [float((gc[i] - base[i]).norm() / (base[i].norm() + 1e-12))
                               for i in range(len(base))]
        self.fh.write(json.dumps(rec) + "\n")
