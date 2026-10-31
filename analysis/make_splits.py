#!/usr/bin/env python
"""R17 — carve and PIN every split used by this project.

Deterministic by construction: fixed seed, sorted input order, written once and then
treated as read-only. Re-running must reproduce byte-identical lists (verified at the end).

Why this script exists: stage 1 found that the SSMIS literature selects checkpoints without a
stated validation protocol. LA/Kvasir ship no test split and ISIC 2018 publishes ground truth
only for its training set, so a validation split MUST be carved from train before any training
run, or checkpoint selection silently touches test.

Outputs -> 04_experiments/outputs/splits/
  <dataset>/{train,val,test}.list           full splits
  <dataset>/labeled_<ratio>.list            labeled subset drawn from train
  <dataset>/PROVENANCE.md                   how each list was produced
"""
import hashlib, json, os, random, sys, glob

ROOT = "/home/u5535699/Desktop/dcslarge/datasets/ssmis"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "splits")
SEED = 20260914          # frozen; never change without a deviation log entry

def write(ds, name, items):
    d = os.path.join(OUT, ds); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    body = "\n".join(items) + "\n"
    with open(p, "w") as f: f.write(body)
    return p, len(items), hashlib.sha256(body.encode()).hexdigest()[:12]

def subsample(train, k, salt):
    """Deterministic k-of-train draw. sorted() first so filesystem order cannot leak in."""
    rng = random.Random(f"{SEED}:{salt}")
    return sorted(rng.sample(sorted(train), k))

report = {}

# ---------------------------------------------------------------- ACDC
# Ships patient-level train/val/test (140/20/40 volumes = 70/10/20 patients). We do NOT re-carve.
# For the LABELED subsets we adopt the community-pinned slice lists that UniMatch / SSL4MIS use
# (acdc/{1,3,7}/labeled.txt), rather than drawing our own: identical labeled slices are what make
# our ACDC numbers comparable to the 2D SSMIS literature, and they are nested 1 c 3 c 7.
UNIMATCH = "/home/u5535699/Desktop/UniMatch-main/more-scenarios/medical/splits/acdc"

def acdc():
    import re
    base = os.path.join(ROOT, "ACDC")
    rd = lambda p: [l.strip() for l in open(p) if l.strip()]
    sp = {k: rd(os.path.join(base, f"{k}.list")) for k in ("train", "val", "test")}
    pid = lambda v: re.search(r"patient\d+", v).group(0)
    pats = {k: {pid(v) for v in vs} for k, vs in sp.items()}
    problems = []
    for a, b in (("train","val"), ("train","test"), ("val","test")):
        ov = pats[a] & pats[b]
        if ov: problems.append(f"patient overlap {a}/{b}: {sorted(ov)}")

    rows = [write("ACDC", f"{k}.list", sorted(sp[k])) for k in ("train","val","test")]

    # train slices (2D training units) + community-pinned labeled subsets
    tsl = sorted(rd(os.path.join(base, "train_slices.list")))
    rows.append(write("ACDC", "train_slices.list", tsl))
    lab = {}
    strip = lambda x: os.path.basename(x).replace(".h5", "")
    for n in (3, 7):
        f = os.path.join(UNIMATCH, str(n), "labeled.txt")
        if not os.path.exists(f):
            problems.append(f"UniMatch labeled list missing for {n} cases"); continue
        sl = sorted(strip(x) for x in rd(f))
        missing = [x for x in sl if x not in set(tsl)]
        if missing:
            problems.append(f"{n}cases: {len(missing)} pinned slices absent from train_slices")
        ps = sorted({pid(x) for x in sl})
        bad = [q for q in ps if q not in pats["train"]]
        if bad:
            problems.append(f"{n}cases: labeled patients not in train split: {bad}")
        lab[n] = (ps, write("ACDC", f"labeled_{n}cases.list", sl))
    if 3 in lab and 7 in lab:
        s3 = set(open(os.path.join(OUT,"ACDC","labeled_3cases.list")).read().split())
        s7 = set(open(os.path.join(OUT,"ACDC","labeled_7cases.list")).read().split())
        if not s3 <= s7: problems.append("labeled 3cases is not nested inside 7cases")

    report["ACDC"] = dict(
        volumes={k: len(v) for k, v in sp.items()},
        patients={k: len(v) for k, v in pats.items()},
        train_slices=len(tsl),
        problems=problems,
        labeled={f"{n}cases": dict(patients=lab[n][0], slices=lab[n][1][1]) for n in lab},
        lists={r[0].split("splits/")[1]: dict(n=r[1], sha=r[2]) for r in rows},
        source=("volumes: shipped SSL4MIS-derived ACDC lists, verified not re-carved. "
                "labeled slice subsets: adopted verbatim from UniMatch "
                "more-scenarios/medical/splits/acdc/{3,7}/labeled.txt for comparability"),
    )

# ---------------------------------------------------------------- Kvasir-SEG
# Official Data-split gives 880 train / 120 val and NO test. We promote the official 120 to
# TEST (it is the set published numbers are reported on, so keeping it untouched preserves
# comparability) and carve a fresh 120-image VAL out of the official 880 train.
def kvasir():
    base = os.path.join(ROOT, "Kvasir-SEG")
    tr = [l.strip() for l in open(os.path.join(base, "train.list")) if l.strip()]
    va = [l.strip() for l in open(os.path.join(base, "val.list")) if l.strip()]
    assert not (set(tr) & set(va)), "official train/val overlap"
    test = sorted(va)
    new_val = subsample(tr, 120, "Kvasir-val")
    new_tr = sorted(set(tr) - set(new_val))
    rows = [write("Kvasir-SEG", "train.list", new_tr),
            write("Kvasir-SEG", "val.list", new_val),
            write("Kvasir-SEG", "test.list", test)]
    lab = {}
    for tag, n in (("5pct", round(0.05*len(new_tr))), ("10pct", round(0.10*len(new_tr)))):
        lab[tag] = write("Kvasir-SEG", f"labeled_{tag}.list",
                         subsample(new_tr, n, f"Kvasir-{tag}"))
    report["Kvasir-SEG"] = dict(
        counts=dict(train=len(new_tr), val=len(new_val), test=len(test)),
        labeled={k: v[1] for k, v in lab.items()},
        lists={r[0].split("splits/")[1]: dict(n=r[1], sha=r[2]) for r in rows},
        source=("official DebeshJha/Kvasir-SEG Data-split (880/120) verified against the "
                "HF mirror kowndinya23/Kvasir-SEG; official 120 kept as TEST, val carved "
                "from the official 880 train"),
        caveat="no patient identifiers ship with Kvasir-SEG, so same-patient leakage across "
               "splits cannot be excluded. Report this.",
    )

# ---------------------------------------------------------------- ISIC 2018 Task 1
# Only the TRAINING ground truth is public (2594 masks). val/test must come out of it.
def isic():
    base = os.path.join(ROOT, "ISIC2018")
    masks = glob.glob(os.path.join(base, "**", "*_segmentation.png"), recursive=True)
    if not masks:
        report["ISIC2018"] = dict(status="SKIPPED - images/masks not extracted yet")
        return
    ids = sorted(os.path.basename(m).replace("_segmentation.png", "") for m in masks)
    n = len(ids)
    n_test = round(0.20 * n); n_val = round(0.10 * n)
    test = subsample(ids, n_test, "ISIC-test")
    rest = sorted(set(ids) - set(test))
    val = subsample(rest, n_val, "ISIC-val")
    train = sorted(set(rest) - set(val))
    rows = [write("ISIC2018", "train.list", train),
            write("ISIC2018", "val.list", val),
            write("ISIC2018", "test.list", test)]
    lab = {}
    for tag, frac in (("5pct", 0.05), ("10pct", 0.10)):
        lab[tag] = write("ISIC2018", f"labeled_{tag}.list",
                         subsample(train, round(frac*len(train)), f"ISIC-{tag}"))
    report["ISIC2018"] = dict(
        total_with_gt=n,
        counts=dict(train=len(train), val=len(val), test=len(test)),
        labeled={k: v[1] for k, v in lab.items()},
        lists={r[0].split("splits/")[1]: dict(n=r[1], sha=r[2]) for r in rows},
        source="ISIC 2018 Task 1 TRAINING ground truth (the only public GT); 70/10/20 carve",
        caveat=("the Task1 archive ships no patient identifiers, so multiple lesions from one "
                "patient may fall in different splits. This cannot be excluded here and must "
                "be stated as a limitation."),
    )

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    acdc(); kvasir(); isic()
    with open(os.path.join(OUT, "splits_report.json"), "w") as f:
        json.dump(dict(seed=SEED, report=report), f, indent=2)
    for ds, r in report.items():
        print(f"\n=== {ds}")
        for k, v in r.items():
            print(f"  {k}: {json.dumps(v) if not isinstance(v,str) else v}"[:400])
    probs = [p for r in report.values() for p in r.get("problems", [])]
    print("\nPROBLEMS:", probs if probs else "none")
