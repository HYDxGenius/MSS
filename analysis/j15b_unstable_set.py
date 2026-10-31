import json, os, glob, statistics as st, itertools
from collections import defaultdict
ROOT="/home/u5535699/Desktop/dcslarge/ssmis-stability/outputs"
runs={}
for d in sorted(glob.glob(os.path.join(ROOT,"R*"))):
    rr=os.path.join(d,"run_record.json"); tr=os.path.join(d,"test_rows.json")
    if not (os.path.exists(rr) and os.path.exists(tr)): continue
    r=json.load(open(rr))["run"]
    runs[r["run_id"]]=(r["dataset"],r["method"],int(r["seed"]),tr)
cells=defaultdict(list)
for rid,(ds,m,s,tr) in runs.items(): cells[(ds,m)].append((s,rid,tr))
cells={k:sorted(v) for k,v in cells.items() if len(v)>=3}

per_cell={}
for k,v in cells.items():
    per=defaultdict(dict); empt=defaultdict(dict)
    for s,rid,tr in v:
        for row in json.load(open(tr)):
            per[(row["id"],row["cls"])][s]=row["dice"]
            empt[(row["id"],row["cls"])][s]=row["pred_empty"]
    keys=[kk for kk in per if len(per[kk])==3]
    per_cell[k]=(per,empt,keys,[s for s,_,_ in v])

print("=== 1. How much of sd(mean) is the empty/non-empty flips? ===")
print(f"{'dataset':11}{'method':22}{'sd_all':>8}{'sd_excl_flip':>13}{'drop%':>7}{'n_flip':>7}")
for k in sorted(per_cell):
    per,empt,keys,seeds=per_cell[k]
    flips={kk for kk in keys if len(set(empt[kk].values()))>1}
    def sdm(ks):
        m=[st.mean(per[kk][s] for kk in ks) for s in seeds]; return st.stdev(m)
    a=sdm(keys); b=sdm([kk for kk in keys if kk not in flips])
    print(f"{k[0]:11}{k[1]:22}{a:8.4f}{b:13.4f}{(1-b/a)*100:7.1f}{len(flips):7d}")

print("\n=== 2. Is the UNSTABLE SET the same across arms? (top-10% unstable items, Jaccard) ===")
for ds in ("ACDC","Kvasir-SEG"):
    ks=[k for k in per_cell if k[0]==ds]
    tops={}
    for k in ks:
        per,empt,keys,seeds=per_cell[k]
        inst={kk:st.stdev(per[kk].values()) for kk in keys}
        n=max(1,len(keys)//10)
        tops[k[1]]=set(sorted(keys,key=lambda kk:-inst[kk])[:n])
    names=sorted(tops)
    print(f"  {ds}  (top-{max(1,len(keys)//10)} of {len(keys)} items)")
    for a,b in itertools.combinations(names,2):
        A,B=tops[a],tops[b]
        print(f"    {a[:20]:22} vs {b[:20]:22} |∩|={len(A&B):2d}  J={len(A&B)/len(A|B):.2f}")
    # how many items appear in >=3 of the 4 arms' unstable sets
    cnt=defaultdict(int)
    for s in tops.values():
        for kk in s: cnt[kk]+=1
    import collections
    h=collections.Counter(cnt.values())
    print(f"    items in >=3 arms' unstable set: {sum(v for c,v in h.items() if c>=3)}"
          f" | in exactly 1 arm: {h.get(1,0)} | union size {len(cnt)}")
    # expected overlap under independence
    n=max(1,len(keys)//10); N=len(keys)
    print(f"    chance |∩| for two arms = {n*n/N:.1f}")
