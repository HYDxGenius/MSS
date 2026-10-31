import json, os, glob, statistics as st
from collections import defaultdict
ROOT="/home/u5535699/Desktop/dcslarge/ssmis-stability/outputs"
# cells with 3 seeds
CELLS={("ACDC","cross_teaching"):["R053","R054","R055"],}
# rebuild from records instead of hardcoding
runs={}
for d in sorted(glob.glob(os.path.join(ROOT,"R*"))):
    rr=os.path.join(d,"run_record.json"); tr=os.path.join(d,"test_rows.json")
    if not (os.path.exists(rr) and os.path.exists(tr)): continue
    r=json.load(open(rr))["run"]
    runs[r["run_id"]]=(r["dataset"],r["method"],int(r["seed"]),tr)
cells=defaultdict(list)
for rid,(ds,m,s,tr) in runs.items(): cells[(ds,m)].append((s,rid,tr))
print(f"{'dataset':11}{'method':22}{'sd(mean)':>9}{'top1%':>7}{'top3%':>7}{'r(inst,diff)':>13}{'flip_empty':>11}{'n_items':>8}")
print("-"*95)
for k in sorted(cells):
    v=sorted(cells[k])
    if len(v)<3: continue
    # per (id,cls) dice across seeds
    per=defaultdict(dict); empt=defaultdict(dict)
    for s,rid,tr in v:
        for row in json.load(open(tr)):
            key=(row["id"],row["cls"])
            per[key][s]=row["dice"]; empt[key][s]=row["pred_empty"]
    keys=[kk for kk in per if len(per[kk])==3]
    seeds=[s for s,_,_ in v]
    means=[st.mean(per[kk][s] for kk in keys) for s in seeds]
    sd_mean=st.stdev(means)
    # per-item contribution to variance of the mean: item deviations
    N=len(keys)
    dev={kk:[per[kk][s]-st.mean(per[kk].values()) for s in seeds] for kk in keys}
    # variance of mean = (1/N^2) * Var(sum of devs); attribute by |contribution|
    contrib={kk: st.stdev(per[kk].values()) for kk in keys}
    order=sorted(keys,key=lambda kk:-contrib[kk])
    tot=sum(contrib[kk]**2 for kk in keys)
    top1=sum(contrib[kk]**2 for kk in order[:max(1,N//100)])/tot
    top3=sum(contrib[kk]**2 for kk in order[:max(1,(3*N)//100)])/tot
    # correlation between instability and difficulty
    diff=[st.mean(per[kk].values()) for kk in keys]; inst=[contrib[kk] for kk in keys]
    mx,my=st.mean(diff),st.mean(inst)
    num=sum((a-mx)*(b-my) for a,b in zip(diff,inst))
    den=(sum((a-mx)**2 for a in diff)*sum((b-my)**2 for b in inst))**.5
    r=num/den if den else float('nan')
    flip=sum(1 for kk in keys if len(set(empt[kk].values()))>1)
    print(f"{k[0]:11}{k[1]:22}{sd_mean:9.4f}{top1*100:7.1f}{top3*100:7.1f}{r:13.3f}{flip:11d}{N:8d}")
