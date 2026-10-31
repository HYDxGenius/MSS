"""Premise check: is pseudo-label / prediction quality size-dependent, and how much of the
unsupervised loss mass do small objects carry? Zero training - GT masks + existing test_rows."""
import json, os, glob, statistics as st
from collections import defaultdict
import numpy as np

SPL="/home/u5535699/Desktop/papers/med-ssl-mathematics/04_experiments/outputs/splits"
ROOT="/home/u5535699/Desktop/dcslarge/datasets/ssmis"
OUT="/home/u5535699/Desktop/dcslarge/ssmis-stability/outputs"

def acdc_sizes():
    import h5py
    s={}
    for sid in [l.strip() for l in open(f"{SPL}/ACDC/test.list") if l.strip()]:
        with h5py.File(f"{ROOT}/ACDC/data/{sid}.h5","r") as h:
            lab=np.asarray(h["label"])
        for c in (1,2,3):
            s[(sid,c)]=int((lab==c).sum())
        s[(sid,0)]=int(lab.size)
    return s

def kvasir_sizes():
    from PIL import Image
    s={}
    base=f"{ROOT}/Kvasir-SEG/masks"
    for i in [l.strip() for l in open(f"{SPL}/Kvasir-SEG/test.list") if l.strip()]:
        p=f"{base}/{i}.jpg"
        if not os.path.exists(p): p=f"{base}/{i}.png"
        a=np.asarray(Image.open(p).convert("L"))
        s[(i,1)]=int((a>127).sum()); s[(i,0)]=int(a.size)
    return s

SIZES={"ACDC":acdc_sizes(),"Kvasir-SEG":kvasir_sizes()}

runs={}
for d in sorted(glob.glob(os.path.join(OUT,"R*"))):
    rr=os.path.join(d,"run_record.json"); tr=os.path.join(d,"test_rows.json")
    if not (os.path.exists(rr) and os.path.exists(tr)): continue
    r=json.load(open(rr))["run"]
    runs[r["run_id"]]=(r["dataset"],r["method"],int(r["seed"]),tr)
cells=defaultdict(list)
for rid,(ds,m,s,tr) in runs.items(): cells[(ds,m)].append((s,rid,tr))

def spearman(x,y):
    def rank(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        i=0
        while i<len(o):
            j=i
            while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
            avg=(i+j)/2+1
            for k in range(i,j+1): r[o[k]]=avg
            i=j+1
        return r
    rx,ry=rank(x),rank(y); n=len(x)
    mx,my=st.mean(rx),st.mean(ry)
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**.5
    return num/den if den else float('nan')

print(f"{'dataset':11}{'method':22}{'rho(size,Dice)':>15}{'Dice Q1(small)':>15}{'Q4(large)':>11}{'gap':>8}{'loss share Q1':>14}")
print("-"*98)
for k in sorted(cells):
    v=sorted(cells[k])
    if len(v)<3: continue
    ds=k[0]; S=SIZES[ds]
    per=defaultdict(list)
    for s,rid,tr in v:
        for row in json.load(open(tr)):
            per[(row["id"],row["cls"])].append(row["dice"])
    keys=[kk for kk in per if len(per[kk])==3 and kk in S and S[kk]>0]
    size=[S[kk] for kk in keys]; dice=[st.mean(per[kk]) for kk in keys]
    rho=spearman(size,dice)
    order=sorted(range(len(keys)),key=lambda i:size[i])
    q=len(order)//4
    q1=[dice[i] for i in order[:q]]; q4=[dice[i] for i in order[-q:]]
    # "loss share": fraction of total GT foreground pixels held by the smallest quartile
    tot=sum(size); share=sum(size[i] for i in order[:q])/tot
    print(f"{ds:11}{k[1]:22}{rho:15.3f}{st.mean(q1):15.4f}{st.mean(q4):11.4f}"
          f"{st.mean(q4)-st.mean(q1):8.4f}{share*100:13.2f}%")

print("\n=== GT foreground size distribution on the TEST split ===")
for ds,S in SIZES.items():
    fg=[v for (i,c),v in S.items() if c>0 and v>0]
    fg.sort()
    n=len(fg)
    print(f"  {ds:11} n={n:4d}  p5={fg[n//20]:7d}  median={fg[n//2]:7d}  p95={fg[19*n//20]:8d}"
          f"  ratio p95/p5={fg[19*n//20]/max(1,fg[n//20]):7.1f}x")
