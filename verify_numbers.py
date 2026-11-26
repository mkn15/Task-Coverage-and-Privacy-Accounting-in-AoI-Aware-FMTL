"""Re-derive every number quoted in the Letter's text from the stored runs."""
import json, numpy as np
from math import comb
G=json.load(open("grid.json")); G3=json.load(open("grid3.json")); G2=json.load(open("grid2.json")); B=json.load(open("baselines.json")); P=json.load(open("proxy_design.json"))
M=dict(G["mean"]); M.update(G3["mean"]); S=dict(G3["sigma"])
def sm(c,w=7): c=np.array(c); return np.convolve(np.pad(c,(w//2,w//2),mode="edge"),np.ones(w)/w,mode="valid")
def rnd(c): i=np.flatnonzero(sm(c)>=0.70); return int(i[0])+1 if len(i) else None
print("no-DP rounds k>=20:",[rnd(M[f"inf_{k}"]) for k in (20,30,40,60,80,95)])
print("eps32 rounds k=40,60,80,95:",[rnd(M[f"32.0_{k}"]) for k in (40,60,80,95)], "eps32 k<=30 reach?",[rnd(M[f"32.0_{k}"]) for k in (5,10,20,30)])
print("eps16 rounds k=60,80,95:",[rnd(M[f"16.0_{k}"]) for k in (60,80,95)])
print("eps8 final acc k=95: %.1f; best over k: %.1f"%(100*np.mean(M["8.0_95"][-5:]), max(100*np.mean(M[k][-5:]) for k in M if k.startswith("8.0"))))
print("noise on mean eps16 k=5,95: %.3f %.3f"%(S["16.0_5"]/5,S["16.0_95"]/95))
print("proxy noise k=5,95: %.3f %.3f"%(G2["sigma"]["16.0_5"]/5,G2["sigma"]["16.0_95"]/95), "ratio %.1f %.1f"%(S["16.0_5"]/G2["sigma"]["16.0_5"],S["16.0_95"]/G2["sigma"]["16.0_95"]))
from collections import defaultdict
A=defaultdict(list)
for e,k,mu,s,fin,cur in B["res"]: A[(e,k,mu)].append(np.mean(fin))
print("TableII eps16 k60/95 mu0.1: %.1f %.1f ; mu0 %.1f %.1f ; local %.1f"%(100*np.mean(A[("16.0",60,0.1)]),100*np.mean(A[("16.0",95,0.1)]),100*np.mean(A[("16.0",60,0.0)]),100*np.mean(A[("16.0",95,0.0)]),100*np.mean(B["local"])))
print("seed-std of mean range (pts): %.1f-%.1f"%(min(100*np.std(v) for v in A.values()),max(100*np.std(v) for v in A.values())))
print("proxy design:",P["best"][1:3],"rounds",P["best"][5],"time %.0f"%P["best"][0],"acc %.0f"%P["acc"],"true eps %.0f"%P["tru"]["60"])
n=100
for eta in (0.1,0.01,0.001):
    print("kmin",eta,next(k for k in range(1,n+1) if 1-comb(n-8,k)/comb(n,k)>=1-eta))
try:
    from sklearn.datasets import load_digits; d=load_digits(); print("digits",d.data.shape, "train/client ~",round((len(d.data)-360)/100,1))
except Exception as e: print("sklearn?",e)
