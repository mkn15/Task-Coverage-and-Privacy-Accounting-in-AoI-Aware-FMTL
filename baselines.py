"""Local-solver comparison under the SAME privacy mechanism (same sigma): DP-FedAvg (mu=0), DP-FedProx (mu>0), and a no-federation local-only reference."""
import numpy as np, json
from multiprocessing import Pool
import fmtl
from fmtl import *
G = json.load(open("grid3.json")); R = 150
def run2(k, sigma, seed, mu, eta=0.5, steps=5, C=1.0, slr=1.0):
    X, y, te, shards, task = data(seed); r = np.random.default_rng(1000 + seed); P = init(r)
    LB = [lab(y[sh]) for sh in shards]; curve = []
    for t in range(R):
        sel = r.permutation(n)[:k]; acc_d = {key: np.zeros_like(v) for key, v in P.items()}
        for j in sel:
            s = task[j]; d = local(P, X[shards[j]], LB[j][s], s, eta, steps, mu, C)
            for key, v in d.items(): acc_d[key] += v
        for key in P:
            if key not in ("Wsh", "bsh") and not any(task[j] == int(key[1]) for j in sel): continue
            P[key] += slr * (acc_d[key] + r.normal(0, sigma * C, P[key].shape)) / k
        curve.append(acc(P, X, y, te)[1])
    return np.array(curve)          # R x S per-task accuracy
def local_only(seed, eta=0.5, steps=5*R // 10):
    X, y, te, shards, task = data(seed); r = np.random.default_rng(2000 + seed); out = np.zeros(S); cnt = np.zeros(S)
    L = lab(y[te])
    for j in range(n):
        s = task[j]; P = init(r); lb = lab(y[shards[j]])[s]
        for _ in range(steps):  # plain local training (no clipping, no noise, no communication)
            d = local(P, X[shards[j]], lb, s, eta, 1, 0.0, 1e9)
            for key, v in d.items(): P[key] += v
        _, pr = fwd(P, X[te], s); out[s] += (pr.argmax(1) == L[s]).mean(); cnt[s] += 1
    return out / cnt
def job(a):
    e, k, mu, s = a; sg = 0.0 if e == "inf" else G["sigma"][f"{e}_{k}"]
    c = run2(k, sg, s, mu); return (e, k, mu, s, c[-5:].mean(0).tolist(), c.mean(1).tolist())
if __name__ == "__main__":
    jobs = [(e, k, mu, s) for e in ("inf", "32.0", "16.0") for k in (60, 95) for mu in (0.0, 0.1, 0.5) for s in range(5)]
    with Pool() as p: res = p.map(job, jobs)
    lo = np.array([local_only(s) for s in range(5)])
    json.dump(dict(res=res, local=lo.tolist()), open("baselines.json", "w"))
    print("done", lo.mean())
