"""What if sigma is chosen with the Poisson proxy (claimed eps0=16) but the PS sees the delivered set? Design under proxy noise, then the TRUE PS-visible eps."""
import json, sys, numpy as np
import fl_age
from fl_age import age_indep
class H2(fl_age.Hypo):
    def __init__(s, md, mu, npts=10001, top=None): super().__init__(md, mu, 10001, top)
fl_age.Hypo = H2
from acct2 import eps_visible, n_worst
G = json.load(open("grid2.json")); n, R, MU, DSTAR, TGT = 100, 150, 0.5, 7.4535, 0.70
KS = G["ks"]
def sm(c, w=7): c = np.array(c); return np.convolve(np.pad(c, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")
def rnd(c): i = np.flatnonzero(sm(c) >= TGT); return int(i[0]) + 1 if len(i) else None
out = {}
rt = {k: rnd(G["mean"][f"16.0_{k}"]) for k in KS}
tru = {k: eps_visible(n_worst(k, n, R), G["sigma"][f"16.0_{k}"]) for k in KS}
print("rounds", rt); print("true eps", {k: round(v, 1) for k, v in tru.items()})
b = None
for k in KS:
    if rt[k] is None: continue
    for m in range(k, n + 1):
        D, Ey = age_indep(k, m, n, mu_dl=MU)
        if D > 1.25 * DSTAR: continue
        T = rt[k] * Ey
        if b is None or T < b[0]: b = (T, m, k, D, Ey, rt[k])
print("best age-constrained under proxy sigma:", b, "true eps", tru[b[2]], "final acc", 100 * np.mean(G["mean"][f"16.0_{b[2]}"][-5:]))
json.dump(dict(best=b, tru={str(k): v for k, v in tru.items()}, rt={str(k): v for k, v in rt.items()}, acc=100 * float(np.mean(G["mean"][f"16.0_{b[2]}"][-5:]))), open("proxy_design.json", "w"))
