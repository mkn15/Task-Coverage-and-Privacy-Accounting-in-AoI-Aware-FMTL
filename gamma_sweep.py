"""Sensitivity of the joint design to the age slack gamma (Section V): best (m,k) and time A for several gamma."""
import json, numpy as np
import fl_age
from fl_age import age_indep
class H2(fl_age.Hypo):
    def __init__(s, md, mu, npts=10001, top=None): super().__init__(md, mu, 10001, top)
fl_age.Hypo = H2
G = json.load(open("grid.json")); G3 = json.load(open("grid3.json")); n, MU, DSTAR, TGT = 100, 0.5, 7.4535, 0.70
M = dict(G["mean"]); M.update(G3["mean"]); KS = G["ks"]
def sm(c, w=7): c = np.array(c); return np.convolve(np.pad(c, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")
def rnd(c): i = np.flatnonzero(sm(c) >= TGT); return int(i[0]) + 1 if len(i) else None
AGE = {(m, k): age_indep(k, m, n, mu_dl=MU) for k in KS for m in range(k, n + 1)}
out = {}
for e in ("inf", "32.0", "16.0"):
    rt = {k: rnd(M[f"{e}_{k}"]) for k in KS}
    for gam in (0.1, 0.25, 0.5, 1e9):
        b = None
        for (m, k), (D, Ey) in AGE.items():
            if rt[k] is None or D > (1 + gam) * DSTAR: continue
            T = rt[k] * Ey
            if b is None or T < b[0]: b = (T, m, k, D)
        out[f"{e}_{gam}"] = b
        print(e, gam, None if b is None else "A=%.0f (m,k)=(%d,%d) age=%.1f" % (b[0], b[1], b[2], b[3]))
json.dump(out, open("gamma_sweep.json", "w"))
