import json, sys, numpy as np
sys.path.insert(0, "..")
import fl_age
from fl_age import age_indep
class H2(fl_age.Hypo):
    def __init__(s, md, mu, npts=10001, top=None): super().__init__(md, mu, 10001, top)
fl_age.Hypo = H2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from acct import coverage, eps
G = json.load(open("grid.json")); G2 = json.load(open("grid2.json")); G["mean"].update(G2["mean"]); G["sigma"].update(G2["sigma"]); R = G["R"]; KS = G["ks"]; n = 100; TGT = 0.70; MU = 0.5; DSTAR = 7.4535
EP = ["inf", "16.0", "12.0", "8.0"]
def sm(c, w=7): c = np.array(c); return np.convolve(np.pad(c, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")
def rnd(c, t=TGT):
    i = np.flatnonzero(sm(c) >= t); return int(i[0]) + 1 if len(i) else None
rt = {(e, k): rnd(G["mean"][f"{e}_{k}"]) for e in EP for k in KS}
pairs = {(m, k): age_indep(k, m, n, mu_dl=MU) for k in KS for m in range(k, n + 1)}
def best(e, gam, kfix=None):
    b = None
    for (m, k), (D, Ey) in pairs.items():
        if rt[(e, k)] is None or D > (1 + gam) * DSTAR: continue
        T = rt[(e, k)] * Ey
        if b is None or T < b[0]: b = (T, m, k, D, Ey, rt[(e, k)])
    return b
out = {"target": TGT, "mu_dl": MU, "rounds": {f"{e}_{k}": rt[(e, k)] for e in EP for k in KS}, "design": {}}
L = [r"\begin{tabular}{@{}clccccc@{}}", r"\toprule", r"$\varepsilon_0$ & design & $(m,k)$ & $\bar\Delta$ & $\mathbb{E}[Y]$ & rounds & time $A$\\ \midrule"]
for e in EP:
    lab = r"$\infty$" if e == "inf" else e.replace(".0", "")
    for gam, nm in ((0.25, r"age-constr. ($\gamma{=}0.25$)"), (1e9, "time-opt.")):
        b = best(e, gam)
        if b is None: continue
        T, m, k, D, Ey, r_ = b; out["design"][f"{e}_{gam}"] = b
        L.append(f"{lab} & {nm} & ({m},{k}) & {D:.1f} & {Ey:.1f} & {r_} & {T:.0f}\\\\")
    # privacy-oblivious: design chosen for eps=inf, run under this eps
    if e != "inf":
        for gam, nm in ((1e9, "obliv. time-opt."),):
            T0, m, k, D, Ey, _ = best("inf", gam)
            r_ = rt[(e, k)]
            if r_ is None: L.append(f"{lab} & {nm} & ({m},{k}) & {D:.1f} & {Ey:.1f} & -- & --\\\\")
            else:
                L.append(f"{lab} & {nm} & ({m},{k}) & {D:.1f} & {Ey:.1f} & {r_} & {r_*Ey:.0f}\\\\"); out["design"][f"obl_{e}_{gam}"] = (r_ * Ey, m, k, D, Ey, r_)
    if e != EP[-1]: L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]; open("tab_design_dp.tex", "w").write("\n".join(L))
# noise on the averaged update and sigma
sig = G["sigma"]
out["noise_on_mean"] = {f"{e}_{k}": sig[f"{e}_{k}"] / k for e in EP if e != "inf" for k in KS}
# ---------- figure ----------
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True, "grid.alpha": .3, "lines.linewidth": 1.2})
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))
col = {"inf": "#4C72B0", "16.0": "#8172B3", "12.0": "#55A868", "8.0": "#DD8452"}; nm = {"inf": "no DP", "16.0": r"$\varepsilon_0{=}16$", "12.0": r"$\varepsilon_0{=}12$", "8.0": r"$\varepsilon_0{=}8$"}
for e in ["inf", "12.0", "8.0"]:
    for k, ls in ((10, ":"), (60, "-")):
        ax[0].plot(np.arange(1, 61), sm(G["mean"][f"{e}_{k}"])[:60], ls, color=col[e], label=f"{nm[e]}, k={k}")
ax[0].set_xlabel("round"); ax[0].set_ylabel("mean task accuracy"); ax[0].legend(ncol=1, fontsize=5.3, loc="lower right"); ax[0].set_title("(a) accuracy vs. round", fontsize=8)
kk = np.arange(2, 96)
for e in EP:
    ks_ = [k for k in KS if rt[(e, k)] is not None]; ax[1].plot(ks_, [rt[(e, k)] for k in ks_], "o-", ms=3, color=col[e], label=nm[e])
ax[1].set_xlabel("$k$"); ax[1].set_ylabel(f"rounds to {int(TGT*100)}% accuracy"); ax[1].set_yscale("log"); ax[1].legend(fontsize=5.8, loc="upper right")
a2 = ax[1].twinx(); a2.plot(kk, [coverage(n, 8, k) for k in kk], "k--", lw=.9); a2.set_ylabel("$P_{cov}$, smallest task ($n_\\tau{=}8$)", fontsize=6.5); a2.set_ylim(0, 1.05); a2.grid(False); a2.tick_params(labelsize=6.5)
ax[1].set_title("(b) rounds and coverage", fontsize=8)
for e in EP:
    ks_, Ts = [], []
    for k in KS:
        if rt[(e, k)] is None: continue
        cand = [rt[(e, k)] * Ey for (m, kk_), (D, Ey) in pairs.items() if kk_ == k]
        if cand: ks_.append(k); Ts.append(min(cand))
    ax[2].plot(ks_, Ts, "o-", ms=3, color=col[e], label=nm[e]); i_ = int(np.argmin(Ts)); ax[2].plot(ks_[i_], Ts[i_], "*", ms=9, color=col[e], mec="k")
ax[2].set_xlabel("$k$"); ax[2].set_ylabel("wall-clock time to 70% accuracy"); ax[2].set_yscale("log"); ax[2].legend(fontsize=5.8); ax[2].set_title("(c) time vs. $k$ (best $m$)", fontsize=8)
fig.tight_layout(); fig.savefig("fig_dp.png", dpi=300, bbox_inches="tight"); plt.close()
json.dump(out, open("final_dp.json", "w"), indent=1, default=float)
print(open("tab_design_dp.tex").read()); print({k: v for k, v in out["rounds"].items()})
