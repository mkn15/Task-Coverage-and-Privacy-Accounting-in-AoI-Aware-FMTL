import json, sys, numpy as np
sys.path.insert(0, "..")
import fl_age
from fl_age import age_indep
class H2(fl_age.Hypo):
    def __init__(s, md, mu, npts=10001, top=None): super().__init__(md, mu, 10001, top)
fl_age.Hypo = H2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from acct import coverage, eps
G = json.load(open("grid.json")); G3 = json.load(open("grid3.json")); G["mean"] = {k: v for k, v in G["mean"].items() if k.startswith("inf")}; G["mean"].update(G3["mean"]); G["sigma"] = G3["sigma"]; GS = {k: v for k, v in json.load(open("grid.json"))["std"].items() if k.startswith("inf")}; GS.update(G3["std"]); R = G["R"]; KS = G["ks"]; n = 100; TGT = 0.70; MU = 0.5; DSTAR = 7.4535
EP = ["inf", "32.0", "16.0"]
def fa(e, k): return 100 * float(np.mean(G["mean"][f"{e}_{k}"][-5:]))
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
L = [r"\begin{tabular}{@{}clcccccc@{}}", r"\toprule", r"$\varepsilon_0$ & design & $(m,k)$ & $\bar\Delta$ & $\mathbb{E}[Y]$ & rounds & time $A$ & acc. (\%) \\ \midrule"]
for e in EP:
    lab = r"$\infty$" if e == "inf" else e.replace(".0", "")
    for gam, nm in ((0.25, r"age-constr. ($\gamma{=}0.25$)"), (1e9, "time-opt.")):
        b = best(e, gam)
        if b is None: continue
        T, m, k, D, Ey, r_ = b; out["design"][f"{e}_{gam}"] = b
        L.append(f"{lab} & {nm} & ({m},{k}) & {D:.1f} & {Ey:.1f} & {r_} & {T:.0f} & {fa(e,k):.0f}\\\\")
    # privacy-oblivious: design chosen for eps=inf, run under this eps
    if e != "inf":
        for gam, nm in ((1e9, "obliv. time-opt."),):
            T0, m, k, D, Ey, _ = best("inf", gam)
            r_ = rt[(e, k)]
            if r_ is None: L.append(f"{lab} & {nm} & ({m},{k}) & {D:.1f} & {Ey:.1f} & -- & -- & {fa(e,k):.0f}\\\\")
            else:
                L.append(f"{lab} & {nm} & ({m},{k}) & {D:.1f} & {Ey:.1f} & {r_} & {r_*Ey:.0f} & {fa(e,k):.0f}\\\\"); out["design"][f"obl_{e}_{gam}"] = (r_ * Ey, m, k, D, Ey, r_)
    if e != EP[-1]: L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]; open("tab_design_dp.tex", "w").write("\n".join(L))
# noise on the averaged update and sigma
sig = G["sigma"]
out["noise_on_mean"] = {f"{e}_{k}": sig[f"{e}_{k}"] / k for e in EP if e != "inf" for k in KS}
# ---------- figure (IEEE style: Times-like fonts, 8 pt, grayscale-safe markers/line styles) ----------
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.linewidth": .35, "grid.alpha": .45, "lines.linewidth": 1.1, "lines.markersize": 4})
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
col = {"inf": "#000000", "32.0": "#1F5FA8", "16.0": "#D55E00"}; mk = {"inf": "o", "32.0": "s", "16.0": "^"}; ls_ = {"inf": "-", "32.0": "--", "16.0": "-."}
nm = {"inf": "no DP", "32.0": r"$\varepsilon_0=32$", "16.0": r"$\varepsilon_0=16$"}
for e in EP:
    for k, a_ in ((10, .45), (60, 1.0)):
        ax[0].plot(np.arange(1, 61), sm(G["mean"][f"{e}_{k}"])[:60], ls_[e], color=col[e], alpha=a_, lw=(1.0 if k == 60 else 0.9))
    mu_ = sm(G["mean"][f"{e}_60"])[:60]; sd_ = sm(GS[f"{e}_60"])[:60]; ax[0].fill_between(np.arange(1, 61), mu_ - sd_, mu_ + sd_, color=col[e], alpha=.15, lw=0)
ax[0].set_xlabel("Round $t$"); ax[0].set_ylabel("Mean task accuracy"); ax[0].set_xlim(0, 60); ax[0].set_ylim(0.4, 0.9)
ax[0].legend([Line2D([], [], color="0.2", lw=1.1), Line2D([], [], color="0.2", lw=1.1, alpha=.45)], [r"$k=60$", r"$k=10$"], loc="upper left", fontsize=7, handlelength=1.6, borderpad=.3, labelspacing=.25, frameon=True, edgecolor="0.6", fancybox=False)
kk = np.arange(2, 96)
a2 = ax[1].twinx(); a2.plot(kk, [coverage(n, 8, k) for k in kk], color="0.45", ls=":", lw=1.1); a2.set_ylim(0, 1.05); a2.grid(False)
a2.set_ylabel(r"Coverage $P_{\rm cov}$ ($n_s=8$)", fontsize=7.5); a2.tick_params(labelsize=7, width=.6, length=2.5)
for e in EP:
    ks_ = [k for k in KS if rt[(e, k)] is not None]; ax[1].plot(ks_, [rt[(e, k)] for k in ks_], ls_[e], marker=mk[e], color=col[e], mfc=col[e] if e != "inf" else "white", zorder=3)
ax[1].set_zorder(a2.get_zorder() + 1); ax[1].patch.set_visible(False)
ax[1].set_xlabel(r"$k$ (updates per round)"); ax[1].set_ylabel(r"Rounds to 70% accuracy"); ax[1].set_yscale("log"); ax[1].set_xlim(0, 100)
ax[1].yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 150])); ax[1].yaxis.set_major_formatter(FixedFormatter(["10", "20", "50", "100", "150"])); ax[1].yaxis.set_minor_locator(NullLocator()); ax[1].set_ylim(10, 170)
for e in EP:
    ks_, Ts = [], []
    for k in KS:
        if rt[(e, k)] is None: continue
        cand = [rt[(e, k)] * Ey for (m, kk_), (D, Ey) in pairs.items() if kk_ == k]
        if cand: ks_.append(k); Ts.append(min(cand))
    ax[2].plot(ks_, Ts, ls_[e], marker=mk[e], color=col[e], mfc=col[e] if e != "inf" else "white")
    i_ = int(np.argmin(Ts)); ax[2].plot(ks_[i_], Ts[i_], marker="*", ms=10, color="gold", mec="k", mew=.7, ls="none", zorder=5)
ax[2].set_xlabel(r"$k$ (updates per round)"); ax[2].set_ylabel(r"Time to 70% accuracy ($1/\lambda$)"); ax[2].set_yscale("log"); ax[2].set_xlim(0, 100)
ax[2].yaxis.set_major_locator(FixedLocator([50, 100, 200, 500, 1000])); ax[2].yaxis.set_major_formatter(FixedFormatter(["50", "100", "200", "500", "1000"])); ax[2].yaxis.set_minor_locator(NullLocator()); ax[2].set_ylim(35, 1300)
hh = [Line2D([], [], color=col[e], ls=ls_[e], marker=mk[e], mfc=col[e] if e != "inf" else "white") for e in EP] + [Line2D([], [], color="0.45", ls=":"), Line2D([], [], color="gold", marker="*", ms=9, mec="k", mew=.7, ls="none")]
fig.legend(hh, [nm[e] for e in EP] + [r"$P_{\rm cov}$ (b, right axis)", "minimum (c)"], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.02), columnspacing=1.6, handletextpad=.5, handlelength=2.2)
fig.tight_layout(rect=(0, .05, 1, .92), w_pad=1.2)
for a, t in zip(ax, ("(a)", "(b)", "(c)")):
    p = a.get_position(); fig.text((p.x0 + p.x1) / 2 + (0.0 if a is not ax[1] else 0.0), 0.015, t, ha="center", va="bottom", fontsize=8)
fig.savefig("fig_dp.png", dpi=600); plt.close()
