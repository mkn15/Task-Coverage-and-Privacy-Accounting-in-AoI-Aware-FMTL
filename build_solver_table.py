"""Builds ../tables/tab_solver.tex (Table II) from baselines.json."""
import json, numpy as np
from collections import defaultdict
d = json.load(open("baselines.json")); A = defaultdict(list)
for e, k, mu, s, fin, cur in d["res"]: A[(e, k, mu)].append(fin)
def cell(e, k, mu):
    v = np.array(A[(e, k, mu)]); return "%.1f (%d)" % (100 * v.mean(1).mean(), round(100 * v.min(1).mean()))
rows = [("inf", 60, "no DP, $k{=}60$"), ("32.0", 60, "$\\varepsilon_0{=}32$, $k{=}60$"), ("32.0", 95, "$\\varepsilon_0{=}32$, $k{=}95$"), ("16.0", 60, "$\\varepsilon_0{=}16$, $k{=}60$"), ("16.0", 95, "$\\varepsilon_0{=}16$, $k{=}95$")]
t = ["\\begin{tabular}{lccc}", "\\hline", "setting & FedAvg ($\\mu{=}0$) & FedProx ($\\mu{=}0.1$) & FedProx ($\\mu{=}0.5$)\\\\", "\\hline"]
for e, k, nm in rows: t.append(nm + " & " + " & ".join(cell(e, k, mu) for mu in (0.0, 0.1, 0.5)) + "\\\\")
lo = np.array(d["local"])
t += ["\\hline", "local-only (no federation, no DP) & \\multicolumn{3}{c}{%.1f (%d)}\\\\" % (100 * lo.mean(1).mean(), round(100 * lo.min(1).mean())), "\\hline", "\\end{tabular}"]
open("../tables/tab_solver.tex", "w").write("\n".join(t) + "\n")
