"""DP-FedProx multitask on digits: shared rep (64->H, ReLU) + S task heads; k-of-n uniform selection; distributed Gaussian noise (sum noise sigma*C)."""
import numpy as np, sys, json
from sklearn.datasets import load_digits
from acct import calibrate, eps
H = 16; n = 100; NT = [35, 25, 18, 14, 8]; S = 5
def lab(d):  # task labelings of the digit
    return [d % 2, (d >= 5).astype(int), d % 3, np.isin(d, [2, 3, 5, 7]).astype(int), d % 4]
NC = [2, 2, 3, 2, 4]
def data(seed):
    X, y = load_digits(return_X_y=True); X = X / 16.0
    r = np.random.default_rng(seed); p = r.permutation(len(X)); te, tr = p[:360], p[360:]
    shards = np.array_split(tr, n); task = np.repeat(np.arange(S), NT)[r.permutation(n)]
    return X, y, te, shards, task
def init(r):
    P = {"Wsh": r.normal(0, .3, (H, 64)), "bsh": np.zeros(H)}
    for s in range(S): P[f"W{s}"] = r.normal(0, .3, (NC[s], H)); P[f"b{s}"] = np.zeros(NC[s])
    return P
def fwd(P, X, s):
    h = np.maximum(X @ P["Wsh"].T + P["bsh"], 0); z = h @ P[f"W{s}"].T + P[f"b{s}"]
    z -= z.max(1, keepdims=True); e = np.exp(z); return h, e / e.sum(1, keepdims=True)
def local(P0, X, yl, s, eta, steps, mu, C):
    P = {k: v.copy() for k, v in P0.items() if k in ("Wsh", "bsh", f"W{s}", f"b{s}")}
    N = len(X)
    for _ in range(steps):
        h, pr = fwd(P, X, s); g = pr.copy(); g[np.arange(N), yl] -= 1; g /= N
        dh = (g @ P[f"W{s}"]) * (h > 0)
        gr = {f"W{s}": g.T @ h, f"b{s}": g.sum(0), "Wsh": dh.T @ X, "bsh": dh.sum(0)}
        for k in P: P[k] -= eta * (gr[k] + mu * (P[k] - P0[k]))
    d = {k: P[k] - P0[k] for k in P}
    nrm = np.sqrt(sum((v**2).sum() for v in d.values())); f = min(1.0, C / (nrm + 1e-12))
    return {k: v * f for k, v in d.items()}
def acc(P, X, y, te):
    L = lab(y[te]); a = []
    for s in range(S):
        _, pr = fwd(P, X[te], s); a.append((pr.argmax(1) == L[s]).mean())
    return np.mean(a), a
def run(k, sigma, R, seed, eta=0.5, steps=5, mu=0.1, C=1.0, slr=1.0):
    X, y, te, shards, task = data(seed); r = np.random.default_rng(1000 + seed); P = init(r)
    LB = [lab(y[sh]) for sh in shards]; curve = []; upd = np.zeros(S)
    for t in range(R):
        sel = r.permutation(n)[:k]; acc_d = {key: np.zeros_like(v) for key, v in P.items()}
        for j in sel:
            s = task[j]; d = local(P, X[shards[j]], LB[j][s], s, eta, steps, mu, C)
            for key, v in d.items(): acc_d[key] += v
            upd[s] += 1
        for key in P:  # noise on the SUM (sum of k client shares each N(0,sigma^2 C^2/k)); heads of absent tasks get no update
            if key in ("Wsh", "bsh"): pass
            else:
                if not any(task[j] == int(key[1]) for j in sel): continue
            P[key] += slr * (acc_d[key] + r.normal(0, sigma * C, P[key].shape)) / k
        curve.append(acc(P, X, y, te)[0])
    return np.array(curve)
if __name__ == "__main__":
    import time; t0 = time.time()
    for k, sg in ((20, 0.0), (20, 1.56), (60, 0.0), (60, 3.93)):
        c = run(k, sg, 60, 0); print(k, sg, np.round(c[[9, 29, 59]], 3), round(time.time() - t0, 1))
