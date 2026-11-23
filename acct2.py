"""Threat-model-aware accounting for the earliest-k rule (all under the SUBSTITUTION neighbouring relation, L2 sensitivity 2C).
A (hidden selection, external observer): fixed-size k-of-n subsampling w/o replacement, Wang-Balle-Kasiviswanathan Thm 9, Gaussian base.
B (PS sees the delivered set): no amplification; client i pays N_i ~ Bin(R,k/n) Gaussian releases; calibrate to the worst client (union bound).
P (previous proxy): Poisson-sampling sampled-Gaussian, add/remove (acct.py)."""
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom
import acct
AL = np.arange(2, 129)
def _lb(a, j): return gammaln(a + 1) - gammaln(j + 1) - gammaln(a - j + 1)
def rdp_fixed(gam, sigma, alphas=AL):
    """WBK Thm 9, Gaussian base: eps(a)=2a/sigma^2 (substitution), eps(inf)=inf."""
    e = lambda a: 2.0 * a / sigma**2
    out = []
    for a in alphas:
        t2 = 2 * np.log(gam) + _lb(a, 2) + np.log(min(4 * np.expm1(e(2)), 2 * np.exp(e(2)))) if e(2) < 700 else 2 * np.log(gam) + _lb(a, 2) + np.log(2) + e(2)
        terms = [0.0, t2]
        for j in range(3, a + 1): terms.append(j * np.log(gam) + _lb(a, j) + (j - 1) * e(j) + np.log(2))
        out.append(logsumexp(terms) / (a - 1))
    return np.array(out)
def _conv(r, alphas, delta):
    return float(np.min(r + np.log((alphas - 1) / alphas) - (np.log(delta) + np.log(alphas)) / (alphas - 1)))
def eps_fixed(k, n, sigma, R, delta=1e-5): return _conv(R * rdp_fixed(k / n, sigma), AL, delta)
def eps_visible(N, sigma, delta=1e-5):
    a = np.arange(2, 513).astype(float); return _conv(N * 2 * a / sigma**2, a, delta)
def n_worst(k, n, R, conf=0.05):
    q = k / n; N = int(binom.ppf(1 - conf / n, R, q)); return max(N, 1)
def _cal(f, target, lo=0.3, hi=5000.0):
    for _ in range(45):
        mid = np.sqrt(lo * hi)
        if f(mid) > target: lo = mid
        else: hi = mid
    return hi
def sigma_fixed(k, n, R, eps0, delta=1e-5): return _cal(lambda s: eps_fixed(k, n, s, R, delta), eps0)
def sigma_visible(k, n, R, eps0, delta=1e-5): return _cal(lambda s: eps_visible(n_worst(k, n, R), s, delta), eps0)
def sigma_poisson(k, n, R, eps0, delta=1e-5): return acct.calibrate(k / n, R, eps0, delta)
if __name__ == "__main__":
    n, R = 100, 150
    for e0 in (8.0, 16.0, 32.0):
        print("eps0", e0, " k | sigma Poisson/fixed/visible | noise on mean sigma/k (x C)")
        for k in (5, 10, 20, 40, 60, 95):
            sp, sf, sv = sigma_poisson(k, n, R, e0), sigma_fixed(k, n, R, e0), sigma_visible(k, n, R, e0)
            print(f"  {k:3d} | {sp:6.2f} {sf:6.2f} {sv:6.2f} | {sp/k:.3f} {sf/k:.3f} {sv/k:.3f}  Nworst={n_worst(k,n,R)}")
