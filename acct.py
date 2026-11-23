"""Coverage + RDP accountant for the sampled Gaussian mechanism (Poisson-sampling proxy for k-of-n)."""
import numpy as np
from scipy.special import gammaln, logsumexp, comb
ALPHAS = np.arange(2, 257)
def coverage(n, nt, k):
    """P(at least one of the nt task-clients is among a uniform k-subset of n)."""
    return 1 - comb(n - nt, k, exact=False) / comb(n, k, exact=False)
def rdp(q, sigma, alphas=ALPHAS):
    out = []
    for a in alphas:
        j = np.arange(a + 1)
        lt = gammaln(a + 1) - gammaln(j + 1) - gammaln(a - j + 1) + (a - j) * np.log1p(-q) + j * np.log(q) + (j * j - j) / (2 * sigma**2)
        out.append(logsumexp(lt) / (a - 1))
    return np.array(out)
def eps(q, sigma, T, delta=1e-5):
    r = T * rdp(q, sigma); a = ALPHAS
    e = r + np.log((a - 1) / a) - (np.log(delta) + np.log(a)) / (a - 1)
    return float(np.min(e))
def calibrate(q, T, target, delta=1e-5):
    lo, hi = 0.3, 200.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if eps(q, mid, T, delta) > target: lo = mid
        else: hi = mid
    return hi
if __name__ == "__main__":
    rng = np.random.default_rng(0); n = 100; nt = 8
    for k in (10, 40, 80):
        s = np.mean([(rng.permutation(n)[:k] < nt).any() for _ in range(40000)])
        print(k, round(coverage(n, nt, k), 4), round(s, 4))
    for k in (10, 20, 40, 60, 80):
        q = k / n; s = calibrate(q, 100, 8.0); print(k, "sigma for eps=8, T=100:", round(s, 2), "eps check", round(eps(q, s, 100), 3))
