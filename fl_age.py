"""Average age for timely FL with downlink latency.

Two downlink models (n clients, m awaited, k earliest used):
  shared : one common downlink delay D ~ Exp(mu_dl) per iteration.
           Y = Z_{m:n} + D + c + U_{k:m}.   (closed form, Theorem 1)
  indep  : independent per-client downlink D_j ~ Exp(mu_dl). Earliest-k is decided by
           W_j = D_j + U_j (hypoexponential).  Y = Z_{m:n} + c + W_{k:m}.
Age convention (as in the paper / [2]): data timestamp = end of local computation, so the age of a
delivered update at the PS equals its uplink delay U; X~ = E[U | client is among the earliest k].
Delta = E[X~] + a E[Y] + Var[Y]/(2E[Y]),  a=(2n-k)/(2k).
"""
import numpy as np
from scipy import stats

N_MAX = 2000
_H = np.concatenate(([0.0], np.cumsum(1.0 / np.arange(1, N_MAX + 1))))
_G = np.concatenate(([0.0], np.cumsum(1.0 / np.arange(1, N_MAX + 1) ** 2)))


def _Z(n, m, lam):
    return (_H[n] - _H[n - m]) / lam, (_G[n] - _G[n - m]) / lam**2


# ---------------- shared downlink (closed form) ----------------
def age_shared(k, m, n, lam=1.0, mu_ul=1.0, c=1.0, mu_dl=np.inf):
    EZ, VZ = _Z(n, m, lam)
    EX = (_H[m] - _H[m - k]) / mu_ul
    VX = (_G[m] - _G[m - k]) / mu_ul**2
    Xt = (1.0 - (m - k) / k * (_H[m] - _H[m - k])) / mu_ul
    ED, VD = (0.0, 0.0) if np.isinf(mu_dl) else (1 / mu_dl, 1 / mu_dl**2)
    EY, VY = EZ + ED + c + EX, VZ + VD + VX
    return Xt + (2 * n - k) / (2 * k) * EY + VY / (2 * EY), EY


# ---------------- independent per-client downlink (numerical) ----------------
class Hypo:
    """W = D + U, D~Exp(md), U~Exp(mu). Precomputed grid for order-statistic integrals."""

    def __init__(self, md, mu, npts=40001, top=None):
        self.md, self.mu = md, mu
        if top is None:
            top = 60.0 / min(md, mu) if np.isfinite(md) else 60.0 / mu
        self.w = np.linspace(0, top, npts)
        w = self.w
        if np.isinf(md):
            self.F = 1 - np.exp(-mu * w)
            self.f = mu * np.exp(-mu * w)
            self.g = np.zeros_like(w) if False else w * 0.0 + 0.0  # E[U|W=w]=w when md=inf
            self.g = w.copy()
        else:
            if abs(md - mu) < 1e-9:
                self.F = 1 - np.exp(-mu * w) * (1 + mu * w)
                self.f = mu**2 * w * np.exp(-mu * w)
                self.g = w / 2
            else:
                self.F = 1 - (mu * np.exp(-md * w) - md * np.exp(-mu * w)) / (mu - md)
                self.f = md * mu / (mu - md) * (np.exp(-md * w) - np.exp(-mu * w))
                d = md - mu  # U|W=w has density ∝ exp(d*u) on [0,w]
                with np.errstate(over="ignore", invalid="ignore"):
                    g = w / (1 - np.exp(-d * w)) - 1 / d
                g = np.where(w < 1e-9, w / 2, g)
                self.g = np.clip(g, 0, w)

    def order_moments(self, k, m):
        """E and Var of W_{k:m}."""
        F = np.clip(self.F, 0, 1)
        dens = stats.beta.pdf(F, k, m - k + 1) * self.f
        e1 = _trapz(self.w * dens, self.w)
        e2 = _trapz(self.w**2 * dens, self.w)
        return e1, e2 - e1**2

    def xtilde(self, k, m):
        """E[U | tagged client among earliest k of m] (by W)."""
        F = np.clip(self.F, 0, 1)
        sel = stats.binom.cdf(k - 1, m - 1, F)
        return (m / k) * _trapz(self.g * sel * self.f, self.w)


_trapz = getattr(np, "trapezoid", None) or np.trapz
_cache = {}


def _hypo(md, mu):
    key = (md, mu)
    if key not in _cache:
        _cache[key] = Hypo(md, mu)
    return _cache[key]


def age_indep(k, m, n, lam=1.0, mu_ul=1.0, c=1.0, mu_dl=np.inf):
    EZ, VZ = _Z(n, m, lam)
    h = _hypo(mu_dl, mu_ul)
    EW, VW = h.order_moments(k, m)
    Xt = h.xtilde(k, m)
    EY, VY = EZ + c + EW, VZ + VW
    return Xt + (2 * n - k) / (2 * k) * EY + VY / (2 * EY), EY


def optimum(fn, n=100, ms=None, **kw):
    ms = ms if ms is not None else range(1, n + 1)
    best = (np.inf, None, None, None)
    for m in ms:
        for k in range(1, m + 1):
            a, ey = fn(k, m, n, **kw)
            if a < best[0]:
                best = (a, m, k, ey)
    return best


# ---------------- Monte-Carlo ground truth ----------------
def simulate(k, m, n, lam=1.0, mu_ul=1.0, c=1.0, mu_dl=np.inf, model="indep",
             iters=200000, seed=0):
    """Event-level simulation of one tagged client. Returns (avg age, mean iteration time)."""
    rng = np.random.default_rng(seed)
    Z = np.sort(rng.exponential(1 / lam, (iters, n)), axis=1)[:, m - 1]
    sel = rng.random(iters) < m / n  # tagged client invited this iteration
    # m clients' uplink delays; column 0 = tagged client (only meaningful if sel)
    U = rng.exponential(1 / mu_ul, (iters, m))
    if model == "shared":
        D = rng.exponential(1 / mu_dl, iters) if np.isfinite(mu_dl) else np.zeros(iters)
        W = U
        Y = Z + c + D + np.partition(W, k - 1, axis=1)[:, k - 1]
    else:
        Dm = rng.exponential(1 / mu_dl, (iters, m)) if np.isfinite(mu_dl) else np.zeros((iters, m))
        W = Dm + U
        Y = Z + c + np.partition(W, k - 1, axis=1)[:, k - 1]
    rank = (W[:, 1:] < W[:, [0]]).sum(axis=1)  # others earlier than tagged
    succ = sel & (rank <= k - 1)
    Xdel = U[:, 0]
    T = np.cumsum(Y)
    idx = np.flatnonzero(succ)
    Td, Xd = T[idx], Xdel[idx]
    gap = np.diff(Td)
    area = Xd[:-1] * gap + gap**2 / 2
    return area.sum() / gap.sum(), Y.mean()
