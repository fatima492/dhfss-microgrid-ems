"""Hesitant fuzzy constructions, coupled temporal operators, risk-averse scores and ranking tools.

Notation (paper): couplings  IND = independent (Cartesian), COM = comonotone (quantile-aligned),
EMP = empirical (same-day); means AR = arithmetic, EIN = Einstein, ALG = algebraic.
HFEs are sorted multisets (numpy arrays); repeated grades are kept.
"""
from __future__ import annotations

from math import lcm

import numpy as np

QUANTILES = (0.10, 0.50, 0.90)


# ------------------------------------------------------------- HFE construction
def hfe_quantile(x, qs=QUANTILES):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return np.sort(np.quantile(x, qs))


def hfe_meansd(x, kappa):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    mu, sd = x.mean(), x.std(ddof=0)
    if kappa == 0:
        return np.array([mu])
    return np.sort(np.clip([mu - kappa * sd, mu, mu + kappa * sd], 0, 1))


def hfe_bootstrap_mean(x, rng, B=400, qs=QUANTILES):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    means = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return np.sort(np.quantile(means, qs))


def build_hfes(X, method="quantile", kappa=1.0, rng=None):
    """X[i, j, k, d] -> H[i][j][k] (sorted multisets)."""
    m, n, p, _ = X.shape
    H = [[[None] * p for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                x = X[i, j, k]
                if method == "quantile":
                    H[i][j][k] = hfe_quantile(x)
                elif method == "meansd":
                    H[i][j][k] = hfe_meansd(x, kappa)
                elif method == "bootstrap":
                    H[i][j][k] = hfe_bootstrap_mean(x, rng)
                elif method == "crisp":
                    H[i][j][k] = np.array([np.nanmean(x)])
                elif method == "full":
                    H[i][j][k] = np.sort(x[~np.isnan(x)])
                else:
                    raise ValueError(method)
    return H


# ------------------------------------------------------------- means
def M_ar(G, w):
    return np.tensordot(w, G, axes=(0, 0))


def M_alg(G, w):
    return 1 - np.prod((1 - G) ** np.asarray(w).reshape((-1,) + (1,) * (G.ndim - 1)), axis=0)


def M_ein(G, w):
    ww = np.asarray(w).reshape((-1,) + (1,) * (G.ndim - 1))
    a = np.prod((1 + G) ** ww, axis=0); b = np.prod((1 - G) ** ww, axis=0)
    return (a - b) / (a + b)


MEANS = {"AR": M_ar, "EIN": M_ein, "ALG": M_alg}


def cap(h, eps):
    return np.minimum(h, 1 - eps) if eps > 0 else h


# ------------------------------------------------------------- couplings
def _combos(hs):
    grids = np.meshgrid(*hs, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=0)


def tuples_ind(hs):
    return _combos(hs)


def tuples_com(hs):
    L = lcm(*[len(h) for h in hs])
    return np.stack([np.repeat(np.sort(h), L // len(h)) for h in hs])


def aggregate_hfe(hs, w, coupling="IND", mean="AR", eps=0.0):
    hs = [cap(np.asarray(h, float), eps) for h in hs]
    T = tuples_ind(hs) if coupling == "IND" else tuples_com(hs)
    return np.sort(MEANS[mean](T, np.asarray(w)))


def score(h):
    return float(np.mean(h))


def deviation(h):
    return float(np.std(h, ddof=0))


def lower_tail_mean(values, alpha, probs=None):
    """Mean of the worst alpha-share of the distribution (CVaR-type lower-tail mean)."""
    v = np.asarray(values, float)
    pr = np.full(len(v), 1.0 / len(v)) if probs is None else np.asarray(probs, float)
    o = np.argsort(v); v = v[o]; pr = pr[o]
    c = np.cumsum(pr); prev = np.r_[0.0, c[:-1]]
    take = np.clip(np.minimum(c, alpha) - prev, 0, None)
    return float((take * v).sum() / alpha)


def aggregate_matrix(H, w, coupling="IND", mean="AR", eps=0.0):
    m, n = len(H), len(H[0])
    Hagg = [[aggregate_hfe(H[i][j], w, coupling, mean, eps) for j in range(n)] for i in range(m)]
    S = np.array([[score(Hagg[i][j]) for j in range(n)] for i in range(m)])
    D = np.array([[deviation(Hagg[i][j]) for j in range(n)] for i in range(m)])
    return Hagg, S, D


# ------------------------------------------------------------- full empirical distributions
def emp_values(X_ijk, w):
    """Empirical (same-day) coupling: one weighted value per day. X_ijk: (p, N)."""
    return np.tensordot(w, X_ijk, axes=(0, 0))


def com_values(X_ijk, w):
    return np.tensordot(w, np.sort(X_ijk, axis=1), axes=(0, 0))


def ind_distribution(X_ijk, w, h=2e-4):
    """Exact-up-to-binning law of sum_k w_k X_k with independent X_k ~ empirical marginals."""
    from numpy.fft import rfft, irfft
    p, N = X_ijk.shape
    grid_n = int(np.ceil(1.0 / h)) + 1
    pmf = None
    for k in range(p):
        vals = w[k] * X_ijk[k]
        b = np.rint(vals / h).astype(int)
        f = np.bincount(b, minlength=grid_n)[:grid_n].astype(float) / N
        if pmf is None:
            pmf = f
        else:
            L = len(pmf) + len(f) - 1; nfft = 1 << (L - 1).bit_length()
            pmf = irfft(rfft(pmf, nfft) * rfft(f, nfft), nfft)[:L]
            pmf = np.clip(pmf, 0, None)
    pmf = pmf / pmf.sum()
    support = np.arange(len(pmf)) * h
    keep = pmf > 1e-15
    return support[keep], pmf[keep]


def coupling_stats(X_ijk, w, alphas=(0.10, 0.25)):
    """Score, deviation and lower-tail means under IND, EMP, COM couplings (arithmetic mean)."""
    out = {}
    e = emp_values(X_ijk, w); c = com_values(X_ijk, w)
    sk = X_ijk.std(axis=1)
    s_ind = float(np.sqrt((np.asarray(w) ** 2 * sk ** 2).sum()))
    sup, pr = ind_distribution(X_ijk, w)
    out["S"] = float(e.mean())
    out["sd"] = {"IND": s_ind, "EMP": float(e.std()), "COM": float(c.std())}
    out["sd_bound_upper"] = float((np.asarray(w) * sk).sum())
    for a in alphas:
        out[f"L{int(a*100)}"] = {"IND": lower_tail_mean(sup, a, pr), "EMP": lower_tail_mean(e, a),
                                 "COM": lower_tail_mean(c, a)}
    return out


def majorization_check(a, b, tol=1e-9):
    """True if vector a is majorized by b (equal sums, top-j partial sums of a <= those of b)."""
    a = np.sort(a)[::-1]; b = np.sort(b)[::-1]
    return bool(abs(a.sum() - b.sum()) < tol * max(1, abs(b.sum())) and
                np.all(np.cumsum(a) <= np.cumsum(b) + tol * max(1, abs(b.sum()))))


# ------------------------------------------------------------- weights
def temporal_weights_exp(lam, theta, wS=(0.30, 0.35, 0.35)):
    k = np.arange(len(wS))
    wo = np.exp(lam * k) / np.exp(lam * k).sum()
    return theta * np.asarray(wS) + (1 - theta) * wo


def roc_weights(order, n):
    """Rank-order centroid weights (Barron & Barrett, 1996); order = criterion indices, most important first."""
    w = np.zeros(n)
    for r, j in enumerate(order):
        w[j] = sum(1.0 / q for q in range(r + 1, n + 1)) / n
    return w


def entropy_weights(S):
    P = S / S.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=0) / np.log(S.shape[0])
    d = 1 - e
    return d / d.sum()


# ------------------------------------------------------------- ranking methods
def mabac(S, w):
    V = w * (S + 1.0)
    g = np.prod(V, axis=0) ** (1.0 / S.shape[0])
    Qm = V - g
    return Qm.sum(axis=1), V, g, Qm


def wsm(S, w):
    return S @ w


def topsis(X, w, cost_cols=()):
    X = np.asarray(X, float)
    R = X / np.sqrt((X ** 2).sum(axis=0)); V = R * w
    is_cost = np.isin(np.arange(X.shape[1]), cost_cols)
    best = np.where(is_cost, V.min(0), V.max(0)); worst = np.where(is_cost, V.max(0), V.min(0))
    dp = np.sqrt(((V - best) ** 2).sum(1)); dm = np.sqrt(((V - worst) ** 2).sum(1))
    return dm / (dp + dm), dp, dm


def _extend(h, L):
    h = np.sort(h)
    return h if len(h) == L else np.sort(np.concatenate([h, np.full(L - len(h), h.min())]))


def hf_topsis(Hagg, w):
    m, n = len(Hagg), len(Hagg[0])
    dp = np.zeros(m); dm = np.zeros(m)
    for j in range(n):
        L = max(len(Hagg[i][j]) for i in range(m))
        E = np.stack([_extend(Hagg[i][j], L) for i in range(m)])
        hp, hn = E.max(axis=0), E.min(axis=0)
        dp += w[j] * np.abs(E - hp).mean(axis=1); dm += w[j] * np.abs(E - hn).mean(axis=1)
    return dm / (dp + dm), dp, dm


def ranking_string(Q, labels):
    order = np.argsort(-np.asarray(Q), kind="stable")
    return " > ".join(labels[i] for i in order)


def ranks(Q):
    order = np.argsort(-np.asarray(Q), kind="stable")
    r = np.empty(len(Q), int); r[order] = np.arange(1, len(Q) + 1)
    return r


# ------------------------------------------------------------- block tensor
def block_tensor(blocks, alts, criteria, periods):
    days = None
    for a in alts:
        b = blocks[a]
        for per in periods:
            d = set(b.loc[b.period == per, "opday"])
            days = d if days is None else days & d
    days = sorted(days)
    X = np.full((len(alts), len(criteria), len(periods), len(days)), np.nan)
    for i, a in enumerate(alts):
        b = blocks[a].set_index(["period", "opday"])
        for k, per in enumerate(periods):
            sub = b.loc[per].loc[days]
            for j, c in enumerate(criteria):
                X[i, j, k] = sub[c].to_numpy()
    return X, days


def scores_quantile(X, omega, qs=QUANTILES):
    """Arithmetic-operator scores for quantile HFEs, via the coupling-invariant score identity."""
    Hq = np.nanquantile(X, qs, axis=-1)                      # (q, m, n, p)
    return np.tensordot(Hq.mean(axis=0), omega, axes=([2], [0]))


# ------------------------------------------------------------- robustness
def smaa(S, n_samples=10_000, conc=None, base=None, seed=2026):
    rng = np.random.default_rng(seed)
    m, n = S.shape
    alpha = np.ones(n) if conc is None else conc * np.asarray(base)
    W = rng.dirichlet(alpha, size=n_samples)
    Q = (S[None, :, :] * W[:, None, :]).sum(-1)
    R = np.argsort(np.argsort(-Q, axis=1), axis=1)
    b = np.stack([(R == r).mean(axis=0) for r in range(m)], axis=1)
    central = np.stack([W[R[:, i] == 0].mean(axis=0) if (R[:, i] == 0).any() else np.full(n, np.nan)
                        for i in range(m)])
    return b, central


def bootstrap_Q(X, omega, w, B=1000, block=28, seed=2026):
    rng = np.random.default_rng(seed)
    m = X.shape[0]; Dn = X.shape[-1]
    Qs = np.zeros((B, m))
    for b in range(B):
        starts = rng.integers(0, Dn - block + 1, size=int(np.ceil(Dn / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:Dn]
        S = scores_quantile(X[..., idx], omega)
        Qs[b] = mabac(S, w)[0]
    return Qs
