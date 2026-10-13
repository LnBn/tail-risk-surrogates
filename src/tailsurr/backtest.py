"""One-day VaR / ES estimators for the residual tail, and coverage tests.

Conventions: losses are positive numbers (loss = -return).  `q` is the VaR
confidence level (e.g. 0.99); the VaR is the q-quantile of the loss and the
ES is the mean loss beyond it.

Tail estimators work on a window of (residual) losses y with the k largest
values treated as the tail and u = the (k+1)-th largest value as threshold,
as in McNeil & Frey (2000).  Within the tail, the required exceedance
probability is p_tail = (1 - q) * n / k.
"""

import numpy as np
from scipy import stats

from .surrogates import constrained_surrogates


def _tail(y, k):
    ys = np.sort(np.asarray(y, dtype=float))
    return ys[-k:], ys[-k - 1], ys.size  # tail values, threshold u, n


def var_es_empirical(y, q):
    """Historical simulation on y (filtered HS when y are GARCH residuals)."""
    var = np.quantile(y, q)
    return var, y[y >= var].mean()


def var_es_normal(q):
    """Standard-normal residuals (plain GARCH-normal model)."""
    var = stats.norm.ppf(q)
    return var, stats.norm.pdf(var) / (1 - q)


def var_es_pareto(y, q, k):
    """Pareto tail with Hill / MLE index zeta above u."""
    tail, u, n = _tail(y, k)
    zeta = k / np.sum(np.log(tail / u))
    p_tail = (1 - q) * n / k
    var = u * p_tail ** (-1 / zeta)
    es = var * zeta / (zeta - 1) if zeta > 1 else np.inf
    return var, es


def var_es_gpd(y, q, k):
    """Generalised Pareto fit to the excesses over u (peaks over threshold)."""
    tail, u, n = _tail(y, k)
    xi, _, beta = stats.genpareto.fit(tail - u, floc=0)
    p_tail = (1 - q) * n / k
    if abs(xi) < 1e-8:
        var = u - beta * np.log(p_tail)
    else:
        var = u + beta / xi * (p_tail ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi) if xi < 1 else np.inf
    return var, es


def surrogate_pool(y, k, family, n_surr=200, rng=None):
    """Pooled values of constrained surrogates of the k-point tail of y.

    Each surrogate element is a draw from the data's conditional law given the
    sufficient statistic, so the pooled exceedance frequency at any level v is
    an unbiased (Rao-Blackwell) estimate of P(tail value > v) under the family.
    """
    tail, u, n = _tail(y, k)
    return constrained_surrogates(tail, family, u, n_surr, rng=rng).ravel()


def var_es_from_pool(pooled, q, n, k):
    """VaR = pooled quantile at the required tail level; ES = mean beyond it."""
    p_tail = (1 - q) * n / k
    var = np.quantile(pooled, 1 - p_tail)
    return var, pooled[pooled >= var].mean()


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------


def _xlogy(a, b):
    return a * np.log(b) if a > 0 else 0.0


def kupiec_pof(hits, q):
    """Kupiec (1995) proportion-of-failures LR test.  Returns (LR, p)."""
    hits = np.asarray(hits, dtype=bool)
    n, x = hits.size, hits.sum()
    p0, phat = 1 - q, x / n
    lr = -2 * (_xlogy(n - x, 1 - p0) + _xlogy(x, p0)
               - _xlogy(n - x, 1 - phat) - _xlogy(x, phat))
    return lr, stats.chi2.sf(lr, 1)


def christoffersen(hits, q):
    """Christoffersen (1998) independence and conditional-coverage LR tests.

    Returns dict with LR_ind, p_ind, LR_cc, p_cc.
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_indep = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_markov = (_xlogy(n00, 1 - pi0) + _xlogy(n01, pi0)
                 + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1))
    lr_ind = -2 * (ll_indep - ll_markov)
    lr_uc, _ = kupiec_pof(h, q)
    lr_cc = lr_uc + lr_ind
    return {"LR_ind": lr_ind, "p_ind": stats.chi2.sf(lr_ind, 1),
            "LR_cc": lr_cc, "p_cc": stats.chi2.sf(lr_cc, 2)}
