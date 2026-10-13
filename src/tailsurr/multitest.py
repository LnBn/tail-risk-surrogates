"""Multiple-testing corrections and p-value combination."""

import numpy as np
from scipy import stats


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (q-values); controls the FDR."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def holm_adjust(p):
    """Holm step-down adjusted p-values; controls the family-wise error rate."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    stepped = p[order] * (m - np.arange(m))
    adj = np.maximum.accumulate(stepped)
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def cauchy_combination(p, weights=None):
    """Cauchy combination test (Liu & Xie 2020).

    T = sum w_i tan((0.5 - p_i) pi) is approximately standard Cauchy under the
    global null *whatever the dependence between the tests*, which matters here
    because tail events of different assets fall on the same crisis days.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    w = np.full(p.size, 1 / p.size) if weights is None else np.asarray(weights)
    t = np.sum(w * np.tan((0.5 - p) * np.pi))
    return float(0.5 - np.arctan(t) / np.pi)


def fisher_combination(p):
    """Fisher's method; valid only for independent tests (shown for contrast)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1)
    return float(stats.chi2.sf(-2 * np.log(p).sum(), 2 * p.size))
