"""Maximum-likelihood fits, sampling and goodness-of-fit for tail families.

All families live on [xmin, inf) with xmin treated as known:

    powerlaw     p(x) = (alpha - 1) / xmin * (x / xmin)^(-alpha)
                 (alpha is the density exponent; the tail index of the
                  survival function is zeta = alpha - 1)
    exponential  p(x) = lam * exp(-lam (x - xmin))
    lognormal    log x ~ Normal(mu, sigma^2) truncated to log x >= log xmin
"""

import numpy as np
from scipy import optimize, special, stats

# ---------------------------------------------------------------------------
# Maximum likelihood
# ---------------------------------------------------------------------------


def _lognormal_negloglik(theta, logx, log_xmin):
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    z = (logx - mu) / sigma
    a = (log_xmin - mu) / sigma
    # -log L (dropping the constant sum(log x) and 0.5 log 2 pi terms)
    return (logx.size * log_sigma + 0.5 * np.sum(z ** 2)
            + logx.size * special.log_ndtr(-a))


def fit_mle(x, family, xmin):
    """Maximum-likelihood parameters for x >= xmin.  Returns a dict."""
    x = np.asarray(x, dtype=float)
    if family == "powerlaw":
        return {"alpha": 1.0 + x.size / np.sum(np.log(x / xmin))}
    if family == "exponential":
        return {"lam": 1.0 / np.mean(x - xmin)}
    if family == "lognormal":
        logx = np.log(x)
        start = np.array([logx.mean(), np.log(logx.std() + 1e-12)])
        res = optimize.minimize(_lognormal_negloglik, start,
                                args=(logx, np.log(xmin)), method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-10,
                                         "maxiter": 4000})
        mu, log_sigma = res.x
        return {"mu": mu, "sigma": np.exp(log_sigma)}
    raise ValueError(f"unknown family {family!r}")


def logpdf(x, family, params, xmin):
    x = np.asarray(x, dtype=float)
    if family == "powerlaw":
        a = params["alpha"]
        return np.log(a - 1) - np.log(xmin) - a * np.log(x / xmin)
    if family == "exponential":
        lam = params["lam"]
        return np.log(lam) - lam * (x - xmin)
    if family == "lognormal":
        mu, s = params["mu"], params["sigma"]
        z = (np.log(x) - mu) / s
        a = (np.log(xmin) - mu) / s
        return (-np.log(x) - np.log(s) - 0.5 * np.log(2 * np.pi) - 0.5 * z ** 2
                - special.log_ndtr(-a))
    raise ValueError(f"unknown family {family!r}")


def sf(x, family, params, xmin):
    """Survival function P(X > x) of the fitted family (x >= xmin)."""
    x = np.asarray(x, dtype=float)
    if family == "powerlaw":
        return (x / xmin) ** (1.0 - params["alpha"])
    if family == "exponential":
        return np.exp(-params["lam"] * (x - xmin))
    if family == "lognormal":
        mu, s = params["mu"], params["sigma"]
        z = (np.log(x) - mu) / s
        a = (np.log(xmin) - mu) / s
        return np.exp(special.log_ndtr(-z) - special.log_ndtr(-a))
    raise ValueError(f"unknown family {family!r}")


def cdf(x, family, params, xmin):
    return 1.0 - sf(x, family, params, xmin)


def sample_family(family, params, xmin, size, rng=None):
    """i.i.d. draws from a family on [xmin, inf) with the given parameters."""
    rng = np.random.default_rng(rng)
    if family == "powerlaw":
        u = rng.random(size)
        return xmin * u ** (-1.0 / (params["alpha"] - 1.0))  # inverse CDF
    if family == "exponential":
        return xmin + rng.exponential(1.0 / params["lam"], size)
    if family == "lognormal":
        mu, s = params["mu"], params["sigma"]
        a = (np.log(xmin) - mu) / s
        logx = stats.truncnorm.rvs(a, np.inf, loc=mu, scale=s, size=size,
                                   random_state=rng)
        return np.exp(logx)
    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------------
# Goodness of fit and threshold selection
# ---------------------------------------------------------------------------


def ks_distance(x, family, params, xmin):
    """Kolmogorov-Smirnov distance between the sample and a fitted CDF.

    Works on the last axis, so x may be (..., N).
    """
    xs = np.sort(np.asarray(x, dtype=float), axis=-1)
    n = xs.shape[-1]
    f = cdf(xs, family, params, xmin)
    i = np.arange(1, n + 1)
    return np.maximum((i / n - f).max(axis=-1), (f - (i - 1) / n).max(axis=-1))


def select_xmin(x, n_tail_min=50, max_tail_frac=1.0):
    """Clauset-Shalizi-Newman (2009) threshold for a continuous power law.

    For every candidate threshold (each observed value leaving at least
    `n_tail_min` points above it) fit alpha by MLE and compute the KS distance
    of the tail; return the threshold minimising it.

    Returns a dict with xmin, alpha, n_tail, ks, and the full scan.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    xs = xs[xs > 0]
    n_all = xs.size
    first = max(0, int(np.floor(n_all * (1 - max_tail_frac))))
    last = n_all - n_tail_min  # candidate index range
    cands, ks, alphas = [], [], []
    for start in range(first, last + 1):
        xmin = xs[start]
        tail = xs[start:]
        # skip duplicated thresholds
        if start > first and xs[start - 1] == xmin:
            continue
        logs = np.log(tail / xmin)
        alpha = 1.0 + tail.size / logs.sum()
        f = 1.0 - (tail / xmin) ** (1.0 - alpha)
        i = np.arange(1, tail.size + 1)
        d = max((i / tail.size - f).max(), (f - (i - 1) / tail.size).max())
        cands.append(xmin)
        ks.append(d)
        alphas.append(alpha)
    best = int(np.argmin(ks))
    xmin = cands[best]
    return {"xmin": xmin, "alpha": alphas[best], "ks": ks[best],
            "n_tail": int(np.sum(xs >= xmin)),
            "scan": {"xmin": np.array(cands), "ks": np.array(ks),
                     "alpha": np.array(alphas)}}


def vuong_test(x, family_a, family_b, xmin):
    """Vuong normalised log-likelihood-ratio test (as used by CSN 2009).

    Returns (R, p): R > 0 favours family_a; p is the two-sided p-value for the
    null that both fit equally well.
    """
    pa = fit_mle(x, family_a, xmin)
    pb = fit_mle(x, family_b, xmin)
    d = logpdf(x, family_a, pa, xmin) - logpdf(x, family_b, pb, xmin)
    n = d.size
    R = d.sum()
    z = R / (np.sqrt(n) * d.std())
    return R, float(special.erfc(abs(z) / np.sqrt(2)))


def umvue_sf(v, x, family, xmin):
    """Analytic unbiased (UMVU) estimate of P(X > v) for power law / exponential.

    For these one-parameter families the sufficient statistic is
    T = sum(u_i) with u = log(x/xmin) (power law) or u = x - xmin (exponential),
    and u_1 | T ~ T * Beta(1, n - 1).  Hence
        P(X > v | T) = (1 - u(v) / T)^(n - 1)   for u(v) < T.
    The marginal of a single element of a constrained surrogate must equal
    this, which gives an exact check of the MCMC sampler.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    if family == "powerlaw":
        T, uv = np.sum(np.log(x / xmin)), np.log(v / xmin)
    elif family == "exponential":
        T, uv = np.sum(x - xmin), v - xmin
    else:
        raise ValueError("analytic UMVUE only for powerlaw / exponential")
    return np.clip(1.0 - uv / T, 0.0, 1.0) ** (x.size - 1)
