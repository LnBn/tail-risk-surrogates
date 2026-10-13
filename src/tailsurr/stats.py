"""Test statistics and surrogate p-values.

Every statistic acts on the last axis, so it can be applied at once to an
observed sample (N,) and to a stack of surrogates (M, N) or (D, M, N).
"""

import numpy as np

from .fitting import fit_mle, ks_distance


def sample_max(x):
    """Largest observation: the most direct extreme-event statistic."""
    return np.max(x, axis=-1)


def top_hill(x, frac=0.1, k_min=5):
    """Hill estimate of the tail index zeta using only the top `frac` of x.

    zeta_hat = 1 / mean(log(x_(i) / x_(k+1))), i = 1..k (largest values).
    For a power law the full-sample Hill estimate is fixed by the constraint,
    but this 'upper-tail Hill' is not: it asks whether the far tail decays at
    the same rate as the tail as a whole.  Lognormal and exponential tails
    get steeper further out (larger zeta), heavier-than-power-law tails get
    flatter.
    """
    xs = np.sort(np.asarray(x, dtype=float), axis=-1)[..., ::-1]  # descending
    n = xs.shape[-1]
    k = max(k_min, int(np.ceil(frac * n)))
    logs = np.log(xs[..., :k] / xs[..., k:k + 1])
    return 1.0 / logs.mean(axis=-1)


def ks_to_fixed_fit(x, family, xmin, params):
    """KS distance to a given fitted law.

    For constrained surrogates the MLE is identical for every surrogate (it is
    a function of the preserved sufficient statistic), so fitting once on the
    observed data and reusing the parameters is exact, not an approximation.
    """
    return ks_distance(x, family, params, xmin)


def tail_statistics(x_obs, surr, family, xmin, frac=0.1):
    """Observed and surrogate values of the tail statistics used in the pilot.

    Returns {name: (observed_value, surrogate_values)}.
    """
    params = fit_mle(x_obs, family, xmin)
    return {
        "max": (sample_max(x_obs), sample_max(surr)),
        "top_hill": (top_hill(x_obs, frac), top_hill(surr, frac)),
        "ks": (ks_to_fixed_fit(x_obs, family, xmin, params),
               ks_to_fixed_fit(surr, family, xmin, params)),
    }


# direction of the test used for each statistic in the pilot
ALTERNATIVE = {"max": "two-sided", "top_hill": "two-sided", "ks": "greater"}


def surrogate_pvalue(obs, surr, alternative="two-sided"):
    """Monte-Carlo p-value with the standard +1 correction.

    obs : array (...,) observed statistic
    surr : array (..., M) statistic over M surrogates
    alternative : "greater" (large obs is extreme), "less", or "two-sided"
                  (twice the smaller one-sided p-value, capped at 1).

    With M surrogates that are exchangeable with the data under the null,
    p = (1 + #{surr >= obs}) / (M + 1) is exactly uniform on {1/(M+1), ..., 1}.
    """
    obs = np.asarray(obs)[..., None]
    surr = np.asarray(surr)
    m = surr.shape[-1]
    p_greater = (1 + np.sum(surr >= obs, axis=-1)) / (m + 1)
    p_less = (1 + np.sum(surr <= obs, axis=-1)) / (m + 1)
    if alternative == "greater":
        return p_greater
    if alternative == "less":
        return p_less
    if alternative == "two-sided":
        return np.minimum(1.0, 2.0 * np.minimum(p_greater, p_less))
    raise ValueError(alternative)


def waiting_time_cv(exceed):
    """Coefficient of variation of waiting times between tail exceedances.

    For i.i.d. data the waiting times are geometric and CV ~ 1; volatility
    clustering makes exceedances bursty (CV > 1).  Accepts (..., T) boolean
    arrays; rows are processed independently.
    """
    exceed = np.atleast_2d(np.asarray(exceed, dtype=bool))
    out = np.empty(exceed.shape[0])
    for r, row in enumerate(exceed):
        w = np.diff(np.flatnonzero(row))
        out[r] = w.std() / w.mean() if w.size > 1 else np.nan
    return out
