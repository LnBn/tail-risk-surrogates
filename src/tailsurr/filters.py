"""Volatility filters: GARCH(1,1), GJR-GARCH(1,1), Student-t innovations.

Each filter is fitted by (quasi-)maximum likelihood with the `arch` package;
the standardised residuals z_t = (r_t - mu) / sigma_t are what the tail tests
use.  The filter for each asset is chosen by BIC, decided *before* looking at
any tail test, so the choice cannot be tuned to the test results.
"""

import numpy as np
from arch import arch_model

SPECS = {
    "GARCH-n": {"o": 0, "dist": "normal"},
    "GJR-n": {"o": 1, "dist": "normal"},
    "GJR-t": {"o": 1, "dist": "t"},
}


def fit_filter(returns, spec):
    """Fit one specification.  Returns (standardised residuals, fit result)."""
    opts = SPECS[spec]
    # scale to unit-ish variance for numerical stability; residuals are scale-free
    scale = float(np.std(returns))
    am = arch_model(returns / scale, mean="Constant", vol="GARCH", p=1, o=opts["o"],
                    q=1, dist=opts["dist"], rescale=False)
    fit = am.fit(disp="off", options={"maxiter": 500})
    z = (fit.resid / fit.conditional_volatility).dropna()
    return z, fit


def best_filter(returns):
    """Fit all specifications and pick the lowest BIC among converged fits.

    Returns (spec name, residuals, {spec: (residuals, BIC, converged)}).
    """
    fits = {}
    for spec in SPECS:
        z, fit = fit_filter(returns, spec)
        fits[spec] = (z, fit.bic, fit.convergence_flag == 0)
    converged = [s for s in fits if fits[s][2]] or list(fits)
    best = min(converged, key=lambda s: fits[s][1])
    return best, fits[best][0], fits
