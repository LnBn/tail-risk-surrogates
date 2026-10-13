"""Threshold selection on a grid, and a selection-aware surrogate KS test.

Clauset-Shalizi-Newman (CSN) choose xmin to minimise the KS distance between
the tail and its fitted power law.  Testing at a data-chosen xmin makes naive
KS p-values optimistic, so CSN re-select xmin inside their bootstrap.  We do
the analogue for constrained surrogates:

    1. pick the tail size k0 (top k0 values) on the data by the CSN rule;
    2. replace the tail by a constrained surrogate of the hypothesised family,
       keeping the values below the threshold unchanged;
    3. re-run the CSN rule on the modified data and compute the KS distance
       of the re-selected tail to its own fitted law;
    4. the p-value is the rank of the observed KS among the surrogate ones.

To keep this cheap, candidate tails are a geometric grid of tail sizes
k (from K_MIN points up to FRAC_MAX of all days) rather than every observed
value, and the scan is vectorised over surrogates.  The p-value is
approximate (the conditioning argument behind exactness does not survive the
re-selection), as is the CSN bootstrap it imitates.

RESULT OF CHECKING IT (scripts/10_check_selection_aware.py): on synthetic
data with an exact power-law tail this selection-aware test is
*anti-conservative* (about 15% rejections at the 5% level), while the naive KS
test at the selected threshold is conservative (about 2%) and the max and
upper-tail Hill tests are close to nominal.  `selection_aware_ks` is therefore
kept only as an experiment and is NOT used for inference; the pipeline uses
the naive tests and reports KS as conservative.
"""

import numpy as np

from .fitting import fit_mle, ks_distance
from .surrogates import constrained_surrogates
from .stats import surrogate_pvalue

K_MIN = 50
FRAC_MAX = 0.20
N_GRID = 40


def tail_size_grid(n_days, k_min=K_MIN, frac_max=FRAC_MAX, n_grid=N_GRID):
    """Candidate tail sizes: a geometric grid of integers."""
    k_max = max(k_min + 1, int(frac_max * n_days))
    return np.unique(np.geomspace(k_min, k_max, n_grid).astype(int))


def _powerlaw_ks_desc(top):
    """Power-law KS distance for tails given as rows sorted in *descending* order.

    top : (M, k); the threshold is the last (smallest) column.
    """
    k = top.shape[1]
    xmin = top[:, -1:]
    logs = np.log(top / xmin)
    alpha = 1 + k / logs.sum(axis=1, keepdims=True)
    asc = top[:, ::-1]
    f = 1 - (asc / xmin) ** (1 - alpha)
    i = np.arange(1, k + 1)
    return np.maximum((i / k - f).max(axis=1), (f - (i - 1) / k).max(axis=1))


def csn_select(desc, grid):
    """CSN rule for each row of `desc` (values sorted descending).

    Returns the chosen tail size per row (array of ints).
    """
    desc = np.atleast_2d(desc)
    ks = np.stack([_powerlaw_ks_desc(desc[:, :k]) for k in grid], axis=1)
    return np.asarray(grid)[np.argmin(ks, axis=1)]


def select_tail(values, n_days):
    """Apply the grid CSN rule to the positive values of one tail side.

    Returns (k, xmin, tail values) where the tail is the top k values.
    """
    desc = np.sort(values[values > 0])[::-1]
    grid = tail_size_grid(n_days)
    grid = grid[grid <= desc.size]
    k = int(csn_select(desc, grid)[0])
    return k, desc[k - 1], desc[:k]


def selection_aware_ks(values, n_days, family, n_surr, rng=None):
    """EXPERIMENTAL selection-aware surrogate KS test (anti-conservative; see above).

    Returns (observed KS, p-value, fraction of surrogates whose re-selected k
    differs from the data's).
    """
    rng = np.random.default_rng(rng)
    desc = np.sort(values[values > 0])[::-1]
    grid = tail_size_grid(n_days)
    grid = grid[grid <= desc.size]
    k0 = int(csn_select(desc, grid)[0])
    xmin0, tail0 = desc[k0 - 1], desc[:k0]
    obs = float(ks_distance(tail0, family, fit_mle(tail0, family, xmin0), xmin0))

    # surrogate tails on top of the unchanged values below the threshold
    surr = constrained_surrogates(tail0, family, xmin0, n_surr, rng=rng)
    k_max = grid.max()
    body = desc[k0:k_max]
    tops = np.concatenate([np.sort(surr, axis=1)[:, ::-1],
                           np.broadcast_to(body, (n_surr, body.size))], axis=1)
    k_new = csn_select(tops, grid)

    d = np.empty(n_surr)
    for m in range(n_surr):
        tail = tops[m, :k_new[m]]
        xmin = tail[-1]
        d[m] = ks_distance(tail, family, fit_mle(tail, family, xmin), xmin)
    return obs, float(surrogate_pvalue(obs, d, "greater")), float(np.mean(k_new != k0))


def bootstrap_zeta(values, n_days, n_boot=200, rng=None):
    """i.i.d. bootstrap of the tail index zeta = alpha - 1, re-selecting k each time.

    Only meaningful for (approximately) i.i.d. data such as GARCH residuals.
    Returns the bootstrap draws.
    """
    rng = np.random.default_rng(rng)
    pos = values[values > 0]
    grid = tail_size_grid(n_days)
    grid = grid[grid <= pos.size]
    out = np.empty(n_boot)
    for b in range(n_boot):
        desc = np.sort(rng.choice(pos, pos.size, replace=True))[::-1]
        k = int(csn_select(desc, grid)[0])
        out[b] = k / np.sum(np.log(desc[:k] / desc[k - 1]))
    return out
