"""Constrained (and typical) surrogates for continuous tail distributions.

Implements the MCMC algorithms of Moore & Altmann (2026), "Constrained surrogates
for arbitrary families of continuous probability distributions", for three
families defined on [xmin, inf):

    family        sufficient statistic (fixed)        move (on transformed u)
    ------------  ----------------------------------  -----------------------------
    exponential   sum(x)                              pair move, u = x - xmin
    powerlaw      prod(x)  <=> sum(log x)             pair move, u = log(x / xmin)
    lognormal     sum(log x) and sum((log x)^2)       triple move, u = log(x / xmin)

A constrained surrogate is a draw from the distribution of the data *conditional
on the sufficient statistic*.  That conditional law does not depend on the
unknown parameters, so the surrogates represent the whole family at once.

All samplers run many independent Markov chains in parallel: the state is a 2-D
array of shape (n_chains, N), and each MCMC step updates every chain at once
with numpy fancy indexing.  Every chain starts at the observed data.
"""

import numpy as np

from .fitting import fit_mle, sample_family

FAMILIES = ("exponential", "powerlaw", "lognormal")

# Orthonormal basis of the plane perpendicular to (1, 1, 1) (first two columns of
# the rotation matrix R in Appendix A of the paper).
_E1 = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)
_E2 = np.array([2.0, -1.0, -1.0]) / np.sqrt(6.0)


def default_n_steps(n):
    """Number of MCMC transitions per surrogate: 2 * r_tilde with eps = 2^-10.

    Eq. (4) of the paper gives r_tilde = (N/2) (log2 N - log2 eps); the authors
    use twice this, i.e. N (log2 N + 10).
    """
    return int(np.ceil(n * (np.log2(n) + 10)))


def _two_distinct(rng, n_chains, n):
    """Two distinct random indices in [0, n) for each chain."""
    i = rng.integers(0, n, n_chains)
    j = rng.integers(0, n - 1, n_chains)
    j += j >= i  # skip over i so that j != i
    return i, j


def _three_distinct(rng, n_chains, n):
    """Three distinct random indices in [0, n) for each chain."""
    i, j = _two_distinct(rng, n_chains, n)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    k = rng.integers(0, n - 2, n_chains)
    k += k >= lo  # skip the smaller index first ...
    k += k >= hi  # ... then the larger one
    return i, j, k


def sum_preserving_moves(u, n_steps, rng):
    """Randomise rows of u >= 0 while keeping each row sum fixed (in place).

    Each step picks two entries (u_i, u_j) per chain and replaces them by
    (s, b - s) with s ~ Uniform[0, b], where b = u_i + u_j (Appendix A.1:
    the Jacobian is 1, so s is uniform).
    """
    n_chains, n = u.shape
    rows = np.arange(n_chains)
    for _ in range(n_steps):
        i, j = _two_distinct(rng, n_chains, n)
        total = u[rows, i] + u[rows, j]
        s = rng.random(n_chains) * total
        u[rows, i] = s
        u[rows, j] = total - s
    return u


def sum_and_square_preserving_moves(u, n_steps, rng, max_tries=50):
    """Randomise rows of u >= 0 keeping each row's sum and sum of squares fixed.

    Each step picks three entries per chain.  The set of triples with the same
    sum and sum of squares is a circle centred at (m1/3)(1,1,1) in the plane
    perpendicular to (1,1,1); the paper shows the uniform angle gives the
    correct (symmetric) proposal (Appendix A.2).  Points of the circle that
    violate the threshold u >= 0 are re-drawn, i.e. we sample uniformly on the
    allowed arcs; after `max_tries` failures the chain simply stays put, which
    keeps the transition kernel symmetric.
    """
    n_chains, n = u.shape
    rows = np.arange(n_chains)
    for _ in range(n_steps):
        i, j, k = _three_distinct(rng, n_chains, n)
        trip = np.stack([u[rows, i], u[rows, j], u[rows, k]], axis=1)
        m1 = trip.sum(axis=1)
        m2 = (trip ** 2).sum(axis=1)
        centre = m1 / 3.0
        radius = np.sqrt(np.maximum(m2 - m1 ** 2 / 3.0, 0.0))

        new = trip.copy()
        todo = np.arange(n_chains)  # chains still looking for a valid point
        for _ in range(max_tries):
            theta = rng.uniform(0.0, 2.0 * np.pi, todo.size)
            cand = (centre[todo, None]
                    + radius[todo, None] * (np.cos(theta)[:, None] * _E1
                                            + np.sin(theta)[:, None] * _E2))
            ok = (cand >= 0.0).all(axis=1)
            new[todo[ok]] = cand[ok]
            todo = todo[~ok]
            if todo.size == 0:
                break
        u[rows, i], u[rows, j], u[rows, k] = new[:, 0], new[:, 1], new[:, 2]
    return u


def _to_u(x, family, xmin):
    """Map data on [xmin, inf) to the non-negative variable the moves act on."""
    if family == "exponential":
        return x - xmin
    return np.log(x / xmin)  # powerlaw and lognormal work in log space


def _from_u(u, family, xmin):
    if family == "exponential":
        return u + xmin
    return xmin * np.exp(u)


def constrained_surrogates(x, family, xmin, n_surr, n_steps=None, rng=None):
    """Draw `n_surr` independent constrained surrogates of the sample(s) `x`.

    Parameters
    ----------
    x : array, shape (N,) or (n_datasets, N)
        Observations, all >= xmin.  A 2-D array is treated as independent
        datasets that are processed together (used for simulation studies).
    family : {"exponential", "powerlaw", "lognormal"}
    xmin : float
        Lower threshold of the family's support (the tail cut-off).
    n_surr : int
        Number of surrogates per dataset.
    n_steps : int, optional
        MCMC transitions per chain (default `default_n_steps(N)`).

    Returns
    -------
    array of shape (n_surr, N), or (n_datasets, n_surr, N) for 2-D input.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    x2d = np.atleast_2d(x)
    n_data, n = x2d.shape
    if np.any(x2d < xmin):
        raise ValueError("all observations must be >= xmin")
    if n_steps is None:
        n_steps = default_n_steps(n)

    # one chain per (dataset, surrogate), each initialised at its dataset
    u = np.repeat(_to_u(x2d, family, xmin), n_surr, axis=0)
    if family == "lognormal":
        sum_and_square_preserving_moves(u, n_steps, rng)
    else:
        sum_preserving_moves(u, n_steps, rng)

    out = _from_u(u, family, xmin).reshape(n_data, n_surr, n)
    return out[0] if squeeze else out


def typical_surrogates(x, family, xmin, n_surr, rng=None):
    """Parametric-bootstrap ('typical') surrogates: fit by MLE, then resample.

    This is the standard approach (e.g. Clauset, Shalizi & Newman 2009 without
    the semi-parametric bulk) and serves as the baseline in the paper.
    Accepts the same shapes as `constrained_surrogates`.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    x2d = np.atleast_2d(x)
    n_data, n = x2d.shape
    out = np.empty((n_data, n_surr, n))
    for d in range(n_data):
        params = fit_mle(x2d[d], family, xmin)
        out[d] = sample_family(family, params, xmin, (n_surr, n), rng)
    return out[0] if squeeze else out


def shuffle_surrogates(x, n_surr, rng=None):
    """Random permutations of a time series (null: exchangeable / i.i.d.)."""
    rng = np.random.default_rng(rng)
    x = np.asarray(x)
    return np.stack([rng.permutation(x) for _ in range(n_surr)])
