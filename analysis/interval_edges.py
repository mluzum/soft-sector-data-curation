"""Edges of contiguous intervals from their means and event fractions.

ALICE (JHEP 11 (2025) 076, Fig. 3) and CMS (Rep. Prog. Phys. 87 (2024)
077801, Fig. 2) tabulate, for intervals of a centrality estimator, the
fraction of events in each interval and the interval's mean N_ch / N_ch^0,
not the interval edges.  The model's classes need the edges on the N_ch /
N_ch^0 scale (centrality.abscissa_bins).  Midway between the means is wrong
in a steep tail: the mean of an interval sits below its centre where the
distribution falls (by ~0.07 of the width where the fraction falls by 2.5
per interval), and the model's fraction in a class shifted that way is a
few percent high (5% at the end of a toy knee).

Here the density within each interval is exponential in the label, with the
log-slope from the neighbouring intervals' densities (fraction over width),
and the edges solve: every interval's mean equals the tabulated one, within
a tolerance (the tabulated means carry the noise of the spectra fits), with
smoothly varying widths (penalized second differences).  On a toy knee with
equal estimator intervals: edges within 5e-5 and fractions within 0.1% of
the truth noise-free; with 1e-4 noise on the means, fractions within 1%
where midpoints are 4.5% off (tests/test_interval_edges.py).
"""
import numpy as np
from scipy.optimize import least_squares


def centroid_offset(u):
    """(mean - centre) / width of a density exp(u t / w) on [0, w]."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    small = np.abs(u) < 1e-4
    us = u[~small]
    out[~small] = 1.0 / (1.0 - np.exp(-us)) - 1.0 / us - 0.5
    out[small] = u[small] / 12.0
    return out


def _log_slopes(centres, density):
    lnd = np.log(np.maximum(density, 1e-300))
    lam = np.empty_like(centres)
    if len(centres) == 1:
        lam[:] = 0.0
        return lam
    lam[1:-1] = (lnd[2:] - lnd[:-2]) / (centres[2:] - centres[:-2])
    lam[0] = (lnd[1] - lnd[0]) / (centres[1] - centres[0])
    lam[-1] = (lnd[-1] - lnd[-2]) / (centres[-1] - centres[-2])
    return lam


def interval_means(edges, fractions):
    """The means of the intervals between `edges` whose events are
    `fractions`, the density exponential within each interval with the
    log-slope of the neighbouring densities."""
    edges = np.asarray(edges, dtype=float)
    w = np.diff(edges)
    c = 0.5 * (edges[:-1] + edges[1:])
    lam = _log_slopes(c, np.asarray(fractions, dtype=float) / w)
    return c + w * centroid_offset(lam * w)


def edges_from_means(means, fractions, tol=2e-4, smooth=0.03):
    """Contiguous interval edges (len(means) + 1, increasing) whose interval
    means (interval_means) equal `means` within `tol` (absolute, in the
    label's units) with widths varying smoothly (second differences
    penalized at `smooth` x the median spacing).  Returns (edges, worst
    |mean - tabulated|)."""
    x = np.asarray(means, dtype=float)
    y = np.asarray(fractions, dtype=float)
    if len(x) < 3 or not np.all(np.diff(x) > 0) or not np.all(y > 0):
        raise ValueError('edges_from_means: at least three increasing means with positive fractions')
    mids = 0.5 * (x[:-1] + x[1:])
    e0 = np.concatenate([[x[0] - (mids[0] - x[0])], mids, [x[-1] + (x[-1] - mids[-1])]])
    scale = float(np.median(np.diff(x)))

    def residuals(e):
        w = np.diff(e)
        if np.any(w <= 0):
            return np.full(2 * len(x) - 1, 1e6)
        return np.concatenate([(interval_means(e, y) - x) / tol, np.diff(w, 2) / (smooth * scale)])

    sol = least_squares(residuals, e0, x_scale=scale, max_nfev=5000)
    e = sol.x
    if not np.all(np.diff(e) > 0):
        raise ValueError('edges_from_means: no increasing solution')
    return e, float(np.max(np.abs(interval_means(e, y) - x)))
