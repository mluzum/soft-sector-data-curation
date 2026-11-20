"""Higher moments of the event-by-event mean transverse momentum [pT].

Per event, over distinct particles with acceptance weights a_i (the
probability to be in the record's window: the eta-window Jacobian over the
plateau on 2D, a cut cell's share on 3D, over nsamples):

    T_n = sum_{i1 != ... != in} a_i1 ... a_in (p_i1 - M) ... (p_in - M),
    W_n = sum_{i1 != ... != in} a_i1 ... a_in,

and the class values <c_n> = sum_e w_e T_n / sum_e w_e W_n (events weighted
by their numbers of n-tuples), M the class mean <[pT]> = sum S1 / sum S0.
These are ATLAS's central moments <(delta pT)^n> (PRL 133 (2024) 252301,
Eq. 1 and the text before it) and, with the same M, ALICE's cumulants of
[pT] (JHEP 11 (2025) 076, Eqs. 2-5): c2 = <[pT]^(2)> - <[pT]>^2 is <c_2>,
c3 = <c_3>, and ALICE's fourth cumulant is <c_4> - 3 <c_2>^2.  Quantities
(Eq. 1 of ATLAS, Eq. 6 of ALICE):

    c2, c3, c4     <c_n> (GeV^n; c4 the central fourth moment)
    k2, k3         <c_n> / M^n            (ATLAS; normalization 'mean')
                   <c_n> / <[pT]^(n)>     (ALICE; normalization 'moment')
    gamma          <c_3> / <c_2>^(3/2)    standardized skewness
    Gamma          <c_3> M / <c_2>^2      intensive skewness
    kappa          <c_4> / <c_2>^2        (c4 + 3 c2^2) / c2^2 with ALICE's
                                          fourth cumulant c4

With `multiplicity_baseline` (ATLAS: the moments in unit N_ch^rec
intervals; STAR: <pT> per reference-multiplicity bin) M is the mean at the
event's multiplicity along a weighted linear regression within the class
(as analysis.observables_2d.compute_pT_fluct).

The distinct sums come from the per-event power sums
P(m, k) = sum_i a_i^m (p_i - M)^k, k = m <= 4 (set partitions of n
identical members):

    T_2 = P11^2 - P22
    T_3 = P11^3 - 3 P22 P11 + 2 P33
    T_4 = P11^4 - 6 P22 P11^2 + 3 P22^2 + 8 P33 P11 - 6 P44

(W_n: the same with p - M -> 1), and P(m, k) expands binomially in the
sums R(m, j) = sum_i a_i^m p_i^j, j <= m, which the prepares form from the
containers' cells: counts, sum pT, and sum pT^2 where stored; the third
and fourth powers of a cell's particles from its mean and variance
(entering only coincidence terms, relative order 1/M_eff per power).
"""
import numpy as np
from math import comb

QUANTITIES = ('c2', 'c3', 'c4', 'k2', 'k3', 'gamma', 'Gamma', 'kappa')
ORDER = {'c2': 2, 'c3': 3, 'c4': 4, 'k2': 2, 'k3': 3, 'gamma': 3, 'Gamma': 3, 'kappa': 4}
NORMALIZATIONS = ('mean', 'moment')


def settings(params):
    """(quantity, normalization, (eta, pT) window) of a record; ValueError
    for a combination the module does not compute."""
    q = params.get('pT_moment')
    if q not in QUANTITIES:
        raise ValueError(f"pT_moment {q!r} (one of {', '.join(QUANTITIES)})")
    norm = params.get('moment_normalization') or 'mean'
    if norm not in NORMALIZATIONS:
        raise ValueError(f"moment_normalization {norm!r} (mean: ATLAS's <[pT]>^n; moment: ALICE's <[pT]^(n)>)")
    if norm != 'mean' and q not in ('k2', 'k3'):
        raise ValueError("moment_normalization applies to k2 and k3")
    if params.get('dimensionality', 'integrated') != 'integrated':
        raise ValueError("[pT] moments are implemented for integrated records")
    full = params.get('poi_range') or {}
    eta, pt = tuple(full.get('eta') or (None, None)), tuple(full.get('pT') or (None, None))
    if None in eta or None in pt:
        raise ValueError("[pT] moments need kinematic_range eta and pT")
    return q, norm, (eta, pt)


def check_record(record):
    """resolve()'s check of a pT_moment record."""
    kr = record.get('kinematic_range') or {}

    def rng(x):
        return (x.get('min'), x.get('max')) if isinstance(x, dict) else (None, None)
    return settings({'pT_moment': record.get('pT_moment'),
                     'moment_normalization': record.get('moment_normalization'),
                     'dimensionality': record.get('dimensionality', 'integrated'),
                     'poi_range': {'eta': rng(kr.get('eta')), 'pT': rng(kr.get('pT'))}})


def cell_powers(c, s1, s2, k):
    """sum over a cell's particles of pT^k (k <= 4), from its count, sum pT
    and sum pT^2 (s2 None: zero spread within the cell)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        m1 = np.where(c > 0, s1 / np.where(c > 0, c, 1.0), 0.0)
        var = 0.0 if s2 is None else np.maximum(np.where(c > 0, s2 / np.where(c > 0, c, 1.0), 0.0) - m1 ** 2, 0.0)
    if k == 0:
        return c
    if k == 1:
        return s1
    if k == 2:
        return c * (m1 ** 2 + var) if s2 is None else s2
    if k == 3:
        return c * (m1 ** 3 + 3.0 * m1 * var)
    return c * (m1 ** 4 + 6.0 * m1 ** 2 * var + 3.0 * var ** 2)


def power_sums(c, s1, s2, a, ns, order):
    """R[(m, j)] per event: sum_i (a_i / ns)^m p_i^j, m = 1..order, j = 0..m.

    c, s1, s2: (n_ev, cells) per-cell count, sum pT, sum pT^2 (s2 may be
    None); a: (cells,) window weights."""
    ns = np.asarray(ns, dtype=float)
    out = {}
    pw = {j: cell_powers(c, s1, s2, j) for j in range(order + 1)}
    for m in range(1, order + 1):
        am = np.asarray(a, dtype=float) ** m
        for j in range(m + 1):
            out[(m, j)] = (pw[j] @ am) / ns ** m
    return out


def _P(R, m, k, M):
    """sum_i a^m (p - M)^k from R (M scalar or per event)."""
    return sum(comb(k, j) * (-M) ** (k - j) * R[(m, j)] for j in range(k + 1))


def distinct(P, n):
    """sum over distinct n-tuples of prod a (p - M) from P(m) = P(m, m)."""
    if n == 2:
        return P[1] ** 2 - P[2]
    if n == 3:
        return P[1] ** 3 - 3.0 * P[2] * P[1] + 2.0 * P[3]
    return P[1] ** 4 - 6.0 * P[2] * P[1] ** 2 + 3.0 * P[2] ** 2 + 8.0 * P[3] * P[1] - 6.0 * P[4]


def core(d, w):
    """One value per class."""
    q, order = d['quantity'], ORDER[d['quantity']]
    R, mult = d['R'], d.get('mult')
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return np.array([np.nan])
    S0, S1 = R[(1, 0)], R[(1, 1)]
    den = np.sum(w * S0)
    if not den > 0:
        return np.array([np.nan])
    M = np.sum(w * S1) / den
    m_bar = M
    if mult is not None:
        y = np.divide(S1, S0, out=np.full_like(S1, m_bar), where=S0 > 0)
        u = w * S0
        dm = mult - np.sum(u * mult) / u.sum()
        var = np.sum(u * dm ** 2)
        if var > 0:
            M = m_bar + np.sum(u * dm * (y - m_bar)) / var * dm
    one = {m: R[(m, 0)] for m in range(1, 5) if (m, 0) in R}
    c = {}
    for n in range(2, order + 1):
        P = {m: _P(R, m, m, M) for m in range(1, n + 1)}
        W = distinct(one, n)
        c[n] = np.sum(w * distinct(P, n)) / np.sum(w * W)
    with np.errstate(divide='ignore', invalid='ignore'):
        if q in ('c2', 'c3', 'c4'):
            return np.array([c[order]])
        if q in ('k2', 'k3'):
            if d.get('normalization', 'mean') == 'moment':
                raw = {m: _P(R, m, m, 0.0) for m in range(1, order + 1)}
                norm = np.sum(w * distinct(raw, order)) / np.sum(w * distinct(one, order))
            else:
                norm = m_bar ** order
            return np.array([c[order] / norm])
        if q == 'gamma':
            return np.array([c[3] / np.sign(c[2]) / np.abs(c[2]) ** 1.5])
        if q == 'Gamma':
            return np.array([c[3] * m_bar / c[2] ** 2])
        return np.array([c[4] / c[2] ** 2])
