"""Radial-flow fluctuations: v0, v0(pT) and the normalized [pT]-spectrum covariance.

Two subevents a and b (record `subevents.A`, `subevents.B`).  Per event
and subevent X:
    [pT]_X  = P_X / Nr_X,           P_X = sum w pT, Nr_X = sum w  (reference pT)
    n_X(b)  = N_X(b) / N_X                                         (spectrum pT range)
and, in a centrality class (event weights w_e),
    v0^2      = <d[pT]_a d[pT]_b> / (m_a m_b),         m_X = <[pT]_X>
    cov(b)    = (<dn_a(b) d[pT]_b> + <dn_b(b) d[pT]_a>) / 2
    ncov(b)   = cov(b) / (<n(b)> sqrt(m_a m_b))  = v0(pT) v0
    v0(pT)    = ncov / v0,   v0(pT)/v0 = ncov / v0^2
(ATLAS, PRL 136 (2026) 032301, Eqs. 1-2 and the text after them).

Deviations d are taken from the class mean, or -- `multiplicity_baseline`,
the counterpart of measuring in narrow multiplicity bins -- from a weighted
linear regression on the event multiplicity within the class.

Both deviations are exact weighted sums over particles,
    d[pT]_X = sum_i w_X,i (pT_i - m_X) / Nr_X,
    dn_X(b) = sum_i w_X,i (1[i in b] - <n_X(b)>) / N_X,
so the pairs of a particle with itself -- present when the two windows
share particles, as on the boost-invariant 2D container where every window
holds every particle with its Jacobian weight -- are removed exactly from
the per-cell sums of w_a w_b times (1, pT, pT^2).  Disjoint 3D subevents
have none.

The prepares (analysis.pipeline_dispatch) reduce a container to per-event
cell arrays (`cells`): counts c, sum pT s1, sum pT^2 s2, shape
(n_events, K), and per-subevent cell weights; `components` turns them into
the per-event sums the core needs.
"""
import numpy as np


def components(c, s1, s2, W, F, bins, W_mult=None, F_mult=None):
    """Per-event sums for the core.

    c, s1, s2 : (n_ev, K) counts, sum pT, sum pT^2 per cell.
    W         : {'A': (n_ev | 1, K), 'B': ...} subevent cell weights (eta
                window, species, Jacobian; no pT selection).
    F         : {'spec': (K,), 'ref': (K,)} pT-range fractions of each cell
                (spectrum range normalizing n(pT); reference range of [pT]).
    bins      : list of (K,) pT-bin fractions (empty for integrated records).
    W_mult, F_mult : cell weights / pT fractions of the baseline multiplicity.
    """
    out = {}
    Fs, Fr = F['spec'], F['ref']
    for X in 'AB':
        wX = W[X]
        out[f'N_{X}'] = (wX * Fs * c).sum(axis=1)
        out[f'Nr_{X}'] = (wX * Fr * c).sum(axis=1)
        out[f'P_{X}'] = (wX * Fr * s1).sum(axis=1)
        out[f'Nb_{X}'] = (np.stack([(wX * Fb * c).sum(axis=1) for Fb in bins], axis=1)
                          if bins else np.zeros((c.shape[0], 0)))
    ww = W['A'] * W['B']
    # pairs of a particle with itself: reference x reference, bin x reference,
    # spectrum x reference
    out['Srr0'] = (ww * Fr * Fr * c).sum(axis=1)
    out['Srr1'] = (ww * Fr * Fr * s1).sum(axis=1)
    out['Srr2'] = (ww * Fr * Fr * s2).sum(axis=1)
    out['Sbr0'] = (np.stack([(ww * Fb * Fr * c).sum(axis=1) for Fb in bins], axis=1)
                   if bins else np.zeros((c.shape[0], 0)))
    out['Sbr1'] = (np.stack([(ww * Fb * Fr * s1).sum(axis=1) for Fb in bins], axis=1)
                   if bins else np.zeros((c.shape[0], 0)))
    out['Ssr0'] = (ww * Fs * Fr * c).sum(axis=1)
    out['Ssr1'] = (ww * Fs * Fr * s1).sum(axis=1)
    out['M'] = ((W_mult * F_mult * c).sum(axis=1) if W_mult is not None
                else np.full(c.shape[0], np.nan))
    return out


def _baseline(x, M, w, use_mult):
    """Per-event baseline of x (n_ev,) or (n_ev, k): the weighted class mean,
    or the weighted linear regression on M."""
    xw = np.average(x, axis=0, weights=w)
    if not use_mult:
        return np.broadcast_to(xw, x.shape)
    Mw = np.average(M, weights=w)
    dM = M - Mw
    var = np.average(dM ** 2, weights=w)
    if var <= 0:
        return np.broadcast_to(xw, x.shape)
    dMx = dM if x.ndim == 1 else dM[:, None]
    slope = np.average(dMx * (x - xw), axis=0, weights=w) / var
    return xw + slope * dMx


def core(data, w):
    """One centrality class: the record's quantity, per pT bin (differential)
    or one value (integrated v0)."""
    q = data['quantity']
    nb = data['Nb_A'].shape[1]
    nan = np.full(max(nb, 1), np.nan)
    sel = w > 0
    if sel.sum() < 3:
        return nan
    d = {k: v[sel] for k, v in data.items() if isinstance(v, np.ndarray) and v.shape[:1] == sel.shape}
    w = w[sel]
    if np.any(d['Nr_A'] <= 0) or np.any(d['Nr_B'] <= 0):
        keep = (d['Nr_A'] > 0) & (d['Nr_B'] > 0) & (d['N_A'] > 0) & (d['N_B'] > 0)
        d = {k: v[keep] for k, v in d.items()}
        w = w[keep]
        if w.sum() <= 0:
            return nan
    use_mult = bool(data.get('use_multiplicity')) and np.all(np.isfinite(d['M']))
    pA, pB = d['P_A'] / d['Nr_A'], d['P_B'] / d['Nr_B']
    m_A, m_B = np.average(pA, weights=w), np.average(pB, weights=w)
    bA, bB = _baseline(pA, d['M'], w, use_mult), _baseline(pB, d['M'], w, use_mult)
    # <d[pT]_a d[pT]_b> without the pairs of a particle with itself
    self_pp = (d['Srr2'] - (bA + bB) * d['Srr1'] + bA * bB * d['Srr0']) / (d['Nr_A'] * d['Nr_B'])
    v0sq = np.average((pA - bA) * (pB - bB) - self_pp, weights=w) / (m_A * m_B)
    v0 = np.sqrt(v0sq) if v0sq > 0 else np.nan
    if q == 'v0':
        return np.array([v0])
    nA, nB = d['Nb_A'] / d['N_A'][:, None], d['Nb_B'] / d['N_B'][:, None]
    nbA, nbB = _baseline(nA, d['M'], w, use_mult), _baseline(nB, d['M'], w, use_mult)
    # dn_X(b) d[pT]_Y: the self pairs are the particles in the bin and in the
    # reference, minus <n_X(b)> times those in the spectrum and the reference
    self_AB = (((d['Sbr1'] - bB[:, None] * d['Sbr0'])
                - nbA * (d['Ssr1'] - bB * d['Ssr0'])[:, None]) / (d['N_A'] * d['Nr_B'])[:, None])
    self_BA = (((d['Sbr1'] - bA[:, None] * d['Sbr0'])
                - nbB * (d['Ssr1'] - bA * d['Ssr0'])[:, None]) / (d['N_B'] * d['Nr_A'])[:, None])
    cov = 0.5 * np.average((nA - nbA) * (pB - bB)[:, None] - self_AB
                           + (nB - nbB) * (pA - bA)[:, None] - self_BA, axis=0, weights=w)
    n_mean = 0.5 * (np.average(nA, axis=0, weights=w) + np.average(nB, axis=0, weights=w))
    ncov = cov / (n_mean * np.sqrt(m_A * m_B))
    if q == 'normalized_covariance':
        return ncov
    if q == 'v0_pT':
        return ncov / v0
    if q == 'v0_pT_over_v0':
        return ncov / v0sq if v0sq > 0 else nan
    raise ValueError(f'radial_flow quantity {q!r}')
