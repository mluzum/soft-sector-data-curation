"""Multi-particle correlators of particles of interest (POI) and a reference (REF).

A k-particle correlator sums prod_m w_m exp(i h_m phi_m) over k DISTINCT
particles, member 1 (or none) from the POI sample and the others from the
reference sample.  The sum over distinct particles is the inclusion-exclusion
over set partitions of the members:

    sum_distinct = sum_{partitions pi} mu(pi) prod_{blocks B} S(B)
    mu(pi)       = prod_B (-1)^{|B|-1} (|B|-1)!

where S(B) is a single-particle sum over the particles that belong to every
member's sample in B (POI and REF overlap when both are in B), weighted by
the product of the members' weights, at the summed harmonic.  This holds
for any POI/REF overlap -- identified POI inside a charged reference, POI
outside it, or POI = REF -- and for particle weights (the 2D boost-invariant
container weights every particle by its window Jacobian).

Backends give S(h, n_poi, n_ref):
- Container2D: QVectorData species/pT slices and window weights;
- Container3D: species-selected unit-weight grids and eta/pT windows.

Differential and integrated v_n{2}, v_n{4} of the POI (Bilandzic et al.,
PRC 83, 044913; PRC 89, 064904):
    d2 = <<2'>>,  c2 = <<2>>,                      v'{2} = d2 / sqrt(c2)
    d4 = <<4'>> - 2 <<2'>><<2>>,  c4 = <<4>> - 2 <<2>>^2,  v'{4} = -d4 / (-c4)^(3/4)
With POI = REF they reduce to v{2} = sqrt(c2) and v{4} = (-c4)^(1/4).  Signs
follow the existing convention: sign(c2) sqrt|c2| and sign(-c4) |c4|^(1/4)
for POI = REF.
"""
from math import factorial

import numpy as np


def set_partitions(items):
    """All partitions of a list into blocks (lists)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


_PARTITIONS = {k: [[tuple(b) for b in p] for p in set_partitions(list(range(k)))] for k in (1, 2, 3, 4)}


def distinct_sum(S, harmonics, roles):
    """Sum over distinct particles of prod_m w_m e^{i h_m phi_m}.

    S(h, n_poi, n_ref) -> (n_events,) complex: single-particle sum over the
    particles in every member's sample, weights to the members' powers.
    roles: 'P' or 'R' per member."""
    total = 0.0
    for part in _PARTITIONS[len(harmonics)]:
        coef = 1
        term = 1.0
        for b in part:
            coef *= (-1) ** (len(b) - 1) * factorial(len(b) - 1)
            nP = sum(1 for m in b if roles[m] == 'P')
            term = term * S(sum(harmonics[m] for m in b), nP, len(b) - nP)
        total = total + coef * term
    return total


def correlator_arrays(S, n_harm, order, which=('poi', 'ref')):
    """Per-event numerators and denominators for harmonics 1..n_harm-1.

    'poi': N2p (n_events, n_harm-1) complex, D2p (n_events,) real, and with
    order 4 N4p, D4p; 'ref': N2, D2 (and N4, D4)."""
    out = {}
    hs = range(1, n_harm)
    specs = []
    if 'poi' in which:
        specs += [('2p', (1, -1), 'PR')] + ([('4p', (1, 1, -1, -1), 'PRRR')] if order >= 4 else [])
    if 'ref' in which:
        specs += [('2', (1, -1), 'RR')] + ([('4', (1, 1, -1, -1), 'RRRR')] if order >= 4 else [])
    for tag, sign, roles in specs:
        out['N' + tag] = np.stack([distinct_sum(S, tuple(h * x for x in sign), roles) for h in hs], axis=1)
        out['D' + tag] = np.real(distinct_sum(S, (0,) * len(sign), roles))
        if len(sign) == 4:
            _note_missing_q2n(S, [h for h in hs if 2 * h >= n_harm], n_harm)
    return out


_NOTED = set()


def _note_missing_q2n(S, harms, n_harm):
    """<4>_{n,n,-n,-n} needs Q_{2n} where two members coincide: only
    weight-squared self-correlation terms, of relative order 1/M_eff
    (M_eff = (sum w)^2 / sum w^2).  A container that does not store Q_{2n}
    drops them; say so once, with M_eff (oversampled 2D JETSCAPE containers:
    ~2e4, a < 1% effect on c3{4}; unit weights: M_eff = M)."""
    key = (tuple(harms), n_harm)
    if not harms or key in _NOTED:
        return
    _NOTED.add(key)
    w1, w2 = np.real(S(0, 0, 1)), np.real(S(0, 0, 2))
    ok = w2 > 0
    m_eff = float(np.median(w1[ok] ** 2 / w2[ok])) if ok.any() else float('nan')
    print(f"  note: v_n{{4}} for n = {', '.join(map(str, harms))}: Q_2n is not stored (harmonics < {n_harm}); "
          f"its self-correlation terms (relative order 1/M_eff, median M_eff = {m_eff:.3g}) are dropped")


def sc_arrays(S, m, n):
    """Per-event sums for the symmetric cumulant SC(m, n) = <<4>>_{m,n,-m,-n}
    - <<2>>_m <<2>>_n over distinct particles of one sample: N4, D4, N2m, N2n,
    D2 (Bilandzic et al., PRC 89, 064904)."""
    return {'N4': distinct_sum(S, (m, n, -m, -n), 'RRRR'), 'D4': np.real(distinct_sum(S, (0, 0, 0, 0), 'RRRR')),
            'N2m': distinct_sum(S, (m, -m), 'RR'), 'N2n': distinct_sum(S, (n, -n), 'RR'),
            'D2': np.real(distinct_sum(S, (0, 0), 'RR'))}


def two_particle_gap(S_AB, h):
    """Per-event <2> between two subevents (POI role = A, REF role = B):
    numerator and pair count over distinct particles."""
    return distinct_sum(S_AB, (h, -h), 'PR'), np.real(distinct_sum(S_AB, (0, 0), 'PR'))


def _avg(num, den, w):
    """<<k>> = sum_e w N_e / sum_e w D_e along the event axis (0)."""
    d = np.tensordot(w, den, axes=(0, 0))
    n = np.tensordot(w, np.real(num), axes=(0, 0))
    return np.where(np.abs(d) > 1e-300, n / np.where(np.abs(d) > 1e-300, d, 1.0), np.nan)


def vn_poi(arrays, w, order):
    """v'_n{order} for harmonics 1..N-1 from correlator_arrays output.

    Arrays may carry a trailing bin axis (N: (n_events, n_harm-1, n_bins),
    D: (n_events, n_bins)); the result is then (n_harm-1, n_bins)."""
    w = np.asarray(w, dtype=float)
    d2 = _avg(arrays['N2p'], arrays['D2p'], w)
    c2 = _avg(arrays['N2'], arrays['D2'], w)
    if arrays['N2p'].ndim == 3:            # binned POI, integrated reference
        c2 = c2[:, None]
    if order == 2:
        root = np.sqrt(np.abs(c2))
        return np.where(root > 0, d2 * np.sign(c2) / np.where(root > 0, root, 1.0), 0.0)
    four_p = _avg(arrays['N4p'], arrays['D4p'], w)
    four = _avg(arrays['N4'], arrays['D4'], w)
    if arrays['N4p'].ndim == 3:
        four = four[:, None]
    d4 = four_p - 2.0 * d2 * c2
    c4 = four - 2.0 * c2 ** 2
    den = np.abs(c4) ** 0.75
    return np.where(den > 0, -d4 / np.where(den > 0, den, 1.0), 0.0)


# ── backends ────────────────────────────────────────────────────────────────

class Container2D:
    """Single-particle sums on a boost-invariant QVectorData.

    poi / ref: dicts with 'idx' (species-axis indices), 'pt' (lo, hi) slice
    indices and 'jac' (n_species, n_pt) window Jacobian.  A particle's weight
    is jac / (2 rap_max nsamples); POI and REF share particles on the common
    species and pT bins (every particle is in every window, with its weight).
    """

    def __init__(self, qv, poi, ref):
        self.qv, self.poi, self.ref = qv, poi, ref
        self.ns = np.asarray(qv.nsamples, dtype=float)
        self.cache = {}

    def __call__(self, h, nP, nR):
        key = (abs(h), nP, nR)
        if key not in self.cache:
            self.cache[key] = self._sum(abs(h), nP, nR)
        v = self.cache[key]
        return np.conj(v) if h < 0 else v

    def _sum(self, h, nP, nR):
        qv = self.qv
        n_ev = qv.Qn.shape[0]
        if h >= qv.Qn.shape[2]:
            return np.zeros(n_ev, complex)       # harmonic not stored (cf. _core_vn4_single)
        sets = ([self.poi] if nP else []) + ([self.ref] if nR else [])
        idx = sorted(set(sets[0]['idx']).intersection(*[s['idx'] for s in sets[1:]]))
        lo = max(s['pt'][0] for s in sets)
        hi = min(s['pt'][1] for s in sets)
        if not idx or hi <= lo:
            return np.zeros(n_ev, complex)
        wgt = np.ones((len(idx), hi - lo))
        for s, k in ((self.poi, nP), (self.ref, nR)):
            if k:
                wgt = wgt * (s['jac'][idx, lo:hi] / (2.0 * qv.rap_max)) ** k
        raw = qv.Qn[:, idx, h, lo:hi].astype(np.complex128)
        return np.einsum('esp,sp->e', raw, wgt) / self.ns ** (nP + nR)


class Container3D:
    """Single-particle sums on unit-weight QGridData (one per sample).

    poi_grid / ref_grid: species-selected grids; windows (eta_lo, eta_hi,
    pt_lo, pt_hi); overlap: whether POI particles are reference particles
    (POI species within the reference species).  Unit weights: every power
    of the weight is 1."""

    def __init__(self, poi_grid, poi_win, ref_grid, ref_win, overlap):
        self.pg, self.pw, self.rg, self.rw, self.overlap = poi_grid, poi_win, ref_grid, ref_win, overlap
        self.cache = {}

    def __call__(self, h, nP, nR):
        key = (abs(h), bool(nP), bool(nR))
        if key not in self.cache:
            self.cache[key] = self._sum(*key)
        v = self.cache[key]
        return np.conj(v) if h < 0 else v

    def _sum(self, h, P, R):
        grid = self.pg if P else self.rg
        n_ev = grid.n_events
        if h >= grid.Qn.shape[-1]:
            return np.zeros(n_ev, complex)
        if P and R:
            if not self.overlap:
                return np.zeros(n_ev, complex)
            e0, e1 = max(self.pw[0], self.rw[0]), min(self.pw[1], self.rw[1])
            p0, p1 = max(self.pw[2], self.rw[2]), min(self.pw[3], self.rw[3])
            if e1 <= e0 or p1 <= p0:
                return np.zeros(n_ev, complex)
            return grid.acceptance(e0, e1, p0, p1)[:, h]
        win = self.pw if P else self.rw
        return grid.acceptance(*win)[:, h]
