"""Linear and mode-coupled flow harmonics (nonlinear response).

The higher harmonic V_k holds a part driven by products of lower harmonics:
V_k = V_k^L + chi_{k,nm} V_n V_m (k = n + m), or with three, V_k = V_k^L +
chi_{k,nml} V_n V_m V_l (k = n + m + l: chi_{6,222}, chi_{7,223}).  The
measurements (STAR PLB 809 135728, PLB 839 137755; ALICE PLB 773 68, JHEP
05 (2020) 085) use, over distinct particles,

  P2 = <<cos k(phi_1 - phi_2)>>                        v_k{2}^2
  C  = <<cos(k phi_1 - n phi_2 - m phi_3 [- l phi_4])>>  C_{k,nm[l]}
  PN = <<cos(n phi_1 + m phi_2 [+ l phi_3] - n phi_4 - m phi_5 [- l phi_6])>>
                                                       <v_n^2 v_m^2 [v_l^2]>

either in one window ('standard') or with two subevents A and B (particles
AB): particle 1 of P2 and C in A, the others in B; the first half of PN
(the positive harmonics) in A, the rest in B.  Both orientations (A <-> B)
are pooled.  Then

  v_inclusive    = sign(P2) sqrt|P2|
  v_mode_coupled = C / sqrt(PN)          (signed root of PN)
  v_linear       = signed sqrt(v_inclusive^2 - v_mode_coupled^2)
  rho            = v_mode_coupled / v_inclusive   (<cos(k Psi_k - n Psi_n - ...)>)
  chi            = v_mode_coupled / sqrt(PN) = C / PN, or with
                   chi_denominator 'factorized': v_mode_coupled /
                   sqrt(<v_n^2> <v_m^2> [<v_l^2>]) (STAR's differential
                   chi_{5,23})

pT-differential: particle 1 of P2 and C in the pT bin (and its subevent);
the normalizations stay integrated: v_inclusive(pT) = P2(pT) / v_k{2}, the
reference v_k{2} = sign(P2) sqrt|P2| over the record's pT range.

Every correlator is a sum over distinct particles from analysis.flow_pt
(CellSums; set partitions, distinct, up to four members, the generating
function, distinct_gf, beyond): windows weight each particle (2D: the
eta-window Jacobian times the pT share; 3D: the cells' overlap shares),
coincidence terms are removed exactly.  Events are weighted by their tuple
counts.
"""
import numpy as np

from analysis.flow_pt import distinct, distinct_gf

QUANTITIES = ('C', 'v_inclusive', 'v_mode_coupled', 'v_linear', 'chi', 'rho')


def _sroot(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def harmonics_ok(harms):
    """[k, n, m] or [k, n, m, l], k the sum of the others, all >= 1."""
    return len(harms) in (3, 4) and min(harms) >= 1 and harms[0] == sum(harms[1:])


def _dsum(S, members):
    """Sum over distinct particles of prod a_W e^{i h phi}, members [(window,
    h)]: set partitions up to four members, the generating function beyond."""
    if len(members) <= 4:
        return distinct(S, [(w, h, 0) for w, h in members])
    types = {}
    for w, h in members:
        types[(w, h)] = types.get((w, h), 0) + 1
    return distinct_gf(S, [(w, h, c) for (w, h), c in types.items()])


def event_sums(S, harmonics, orientations, bins):
    """Per-event numerators and tuple counts.

    harmonics: [k, n, m(, l)]; orientations: [(X, Y)] window-name pairs (two
    subevents: ('A', 'B'), ('B', 'A'); one window: ('full', 'full')); bins:
    per pT bin, the POI window names {X: name} of each orientation's X
    (empty: integrated).
    """
    k, low = harmonics[0], list(harmonics[1:])
    out = {}

    def add(key, val):
        out[key] = out.get(key, 0.0) + np.real(val)

    for X, Y in orientations:
        add('P2', _dsum(S, [(X, k), (Y, -k)]))
        add('W2', _dsum(S, [(X, 0), (Y, 0)]))
        add('C', _dsum(S, [(X, k)] + [(Y, -h) for h in low]))
        add('WC', _dsum(S, [(X, 0)] + [(Y, 0)] * len(low)))
        add('PN', _dsum(S, [(X, h) for h in low] + [(Y, -h) for h in low]))
        add('WN', _dsum(S, [(X, 0)] * len(low) + [(Y, 0)] * len(low)))
        for j, h in enumerate(low):
            add(f'Pl{j}', _dsum(S, [(X, h), (Y, -h)]))
        for b, poi in enumerate(bins):
            Xb = poi[X]
            add(f'P2_{b}', _dsum(S, [(Xb, k), (Y, -k)]))
            add(f'W2_{b}', _dsum(S, [(Xb, 0), (Y, 0)]))
            add(f'C_{b}', _dsum(S, [(Xb, k)] + [(Y, -h) for h in low]))
            add(f'WC_{b}', _dsum(S, [(Xb, 0)] + [(Y, 0)] * len(low)))
    out['n_low'] = len(low)
    return out


def _avg(num, den, w):
    d = np.sum(w * den)
    return np.sum(w * num) / d if d > 0 else np.nan


def core(d, w):
    """One centrality class: the record's quantity, one value per pT bin
    (one value when integrated)."""
    q = d['quantity']
    P2 = _avg(d['P2'], d['W2'], w)
    PN = _avg(d['PN'], d['WN'], w)
    vref = _sroot(P2)
    if d['factorized']:
        norm = _sroot(np.prod([_avg(d[f'Pl{j}'], d['W2'], w) for j in range(d['n_low'])]))
    else:
        norm = _sroot(PN)
    if d['n_bins']:
        p2 = np.array([_avg(d[f'P2_{b}'], d[f'W2_{b}'], w) for b in range(d['n_bins'])])
        c = np.array([_avg(d[f'C_{b}'], d[f'WC_{b}'], w) for b in range(d['n_bins'])])
        with np.errstate(divide='ignore', invalid='ignore'):
            v_inc = p2 / vref
    else:
        c = np.array([_avg(d['C'], d['WC'], w)])
        v_inc = np.array([vref])
    with np.errstate(divide='ignore', invalid='ignore'):
        v_mc = c / _sroot(PN)
        if q == 'C':
            return c
        if q == 'v_inclusive':
            return v_inc
        if q == 'v_mode_coupled':
            return v_mc
        if q == 'v_linear':
            return _sroot(v_inc ** 2 - v_mc ** 2)
        if q == 'rho':
            return v_mc / v_inc
        if q == 'chi':
            return v_mc / norm
    raise ValueError(f"mode coupling: quantity {q!r} (one of {', '.join(QUANTITIES)})")
