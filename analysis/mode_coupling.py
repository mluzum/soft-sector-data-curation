"""Linear and mode-coupled flow harmonics (nonlinear response).

The higher harmonic V_k (k = n + m) holds a part driven by the product of
lower harmonics: V_k = V_k^L + chi_{k,nm} V_n V_m.  The measurements (STAR
PLB 809 135728, PLB 839 137755; ALICE, ATLAS) use, over distinct particles,

  P2 = <<cos k(phi_1 - phi_2)>>                      v_k{2}^2
  C  = <<cos(k phi_1 - n phi_2 - m phi_3)>>          C_{k,nm}
  P4 = <<cos(n phi_1 + m phi_2 - n phi_3 - m phi_4)>>  <v_n^2 v_m^2>

either in one window ('standard') or with two subevents A and B (particles
AB): particle 1 of P2 and C in A, the others in B; particles 1, 2 of P4 in A,
3, 4 in B.  Both orientations (A <-> B) are pooled.  Then

  v_inclusive    = sign(P2) sqrt|P2|
  v_mode_coupled = C / sqrt(P4)          (signed root of P4)
  v_linear       = signed sqrt(v_inclusive^2 - v_mode_coupled^2)
  rho            = v_mode_coupled / v_inclusive   (<cos(k Psi_k - n Psi_n - m Psi_m)>)
  chi            = v_mode_coupled / sqrt(P4) = C / P4, or with
                   chi_denominator 'factorized': v_mode_coupled /
                   sqrt(<v_n^2> <v_m^2>) (STAR's differential chi_{5,23})

pT-differential: particle 1 of P2 and C in the pT bin (and its subevent);
the normalizations stay integrated: v_inclusive(pT) = P2(pT) / v_k{2}, the
reference v_k{2} = sign(P2) sqrt|P2| over the record's pT range.

Every correlator is a sum over distinct particles from analysis.flow_pt
(CellSums, distinct): windows weight each particle (2D: the eta-window
Jacobian times the pT share; 3D: the cells' overlap shares), coincidence
terms are removed exactly.  Events are weighted by their tuple counts.
"""
import numpy as np

from analysis.flow_pt import distinct

QUANTITIES = ('C', 'v_inclusive', 'v_mode_coupled', 'v_linear', 'chi', 'rho')


def _sroot(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def event_sums(S, k, n, m, orientations, bins):
    """Per-event numerators and tuple counts.

    orientations: [(X, Y)] window-name pairs (two subevents: ('A', 'B'),
    ('B', 'A'); one window: ('full', 'full')); bins: per pT bin, the POI
    window names {X: name} of each orientation's X (empty: integrated).
    """
    out = {}

    def add(key, val):
        out[key] = out.get(key, 0.0) + np.real(val)

    for X, Y in orientations:
        add('P2', distinct(S, [(X, k, 0), (Y, -k, 0)]))
        add('W2', distinct(S, [(X, 0, 0), (Y, 0, 0)]))
        add('C', distinct(S, [(X, k, 0), (Y, -n, 0), (Y, -m, 0)]))
        add('W3', distinct(S, [(X, 0, 0), (Y, 0, 0), (Y, 0, 0)]))
        add('P4', distinct(S, [(X, n, 0), (X, m, 0), (Y, -n, 0), (Y, -m, 0)]))
        add('W4', distinct(S, [(X, 0, 0), (X, 0, 0), (Y, 0, 0), (Y, 0, 0)]))
        add('Pn', distinct(S, [(X, n, 0), (Y, -n, 0)]))
        add('Pm', distinct(S, [(X, m, 0), (Y, -m, 0)]))
        for b, poi in enumerate(bins):
            Xb = poi[X]
            add(f'P2_{b}', distinct(S, [(Xb, k, 0), (Y, -k, 0)]))
            add(f'W2_{b}', distinct(S, [(Xb, 0, 0), (Y, 0, 0)]))
            add(f'C_{b}', distinct(S, [(Xb, k, 0), (Y, -n, 0), (Y, -m, 0)]))
            add(f'W3_{b}', distinct(S, [(Xb, 0, 0), (Y, 0, 0), (Y, 0, 0)]))
    return out


def _avg(num, den, w):
    d = np.sum(w * den)
    return np.sum(w * num) / d if d > 0 else np.nan


def core(d, w):
    """One centrality class: the record's quantity, one value per pT bin
    (one value when integrated)."""
    q = d['quantity']
    P2 = _avg(d['P2'], d['W2'], w)
    P4 = _avg(d['P4'], d['W4'], w)
    vref = _sroot(P2)
    if d['factorized']:
        norm = _sroot(_avg(d['Pn'], d['W2'], w) * _avg(d['Pm'], d['W2'], w))
    else:
        norm = _sroot(P4)
    if d['n_bins']:
        p2 = np.array([_avg(d[f'P2_{b}'], d[f'W2_{b}'], w) for b in range(d['n_bins'])])
        c = np.array([_avg(d[f'C_{b}'], d[f'W3_{b}'], w) for b in range(d['n_bins'])])
        with np.errstate(divide='ignore', invalid='ignore'):
            v_inc = p2 / vref
    else:
        c = np.array([_avg(d['C'], d['W3'], w)])
        v_inc = np.array([vref])
    with np.errstate(divide='ignore', invalid='ignore'):
        v_mc = c / _sroot(P4)
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
