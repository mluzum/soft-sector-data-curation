"""Pseudorapidity-differential flow with side-dependent reference regions, and its decorrelation.

ALICE (PLB 850 (2024) 138477, arXiv:2307.11116) correlates particles of
interest in narrow eta bins with reference particles on the other side of
the detector, and normalizes by the correlation of that reference region
with its partner (Eqs. 1, 2):

  v'_n(eta) = <<cos n(phi'(eta) - phi^Y)>> / sqrt(<<cos n(phi^X - phi^Y)>>),

Y the reference region of the bin's side (`flow_reference.segments`, each
with `applies_to_eta`, `eta`, `pT`) and X its `partner` region.  Without
longitudinal decorrelation X and Y carry the same flow; with it the
cross-correlation, not |Q_Y|^2, is what the measurement divides by.

The decorrelation ratio (Eq. 5; the r_n(eta) of CMS and ATLAS):

  r_{n|n}(eta) = <<cos n(phi'(-eta) - phi^Y)>> / <<cos n(phi'(eta) - phi^Y)>>,

the POI at -eta on the side opposite the reference Y, at +eta on its side;
both orientations (Y at +-|eta_ref|) pooled.

The four-plane correlator (STAR, PRC 113 (2026) 024908, Eqs. 17-19, the
particle-level calculation; Bhalerao et al.'s T_n):

  T_n{ba; dc} = <<sin n(phi_b - Psi_a) sin n(Psi_d - phi_c)>>
                / (<cos n(phi_b - Psi_a)> <cos n(Psi_d - phi_c)>),

<<..>> = <sin sin> - 1/2 <cos n(phi_b - Psi_d)><cos n(Psi_a - phi_c)> +
1/2 <cos n(Psi_a - Psi_d)><cos n(phi_b - phi_c)>, a and d the reference
windows at -|eta_ref| and +|eta_ref|, b and c the record's |eta| window at
negative and positive eta.  The scalar-product analog (the event-plane
result at low resolution of a and d): each e^{i n Psi_X} becomes the flow
vector of X, so with u_X = e^{i n phi} summed over X,

  <sin sin> = 1/2 Re(<u_b u_c u_a* u_d*> - <u_b u_c* u_a* u_d>),

and every correlator holds one particle of a and one of d, whose
normalization cancels in the ratio.  T_n is unchanged by the mirror
(a <-> d, b <-> c).  Zero without decorrelation.

Sums over distinct particles (analysis.flow_pt.CellSums), events weighted
by their pair counts.  One value per eta bin; signed roots.
"""
import numpy as np


def _sroot(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def _avg(num, den, w):
    w = np.asarray(w, float)
    d = np.tensordot(w, den, axes=(0, 0))
    n = np.tensordot(w, num, axes=(0, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)


def core_vn2(d, w):
    """v'_n(eta) per bin: num/cnt (n_ev, n_bins), the partner correlation
    xy/cxy (n_ev, n_bins)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return _avg(d['num'], d['cnt'], w) / _sroot(_avg(d['xy'], d['cxy'], w))


def core_rnn(d, w):
    """r_{n|n}(eta) per |eta| bin: opposite-side over same-side pair
    correlations, both orientations pooled."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return _avg(d['opp'], d['copp'], w) / _avg(d['same'], d['csame'], w)


TN_TERMS = ('f1', 'f2', 'bd', 'ac', 'ad', 'bc', 'ba', 'dc')


def core_tn(d, w):
    """T_n{ba; dc}: the four-plane cumulant over its resolution product,
    each correlator a tuple-weighted average over distinct particles (one
    per window, b and c in the record's window, a and d the reference)."""
    c = {k: _avg(d[k], d['c' + k], w) for k in TN_TERMS}
    with np.errstate(divide='ignore', invalid='ignore'):
        return (0.5 * (c['f1'] - c['f2']) - 0.5 * c['bd'] * c['ac'] + 0.5 * c['ad'] * c['bc']) / (c['ba'] * c['dc'])
