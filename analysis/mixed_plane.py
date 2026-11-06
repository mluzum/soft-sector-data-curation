"""Flow relative to a lower-order plane (mixed harmonic): v_n{Psi_m}, m dividing n.

STAR measures v_2 of identified hadrons at sqrt(sNN) = 3-4.5 GeV relative to
the first-order event plane of its forward EPD (PLB 827 137003; PRL 2025,
arXiv:2504.02531): v_2{Psi_1} = <cos 2(phi - Psi_1^A)> / R_21(A), the
resolution from three subevents A, B, C,
R_21(A) = sqrt(<cos 2(Psi_1^A - Psi_1^B)><cos 2(Psi_1^A - Psi_1^C)> /
<cos 2(Psi_1^B - Psi_1^C)>).  Computed as the scalar-product analog (the
event-plane result at low resolution, as the repository serves event-plane
records): with k = n / m and D_X the sum over distinct k-tuples of subevent
X of e^{i m (phi_1 + .. + phi_k)},

  v_n{Psi_m} = <Re q_n D_A*> / sqrt(<Re D_A D_B*><Re D_A D_C*> / <Re D_B D_C*>),

each average over events weighted by its tuple count (POI x k-tuples of A;
k-tuples of two subevents).  Large multiplicity: <v_n v_m^k cos n(Psi_n -
Psi_m)> / sqrt(<v_m^{2k}>) with v_m that of A, i.e. <v_n cos n(Psi_n -
Psi_m)> when v_m does not fluctuate.  The sign of Psi_m drops out for even
k (Psi_1 + pi gives the same 2 Psi_1).  The POI and subevent A must not
share particles (refused).
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


def core(d, w):
    """One centrality class: v_n{Psi_m} per POI bin."""
    num = _avg(d['num'], d['cnt'], w)                    # (n_bins,)
    ab = _avg(d['ab'], d['cab'], w)
    ac = _avg(d['ac'], d['cac'], w)
    bc = _avg(d['bc'], d['cbc'], w)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.atleast_1d(num / _sroot(ab * ac / bc))
