"""Flow of particles of interest from three sets of two-particle correlations (3x2PC).

PHENIX (the 3x2PC method) and ALICE (the three-subevent 2PC of
Nature Commun. 17 (2026) 2585, arXiv:2411.09323, Eq. A.2) combine the
correlations of the POI (A) with two reference regions B and C, and of B
with C:

  v_n(A) = sqrt( c_n(A, B) c_n(A, C) / c_n(B, C) ),
  c_n(X, Y) = <<cos n(phi_X - phi_Y)>>   (V_{n Delta} of the 2PC),

each over distinct pairs, events weighted by their pair counts.  Without
non-flow c_n(A, B) = <v_A v_B>, so v_n(A) = v_A when the flows do not
fluctuate independently.  The experiments remove non-flow from each
correlation (template fits, low-multiplicity subtraction); a model without
non-flow compares directly.  One value per POI bin; signed root.
"""
import numpy as np


def _avg(num, den, w):
    w = np.asarray(w, float)
    d = np.tensordot(w, den, axes=(0, 0))
    n = np.tensordot(w, num, axes=(0, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)


def core(d, w):
    ab = _avg(d['ab'], d['cab'], w)          # (n_bins,)
    ac = _avg(d['ac'], d['cac'], w)
    bc = _avg(d['bc'], d['cbc'], w)          # scalar
    with np.errstate(divide='ignore', invalid='ignore'):
        x = ab * ac / bc
        return np.atleast_1d(np.sign(x) * np.sqrt(np.abs(x)))
