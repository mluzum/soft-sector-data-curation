"""Gulbrandsen recursion for exact multi-particle correlators from Q-vectors.

Computes ⟨k⟩_harmonic_pattern for arbitrary weight powers, handling all-
distinct index constraints through the recursive structure (Gulbrandsen,
as used in FlowNICA/CumulantFlow and the ALICE high-order flow framework).

This is an INDEPENDENT algorithm from the explicit direct-Q expressions —
agreement between the two validates both implementations.

Usage:
    Q = {(h, p): complex_vector}   # harmonic h, weight power p
    corr = recursion(Q, n=6, harm=[2, 2, 2, 2, 2, 2])
"""
import numpy as np


def recursion(Q, n, harm, mult=1, skip=0):
    """Recursive computation of multi-particle azimuthal correlators.

    Parameters
    ----------
    Q : dict {(harmonic, power): complex array}
        Q-vectors indexed by (harmonic, weight-power).
    n : int
        Number of particles in the correlator.
    harm : list of int
        Harmonic assignment per particle.
    mult : int
        Current weight multiplicity (internal).
    skip : int
        Position to skip (internal).
    """
    nm1 = n - 1
    if harm[nm1] >= 0:
        c = Q[(harm[nm1], mult)]
    else:
        c = np.conj(Q[(-harm[nm1], mult)])
    if nm1 == 0:
        return c

    c = c * recursion(Q, nm1, harm, mult, skip)
    if nm1 == skip:
        return c

    mult_p1 = mult + 1
    nm2 = n - 2
    counter1 = 0
    hhold = harm[counter1]
    harm[counter1] = harm[nm2]
    harm[nm2] = hhold + harm[nm1]

    c2 = recursion(Q, nm1, harm, mult_p1, nm2)

    counter2 = n - 3
    while counter2 >= skip:
        harm[nm2] = harm[counter1]
        harm[counter1] = hhold
        counter1 += 1
        hhold = harm[counter1]
        harm[counter1] = harm[nm2]
        harm[nm2] = hhold + harm[nm1]
        c2 = c2 + recursion(Q, nm1, harm, mult_p1, counter2)
        counter2 -= 1

    harm[nm2] = harm[counter1]
    harm[counter1] = hhold

    import numpy as _np
    if mult == 1:
        return c - c2
    return c - float(mult) * c2


def correlation(Q, k, harm_list):
    """Compute the k-particle normalized correlator.

    Returns ⟨k⟩ / ⟨denominator⟩ where the numerator uses the assigned
    harmonics and the denominator uses all-zero harmonics (= multiplicities).
    """
    num = recursion(Q, k, list(harm_list))
    den = recursion(Q, k, [0] * k).real
    import numpy as _np
    return num / _np.maximum(_np.real(den), 1e-60)


def build_qdict(qv_harmonics, weight_powers=None, jac=None):
    """Build Q-dict from a (n_events, n_harm) array.

    qv_harmonics: (n_events, n_harm) complex — slot h holds harmonic-h.
    weight_powers: list of ints; if given, also builds higher-power entries
                   by multiplying by jac (per-event weight array per bin).
                   For unit weights pass None.
    """
    Q = {}
    ne, nh = qv_harmonics.shape
    max_p = max(weight_powers) if weight_powers else 1
    for h in range(-(nh - 1), nh):
        for pw in range(1, max_p + 1):
            if abs(h) < nh:
                if pw == 1 or weight_powers is None:
                    Q[(h, pw)] = (
                        qv_harmonics[:, h].copy() if h >= 0
                        else np.conj(qv_harmonics[:, -h]))
                elif jac is not None:
                    base = (qv_harmonics[:, h].copy() if h >= 0
                            else np.conj(qv_harmonics[:, -h]))
                    Q[(h, pw)] = base * jac[:, pw - 1]
    return Q
