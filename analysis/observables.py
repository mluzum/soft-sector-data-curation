import numpy as np


def vn2(Qn, Q0, dQ0):
    """v_n{2} standard (no η-gap).

    Follows qnscript.integratedvn2:
      c = ⟨|Qn|² - dQ0⟩ / ⟨Q0² - dQ0⟩
      v = sign(c) · √|c|

    Parameters
    ----------
    Qn : (n_events,) complex — Q_n = Σ w e^{inφ}
    Q0 : (n_events,) float — Σ w (single weighted)
    dQ0 : (n_events,) float — Σ w² (double weighted, self-correction)
    """
    num = np.abs(Qn) ** 2 - dQ0
    den = Q0 * Q0 - dQ0
    c = np.sum(num) / max(np.sum(den), 1e-60)
    return np.sign(c) * np.sqrt(np.abs(c))


def vn2_gap(Qn_A, Qn_B, MA, MB, dQ_cross):
    """v_n{2} with η-gap using two subevents A, B.

    Self-correction δ_AB = Σ w_A · w_B.
    For 2D (boost-invariant): same particles, different Jacobian weights per η range.
      δ_AB = Σ_i w_A,i · w_B,i = Σ n_raw · jac_A · jac_B / (2·rap_max·ns)²
    For 3D (disjoint subevents): δ_AB = 0.

      c = ⟨Re(Q_A · Q_B*) - δ_AB⟩ / ⟨M_A · M_B - δ_AB⟩
      v = sign(c) · √|c|

    Parameters
    ----------
    Qn_A : (n_events,) complex — Q_n for subevent A
    Qn_B : (n_events,) complex — Q_n for subevent B
    MA : (n_events,) float — M_A = Σ w_A
    MB : (n_events,) float — M_B = Σ w_B
    dQ_cross : (n_events,) float — δ_AB = Σ w_A · w_B
    """
    num = np.real(Qn_A * np.conj(Qn_B)) - dQ_cross
    den = MA * MB - dQ_cross
    c = np.sum(num) / max(np.sum(den), 1e-60)
    return np.sign(c) * np.sqrt(np.abs(c))


def vn4(Qn, dQn, tQn, dQ2n, Q0, dQ0, tQ0, qQ0):
    """v_n{4} standard (no η-gap).

    Follows qnscript.integratedvn4 (Bilandzic Eq. 3.56/B16-B17).
    Requires multi-weighted Q-vectors for proper self-correction.

    v_n{4} = sign(-c_n{4})^{1/4} · | -c_n{4} |^{1/4}
    where c_n{4} = c_n{4, raw} and c_n{2} = c_n{2, raw}
    and v_n{4} = (2·c_n{2}² - c_n{4})^{1/4}
    """
    c4 = np.abs(Qn) ** 4 + np.abs(dQ2n) ** 2
    c4 += -2 * np.real(dQ2n * np.conj(Qn) * np.conj(Qn))
    c4 += -4 * dQ0 * np.abs(Qn) ** 2
    c4 += 8 * np.real(tQn * np.conj(Qn))
    c4 += 2 * dQ0 * dQ0 - 6 * qQ0
    c4 = np.sum(c4)

    denom = (np.power(Q0, 4) - 6 * dQ0 * Q0 * Q0
             + 3 * dQ0 * dQ0 + 8 * Q0 * tQ0 - 6 * qQ0)
    c4 /= max(np.sum(denom), 1e-60)

    c2 = np.abs(Qn) ** 2 - dQ0
    c2 = np.sum(c2) / max(np.sum(Q0 * Q0 - dQ0), 1e-60)

    vn4 = 2 * c2 ** 2 - c4
    return np.sign(vn4) * np.power(np.abs(vn4), 0.25)


def diff_vn2(Qn_int, pn, Q0, mq, dmq, dQ0):
    """Differential v_n{2}(pT).

    Follows qnscript.diffvn2, Eq. B.13-B.17 in Bilandzic thesis.
    """
    dn = pn * np.conj(Qn_int) - dmq
    dn_den = mq * Q0 - dmq
    dn = np.real(dn) / dn_den
    dn = np.mean(dn)

    c2 = np.abs(Qn_int) ** 2 - dQ0
    c2 = np.sum(c2) / max(np.sum(Q0 * Q0 - dQ0), 1e-60)
    vn2 = np.sign(c2) * np.sqrt(np.abs(c2))

    return dn / max(vn2, 1e-60)


def compute_centrality_weights(sort_key, cent_edges):
    """Binary centrality weights from percentile edges.

    Parameters
    ----------
    sort_key : (n_events,) float — quantity to sort by (e.g. multiplicity)
    cent_edges : list of float — centrality percentile boundaries [0, 5, 10, ..., 100]

    Returns
    -------
    weights : (n_events, n_cent) float — one-hot per centrality bin
    cent_labels : list of str
    """
    n = len(sort_key)
    sorted_idx = np.argsort(sort_key)[::-1]
    weights = np.zeros((n, len(cent_edges) - 1), dtype=np.float64)
    for i in range(len(cent_edges) - 1):
        lo = int(cent_edges[i] / 100 * n)
        hi = int(cent_edges[i + 1] / 100 * n)
        weights[sorted_idx[lo:hi], i] = 1.0
    cent_labels = [f"{cent_edges[i]:.0f}-{cent_edges[i+1]:.0f}%"
                   for i in range(len(cent_edges) - 1)]
    return weights, cent_labels


def jackknife(function, arg, n_blocks=None):
    """Delete-1 jackknife error estimate.

    Follows qnscript.jackknifeerror.
    """
    nevents = len(arg)
    val = function(arg)
    jkdist = np.array([function(np.delete(arg, i, axis=0))
                        for i in range(nevents)])
    error = np.sqrt(nevents - 1) * np.std(jkdist, axis=0)
    corrval = nevents * val - (nevents - 1) * np.mean(jkdist, axis=0)
    return corrval, error


def block_jackknife(function, arg, n_blocks=20):
    """Block jackknife for fewer compute passes.

    Returns (bias-corrected value, error, covariance matrix).
    """
    nevents = len(arg)
    block_size = nevents // n_blocks
    val = function(arg)
    theta_i = []
    for i in range(n_blocks):
        lo = i * block_size
        hi = nevents if i == n_blocks - 1 else (i + 1) * block_size
        mask = np.ones(nevents, dtype=bool)
        mask[lo:hi] = False
        theta_i.append(function(arg[mask]))
    theta_i = np.array(theta_i)
    n_b = len(theta_i)
    bias = (n_b - 1) * (np.mean(theta_i, axis=0) - val)
    corrval = val - bias
    diff = theta_i - np.mean(theta_i, axis=0)
    cov = (n_b - 1) / n_b * np.dot(diff.T, diff)
    error = np.sqrt(np.maximum(np.diag(cov), 0))
    return corrval, error, cov
