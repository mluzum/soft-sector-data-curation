import numpy as np
from scipy import integrate
from analysis.jackknife import jackknife_blocks
from analysis.dispatcher import register


def _integrated_jacobian(pt_val, mass, eta_min, eta_max):
    def integrand(eta):
        return pt_val * np.cosh(eta) / np.sqrt(mass**2 + pt_val**2 * np.cosh(eta)**2)
    result, _ = integrate.quad(integrand, eta_min, eta_max, limit=200)
    return result


def _make_jacobian_list(pt_centers, masses, eta_min, eta_max):
    jac = np.zeros((len(masses), len(pt_centers)), dtype=np.float64)
    for i, mass in enumerate(masses):
        for j, pt in enumerate(pt_centers):
            jac[i, j] = _integrated_jacobian(pt, mass, eta_min, eta_max)
    return jac


def cut_qvectors(qv, eta_min, eta_max):
    jac = _make_jacobian_list(qv.pt_centers, qv.masses, eta_min, eta_max)
    Qn = qv.Qn.astype(np.complex128)
    Qn /= 2.0 * qv.rap_max  # per-rapidity normalization
    jac_expanded = jac[np.newaxis, :, np.newaxis, :]
    Qn *= jac_expanded
    Qn /= qv.nsamples[:, None, None, None]
    return Qn, jac


def rapidity_window_extent(pt, mass, y_min, y_max, assumed_mass):
    """True-rapidity extent of a window in rapidity computed with an assumed
    mass (e.g. y_pi), for particles of the given mass and pT.  At fixed pT,
    p_z = mT sinh y = mT_a sinh y_a, so y = asinh(mT_a / mT * sinh y_a).
    assumed_mass = 0 gives the eta window."""
    pt = np.asarray(pt, dtype=float)
    r = np.sqrt(pt ** 2 + assumed_mass ** 2) / np.sqrt(pt ** 2 + mass ** 2)
    return np.arcsinh(r * np.sinh(y_max)) - np.arcsinh(r * np.sinh(y_min))


def cut_qvectors_rapidity(qv, y_min, y_max, assumed_mass):
    """cut_qvectors for a window in rapidity computed with an assumed mass:
    each species and pT bin weighted by the true-rapidity extent of the
    window (the container is flat in true rapidity)."""
    jac = np.array([rapidity_window_extent(qv.pt_centers, m, y_min, y_max, assumed_mass)
                    for m in qv.masses])
    Qn = qv.Qn.astype(np.complex128)
    Qn /= 2.0 * qv.rap_max
    Qn *= jac[np.newaxis, :, np.newaxis, :]
    Qn /= qv.nsamples[:, None, None, None]
    return Qn, jac


def sum_over_species_pt(Qn):
    return Qn.sum(axis=(1, 3))


def _compute_dq0(qv, Qn_corrected, jac, pt_lo=None, pt_hi=None):
    if pt_lo is None:
        pt_lo = 0
    if pt_hi is None:
        pt_hi = Qn_corrected.shape[3]
    pt_hi = min(pt_hi, jac.shape[1])
    w_sq = (jac[np.newaxis, :, pt_lo:pt_hi] /
            (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis])) ** 2
    Q0_raw = qv.Qn[:, :, 0, pt_lo:pt_hi].real
    return np.sum(Q0_raw * w_sq, axis=(1, 2))


def _compute_weighted_qvectors(qv, jac, power, pt_lo=None, pt_hi=None):
    """Compute Σ w^power · e^{inφ} for all harmonics.

    In 2D boost-invariant sim, the same underlying particles contribute
    regardless of η cut, just with different Jacobian weights.  Nested
    cutQn raises the per-particle weight to the given power.

    Returns (n_events, n_harm) complex array.
    """
    if pt_lo is None:
        pt_lo = 0
    if pt_hi is None:
        pt_hi = jac.shape[1]
    pt_hi = min(pt_hi, jac.shape[1])

    Qn_raw = qv.Qn.astype(np.complex128)[:, :, :, pt_lo:pt_hi]
    jac_pow = (jac ** power)[np.newaxis, :, np.newaxis, pt_lo:pt_hi]
    norm = (2.0 * qv.rap_max) ** power

    weighted = Qn_raw * jac_pow
    weighted /= norm
    weighted /= qv.nsamples[:, None, None, None] ** power

    return weighted.sum(axis=(1, 3))


def _compute_dq0_cross(qv, jac_A, jac_B, pt_lo=None, pt_hi=None):
    """Cross-term δ_AB = Σ w_A · w_B for two η ranges (same particles).

    In 2D boost-invariant sim, the same particles contribute to both subevents
    with different Jacobian weights.  This computes the exact cross-term:
      δ_AB = Σ_i w_A,i · w_B,i = Σ n_raw · jac_A · jac_B / (2·rap_max·ns)²

    For symmetric η ranges (jac_A == jac_B), δ_AB = dQ0_A = dQ0_B.
    For asymmetric η ranges (e.g. trigger at mid-rapidity, reference forward),
    this gives the correct value regardless.
    """
    if pt_lo is None:
        pt_lo = 0
    if pt_hi is None:
        pt_hi = jac_A.shape[1]
    pt_hi = min(pt_hi, jac_A.shape[1])
    n_events = qv.n_events
    wA = jac_A[np.newaxis, :, pt_lo:pt_hi] / (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis])
    wB = jac_B[np.newaxis, :, pt_lo:pt_hi] / (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis])
    Q0_raw = qv.Qn[:, :, 0, pt_lo:pt_hi].real
    return np.sum(Q0_raw * wA * wB, axis=(1, 2))


def _core_vn4_single(Q1, Q2, Q3, Q4, event_weights):
    """Core v_n{4} single-acceptance, no jackknife.

    Args:
        Q1, Q2, Q3, Q4: (n_events, n_harm) complex — weighted Q-vectors
        event_weights: (n_events,) float — single centrality bin, or None

    Returns: (n_harm-1,) float — v₂{4} through v_N{4}.
    """
    n_harm = Q1.shape[1]
    result = np.zeros(n_harm)
    for h in range(1, n_harm):
        Qn = Q1[:, h]
        dQn = Q2[:, h]
        tQn = Q3[:, h]
        dQ2n = Q2[:, 2 * h] if 2 * h < n_harm else np.zeros_like(Qn)
        Q0 = Q1[:, 0].real
        dQ0 = Q2[:, 0].real
        tQ0 = Q3[:, 0].real
        qQ0 = Q4[:, 0].real

        c4 = (np.abs(Qn) ** 4 + np.abs(dQ2n) ** 2
              - 2 * np.real(dQ2n * np.conj(Qn) * np.conj(Qn))
              - 4 * dQ0 * np.abs(Qn) ** 2 + 8 * np.real(tQn * np.conj(Qn))
              + 2 * dQ0 * dQ0 - 6 * qQ0)
        denom = (np.power(Q0, 4) - 6 * dQ0 * Q0 * Q0
                 + 3 * dQ0 * dQ0 + 8 * Q0 * tQ0 - 6 * qQ0)

        if event_weights is not None:
            c4_mean = np.average(c4, weights=event_weights)
            denom_mean = np.average(denom, weights=event_weights)
            c2 = (np.average(np.abs(Qn) ** 2 - dQ0, weights=event_weights)
                  / max(np.average(Q0 * Q0 - dQ0, weights=event_weights), 1e-60))
        else:
            c4_mean = np.mean(c4)
            denom_mean = np.mean(denom)
            c2 = np.mean(np.abs(Qn) ** 2 - dQ0) / max(np.mean(Q0 * Q0 - dQ0), 1e-60)

        v4 = 2 * c2 ** 2 - c4_mean / max(denom_mean, 1e-60)
        result[h] = np.sign(v4) * np.power(np.abs(v4), 0.25)
    return result[1:]


def _zero_pad_harmonics(q_vec, min_slots):
    """Zero-pad a (n_events, n_harm) Q-vector array to min_slots columns.

    Missing harmonics contribute zero to multi-particle correlators,
    equivalent to neglecting those self-correlation terms (suppressed by
    powers of 1/M; negligible at heavy-ion multiplicities).
    """
    import warnings
    nh = q_vec.shape[-1]
    if nh >= min_slots:
        return q_vec
    warnings.warn(
        f"Q-vector has {nh} harmonic slots; padding to {min_slots} "
        f"with zeros (neglects self-correlation terms suppressed by "
        f"1/M -- negligible at heavy-ion multiplicity)",
        UserWarning, stacklevel=2)
    out = np.zeros(q_vec.shape[:-1] + (min_slots,), dtype=q_vec.dtype)
    out[..., :nh] = q_vec
    return out


def _core_vn6_single(Q1, event_weights=None, _pad=True):
    """Core v_n{6} single-acceptance, no jackknife.

    Not used by the pipeline since November 2026 (exact weighted distinct-
    particle sums, pipeline_dispatch._prepare_multi): it needs unit weights and
    an integer multiplicity, and averages <6> (<8>) over events unweighted
    while <2>, <4> are weighted by their tuple counts.

    c_n{6} = <6>_corr - 9 <2><4>_corr + 12 <2>^3,  v_n{6}=(c_n{6}/4)^(1/6)
    where <6>_corr is the exact direct-Q 6-particle correlator
    (Bilandzic/Snellings/Voloshin PRC 83, 044913 -- verified against
    brute-force enumeration to machine precision) and <4>_corr is the
    fully self-correlation-subtracted 4-particle cumulant ratio
    (same convention as _core_vn4_single / Everett qnscript).

    Q1..Q4: (n_events, n_harm) complex weighted vectors (slot 0=counts);
            unit weights -> pass four copies.  (For the 2D path the Jacobian
            weights are already inside Q1 via cut_qvectors; only the
            higher-weight-power self-correlation terms are neglected.)
    Requires harmonic axis up to 3n (for n=2: slot 6 -> n_harm>=7).
    Returns array over available h: v_n{6}, NaN where c_n{6}<=0 or M<6.
    """
    if _pad and Q1.shape[1] < 7:
        Q1 = _zero_pad_harmonics(Q1, max(Q1.shape[1], 7))
    n_harm = Q1.shape[1]
    out = []
    for h in range(1, n_harm):
        if 3 * h >= n_harm:
            break
        Qn = Q1[:, h]
        Q2n = Q1[:, 2 * h]
        Q3n = Q1[:, 3 * h]
        # weight-power vectors at harmonic n (for <4> corrections):
        # Unit-weight self-correlation vectors (= Qn; corrections ~1/M)
        dQ2n = Q1[:, 2 * h] if 2*h < n_harm else np.zeros_like(Qn)
        dQ0 = Q1[:, 0].real
        M = Q1[:, 0].real

        valid = (M >= 6) & (np.abs(M - np.round(M)) < 1e-6)
        P6 = np.where(valid, M * (M - 1) * (M - 2) * (M - 3)
                      * (M - 4) * (M - 5), 1.0)

        corr6 = ((np.abs(Qn) ** 6
                  + 9.0 * np.abs(Q2n) ** 2 * np.abs(Qn) ** 2
                  - 6.0 * np.real(Q2n * Qn * np.conj(Qn) ** 3)
                  + 4.0 * np.real(Q3n * np.conj(Qn) ** 3)
                  - 12.0 * np.real(Q3n * np.conj(Q2n) * np.conj(Qn))
                  + 18.0 * (M - 4) * np.real(Q2n * np.conj(Qn) ** 2)
                  + 4.0 * np.abs(Q3n) ** 2) / P6
                 - 9.0 * (np.abs(Qn) ** 4 + np.abs(Q2n) ** 2)
                 / np.where(valid, M * (M - 1) * (M - 2) * (M - 3) * (M - 5),
                            1.0)
                 + 18.0 * np.abs(Qn) ** 2
                 / np.where(valid, M * (M - 1) * (M - 3) * (M - 4), 1.0)
                 - 6.0 / np.where(valid, (M - 1) * (M - 2) * (M - 3), 1.0))
        corr6 = np.where(valid, corr6, 0.0)

        # corrected <2> and <4> (Everett qnscript conventions, validated):
        if event_weights is not None and np.any(event_weights > 0):
            w = event_weights
            m2 = np.sum(w * (np.abs(Qn) ** 2 - dQ0)) \
                / max(np.sum(w * (M * M - dQ0)), 1e-60)
            c4_num = np.sum(w * (
                np.abs(Qn) ** 4 + np.abs(Q2n) ** 2
                - 2.0 * np.real(dQ2n * np.conj(Qn) ** 2)
                - 4.0 * dQ0 * np.abs(Qn) ** 2
                + 8.0 * np.real(Qn * np.conj(Qn))
                + 2.0 * dQ0 ** 2 - 6.0 * dQ0))
            c4_den = np.sum(w * (
                M ** 4 - 6.0 * dQ0 * M * M + 3.0 * dQ0 ** 2
                + 8.0 * M * dQ0 - 6.0 * dQ0))
            m6 = np.average(corr6, weights=w)
        else:
            m2 = np.nansum(np.abs(Qn) ** 2 - dQ0) \
                / max(np.nansum(M * M - dQ0), 1e-60)
            c4_num = np.nansum(
                np.abs(Qn) ** 4 + np.abs(Q2n) ** 2
                - 2.0 * np.real(dQ2n * np.conj(Qn) ** 2)
                - 4.0 * dQ0 * np.abs(Qn) ** 2
                + 8.0 * np.real(Qn * np.conj(Qn))
                + 2.0 * dQ0 ** 2 - 6.0 * dQ0)
            c4_den = np.nansum(
                M ** 4 - 6.0 * dQ0 * M * M + 3.0 * dQ0 ** 2
                + 8.0 * M * dQ0 - 6.0 * dQ0)
            m6 = np.nanmean(corr6)
        m4 = c4_num / max(c4_den, 1e-60)

        c6 = m6 - 9.0 * m2 * m4 + 12.0 * m2 ** 3
        # sign-preserving root: negative c6 (non-flow dominated) gives
        # negative v{6} -- the sign is physical, never NaN.
        out.append(np.sign(c6) * np.abs(c6 / 4.0) ** (1.0 / 6.0))
    return np.array(out)


def _core_vn8_single(Q1, event_weights=None, _pad=True):
    """Core v_n{8} single-acceptance, no jackknife.

    Not used by the pipeline since November 2026 (exact weighted distinct-
    particle sums, pipeline_dispatch._prepare_multi): it needs unit weights and
    an integer multiplicity, and averages <6> (<8>) over events unweighted
    while <2>, <4> are weighted by their tuple counts.

    c_n{8} = <8>_corr - 16<6><2> - 18<4>_corr^2 + 144<4>_corr<2>^2
             - 144<2>^4 ,   v_n{8} = (-c_n{8}/33)^(1/8).
    <8>_corr: thesis Eq. I.28 (verified vs enumeration at machine precision).
    <6>_corr: PRC 83, 044913 exact expression.
    <4>_corr: fully subtracted (Everett/qnscript convention).
    """
    if _pad and Q1.shape[1] < 9:
        Q1 = _zero_pad_harmonics(Q1, max(Q1.shape[1], 9))
    n_harm = Q1.shape[1]
    out = []
    for h in range(1, n_harm):
        if 4 * h >= n_harm:
            break
        Qn = Q1[:, h]
        Q2n = Q1[:, 2 * h]
        Q3n = Q1[:, 3 * h]
        Q4n = Q1[:, 4 * h]
        M = Q1[:, 0].real
        valid = (M >= 8) & (np.abs(M - np.round(M)) < 1e-6)
        P8 = np.where(valid, M*(M-1)*(M-2)*(M-3)*(M-4)*(M-5)*(M-6)*(M-7), 1.0)
        P6 = np.where(valid, M*(M-1)*(M-2)*(M-3)*(M-4)*(M-5), 1.0)

        corr6 = ((np.abs(Qn)**6 + 9*np.abs(Q2n)**2*np.abs(Qn)**2
                  - 6*np.real(Q2n*Qn*np.conj(Qn)**3)
                  + 4*np.real(Q3n*np.conj(Qn)**3)
                  - 12*np.real(Q3n*np.conj(Q2n)*np.conj(Qn))
                  + 18*(M-4)*np.real(Q2n*np.conj(Qn)**2)
                  + 4*abs(Q3n)**2) / P6
                 - 9*(abs(Qn)**4+abs(Q2n)**2)
                 / np.where(valid, M*(M-1)*(M-2)*(M-3)*(M-5), 1.0)
                 + 18*abs(Qn)**2
                 / np.where(valid, M*(M-1)*(M-3)*(M-4), 1.0)
                 - 6/np.where(valid, (M-1)*(M-2)*(M-3), 1.0))
        corr6 = np.where(valid, corr6, 0.0)

        corr8 = ((abs(Qn)**8 - 12*np.real(Q2n*Qn**2*np.conj(Qn)**4)
                  + 6*np.real(Q2n**2*np.conj(Qn)**4)
                  + 16*np.real(Q3n*Qn*np.conj(Qn)**4)
                  - 96*np.real(Q3n*Qn*np.conj(Q2n)*np.conj(Qn)**2)
                  - 12*np.real(Q4n*np.conj(Qn)**4)
                  - 36*np.real(Q2n**2*np.conj(Q2n)*np.conj(Qn)**2)
                  + 96*(M-6)*np.real(Q2n*Qn*np.conj(Qn)**3)
                  + 72*np.real(Q4n*np.conj(Q2n)*np.conj(Qn)**2)
                  + 48*np.real(Q3n*Qn*np.conj(Q2n)**2)
                  - 64*(M-6)*np.real(Q3n*np.conj(Qn)**3)
                  + 192*(M-6)*np.real(Q3n*np.conj(Q2n)*np.conj(Qn))
                  - 96*np.real(Q4n*np.conj(Q3n)*np.conj(Qn))
                  - 36*np.real(Q4n*np.conj(Q2n)**2)) / P8
                 + (-144*(M-7)*(M-4)*np.real(Q2n*np.conj(Qn)**2)
                    + 36*abs(Q4n)**2 + 64*abs(Q3n)**2*abs(Qn)**2
                    - 64*(M-6)*abs(Q3n)**2 + 9*abs(Q2n)**4
                    + 36*abs(Qn)**4*abs(Q2n)**2
                    - 144*(M-6)*abs(Q2n)**2*abs(Qn)**2
                    + 72*(M-7)*(M-4)*(abs(Q2n)**2+abs(Qn)**4)
                    - 16*(M-6)*abs(Qn)**6
                    - 96*(M-7)*(M-6)*(M-2)*abs(Qn)**2
                    + 24*M*(M-7)*(M-6)*(M-5)) / P8)
        corr8 = np.where(valid, corr8, 0.0)

        # <4> with FULL self-correlation subtraction (unit weights:
        # dQ0=tQ0=qQ0=M, dQ2n=Q2n, tQn=Qn -- Everett qnscript convention).
        n4_num = (abs(Qn)**4 + abs(Q2n)**2 - 2*np.real(Q2n*np.conj(Qn)**2)
                  - 4*M*abs(Qn)**2 + 8*abs(Qn)**2 + 2*M*M - 6*M)
        n4_den = M**4 - 6*M**3 + 3*M*M + 8*M*M - 6*M
        if event_weights is not None and np.any(event_weights > 0):
            w = event_weights
            m8 = np.average(corr8, weights=w)
            m6 = np.average(corr6, weights=w)
            m2 = np.sum(w*(abs(Qn)**2-M))/max(np.sum(w*(M*M-M)), 1e-60)
            m4 = np.sum(w*n4_num)/max(np.sum(w*n4_den), 1e-60)
        else:
            m8 = np.nanmean(corr8); m6 = np.nanmean(corr6)
            m2 = np.nansum(abs(Qn)**2-M)/max(np.nansum(M*M-M), 1e-60)
            m4 = np.nansum(n4_num)/max(np.nansum(n4_den), 1e-60)

        c8 = m8 - 16*m6*m2 - 18*m4*m4 + 144*m4*m2*m2 - 144*m2**4
        arg = -c8 / 33.0
        # sign-preserving root: arg<0 (non-flow dominated) gives negative
        # v{8} -- the sign is physical, never NaN.
        out.append(np.sign(arg) * np.abs(arg) ** (1.0 / 8.0))
    return np.array(out)


@register("vn6_single")
def compute_vn6_single_per_cent_bin(Q1, event_weights=None):
    """v_n{6} for one centrality bin (zero-pads missing harmonics)."""
    return _core_vn6_single(Q1, event_weights)


@register("vn4_single")
def compute_vn4_integrated(Qn_corrected, qv, jac, event_weights=None):
    Q1 = _compute_weighted_qvectors(qv, jac, 1)
    Q2 = _compute_weighted_qvectors(qv, jac, 2)
    Q3 = _compute_weighted_qvectors(qv, jac, 3)
    Q4 = _compute_weighted_qvectors(qv, jac, 4)
    return _core_vn4_single(Q1, Q2, Q3, Q4, event_weights)


def _core_vn4_gap(Q1_A, Q2_A, Q3_A, Q4_A, Q1_B, Q2_B, Q3_B, Q4_B, dQ_cross, event_weights):
    """Core v_n{4} AABB two-subevent, no jackknife.

    Args:
        All Qi_X: (n_events, n_harm) complex — weighted Q-vectors per subevent
        dQ_cross: (n_events,) float — cross-term δ_AB
        event_weights: (n_events,) — single centrality bin, or None

    Returns: (n_harm-1,) float — v₂{4}AABB through v_N{4}AABB.
    """
    n_harm = Q1_A.shape[1]
    result = np.zeros(n_harm)
    MA = Q1_A[:, 0].real
    MB = Q1_B[:, 0].real
    Q0_A = Q1_A[:, 0].real
    Q0_B = Q1_B[:, 0].real
    dQ0_A = Q2_A[:, 0].real
    dQ0_B = Q2_B[:, 0].real
    w = event_weights
    for h in range(1, n_harm):
        Qn_A = Q1_A[:, h]
        Qn_B = Q1_B[:, h]
        dQ2n_A = Q2_A[:, 2 * h] if 2 * h < n_harm else np.zeros(len(Q1_A), dtype=complex)
        dQ2n_B = Q2_B[:, 2 * h] if 2 * h < n_harm else np.zeros(len(Q1_B), dtype=complex)

        c2_num = np.real(Qn_A * np.conj(Qn_B)) - dQ_cross
        c2_den = MA * MB - dQ_cross
        c4_num = ((Qn_A ** 2 - dQ2n_A) * (np.conj(Qn_B) ** 2 - dQ2n_B))
        c4_den = (Q0_A ** 2 - dQ0_A) * (Q0_B ** 2 - dQ0_B)

        if w is not None and w.sum() > 1:
            c2 = np.average(c2_num, weights=w) / max(np.average(c2_den, weights=w), 1e-60)
            c4 = np.average(c4_num.real if np.iscomplexobj(c4_num) else c4_num, weights=w) / max(np.average(c4_den, weights=w), 1e-60)
        else:
            c2 = np.mean(c2_num) / max(np.mean(c2_den), 1e-60)
            c4 = np.mean(c4_num.real if np.iscomplexobj(c4_num) else c4_num) / max(np.mean(c4_den), 1e-60)

        v4 = 2 * c2 ** 2 - c4
        result[h] = np.sign(v4) * np.power(np.abs(v4), 0.25)
    return result[1:]


@register("vn4_gap")
def compute_vn4_subevent_per_cent(qv, jac_A, jac_B, event_weights, n_jk_blocks=20):
    Q1_A = _compute_weighted_qvectors(qv, jac_A, 1)
    Q2_A = _compute_weighted_qvectors(qv, jac_A, 2)
    Q3_A = _compute_weighted_qvectors(qv, jac_A, 3)
    Q4_A = _compute_weighted_qvectors(qv, jac_A, 4)
    Q1_B = _compute_weighted_qvectors(qv, jac_B, 1)
    Q2_B = _compute_weighted_qvectors(qv, jac_B, 2)
    Q3_B = _compute_weighted_qvectors(qv, jac_B, 3)
    Q4_B = _compute_weighted_qvectors(qv, jac_B, 4)
    dQ_cross = _compute_dq0_cross(qv, jac_A, jac_B)

    n_events = len(Q1_A)
    n_cent = event_weights.shape[1]
    n_harm = Q1_A.shape[1]

    blocks = jackknife_blocks(n_events, min(n_jk_blocks, n_events))
    n_b = len(blocks)

    theta_full = np.zeros((n_cent, n_harm - 1))
    for c in range(n_cent):
        w = event_weights[:, c]
        if w.sum() < 2:
            continue
        theta_full[c] = _core_vn4_gap(Q1_A, Q2_A, Q3_A, Q4_A, Q1_B, Q2_B, Q3_B, Q4_B, dQ_cross, w)

    theta_i = np.zeros((n_b, n_cent, n_harm - 1))
    for i, mask in enumerate(blocks):
        for c in range(n_cent):
            w = event_weights[mask, c]
            if w.sum() < 2:
                continue
            theta_i[i, c] = _core_vn4_gap(
                Q1_A[mask], Q2_A[mask], Q3_A[mask], Q4_A[mask],
                Q1_B[mask], Q2_B[mask], Q3_B[mask], Q4_B[mask],
                dQ_cross[mask], w)

    bias = (n_b - 1) * (np.mean(theta_i, axis=0) - theta_full)
    var = (n_b - 1) / n_b * np.sum((theta_i - np.mean(theta_i, axis=0)) ** 2, axis=0)
    err = np.sqrt(np.maximum(var, 0))
    return theta_full, err, bias


@register("vn2_single")
def compute_vn2_integrated(Qn_corrected, qv, jac, event_weights=None):
    Qn_int = sum_over_species_pt(Qn_corrected)
    dQ0 = _compute_dq0(qv, Qn_corrected, jac)
    return compute_vn2_single_per_cent_bin(Qn_int, dQ0, event_weights)


@register("vn2_single_per_cent")
def compute_vn2_integrated_per_cent(Qn_corrected, qv, jac, event_weights):
    n_cent = event_weights.shape[1]
    n_harm = Qn_corrected.shape[2]
    result = np.zeros((n_cent, n_harm), dtype=np.float64)

    for c in range(n_cent):
        w = event_weights[:, c]
        if w.sum() < 2:
            result[c] = 0.0
            continue
        result[c] = compute_vn2_integrated(Qn_corrected, qv, jac, event_weights=w)

    return result


def compute_vn2_integrated_per_cent_with_jackknife(Qn_corrected, qv, jac, event_weights, n_jk_blocks=20):
    n_events = Qn_corrected.shape[0]
    n_cent = event_weights.shape[1]
    n_harm = Qn_corrected.shape[2]

    blocks = jackknife_blocks(n_events, min(n_jk_blocks, n_events))
    n_b = len(blocks)

    theta_full = compute_vn2_integrated_per_cent(Qn_corrected, qv, jac, event_weights)
    theta_i = np.zeros((n_b, n_cent, n_harm), dtype=np.float64)

    for i, mask in enumerate(blocks):
        theta_i[i] = compute_vn2_integrated_per_cent(
            Qn_corrected[mask], qv, jac, event_weights[mask])
    bias = (n_b - 1) * (np.mean(theta_i, axis=0) - theta_full)
    var = (n_b - 1) / n_b * np.sum((theta_i - np.mean(theta_i, axis=0)) ** 2, axis=0)
    err = np.sqrt(np.maximum(var, 0))
    return theta_full, err, bias


# ── Yield / multiplicity ────────────────────────────────────────────────────

@register("yield_dNch_deta")
def compute_yield_dNch_deta(Q0, delta_eta, w):
    """dNch/dη for a single centrality bin, no jackknife."""
    if w.sum() < 2:
        return np.array([0.0])
    return np.array([np.average(Q0, weights=w) / delta_eta])


# ── Spectra ──────────────────────────────────────────────────────────────────

@register("spectra_invariant_yield")
def compute_spectra_invariant_yield(Q0_pt, pt_centers, pt_widths, delta_eta, w):
    """Invariant yield 1/(2π pT) · d²N/(dpT dη) for one cent bin, no jackknife."""
    n_pT = Q0_pt.shape[1]
    if w.sum() < 2:
        return np.zeros(n_pT)
    norm = 1.0 / (2 * np.pi * pt_centers * pt_widths * delta_eta)
    return np.average(Q0_pt, axis=0, weights=w) * norm


@register("yield_dNch_deta_per_eta")
def compute_yield_dNch_deta_per_eta(Q0_per_eta, delta_eta_per_bin, w):
    """dNch/dη(eta_i) = ⟨Q0(e, i)⟩ / Δη_i per eta bin.

    Q0_per_eta: (n_events, n_eta_bins) float — weighted Σ w_j per event per eta bin.
    delta_eta_per_bin: (n_eta_bins,) float — Δη_i per bin.
    w: (n_events,) float — per-event centrality weight.

    Returns: (n_eta_bins,) float — dNch/dη per eta bin.
    """
    if w.sum() < 2:
        return np.full(len(delta_eta_per_bin), np.nan)
    out = np.zeros(len(delta_eta_per_bin))
    for b in range(len(delta_eta_per_bin)):
        if delta_eta_per_bin[b] > 0:
            out[b] = (w * Q0_per_eta[:, b]).sum() / w.sum() / delta_eta_per_bin[b]
    return out


# ── Mean transverse momentum ────────────────────────────────────────────────

def _compute_weighted_moment(qv, jac, raw, pt_lo=None, pt_hi=None):
    """Apply per-particle Jacobian weights to a raw (event, species, pt) moment.

    raw: counts slot (Qn[...,0,:].real) for the denominator, or the first
    pT moment sum_pt for the numerator.  Weight = jac/(2·rap_max·ns),
    identical to the per-particle weight used everywhere else in 2D.
    """
    if pt_lo is None:
        pt_lo = 0
    if pt_hi is None:
        pt_hi = raw.shape[2]
    pt_hi = min(pt_hi, jac.shape[1])
    w = (jac[np.newaxis, :, pt_lo:pt_hi] /
         (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis]))
    return np.sum(raw[:, :, pt_lo:pt_hi] * w, axis=(1, 2))


def _cross_avg(dq_cross, n_poi_bins, w, per_event_normalised):
    """Weighted mean of the POI/reference self-correlation (diagonal).

    ``Re<Q_poi Q_ref*>`` sums cos n(phi_i - phi_j) over every pair with i in
    the POI and j in the reference.  Pairs with i == j -- the particles
    lying in BOTH windows -- each contribute exactly 1, so the diagonal is
    just the shared-particle count.  Every i != j pair is a genuine
    correlation and must be kept.

    The subtraction is therefore purely geometric: it is the count of
    particles in POI INTERSECT reference.  It is automatically 0 for
    disjoint windows, and equals N_poi when the POI is nested inside the
    reference.  No overlap TEST is involved anywhere -- which is what keeps
    this consistent with the 2D path, where boost invariance makes every
    eta window the same particle set, so the shared count is never zero
    there and must not be treated as if it were.

    For the per-event-normalised form each shared particle carries weight
    1/n_poi, so a fully nested POI contributes exactly 1.
    """
    if dq_cross is None:
        return None
    dq = np.asarray(dq_cross, dtype=float)
    if per_event_normalised:
        nb = np.asarray(n_poi_bins, dtype=float)
        # A 1-D dQ_cross means "same for every bin" -> give it the bin axis
        # so it divides the per-bin multiplicities instead of broadcasting
        # (n_events,) against (n_events, n_bins) into an (n_ev, n_ev, nb)
        # outer product.
        if dq.ndim == 1 and nb.ndim == 2:
            dq = dq[:, None]
        dq = dq / np.maximum(nb, 1e-9)
    return np.average(dq, axis=0, weights=w)


def _core_vn2_sp_diff(q_poi_bins, n_poi_bins, q_ref, w, dQ_cross=None,
                      n_ref=None):
    """Vectorized scalar-product differential: one column per POI bin.

    Standard convention (ATLAS/CMS):
        v_n(pT){SP} = < (Q_poi/n_poi) . Q_ref* > / sqrt(<|Q_ref|^2>)
    q_poi_bins: (n_events, n_harm, n_bins) complex
    n_poi_bins: (n_events, n_bins) float
    q_ref:      (n_events, n_harm) complex, or (n_events, n_harm, n_bins)
                when the reference depends on the differential bin (a
                mirrored sub-event reference, or one reference per
                |eta| segment).
    Returns (n_harm-1)*n_bins flattened, harmonic-major.
    """
    per_bin = q_ref.ndim == 3
    n_harm = q_ref.shape[1]
    n_bins = q_poi_bins.shape[2]
    out = np.zeros((n_harm - 1, n_bins))
    if w.sum() < 2:
        return out.ravel()
    # The reference's OWN auto-correlation must come out of |Q_ref|^2:
    # |Q_B|^2 = v^2 M_B^2 + M_B, so leaving M_B in inflates the denominator
    # by sqrt(1 + 1/(v^2 M_B)) and biases v_n low by that factor.  At
    # v2=0.06, M_B=334 that is 27.4 vs 20.4 -> a 25% deficit, and it grows
    # as multiplicity falls.  The JETSCAPE `_means` form already does this;
    # this form did not.  n_ref is optional so existing callers keep
    # working, but omitting it silently reintroduces the bias.
    ref_pow = np.average(np.abs(q_ref) ** 2, axis=0, weights=w)
    if n_ref is not None:
        nr = np.asarray(n_ref, dtype=float)
        if nr.ndim == 1:
            nr = nr[:, None]
        ref_pow = ref_pow - np.average(nr, axis=0, weights=w)
    # The reference amplitude is a signed root, sign(P) sqrt|P| of the
    # reference power P (the convention of every v_n{k}: a reference whose
    # pair correlation is negative flips the sign instead of giving NaN;
    # until November 2026 NaN).  An empty reference (P = 0) or a bin
    # without POI particles has no v_n: NaN, not num / 1e-30.
    denom_h = np.sign(ref_pow) * np.sqrt(np.abs(ref_pow))
    has_poi = np.average(n_poi_bins, axis=0, weights=w) > 0
    inv_n = 1.0 / np.maximum(n_poi_bins, 1e-9)
    cross = _cross_avg(dQ_cross, n_poi_bins, w, per_event_normalised=True)
    if cross is not None:
        cross = np.broadcast_to(np.asarray(cross, dtype=float).reshape(-1),
                                (n_bins,))
    for h in range(1, n_harm):
        qp = q_poi_bins[:, h, :] * inv_n
        qr = q_ref[:, h, :] if per_bin else q_ref[:, h][:, None]
        num = np.real(np.average(qp * np.conj(qr), axis=0, weights=w))
        if cross is not None:
            num = num - cross
        d = np.broadcast_to(denom_h[h], num.shape)
        out[h - 1, :] = np.where((d != 0) & has_poi, num / np.where(d != 0, d, 1.0), np.nan)
    return out.ravel()


@register("vn2_sp_diff_pT")
@register("vn2_sp_diff_eta")
def compute_vn2_sp_diff(Q_poi_bins, n_poi_bins, q_ref, w):
    """SP differential dispatcher: returns per-(harmonic,bin) array."""
    arr = _core_vn2_sp_diff(Q_poi_bins, n_poi_bins, q_ref, w)
    return arr


def _core_vn2_sp_diff_means(q_poi_bins, n_poi_bins, q_ref, n_ref, w,
                            dQ_cross=None, dQ0_ref=None):
    """Scalar-product differential, JETSCAPE pair-normalized form.

    Verbatim plain-mean version of JETSCAPE's calculate_vn_diff_SP
    (their leave-one-out jackknife averages to the same central value):
        v_n(bin) = <Re[Q_poi.Q_ref*]> / <N_poi.N_ref>
                   / sqrt(<(|Q_ref|^2-N_ref)> / <N_ref.(N_ref-1)>)
    Differs from _core_vn2_sp_diff (per-event N_poi normalization,
    ATLAS/CMS convention) when multiplicities fluctuate a lot within
    the bin (wide centrality, narrow eta/pT bins): up to ~20% on 3D
    tables.  q_ref / n_ref may be (n_events, n_harm) / (n_events,), or
    carry a trailing bin axis when the reference varies per differential
    bin (mirrored sub-event, or per-|eta|-segment reference).
    Returns (n_harm-1)*n_bins flattened, harmonic-major.
    Weighted particles (2D Jacobian weights): dQ0_ref = Sum w^2 replaces
    N_ref as the self-term, and the shared diagonal dQ_cross comes out of
    both <Re[Q_poi.Q_ref*]> and <N_poi.N_ref>.
    """
    per_bin = q_ref.ndim == 3
    n_harm = q_ref.shape[1]
    n_bins = q_poi_bins.shape[2]
    out = np.full((n_harm - 1, n_bins), np.nan)
    if w.sum() < 2:
        return out.ravel()
    self_ref = n_ref if dQ0_ref is None else np.asarray(dQ0_ref, dtype=float)
    # Broadcast the reference multiplicity against (n_events, n_harm,
    # n_bins) for C, and against (n_events, n_bins) for M2.
    if per_bin:
        n_ref_3d = self_ref[:, None, :]       # (n_ev, 1, n_bins)
        n_ref_2d = n_ref                      # (n_ev, n_bins)
    else:
        n_ref_3d = self_ref[:, None]          # (n_ev, 1, 1)
        n_ref_2d = n_ref[:, None]             # (n_ev, 1)
    n2ref = np.abs(q_ref) ** 2 - n_ref_3d
    N2ref = n_ref * n_ref - self_ref
    N2ref = N2ref[:, None, :] if per_bin else N2ref[:, None]
    wq = w[:, None, None] if per_bin else w[:, None]
    C = (np.sum(wq * n2ref, axis=0) /
         np.maximum(np.sum(wq * np.maximum(N2ref, 0.0), axis=0), 1e-30))
    for h in range(1, n_harm):
        qr = q_ref[:, h, :] if per_bin else q_ref[:, h][:, None]
        ch = np.broadcast_to(np.asarray(C[h], dtype=float).reshape(-1), (n_bins,))
        # the reference c2 enters as sign(c2) sqrt|c2| (until November 2026 a
        # negative c2 gave NaN); zero or undefined: NaN (out starts NaN)
        good = np.isfinite(ch) & (ch != 0)
        if not good.any():
            continue
        M1 = np.sum(w[:, None] * np.real(q_poi_bins[:, h, :] *
                                         np.conj(qr)), axis=0)
        if dQ_cross is not None:
            # shared-particle diagonal; see _cross_avg
            dq = np.asarray(dQ_cross, dtype=float)
            if dq.ndim == 1:
                dq = dq[:, None]
            M1 = M1 - np.sum(w[:, None] * dq, axis=0)
        pairs = n_poi_bins * n_ref_2d
        if dQ_cross is not None:
            pairs = pairs - dq
        M2 = np.sum(w[:, None] * pairs, axis=0)
        good_h = good & (M2 > 0)
        root = np.sign(ch) * np.sqrt(np.abs(np.where(good, ch, 1.0)))
        out[h - 1, :] = np.where(good_h, M1 / np.where(M2 > 0, M2, 1.0) / root, np.nan)
    return out.ravel()


@register("vn2_sp_diff_pT_means")
@register("vn2_sp_diff_eta_means")
def compute_vn2_sp_diff_means(Q_poi_bins, n_poi_bins, q_ref, n_ref, w):
    """SP differential dispatcher (JETSCAPE pair-normalized form)."""
    return _core_vn2_sp_diff_means(Q_poi_bins, n_poi_bins, q_ref, n_ref, w)


def _amplitude_vn(q_n, n_poi, w):
    """Single-particle amplitude estimator of |v_n| for one particle set.

        |v_n|^2 = ( <|Q_n|^2> - <M> ) / <M(M-1)>

    Used for records whose method is event_plane but which carry no
    separate flow reference.  It measures the amplitude of the n-th Fourier
    component of the SAME particle sample that forms the flow, so the
    self-correlation term is the whole sample's multiplicity M -- there is
    no second set, hence no intersection to compute.

    The <M> subtraction is essential.  For M uncorrelated particles,
    <|Q_n|^2> = v_n^2 M^2 + M, so omitting it returns sqrt(v_n^2 + 1/M),
    which inflates by +14% at M=1000/v=0.06 and +249% at M=100/v=0.03.
    Integrated and differential bins share this helper so the correction
    cannot drift apart between them.

    IMPORTANT, and easy to confuse: this is the AMPLITUDE estimator, not
    the resolution-corrected event-plane estimator v_n = <Q_n>/<Q2>^(n/2).
    It contains no event plane and therefore no resolution correction.  A
    record that declares a `flow_reference` (a real sub-event or detector
    reference) should be computed through the scalar-product path, which
    does implement the correlation; see `flow_reference` in
    observables/_common.yaml.  This estimator is the right fallback only
    for single-acceptance records with no declared reference.
    """
    m = np.maximum(np.asarray(n_poi, dtype=float), 0.0)
    m_mean = np.average(m, weights=w)
    m2_mean = np.average(m * (m - 1.0), weights=w)
    if not (np.isfinite(m2_mean) and m2_mean > 0):
        return np.nan
    q2_mean = np.average(np.abs(q_n) ** 2, weights=w)
    num = q2_mean - m_mean
    if not np.isfinite(num):
        return np.nan
    # signed root, as v_n{2} (until November 2026 a non-positive num: NaN)
    return float(np.sign(num) * np.sqrt(abs(num) / m2_mean))


def _core_vn2_ep_diff_eta(q_poi_bins, n_poi_bins, w):
    """Event-plane v_n per differential bin, harmonic-major.

        v_n(bin)^2 = ( <|Q_n|^2> - <M> ) / <M(M-1)>      per bin

    The SELF-CORRELATION subtraction of <M> is essential and was missing
    until Sep 2026.  For a bin of M uncorrelated particles with flow v_n,

        <|Q_n|^2> = v_n^2 M^2 + M ,

    so the naive sqrt(<|Q_n|^2>/<M^2>) returns sqrt(v_n^2 + 1/M).  The
    1/M term overtakes v_n^2 once M <~ 1/v_n^2 -- about 400 particles at
    v_n = 0.05 -- and is far worse for the narrow bins that differential
    measurements actually use.  Toy check (see
    tests/test_toy_mc_ep_diff.py), v_true = 0.05:

        M/bin   naive      corrected
          2000   0.0550     0.0502   (+10%  /  +0.4%)
           400   0.0706     0.0499   (+41%  /  -0.2%)
           100   0.1119     0.0505   (+124% /  +1%)
            30   0.1900     0.0534   (+280% /  +7%)

    Using M(M-1) in the denominator (not M^2) makes the estimator
    unbiased for the subtracted form.  Bins with too few surviving events
    are left NaN rather than returning a noise-dominated value.
    """
    n_bins = q_poi_bins.shape[2]
    n_harm = q_poi_bins.shape[1]
    out = np.full((n_harm - 1, n_bins), np.nan)
    if w.sum() < 2:
        return out.ravel()
    for h in range(1, n_harm):
        for b in range(n_bins):
            out[h - 1, b] = _amplitude_vn(q_poi_bins[:, h, b],
                                          n_poi_bins[:, b], w)
    return out.ravel()


@register("vn2_ep_diff_eta")
def compute_vn2_ep_diff_eta(Q_poi_bins, n_poi_bins, w):
    """Event-plane v2 dispatcher for differential_eta.

    Returns: (n_bins,) float — v2 per eta bin.
    """
    return _core_vn2_ep_diff_eta(Q_poi_bins, n_poi_bins, w)


def _sp_harmonic1(q_poi, n_poi, q_ref, n_ref, dq, w):
    """First-harmonic scalar product per column (bins on axis 1, or scalar):
    (<Re(q_poi/n_poi q_ref*)> - <dq/n_poi>) / sqrt(<|q_ref|^2> - <n_ref>).
    dq is the POI/reference diagonal (shared particles, with the reference's
    weight); n_ref the reference's own self-correlation."""
    inv = 1.0 / np.maximum(n_poi, 1e-9)
    num = (np.real(np.average(q_poi * inv * np.conj(q_ref), axis=0, weights=w))
           - np.average(dq * inv, axis=0, weights=w))
    den = np.average(np.abs(q_ref) ** 2, axis=0, weights=w) - np.average(n_ref, axis=0, weights=w)
    return np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1.0)), np.nan)


@register("v1_odd_sp_diff_eta")
def compute_v1_odd_sp_diff_eta(d, w):
    """Rapidity-odd directed flow v1(eta) per POI bin.

    Each bin: scalar product with its eta-odd reference (windows at eta < 0
    weighted -1), which measures v1 times the sign of the reference's own
    directed flow.  Every bin is aligned to one common reference -- the one
    governing the sign-convention window -- by the sign of the correlation
    between the two references, and the overall sign makes v1 averaged
    over the sign-convention window carry the declared sign.

    Returns (n_bins,) float (the harmonic-1 row).
    """
    nb = d['q_poi'].shape[1]
    if w.sum() < 2:
        return np.zeros(nb)
    v = _sp_harmonic1(d['q_poi'], d['n_poi'], d['q_ref'], d['n_ref'], d['dq'], w)
    corr = (np.real(np.average(d['q_ref'] * np.conj(d['q_c'])[:, None], axis=0, weights=w))
            - np.average(d['align_shared'], axis=0, weights=w))
    align = np.sign(corr)
    v_sw = _sp_harmonic1(d['q_sw'], d['n_sw'], d['q_c'], d['n_c'], d['dq_sw'], w)
    s = d['sign'] * (np.sign(v_sw) if v_sw != 0 else 1.0)
    return v * align * s


@register("vn2_ep_diff_pT")
def compute_vn2_ep_diff_pT(Q_poi_bins, n_poi_bins, w):
    """Event-plane v2 dispatcher for differential_pT.

    Same per-bin estimator as differential_eta (the core is agnostic to
    the bin variable); only the prepare step differs (pT vs eta slicing).
    Returns: (n_bins,) float — v2 per pT bin.
    """
    return _core_vn2_ep_diff_eta(Q_poi_bins, n_poi_bins, w)


@register("vn3_ep")
def compute_vn3_ep(Qn, w):
    """Amplitude estimate of |v3|, single-acceptance, integrated.

    Qn: (n_events, n_harm) complex.  See `_amplitude_vn` for the estimator
    and for why it is NOT the resolution-corrected event-plane v3.
    """
    if w.sum() < 2:
        return np.full(1, np.nan)
    return np.array([_amplitude_vn(Qn[:, 3], Qn[:, 0].real, w)])


@register("vn2_ep")
def compute_vn2_ep(Qn, w):
    """Amplitude estimate of |v2|, single-acceptance, integrated.

    Qn: (n_events, n_harm) complex.  See `_amplitude_vn` for the estimator
    (including the self-correlation subtraction) and for why it is not the
    resolution-corrected event-plane v2.
    """
    if w.sum() < 2:
        return np.full(1, np.nan)
    return np.array([_amplitude_vn(Qn[:, 2], Qn[:, 0].real, w)])


@register("vn3_ep")
def compute_vn3_ep(Qn, w):
    """Event-plane v3 single-acceptance, integrated.

    Qn: (n_events, n_harm) complex. v3 = <|Q_3|²/M²>^{1/2} per-event.
    """
    if w.sum() < 2:
        return np.zeros(1)
    M = Qn[:, 0].real
    q3 = Qn[:, 3]
    inv_M = 1.0 / np.maximum(M, 1e-9)
    v2 = np.average((np.abs(q3) ** 2) * (inv_M ** 2), weights=w)
    return np.sqrt(np.maximum(v2, 0.0)).reshape(1)


@register("vn2_sp")
def compute_vn2_sp(q_poi, q_ref, n_poi, n_ref, w, same_window=False,
                   dQ_cross=None, dQ0_poi=None, dQ0_ref=None):
    """Scalar-product v_n{SP} for one centrality class (JETSCAPE3D
    convention: calculate_vn_eta with density-normalized flow vectors,
    cf. arXiv:2601.17234 App. M: vn = <QA.QB*> / (<MA> sqrt(<QB.QB*>)).

    q_poi/q_ref: (n_events, n_harm) complex density-weighted flow vectors;
    n_poi/n_ref: (n_events,) POI/reference multiplicities.
    same_window: POI and reference coincide (their auto-correlation then
      sits in the numerator too and is subtracted, cf. App. M Fig. 36).

    vn = Re(<q_poi.conj(q_ref)> - [same_window]<n_poi>)
         / (sqrt(<|q_ref|^2> - <n_ref>) * <n_poi>)
    The subtractions are the standard self-correlation removals; on
    oversample-averaged 3D tables they are O(1/R) negligible, reproducing
    the JETSCAPE numbers, while on single-shot (toy/R=1) inputs they are
    exact.  NOTE the denominator uses the POI count <MA>, not <MB>.
    """
    if w.sum() < 2:
        return np.zeros(q_poi.shape[1] - 1)
    out = np.zeros(q_poi.shape[1] - 1)
    m_ref = np.average(n_ref, weights=w)
    m_poi = np.average(n_poi, weights=w)
    # Self-correlation terms: unit-weight containers use the multiplicity
    # itself (dQ0 = N); density-weighted 2D containers pass the actual
    # sum of squared weights.
    dq_poi = (np.average(np.asarray(dQ0_poi, dtype=float), weights=w)
              if dQ0_poi is not None else m_poi)
    dq_ref = (np.average(np.asarray(dQ0_ref, dtype=float), weights=w)
              if dQ0_ref is not None else m_ref)
    # The reference amplitude is the signed root sign(P) sqrt|P| (a negative
    # <|Q_ref|^2> - Sum w^2 flips the sign; until November 2026 NaN); an
    # empty reference (P = 0) or no POI: NaN.
    ref_pow = np.average(np.abs(q_ref) ** 2, axis=0, weights=w) - dq_ref
    denom_ref = np.sign(ref_pow) * np.sqrt(np.abs(ref_pow))
    for h in range(1, q_poi.shape[1]):
        num = np.real(np.average(q_poi[:, h] * np.conj(q_ref[:, h]),
                                 weights=w))
        if same_window:
            # POI and reference coincide: subtract their diagonal.
            num -= dq_poi
        if dQ_cross is not None:
            # General nested / partially-overlapping case.
            num -= np.average(np.asarray(dQ_cross, dtype=float), weights=w)
        out[h - 1] = num / (denom_ref[h] * m_poi) if ref_pow[h] != 0 and m_poi > 0 else np.nan
    return out


@register("pT_fluct")
def compute_pT_fluct(s1, s2, n, w, t0=None, t1=None):
    """√⟨⟨ΔpTi ΔpTj⟩⟩ / ⟨pT⟩ for one centrality class (ALICE, arXiv:1407.5530).

    Per-event sums over particles with acceptance weights a_i (the
    probability to be in the record's window: the 2D eta-window Jacobian and
    the share of a cut pT cell; 1 for unit weights):
      n = S0 = Σ a_i,  s1 = S1 = Σ a_i pT_i,  s2 = T2 = Σ a_i² pT_i²,
      t0 = T0 = Σ a_i²,  t1 = T1 = Σ a_i² pT_i  (default n, s1: unit weights).
    Distinct pairs: Σ_{i≠j} a_i a_j (pT_i - M)(pT_j - M) = (S1 - M S0)² -
    (T2 - 2 M T1 + M² T0) over Σ_{i≠j} a_i a_j = S0² - T0, summed over
    events with the class weights; M the class mean pT.  With a_i = 1 this
    is the unit-weight estimator.
    """
    if w.sum() < 2:
        return np.array([np.nan])
    t0 = n if t0 is None else t0
    t1 = s1 if t1 is None else t1
    m_bar = np.average(s1, weights=w) / max(np.average(n, weights=w), 1e-30)
    num = np.sum(w * ((s1 - m_bar * n) ** 2 - (s2 - 2.0 * m_bar * t1 + m_bar ** 2 * t0)))
    den = np.sum(w * np.maximum(n * n - t0, 0.0))
    if den <= 1e-30 or m_bar <= 1e-30:
        return np.array([np.nan])
    C = num / den
    return np.array([np.sqrt(max(C, 0.0)) / m_bar])


@register("dET_deta")
def compute_dET_deta(sum_et, delta_eta, w):
    """dET/dη for one centrality bin: ⟨Σ E_T⟩ / Δη."""
    if w.sum() < 2:
        return np.array([0.0])
    return np.array([np.average(sum_et, weights=w) / delta_eta])


@register("mean_pt")
def compute_mean_pt(num, den, w):
    """⟨pT⟩ for a single centrality bin, no jackknife.

    num: (n_events,) — weighted Σ w_i·pT_i;  den: (n_events,) — weighted Σ w_i.
    """
    if w.sum() < 2:
        return np.array([0.0])
    den_avg = np.average(den, weights=w)
    if den_avg <= 1e-30:
        return np.array([0.0])
    return np.array([np.average(num, weights=w) / den_avg])


def _compute_dmq_full(qv, jac):
    """dmq per sim pT bin for all events, fully vectorized.
    Returns (n_events, n_pt) — Σ species Q0_raw * (jac / (2*rap_max*nsample))²
    """
    w_sq = (jac[np.newaxis, :, :] /
            (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis])) ** 2
    return np.sum(qv.Qn[:, :, 0, :].real * w_sq, axis=1)


@register("vn2_single_mor")
def compute_vn2_single_mor(Qn, dQ0, event_weights):
    """v_n{2} single-acceptance, mean-of-per-event-ratios.

    Verbatim JETSCAPE `calcualte_vn_2` form (average_event_spvn_Qn2D.py):
      corr_e = (|Qn_e,h|^2 - dQ0_e) / (Q0_e * (Q0_e - 1));  v = sqrt(<corr>_w)
    Differs from compute_vn2_single_per_cent_bin (ratio of weighted means)
    at O(event-by-event fluctuation^2) — a few % on 3D-averaged tables.
    The caller must pass JETSCAPE-scale vectors (3D loader scale, where the
    formal self-subtraction is exact); see pipeline_dispatch.

    Qn: (n_events, n_harm) complex — per-event Q-vectors.
    dQ0: (n_events,) float — self-correlation subtraction.
    event_weights: (n_events,) float — centrality weights.

    Returns: (n_harm-1,) float — v_2{2} through v_N{2}.
    """
    n_harm = Qn.shape[1]
    result = np.zeros(n_harm - 1)
    Q0 = Qn[:, 0].real
    for h in range(1, n_harm):
        r = (np.abs(Qn[:, h]) ** 2 - dQ0) / np.maximum(Q0 * (Q0 - 1.0), 1e-60)
        result[h - 1] = np.sqrt(max(np.average(r, weights=event_weights), 0.0))
    return result


@register("vn2_single_per_cent_bin")
def compute_vn2_single_per_cent_bin(Qn, dQ0, event_weights):
    """v_n{2} single-acceptance, single centrality bin.

    Qn: (n_events, n_harm) complex — per-event Q-vectors.
    dQ0: (n_events,) float — self-correlation subtraction.
    event_weights: (n_events,) float — centrality weights.

    Returns: (n_harm-1,) float — harmonic-major v1..v_{N-1}.  The pipeline
    selects a record's harmonic with index ``harmonic-1``; v2 is index 1.
    """
    n_harm = Qn.shape[1]
    result = np.zeros(n_harm)
    Q0 = Qn[:, 0].real
    for h in range(1, n_harm):
        num = np.abs(Qn[:, h]) ** 2 - dQ0
        den = Q0 * Q0 - dQ0
        r = np.average(num, weights=event_weights) / max(np.average(den, weights=event_weights), 1e-60)
        result[h] = np.sign(r) * np.sqrt(np.abs(r))
    return result[1:]


# v_n{2}(pT) for the legacy analysis.Computer2D (layout h*n_pt + b with an unused
# h = 0 row).  The pipeline computes vn2_single_diff_pT with analysis.correlators.
def _core_vn2_diff_pT(Qn_int, Qn_pt_int, Q0, dQ0, dmq, w):
    """Core v_n{2} pT-differential, single centrality bin, no jackknife.

    dmq: (n_events, n_pt) — self-correction per measurement pT bin.
    Returns (n_harm * n_pt,) — flattened v_n{2}(pT) per harmonic × pT bin.
    Qn_int: (n_events, n_harm) — integrated over all pT.
    Qn_pt_int: (n_events, n_harm, n_pt) — per-measurement-pT-bin.
    """
    n_harm = Qn_int.shape[1]
    n_pt = Qn_pt_int.shape[2]
    out = np.zeros((n_harm, n_pt))
    w_sum = np.sum(w)
    if w_sum < 1e-30:
        return out.ravel()
    for h in range(1, n_harm):
        Qn_h = Qn_int[:, h]
        num = np.abs(Qn_h) ** 2 - dQ0
        den = Q0 * Q0 - dQ0
        num_mean = np.average(num, weights=w)
        den_mean = np.average(den, weights=w)
        r_int = num_mean / max(den_mean, 1e-30)
        vn2_int = np.sign(r_int) * abs(r_int) ** 0.5           # signed root
        pn_all = Qn_pt_int[:, h, :]
        mq_all = np.abs(Qn_pt_int[:, 0, :]).real
        dn_all = (pn_all.real * Qn_h.real[:, np.newaxis] +
                  pn_all.imag * Qn_h.imag[:, np.newaxis] - dmq)
        dn_den_all = mq_all * Q0[:, np.newaxis] - dmq
        dn_mean_all = np.sum(dn_all * w[:, np.newaxis], axis=0) / w_sum
        dn_den_mean_all = np.sum(dn_den_all * w[:, np.newaxis], axis=0) / w_sum
        dn_ratio_all = np.where(dn_den_mean_all > 1e-30,
                                dn_mean_all / dn_den_mean_all, 0.0)
        out[h, :] = dn_ratio_all / vn2_int if vn2_int != 0 else 0.0
    return out.ravel()


def compute_vn2_differential_pt_per_cent(Qn_corrected, qv, jac, event_weights, n_jk_blocks=20):
    n_events = Qn_corrected.shape[0]
    n_cent = event_weights.shape[1]
    n_harm = Qn_corrected.shape[2]
    n_pt = Qn_corrected.shape[3]

    Qn_int = sum_over_species_pt(Qn_corrected)
    Q0 = Qn_int[:, 0].real
    dQ0 = _compute_dq0(qv, Qn_corrected, jac)
    Qn_pt_int = Qn_corrected.sum(axis=1)

    dmq = _compute_dmq_full(qv, jac)

    blocks = jackknife_blocks(n_events, min(n_jk_blocks, n_events))
    n_b = len(blocks)

    theta_full = np.zeros((n_cent, n_harm, n_pt))
    for c in range(n_cent):
        w = event_weights[:, c]
        if w.sum() < 2:
            continue
        out = _core_vn2_diff_pT(Qn_int, Qn_pt_int, Q0, dQ0, dmq, w)
        theta_full[c] = out.reshape(n_harm, n_pt)

    theta_i = np.zeros((n_b, n_cent, n_harm, n_pt))
    for i, mask in enumerate(blocks):
        Qn_b = Qn_corrected[mask]
        w_b = event_weights[mask]
        Qn_int_b = sum_over_species_pt(Qn_b)
        Q0_b = Qn_int_b[:, 0].real
        dQ0_b = _compute_dq0(qv, Qn_b, jac)
        Qn_pt_b = Qn_b.sum(axis=1)
        qv_b = qv
        qv_b.nsamples = qv.nsamples[mask]
        qv_b.Qn = qv.Qn[mask]
        dmq_b = _compute_dmq_full(qv_b, jac)

        for c in range(n_cent):
            w = w_b[:, c]
            if w.sum() < 2:
                continue
            out = _core_vn2_diff_pT(Qn_int_b, Qn_pt_b, Q0_b, dQ0_b, dmq_b, w)
            theta_i[i, c] = out.reshape(n_harm, n_pt)

    bias = (n_b - 1) * (np.mean(theta_i, axis=0) - theta_full)
    var = (n_b - 1) / n_b * np.sum((theta_i - np.mean(theta_i, axis=0)) ** 2, axis=0)
    err = np.sqrt(np.maximum(var, 0))
    return theta_full, err, bias

@register("vn2_gap")
def compute_vn2_subevent_per_cent(Qn_A, Qn_B, dQ_cross, event_weights):
    """v_n{2}ab two-subevent, single centrality bin.

    Qn_A: (n_events, n_harm) complex — subevent A Q-vectors.
    Qn_B: (n_events, n_harm) complex — subevent B Q-vectors.
    dQ_cross: (n_events,) float — per-event cross-term δ_AB.
    event_weights: (n_events,) float — centrality weights.

    Returns: (n_harm-1,) float — v₂{2}ab through v_N{2}ab.
    """
    n_harm = Qn_A.shape[1]
    result = np.zeros(n_harm)
    MA = Qn_A[:, 0].real
    MB = Qn_B[:, 0].real
    den_avg = np.average(MA * MB - dQ_cross, weights=event_weights)
    if not den_avg > 0:
        # no pairs (an empty subevent; until November 2026: 0)
        return np.full(n_harm - 1, np.nan)
    for h in range(1, n_harm):
        num = np.real(Qn_A[:, h] * np.conj(Qn_B[:, h])) - dQ_cross
        r = np.average(num, weights=event_weights) / den_avg
        result[h] = np.sign(r) * np.sqrt(np.abs(r))
    return result[1:]



@register("vn2_gap_poi")
def compute_vn2_gap_poi(p_A, p_B, Q_A, Q_B, d_pQ, d_QQ, event_weights):
    """v_n{2} of a particle of interest (POI), two subevents, one centrality bin.

    The POI in each subevent is correlated with the reference in the other,
    both orderings summed, and divided by the reference v_n{2}ab:

        d_n = <Re(p_A Q_B*) + Re(p_B Q_A*) - 2 d_pQ> / <m_A M_B + m_B M_A - 2 d_pQ>
        c_n = <Re(Q_A Q_B*) - d_QQ> / <M_A M_B - d_QQ>
        v_n = d_n / sqrt(c_n)

    p_X, Q_X: (n_events, n_harm) complex POI / reference Q-vectors in
      subevent X (slot 0 the counts);
    d_pQ: (n_events,) POI-reference self-pair term, Sum w_A w_B over particles
      in both samples (zero for disjoint subevents on a 3D grid; on a
      boost-invariant 2D container every particle sits in both windows);
    d_QQ: (n_events,) the same for the two reference samples.

    Returns (n_harm-1,) harmonic-major v_1..v_{N-1} (v_2 at index 1), like
    vn2_gap.  A negative c_n gives sign(c_n) sqrt|c_n| in the denominator.
    No POI-reference pairs (a pT bin beyond the container's range) or no
    reference pairs give NaN (until November 2026: 0).
    """
    n_harm = Q_A.shape[1]
    out = np.full(n_harm - 1, np.nan)
    if np.sum(event_weights) <= 0:
        return out
    mA, mB = p_A[:, 0].real, p_B[:, 0].real
    MA, MB = Q_A[:, 0].real, Q_B[:, 0].real
    den_d = np.average(mA * MB + mB * MA - 2.0 * d_pQ, weights=event_weights)
    den_c = np.average(MA * MB - d_QQ, weights=event_weights)
    if not den_d > 0 or not den_c > 0:
        return out
    for h in range(1, n_harm):
        num_d = np.average(np.real(p_A[:, h] * np.conj(Q_B[:, h]))
                           + np.real(p_B[:, h] * np.conj(Q_A[:, h])) - 2.0 * d_pQ,
                           weights=event_weights)
        num_c = np.average(np.real(Q_A[:, h] * np.conj(Q_B[:, h])) - d_QQ,
                           weights=event_weights)
        d = num_d / den_d
        c = num_c / den_c
        root = np.sign(c) * np.sqrt(abs(c))
        out[h - 1] = d / root if root != 0 else np.nan
    return out


def compute_vn2_subevent_per_cent_batch(Qn_A, Qn_B, dQ_cross, event_weights, n_jk_blocks=20):
    """v_n{2}ab two-subevent, batch jackknife over all centrality bins.

    Args:
        Qn_A, Qn_B: (n_events, n_harm) complex
        dQ_cross: (n_events,) float
        event_weights: (n_events, n_cent) float
        n_jk_blocks: int

    Returns: (theta_full, err, bias) each shape (n_cent, n_harm-1)
    """
    n_events = len(Qn_A)
    n_cent = event_weights.shape[1]
    n_harm = Qn_A.shape[1]

    blocks = jackknife_blocks(n_events, min(n_jk_blocks, n_events))
    n_b = len(blocks)

    theta_full = np.zeros((n_cent, n_harm - 1))
    for c in range(n_cent):
        w = event_weights[:, c]
        if w.sum() < 2:
            continue
        theta_full[c] = compute_vn2_subevent_per_cent(Qn_A, Qn_B, dQ_cross, w)

    theta_i = np.zeros((n_b, n_cent, n_harm - 1))
    for i, mask in enumerate(blocks):
        for c in range(n_cent):
            w = event_weights[mask, c]
            if w.sum() < 2:
                continue
            theta_i[i, c] = compute_vn2_subevent_per_cent(
                Qn_A[mask], Qn_B[mask], dQ_cross[mask], w)

    bias = (n_b - 1) * (np.mean(theta_i, axis=0) - theta_full)
    var = (n_b - 1) / n_b * np.sum((theta_i - np.mean(theta_i, axis=0)) ** 2, axis=0)
    err = np.sqrt(np.maximum(var, 0))
    return theta_full, err, bias


def _core_vn2_diff_eta_one_bin(Qn_corrected, qv, jac, eta_lo, eta_hi, w,
                               pt_lo=None, pt_hi=None):
    """v_n{2} single η bin, single centrality bin, no jackknife.

    Re-cuts the 2D container on the η sub-window (new Jacobian weights)
    and restricts to the record's fixed pT acceptance so that dQ0 is
    computed with exactly matching weights.  CF-free (consistent).
    Returns (n_harm-1,) — v₂{2} through v_N{2} in this η bin.
    """
    if pt_lo is None:
        pt_lo = 0
    if pt_hi is None:
        pt_hi = jac.shape[1]
    Qn_sub, jac_sub = cut_qvectors(qv, eta_lo, eta_hi)
    Qn_int = sum_over_species_pt(Qn_sub[:, :, :, pt_lo:pt_hi])
    Q0 = Qn_int[:, 0].real
    dQ0 = _compute_dq0(qv, Qn_sub, jac_sub, pt_lo=pt_lo, pt_hi=pt_hi)
    n_harm = Qn_int.shape[1]
    result = np.zeros(n_harm)
    for h in range(1, n_harm):
        num = np.abs(Qn_int[:, h]) ** 2 - dQ0
        den = Q0 * Q0 - dQ0
        r = np.average(num, weights=w) / max(np.average(den, weights=w), 1e-60)
        result[h] = np.sign(r) * np.sqrt(np.abs(r))
    return result[1:]


def _core_vn2_gap_diff_pT_all_bins(Qn_A_int_per_bin, Qn_B_int_per_bin, dQ_cross_per_bin, w):
    """Vectorized v_n{2}ab pT-differential, single centrality bin, no jackknife.

    All per-pT-bin pre-computation is done in prepare_vn2_gap_diff_pT_2d
    so this function only needs weighted averages across events.

    Args:
        Qn_A_int_per_bin: (n_events, n_harm, n_pT) complex — subevent A per pT bin
        Qn_B_int_per_bin: (n_events, n_harm, n_pT) complex — subevent B per pT bin
        dQ_cross_per_bin: (n_events, n_pT) float — cross-term self-correction per pT bin
        w: (n_events,) float — centrality weights
    Returns ((n_harm-1) * n_pT,) flattened — v₂{2}ab through v_N{2}ab per pT bin.
    """
    n_pT = Qn_A_int_per_bin.shape[2]
    n_harm = Qn_A_int_per_bin.shape[1]
    MA_all = np.abs(Qn_A_int_per_bin[:, 0, :]).real
    MB_all = np.abs(Qn_B_int_per_bin[:, 0, :]).real
    w_sum = np.sum(w)
    if w_sum < 1e-30:
        return np.zeros((n_harm - 1) * n_pT)
    out = np.zeros((n_harm - 1, n_pT))
    for h in range(1, n_harm):
        num_all = (Qn_A_int_per_bin[:, h, :].real * Qn_B_int_per_bin[:, h, :].real +
                   Qn_A_int_per_bin[:, h, :].imag * Qn_B_int_per_bin[:, h, :].imag
                   ) - dQ_cross_per_bin
        den_all = MA_all * MB_all - dQ_cross_per_bin
        num_mean = np.sum(num_all * w[:, np.newaxis], axis=0) / w_sum
        den_mean = np.sum(den_all * w[:, np.newaxis], axis=0) / w_sum
        r = np.where(den_mean > 1e-60, num_mean / den_mean, 0.0)
        out[h - 1, :] = np.sign(r) * np.sqrt(np.abs(r))
    return out.ravel()


def _core_vn2_gap_diff_eta_all_bins(Qn_A_int, Qn_B_int_per_bin, dQ_cross_per_bin, w):
    """Vectorized v_n{2}ab η-differential, single centrality bin, no jackknife.

    All per-η-bin pre-computation is done in prepare_vn2_gap_diff_eta_2d
    so this function only needs weighted averages across events.

    Args:
        Qn_A_int: (n_events, n_harm) complex — subevent A, integrated over pT
        Qn_B_int_per_bin: (n_events, n_harm, n_eta) complex — subevent B per η bin
        dQ_cross_per_bin: (n_eta,) float — cross-term self-correction per η bin
        w: (n_events,) float — centrality weights
    Returns ((n_harm-1) * n_eta,) flattened — v₂{2}ab through v_N{2}ab per η bin.
    """
    n_eta = Qn_B_int_per_bin.shape[2]
    n_harm = Qn_A_int.shape[1]
    MA = np.abs(Qn_A_int[:, 0]).real
    MB_all = np.abs(Qn_B_int_per_bin[:, 0, :]).real
    w_sum = np.sum(w)
    if w_sum < 1e-30:
        return np.zeros((n_harm - 1) * n_eta)
    out = np.zeros((n_harm - 1, n_eta))
    for h in range(1, n_harm):
        Qn_A_h = Qn_A_int[:, h]
        num_all = (Qn_A_h.real[:, np.newaxis] * Qn_B_int_per_bin[:, h, :].real +
                   Qn_A_h.imag[:, np.newaxis] * Qn_B_int_per_bin[:, h, :].imag
                   ) - dQ_cross_per_bin
        den_all = MA[:, np.newaxis] * MB_all - dQ_cross_per_bin
        num_mean = np.sum(num_all * w[:, np.newaxis], axis=0) / w_sum
        den_mean = np.sum(den_all * w[:, np.newaxis], axis=0) / w_sum
        r = np.where(den_mean > 1e-60, num_mean / den_mean, 0.0)
        out[h - 1, :] = np.sign(r) * np.sqrt(np.abs(r))
    return out.ravel()


@register("vn2_gap_diff_pT")
def compute_vn2_gap_diff_pT(Qn_corrected_A, Qn_corrected_B, qv, jac_A, jac_B,
                             event_weights, pT_bins, eta_range_A, eta_range_B,
                             n_jk_blocks=20):
    n_events = Qn_corrected_A.shape[0]
    n_cent = event_weights.shape[1]
    n_harm = Qn_corrected_A.shape[2]
    n_pT = len(pT_bins)

    Qn_A_int_per_bin = np.zeros((n_events, n_harm, n_pT), dtype=np.complex128)
    Qn_B_int_per_bin = np.zeros((n_events, n_harm, n_pT), dtype=np.complex128)
    dQ_cross_per_bin = np.zeros((n_events, n_pT))
    for b, (lo, hi) in enumerate(pT_bins):
        lo_b = int(np.searchsorted(qv.pt_centers, lo))
        hi_b = int(np.searchsorted(qv.pt_centers, hi, side='right'))
        Qn_A_int_per_bin[:, :, b] = sum_over_species_pt(Qn_corrected_A[:, :, :, lo_b:hi_b])
        Qn_B_int_per_bin[:, :, b] = sum_over_species_pt(Qn_corrected_B[:, :, :, lo_b:hi_b])
        dQ_cross_per_bin[:, b] = _compute_dq0_cross(qv, jac_A, jac_B, lo_b, min(hi_b, jac_A.shape[1]))

    blocks = jackknife_blocks(n_events, min(n_jk_blocks, n_events))
    n_b = len(blocks)

    theta_full = np.zeros((n_cent, n_harm - 1, n_pT))
    for c in range(n_cent):
        w = event_weights[:, c]
        if w.sum() < 2:
            continue
        out = _core_vn2_gap_diff_pT_all_bins(
            Qn_A_int_per_bin, Qn_B_int_per_bin, dQ_cross_per_bin, w)
        theta_full[c] = out.reshape(n_harm - 1, n_pT)

    theta_i = np.zeros((n_b, n_cent, n_harm - 1, n_pT))
    for i, mask in enumerate(blocks):
        Qn_A_int_b = Qn_A_int_per_bin[mask]
        Qn_B_int_b = Qn_B_int_per_bin[mask]
        dQ_cross_b = dQ_cross_per_bin[mask]
        w_b = event_weights[mask]
        for c in range(n_cent):
            w = w_b[:, c]
            if w.sum() < 2:
                continue
            out = _core_vn2_gap_diff_pT_all_bins(
                Qn_A_int_b, Qn_B_int_b, dQ_cross_b, w)
            theta_i[i, c] = out.reshape(n_harm - 1, n_pT)

    bias = (n_b - 1) * (np.mean(theta_i, axis=0) - theta_full)
    var = (n_b - 1) / n_b * np.sum((theta_i - np.mean(theta_i, axis=0)) ** 2, axis=0)
    err = np.sqrt(np.maximum(var, 0))
    return theta_full, err, bias


@register("vn2_gap_diff_eta")
def compute_vn2_gap_diff_eta(Qn_corrected_A, Qn_corrected_B, qv, jac_A, jac_B,
                              event_weights, eta_bins, pT_range_A, pT_range_B,
                              n_jk_blocks=20):
    n_events = Qn_corrected_A.shape[0]
    n_cent = event_weights.shape[1]
    n_harm = Qn_corrected_A.shape[2]
    n_eta = len(eta_bins)

    Qn_A_int = sum_over_species_pt(Qn_corrected_A)
    pt_lo = int(np.searchsorted(qv.pt_centers, pT_range_A[0]))
    pt_hi = int(np.searchsorted(qv.pt_centers, pT_range_A[1], side='right'))

    # Pre-compute per-η-bin Qn_B_int and dQ_cross
    Qn_B_int_per_bin = np.zeros((n_events, n_harm, n_eta), dtype=np.complex128)
    dQ_cross_per_bin = np.zeros((n_events, n_eta))
    for b, (lo, hi) in enumerate(eta_bins):
        Qn_B_cut = cut_qvectors(qv, lo, hi)[0]
        Qn_B_int_per_bin[:, :, b] = sum_over_species_pt(Qn_B_cut)
        jac_B_sub = _make_jacobian_list(qv.pt_centers, qv.masses, lo, hi)
        dQ_cross_per_bin[:, b] = _compute_dq0_cross(qv, jac_A, jac_B_sub,
                                                     pt_lo, min(pt_hi, jac_A.shape[1]))

    blocks = jackknife_blocks(n_events, min(n_jk_blocks, n_events))
    n_b = len(blocks)

    theta_full = np.zeros((n_cent, n_harm - 1, n_eta))
    for c in range(n_cent):
        w = event_weights[:, c]
        if w.sum() < 2:
            continue
        out = _core_vn2_gap_diff_eta_all_bins(
            Qn_A_int, Qn_B_int_per_bin, dQ_cross_per_bin, w)
        theta_full[c] = out.reshape(n_harm - 1, n_eta)

    bias = (n_b - 1) * (np.mean(theta_i, axis=0) - theta_full)
    var = (n_b - 1) / n_b * np.sum((theta_i - np.mean(theta_i, axis=0)) ** 2, axis=0)
    err = np.sqrt(np.maximum(var, 0))
    return theta_full, err, bias


# ── Symmetric cumulants SC(n,m) ─────────────────────────────────────────────


# ── Event-plane correlations NAC(n,m) (subevent) ─────────────────────────────


def _core_nac_mn(Q1_A, Q1_B, event_weights, n, m):
    """Core NAC(n,m) subevent estimator, one centrality bin, no JK.

    NAC(n,m) = <cos(n Psi_n^A - m Psi_m^B)>
            = Re(<Q_n^A · conj(Q_m^B)>) / <M_A · M_B>

    The two subevents must be disjoint (no shared particles) for the
    direct formula to be unbiased; the 2D prepare uses the per-subevent
    kinematic ranges to build independent Q-vectors via cut_qvectors,
    and the cross-term dQ_cross is zero for disjoint windows.

    Args:
        Q1_A, Q1_B: (n_events, n_harm) complex — weighted Q^1 vectors
            for subevents A and B; slot 0 = M (real), slots >=1 = Q_n.
        event_weights: (n_events,) float — per-event centrality weight
        n, m: positive int — the two harmonics (may differ, may be equal)

    Returns: scalar float — NAC(n,m).
    """
    M_A = Q1_A[:, 0].real
    M_B = Q1_B[:, 0].real
    Qn_A = Q1_A[:, n]
    Qm_B = Q1_B[:, m]
    num = (Qn_A * np.conj(Qm_B)).real
    den = M_A * M_B
    # Guard against zero-multiplicity events in either subevent.
    valid = (M_A > 0) & (M_B > 0)
    if event_weights is not None:
        w = event_weights * valid
        wsum = w.sum()
        if wsum < 1e-30:
            return np.nan
        return np.sum(w * num) / np.sum(w * den)
    num_v = num[valid]
    den_v = den[valid]
    if len(num_v) == 0:
        return np.nan
    return num_v.mean() / den_v.mean()


@register("nac_mn")
def compute_nac_mn(Q1_A, Q1_B, event_weights, n, m):
    """NAC(n,m) subevent for one centrality bin (length-1 array)."""
    return np.array([_core_nac_mn(Q1_A, Q1_B, event_weights, n, m)])

