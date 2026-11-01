"""Pipeline dispatch: maps registered keys to core functions + data prep.

Separate module to avoid circular imports between dispatcher.py
(which observables_2d.py imports) and observables_2d.py (which
contains core functions).
"""

import numpy as np

from analysis.observables_2d import (
    compute_pT_fluct,
    _core_vn8_single,
    _core_vn6_single,
    compute_dET_deta,
    compute_vn2_sp,
    compute_vn2_ep,
    compute_vn2_ep_diff_eta,
    compute_vn2_ep_diff_pT,
    compute_vn3_ep,
    _core_vn2_sp_diff,
    compute_vn2_sp_diff_means,
    compute_vn2_single_per_cent_bin,
    compute_vn2_single_mor,
    compute_vn2_subevent_per_cent,
    _core_vn4_single,
    _core_vn4_gap,
    _core_vn2_diff_pT,
    _core_vn2_diff_eta_all_bins,
    _core_vn2_gap_diff_pT_all_bins,
    _core_vn2_gap_diff_eta_all_bins,
    _core_vn4_one_diff_bin,
    cut_qvectors,
    cut_qvectors_rapidity,
    sum_over_species_pt,
    _compute_dq0,
    _compute_dq0_cross,
    _compute_weighted_qvectors,
    _make_jacobian_list,
    compute_yield_dNch_deta,
    compute_yield_dNch_deta_per_eta,
    compute_v1_odd_sp_diff_eta,
    compute_spectra_invariant_yield,
    compute_mean_pt,
    compute_sc_mn,
    compute_nac_mn,
)

from analysis.qvector import QGridData, _snap_searchsorted

PTCUTS = np.array([
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45,
    1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
    2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
    3., 3.2, 3.4, 3.6, 3.8, 4., 10.
], dtype=float)


# ── Species selection ──────────────────────────────────────────────
# Sim species indices: 0=pi+, 1=K+, 2=p, 3=Sigma, 4=Xi
# The sim stores only positively charged hadrons.  For experimental
# records that include both charges (e.g. "pi+ + pi-"), apply the
# charge factor below (charge-symmetry approximation at midrapidity).
# The mapping now lives in analysis.species.resolve_species so it is
# shared identically with the 3D (PDG-based) selection.
from analysis.species import (
    resolve_species, resolve_species_on_axis, charged_species_indices,
    is_charged,
)  # noqa: E402,F401


def _get_species_selection(particle_id, qv=None):
    """Map particle_id dict to sim species indices and charge factor.

    When the container carries a PDG-labelled species axis (``pdg_ids``),
    exact charges are resolved (e.g. pi- is a separate index from pi+);
    otherwise the legacy positives-only charge-symmetry convention is used.

    Args:
        particle_id: dict like {'species': 'pi+ + pi-'} or None
        qv: optional QVectorData (or similar) carrying ``pdg_ids``

    Returns:
        (species_indices, charge_factor, found) where:
          species_indices: list of sim indices or None (meaning all species)
          charge_factor: float multiplier for charge symmetry
          found: True if species was found in mapping, False if fallback
    """
    if particle_id is None:
        return None, 1.0, False
    species_str = particle_id.get('species')
    if species_str is None:
        return None, 1.0, False
    pdg_ids = getattr(qv, 'pdg_ids', None) if qv is not None else None
    sim_indices, cf, found = resolve_species_on_axis(species_str, pdg_ids)
    if not found:
        import warnings
        warnings.warn(f"Species '{species_str}' not in sim data — returning zero yield")
        return [], 0.0, False
    return sim_indices, cf, True


def _find_pt_slice(pt_min, pt_max):
    lo = int(np.searchsorted(PTCUTS, pt_min, side='left'))
    hi = int(np.searchsorted(PTCUTS, min(pt_max, PTCUTS[-1]), side='right'))
    return lo, hi


def _compute_dq0_species(qv, jac, species_idx, charge_factor, pt_lo, pt_hi):
    """Σ w² for a selected species subset (container-aware)."""
    w = (jac[np.newaxis, :, :]
         / (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis]))
    q0 = qv.Qn[:, :, 0, :].real
    if species_idx is not None:
        q0 = q0[:, species_idx, :]
        w = w[:, species_idx, :]
    return ((q0[:, :, pt_lo:pt_hi] * w[:, :, pt_lo:pt_hi] ** 2)
            .sum(axis=(1, 2)) * charge_factor ** 2)


def _compute_dq0_cross_species(qv, jac_a, jac_b, species_idx,
                               charge_factor, pt_lo, pt_hi):
    """Σ w_A w_B for a selected species subset (container-aware)."""
    wa = (jac_a[np.newaxis, :, :]
          / (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis]))
    wb = (jac_b[np.newaxis, :, :]
          / (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis]))
    q0 = qv.Qn[:, :, 0, :].real
    if species_idx is not None:
        q0 = q0[:, species_idx, :]
        wa = wa[:, species_idx, :]
        wb = wb[:, species_idx, :]
    return ((q0[:, :, pt_lo:pt_hi]
             * wa[:, :, pt_lo:pt_hi] * wb[:, :, pt_lo:pt_hi])
            .sum(axis=(1, 2)) * charge_factor ** 2)


def _charged_qn_and_dq0(qv, eta_lo, eta_hi, pt_lo, pt_hi):
    """Charged-hadron Q-vector and Σw² in an (η, pT) window."""
    Qn_cut, jac = cut_qvectors(qv, eta_lo, eta_hi)
    idx, cf = charged_species_indices(getattr(qv, 'pdg_ids', None))
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, pt_lo, pt_hi)
    Qn_pt = Qn_cut[:, idx, :, lo:hi].sum(axis=3)
    Qn = cf * Qn_pt.sum(axis=1)
    dQ0 = _compute_dq0_species(qv, jac, idx, cf, lo, hi)
    return Qn, dQ0, Qn_cut, jac, lo, hi


def _find_pt_slice_edges(edges, pt_min, pt_max):
    """Container-aware pT slicing (works for any sim pT grid).

    Equivalent to _find_pt_slice when edges == PTCUTS.  Both bounds get
    float-noise tolerance (cf. _snap_searchsorted in analysis.qvector):
    a query within 1e-9 above a grid edge snaps down (lo), within 1e-9
    below snaps up (hi) — otherwise repr noise (e.g. 2.9999999999999996
    for 3.0) silently drops a whole pT bin.
    """
    lo = int(_snap_searchsorted(np.asarray(edges), pt_min, pt_min)[0])
    qhi = min(pt_max, edges[-1])
    hi = int(np.searchsorted(edges, qhi, side='right'))
    if hi < len(edges) and 0.0 <= edges[hi] - qhi <= 1e-9:
        hi += 1
    return max(lo, 0), min(hi, len(edges) - 1)


def _grid_slices(grid, eta, pt):
    """Eta/pT slice pair for 3D grid prepares with float-noise tolerance.

    Same snapping as QGridData.acceptance (see analysis.qvector): record
    queries within 1e-9 of a grid edge snap to it, so repr noise never
    includes/excludes a whole cell.
    """
    ei = _snap_searchsorted(grid.eta_edges, eta[0], eta[1])
    pi_lo, pi_hi = _find_pt_slice_edges(grid.pt_edges, pt[0], pt[1])
    return slice(ei[0], ei[1]), slice(pi_lo, pi_hi)


def _extract_subevent_eta(params):
    """Return (eta_min_A, eta_max_A, eta_min_B, eta_max_B) from params."""
    sr = params.get('subevent_ranges', {})
    if len(sr) >= 2:
        labels = sorted(sr.keys())
        eA = sr[labels[0]]['eta']
        eB = sr[labels[1]]['eta']
    else:
        sk = params.get('single_range', {})
        eA = sk.get('eta', (-1, 1))
        eB = eA
    return eA[0], eA[1], eB[0], eB[1]


# ── prepare_data functions ────────────────────────────────────────────

def prepare_vn2_single_2d(qv, params):
    """Returns {Qn, dQ0} for v_n{2} single-acceptance.

    Uses the container's species axis: the legacy positives-only Qns
    container sums π/K/p and applies charge symmetry ×2; the 13-species
    isobar container sums the exact charged set with no charge factor.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    Qn, dQ0, _, _, _, _ = _charged_qn_and_dq0(
        qv, eta[0], eta[1], pt[0], pt[1])
    return {'Qn': Qn, 'dQ0': dQ0}


def prepare_vn2_gap_2d(qv, params):
    """Returns {Qn_A, Qn_B, dQ_cross} for v_n{2}ab."""
    sk = params.get('single_range', {})
    pt = sk.get('pT', (0.1, 10))
    eA0, eA1, eB0, eB1 = _extract_subevent_eta(params)
    idx, cf = charged_species_indices(getattr(qv, 'pdg_ids', None))
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, pt[0], pt[1])

    Qn_cut_A, jac_A = cut_qvectors(qv, eA0, eA1)
    Qn_cut_B, jac_B = cut_qvectors(qv, eB0, eB1)

    Qn_A = cf * Qn_cut_A[:, idx, :, lo:hi].sum(axis=(1, 3))
    Qn_B = cf * Qn_cut_B[:, idx, :, lo:hi].sum(axis=(1, 3))

    dQ0_A = _compute_dq0_species(qv, jac_A, idx, cf, lo, hi)
    if np.allclose(jac_A, jac_B):
        dQ_cross = dQ0_A.copy()
    else:
        dQ_cross = _compute_dq0_cross_species(
            qv, jac_A, jac_B, idx, cf, lo, hi)

    return {'Qn_A': Qn_A, 'Qn_B': Qn_B, 'dQ_cross': dQ_cross}


def prepare_vn4_single_2d(qv, params):
    """Returns {Q1, Q2, Q3, Q4} for v_n{4} single-acceptance."""
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    jac = _make_jacobian_list(qv.pt_centers, qv.masses, eta[0], eta[1])
    return {
        'Q1': _compute_weighted_qvectors(qv, jac, 1),
        'Q2': _compute_weighted_qvectors(qv, jac, 2),
        'Q3': _compute_weighted_qvectors(qv, jac, 3),
        'Q4': _compute_weighted_qvectors(qv, jac, 4),
    }


def prepare_vn4_gap_2d(qv, params):
    """Returns {Q1_A..Q4_A, Q1_B..Q4_B, dQ_cross} for v_n{4}AABB."""
    eA0, eA1, eB0, eB1 = _extract_subevent_eta(params)
    jac_A = _make_jacobian_list(qv.pt_centers, qv.masses, eA0, eA1)
    jac_B = _make_jacobian_list(qv.pt_centers, qv.masses, eB0, eB1)
    return {
        'Q1_A': _compute_weighted_qvectors(qv, jac_A, 1),
        'Q2_A': _compute_weighted_qvectors(qv, jac_A, 2),
        'Q3_A': _compute_weighted_qvectors(qv, jac_A, 3),
        'Q4_A': _compute_weighted_qvectors(qv, jac_A, 4),
        'Q1_B': _compute_weighted_qvectors(qv, jac_B, 1),
        'Q2_B': _compute_weighted_qvectors(qv, jac_B, 2),
        'Q3_B': _compute_weighted_qvectors(qv, jac_B, 3),
        'Q4_B': _compute_weighted_qvectors(qv, jac_B, 4),
        'dQ_cross': _compute_dq0_cross(qv, jac_A, jac_B),
    }


def _extract_2d_pT_bins(params):
    """Return list of (pt_lo, pt_hi) from differential_bins in params."""
    bins = params.get('differential_bins', [])
    if bins:
        return bins
    pT = params.get('single_range', {}).get('pT', {})
    if isinstance(pT, dict):
        edges = pT.get('bins', [])
        if len(edges) > 1:
            return [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
        pT_min = pT.get('min')
        pT_max = pT.get('max')
        if pT_min is not None and pT_max is not None:
            return [(pT_min, pT_max)]
    elif isinstance(pT, (list, tuple)) and len(pT) == 2:
        return [(pT[0], pT[1])]
    return []


def prepare_vn2_diff_pT_2d(qv, params):
    """Data prep for v_n{2} pT-differential.

    Maps measurement pT bins to sim pT indices, accumulating Q-vectors
    and self-correction (dmq) per measurement bin.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    Qn_full, jac = cut_qvectors(qv, eta[0], eta[1])
    Qn_int = sum_over_species_pt(Qn_full)
    Q0 = Qn_int[:, 0].real
    dQ0 = _compute_dq0(qv, Qn_full, jac)

    meas_bins = _extract_2d_pT_bins(params)
    if not meas_bins:
        # Fallback: no measurement bins specified, use all sim bins
        Qn_pt_int = Qn_full.sum(axis=1)
        pt_centers = qv.pt_centers
        meas_bins = [(pt_centers[0], pt_centers[-1])]

    pt_centers = qv.pt_centers
    n_meas = len(meas_bins)
    Qn_pt_sim = Qn_full.sum(axis=1)  # (n_events, n_harm, n_pt_sim)
    n_harm = Qn_pt_sim.shape[1]
    Qn_pt_meas = np.zeros((Qn_pt_sim.shape[0], n_harm, n_meas), dtype=np.complex128)

    # Pre-compute dmq per sim pT bin (fully vectorized)
    w_sq = (jac[np.newaxis, :, :] /
            (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis])) ** 2
    dmq_sim = np.sum(qv.Qn[:, :, 0, :].real * w_sq, axis=1)
    dmq_meas = np.zeros((dmq_sim.shape[0], n_meas))

    for b, (lo, hi) in enumerate(meas_bins):
        lo_b = int(np.searchsorted(pt_centers, lo))
        hi_b = int(np.searchsorted(pt_centers, hi, side='right'))
        Qn_pt_meas[:, :, b] = Qn_pt_sim[:, :, lo_b:hi_b].sum(axis=2)
        dmq_meas[:, b] = dmq_sim[:, lo_b:hi_b].sum(axis=1)

    return {
        'Qn_int': Qn_int,
        'Qn_pt_int': Qn_pt_meas,
        'Q0': Q0,
        'dQ0': dQ0,
        'dmq': dmq_meas,
    }


def prepare_vn2_diff_eta_2d(qv, params):
    """Data prep for v_n{2} η-differential."""
    sk = params.get('single_range', {})
    pt = sk.get('pT', (0.1, 10))
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, pt[0], pt[1])
    eta = sk.get('eta', (-1, 1))
    Qn_full, jac = cut_qvectors(qv, eta[0], eta[1])
    Qn_full = Qn_full[:, :, :, lo:hi]
    return {'Qn_corrected': Qn_full, 'qv': qv, 'jac': jac,
            'params': params}


def prepare_vn2_gap_diff_pT_2d(qv, params):
    """Data prep for v_n{2}ab pT-differential.

    Pre-computes Qn_A_int, Qn_B_int, and dQ_cross per pT bin so the core
    function only needs weighted averages (no per-bin cuts or jacobians).
    """
    eA0, eA1, eB0, eB1 = _extract_subevent_eta(params)
    Qn_A_full, jac_A = cut_qvectors(qv, eA0, eA1)
    Qn_B_full, jac_B = cut_qvectors(qv, eB0, eB1)
    pt_bins = _extract_2d_pT_bins(params)
    n_pT = len(pt_bins)
    if n_pT == 0:
        return {'Qn_A_int_per_bin': None, 'Qn_B_int_per_bin': None,
                'dQ_cross_per_bin': None, 'params': params}

    n_events = qv.n_events
    n_harm = Qn_A_full.shape[2]
    Qn_A_int_per_bin = np.zeros((n_events, n_harm, n_pT), dtype=np.complex128)
    Qn_B_int_per_bin = np.zeros((n_events, n_harm, n_pT), dtype=np.complex128)
    dQ_cross_per_bin = np.zeros((n_events, n_pT))
    for b, (lo, hi) in enumerate(pt_bins):
        lo_b = int(np.searchsorted(qv.pt_centers, lo))
        hi_b = int(np.searchsorted(qv.pt_centers, hi, side='right'))
        Qn_A_int_per_bin[:, :, b] = sum_over_species_pt(Qn_A_full[:, :, :, lo_b:hi_b])
        Qn_B_int_per_bin[:, :, b] = sum_over_species_pt(Qn_B_full[:, :, :, lo_b:hi_b])
        dQ_cross_per_bin[:, b] = _compute_dq0_cross(qv, jac_A, jac_B, lo_b, min(hi_b, jac_A.shape[1]))
    return {
        'Qn_A_int_per_bin': Qn_A_int_per_bin,
        'Qn_B_int_per_bin': Qn_B_int_per_bin,
        'dQ_cross_per_bin': dQ_cross_per_bin,
    }


def prepare_vn2_gap_diff_eta_2d(qv, params):
    """Data prep for v_n{2}ab η-differential.

    Pre-computes Qn_B_int and dQ_cross per η bin so the core function
    only needs weighted averages (no per-bin cuts or jacobians).
    """
    eA0, eA1, eB0, eB1 = _extract_subevent_eta(params)
    Qn_A_full, jac_A = cut_qvectors(qv, eA0, eA1)
    Qn_A_int = sum_over_species_pt(Qn_A_full)
    pT_range = params.get('single_range', {}).get('pT', (0.1, 10.0))
    pt_lo = int(np.searchsorted(qv.pt_centers, pT_range[0], side='left'))
    pt_hi = int(np.searchsorted(qv.pt_centers, min(pT_range[1], PTCUTS[-1]), side='right'))

    eta_edges = params.get('differential_bins', [])
    eta_bins = [(lo, hi) for lo, hi in eta_edges]
    n_eta = len(eta_bins)
    if n_eta == 0:
        return {'Qn_A_int': Qn_A_int, 'Qn_B_int_per_bin': None,
                'dQ_cross_per_bin': None, 'params': params}

    n_harm = Qn_A_int.shape[1]
    n_events = qv.n_events
    Qn_B_int_per_bin = np.zeros((n_events, n_harm, n_eta), dtype=np.complex128)
    dQ_cross_per_bin = np.zeros((n_events, n_eta))
    for b, (lo, hi) in enumerate(eta_bins):
        Qn_B_cut = cut_qvectors(qv, lo, hi)[0]
        Qn_B_int_per_bin[:, :, b] = sum_over_species_pt(Qn_B_cut)
        jac_B_sub = _make_jacobian_list(qv.pt_centers, qv.masses, lo, hi)
        dQ_cross_per_bin[:, b] = _compute_dq0_cross(qv, jac_A, jac_B_sub,
                                                     pt_lo, min(pt_hi, jac_A.shape[1]))
    return {
        'Qn_A_int': Qn_A_int,
        'Qn_B_int_per_bin': Qn_B_int_per_bin,
        'dQ_cross_per_bin': dQ_cross_per_bin,
        'params': params,
    }


def prepare_vn4_diff_2d(qv, params):
    """Data prep for v_n{4} pT/η differential (single acceptance)."""
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    jac = _make_jacobian_list(qv.pt_centers, qv.masses, eta[0], eta[1])
    Q1_ref = _compute_weighted_qvectors(qv, jac, 1)
    Q2_ref = _compute_weighted_qvectors(qv, jac, 2)
    Q3_ref = _compute_weighted_qvectors(qv, jac, 3)
    return {
        'Q1_ref': Q1_ref,
        'Q2_ref': Q2_ref,
        'Q3_ref': Q3_ref,
        'qv': qv,
        'jac': jac,
        'params': params,
    }


# ── prepare_data_3d functions (grid-based) ───────────────────────────

def prepare_vn2_single_3d(grid, params):
    """Returns {Qn, dQ0} from a QGridData for v_n{2} single-acceptance.

    For unit-weighted particles dQ0 = Q0 (Σ 1² = Σ 1 = N).  Charge-symmetry
    scaling (grid.charge_factor, applied by MultiSpeciesGrid.select) must be
    squared on the self-correlation term, so dQ0 = Q0 * charge_factor — this
    mirrors the 2D prepares' inline CF**2 on dQ0 and keeps v_n{2} invariant.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    Qn = grid.acceptance(eta[0], eta[1], pt[0], pt[1])
    cf = getattr(grid, 'charge_factor', 1.0)
    dQ0 = Qn[:, 0].real.copy() * cf  # Σ 1² = Q0; CF**2 via Qn (×CF) × dQ0 (×CF)
    return {'Qn': Qn, 'dQ0': dQ0}


def prepare_vn2_gap_3d(grid, params):
    """Returns {Qn_A, Qn_B, dQ_cross} from a QGridData for v_n{2}ab.

    Disjoint η ranges → no shared particles → dQ_cross = 0.
    """
    sr = params.get('subevent_ranges', {})
    if len(sr) < 2:
        eA = params.get('single_range', {}).get('eta', (-1, 1))
        eB = eA
    else:
        labels = sorted(sr.keys())
        eA = sr[labels[0]]['eta']
        eB = sr[labels[1]]['eta']
    pt = params.get('single_range', {}).get('pT', (0.1, 10))
    pt0, pt1 = pt[0], pt[1]
    Qn_A = grid.acceptance(eA[0], eA[1], pt0, pt1)
    Qn_B = grid.acceptance(eB[0], eB[1], pt0, pt1)
    dQ_cross = np.zeros(grid.n_events, dtype=np.float64)
    return {'Qn_A': Qn_A, 'Qn_B': Qn_B, 'dQ_cross': dQ_cross}


def prepare_vn4_single_3d(grid, params):
    """Returns {Q1, Q2, Q3, Q4} for v_n{4} single-acceptance.

    3D particles are unit-weighted → Q1 = Q2 = Q3 = Q4.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    Qn = grid.acceptance(eta[0], eta[1], pt[0], pt[1])
    return {
        'Q1': Qn,
        'Q2': Qn.copy(),
        'Q3': Qn.copy(),
        'Q4': Qn.copy(),
    }


def prepare_vn4_gap_3d(grid, params):
    """Returns {Q1_A..Q4_A, Q1_B..Q4_B, dQ_cross} for v_n{4}AABB.

    Unit-weighted 3D particles → Q1 = Q2 = Q3 = Q4 per subevent.
    Disjoint η ranges → dQ_cross = 0.
    """
    sr = params.get('subevent_ranges', {})
    labels = sorted(sr.keys())
    eA = sr[labels[0]]['eta']
    eB = sr[labels[1]]['eta']
    pt = params.get('single_range', {}).get('pT', (0.1, 10))
    pt0, pt1 = pt[0], pt[1]
    Qn_A = grid.acceptance(eA[0], eA[1], pt0, pt1)
    Qn_B = grid.acceptance(eB[0], eB[1], pt0, pt1)
    dQ_cross = np.zeros(grid.n_events, dtype=np.float64)
    return {
        'Q1_A': Qn_A, 'Q2_A': Qn_A.copy(), 'Q3_A': Qn_A.copy(), 'Q4_A': Qn_A.copy(),
        'Q1_B': Qn_B, 'Q2_B': Qn_B.copy(), 'Q3_B': Qn_B.copy(), 'Q4_B': Qn_B.copy(),
        'dQ_cross': dQ_cross,
    }


# ── 3D differential aliases (per-bin params → delegate to integrated 3D) ──

def _prepare_vn2_diff_3d(grid, params):
    """For vn2_single_diff_pT/eta: per-bin params already narrow single_range."""
    return prepare_vn2_single_3d(grid, params)

def _prepare_vn2_gap_diff_3d(grid, params):
    """For vn2_gap_diff_pT/eta: per-bin params narrow subevent's range."""
    return prepare_vn2_gap_3d(grid, params)

def _prepare_vn4_diff_3d(grid, params):
    """For vn4_single_diff_pT/eta: per-bin params narrow single_range."""
    return prepare_vn4_single_3d(grid, params)


# ── core_fn wrappers (uniform interface: data_dict, w) ────────────────

def _wrap_vn2_single(data, w):
    return compute_vn2_single_per_cent_bin(data['Qn'], data['dQ0'], w)


def _wrap_vn2_single_mor(data, w):
    return compute_vn2_single_mor(data['Qn'], data['dQ0'], w)


def _wrap_vn2_gap(data, w):
    return compute_vn2_subevent_per_cent(data['Qn_A'], data['Qn_B'], data['dQ_cross'], w)


def _wrap_vn4_single(data, w):
    if 'Qn' in data:
        raise ValueError("vn4 requires higher-power Q-vectors (Q1..Q4); "
                         "not available from 3D data")
    return _core_vn4_single(data['Q1'], data['Q2'], data['Q3'], data['Q4'], w)


def _wrap_vn4_gap(data, w):
    if 'Qn_A' in data and 'Qn' not in data:
        raise ValueError("vn4 gap requires higher-power Q-vectors (Q1_A..Q4_B); "
                         "not available from 3D data")
    return _core_vn4_gap(
        data['Q1_A'], data['Q2_A'], data['Q3_A'], data['Q4_A'],
        data['Q1_B'], data['Q2_B'], data['Q3_B'], data['Q4_B'],
        data['dQ_cross'], w,
    )


def _wrap_vn2_diff_pT(data, w):
    return _core_vn2_diff_pT(
        data['Qn_int'], data['Qn_pt_int'], data['Q0'], data['dQ0'],
        data['dmq'], w)


def _wrap_vn2_diff_eta(data, w):
    # η bins come from params on the data dict
    params = data.get('params', {})
    sk = params.get('single_range', {})
    eta_cfg = sk.get('eta', {})
    # single_range['eta'] is normally a (lo, hi) tuple; a dict with 'bins'
    # appears when differential η edges were stashed there.
    eta_edges = eta_cfg.get('bins', []) if isinstance(eta_cfg, dict) else []
    if not eta_edges:
        eta_edges = params.get('differential_bins', [])
        eta_bins = [(lo, hi) for lo, hi in eta_edges]
    else:
        eta_bins = [(eta_edges[i], eta_edges[i+1]) for i in range(len(eta_edges)-1)]
    # Fixed pT acceptance for the differential measurement
    pt = params.get('single_range', {}).get('pT', (0.1, 10.0))
    pt_lo, pt_hi = _find_pt_slice_edges(data['qv'].pt_bin_edges, pt[0], pt[1])
    return _core_vn2_diff_eta_all_bins(
        data['Qn_corrected'], data['qv'], data['jac'], eta_bins, w,
        pt_lo=pt_lo, pt_hi=pt_hi,
    )


def _wrap_vn2_gap_diff_pT(data, w):
    if data.get('Qn_A_int_per_bin') is None:
        return np.array([])
    return _core_vn2_gap_diff_pT_all_bins(
        data['Qn_A_int_per_bin'], data['Qn_B_int_per_bin'],
        data['dQ_cross_per_bin'], w)


def _wrap_vn2_gap_diff_eta(data, w):
    return _core_vn2_gap_diff_eta_all_bins(
        data['Qn_A_int'], data['Qn_B_int_per_bin'],
        data['dQ_cross_per_bin'], w)


def _wrap_vn4_diff(data, w):
    params = data['params']
    jac = data['jac']
    qv = data['qv']
    pT_centers = qv.pt_centers
    masses = qv.masses
    diff_var = params.get('differential_var', 'pT')
    diff_bins = params.get('differential_bins', [])
    Q1_ref = data['Q1_ref']
    Q2_ref = data['Q2_ref']
    Q3_ref = data['Q3_ref']
    pT_range = params.get('single_range', {}).get('pT', (0.1, 10.0))

    n_bins = len(diff_bins)
    n_harm = Q1_ref.shape[1]
    out = np.zeros((n_harm - 1, n_bins))
    n_events = Q1_ref.shape[0]

    for b, (lo, hi) in enumerate(diff_bins):
        if diff_var == 'pT':
            lo_bin = int(np.searchsorted(pT_centers, lo))
            hi_bin = int(np.searchsorted(pT_centers, hi, side='right'))
            p1 = np.zeros((n_events, n_harm), dtype=complex)
            p2 = np.zeros((n_events, n_harm), dtype=complex)
            for p in range(lo_bin, min(hi_bin, len(pT_centers))):
                w_p = jac[:, p] / (2.0 * qv.rap_max * qv.nsamples[:, None])
                w_factor = w_p[:, :, np.newaxis]
                qn_p = qv.Qn[:, :, :, p].astype(np.complex128)
                p1 += (qn_p * w_factor).sum(axis=1)
                p2 += (qn_p * w_factor ** 2).sum(axis=1)
        else:
            lo_idx = int(np.searchsorted(pT_centers, pT_range[0], side='left'))
            hi_idx = int(np.searchsorted(pT_centers, min(pT_range[1], PTCUTS[-1]), side='right'))
            Qn_cut, _ = cut_qvectors(qv, lo, hi)
            Qn_int = Qn_cut[:, :, :, lo_idx:hi_idx].sum(axis=3)
            p1 = 2.0 * Qn_int[:, :3, :].sum(axis=1)
            p2 = 2.0 * Qn_int[:, :3, :].sum(axis=1)

        out[:, b] = _core_vn4_one_diff_bin(p1[:, :n_harm], p2[:, :n_harm],
                                            Q1_ref, Q2_ref, Q3_ref, w)
    return out.ravel()


# ── Yield / multiplicity prepare + wrappers ─────────────────────────────────

def prepare_yield_2d(qv, params):
    """Extract Q0 in the eta/pT acceptance for dNch/deta.

    dy-mode: when the record specifies kinematic_range.y and no eta, return
    raw (un-Jacobian-weighted) counts normalized by the rapidity width —
    the natural dN/dy of a flat-y container.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    species_idx, charge_factor, species_found = _get_species_selection(
        params.get('particle_id'), qv)
    if not species_found:
        z = np.zeros(qv.n_events)
        return {'Q0': z, 'delta_eta': 1.0}
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, *sk.get('pT', (0.1, 10)))
    if eta[0] is None or eta[1] is None:
        # dy mode: no Jacobian; normalize by total rapidity width
        counts_raw = qv.Qn[:, :, 0, :].real
        if species_idx is not None:
            counts_raw = counts_raw[:, species_idx, :]
        Q0 = counts_raw[:, :, lo:hi].sum(axis=(1, 2)) \
            / qv.nsamples * charge_factor
        return {'Q0': Q0, 'delta_eta': 2.0 * qv.rap_max}
    pt = sk.get('pT', (0.1, 10))
    Qn, _ = cut_qvectors(qv, eta[0], eta[1])
    if species_idx is not None:
        Qn = Qn[:, species_idx, :, :]
    Q0 = sum_over_species_pt(Qn[:, :, :, lo:hi])[:, 0].real * charge_factor
    delta_eta = abs(eta[1] - eta[0])
    return {'Q0': Q0, 'delta_eta': delta_eta}


def prepare_yield_3d(grid, params):
    """Extract Q0 from grid for dNch/dη, normalized by nsamples.

    The charge-symmetry factor is already baked into ``grid`` by
    ``MultiSpeciesGrid.select`` (grid.charge_factor), so no extra factor is
    applied here.  A raw combined grid carries charge_factor=1.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    if eta[0] is None or eta[1] is None:
        return {'Q0': np.zeros(grid.n_events), 'delta_eta': 1.0}
    pt = sk.get('pT', (0.1, 10))
    Qn = grid.acceptance(eta[0], eta[1], pt[0], pt[1])
    Q0 = Qn[:, 0].real / grid.nsamples
    delta_eta = abs(eta[1] - eta[0])
    return {'Q0': Q0, 'delta_eta': delta_eta}


def prepare_yield_3d_per_eta(grid, params):
    """Per-eta-bin Q0 from a 3D grid for dNch/deta(eta).

    Reads ``params['differential_bins']`` (the eta bin edges from the
    curated record's ``kinematic_range.eta.bins``).  For each eta bin
    [lo, hi], extracts Q0 from the grid (integrated over the record's
    pT range).  Returns a per-event per-eta-bin Q0 array plus the bin
    widths so the wrapper can divide by delta_eta per bin.

    The charge-symmetry factor is already baked into ``grid`` by
    ``MultiSpeciesGrid.select`` (grid.charge_factor), so no extra
    factor is applied here.
    """
    n_events = grid.n_events
    z = np.zeros(n_events)
    sk = params.get('single_range', {})
    bins = params.get('differential_bins') or []
    n_eta_bins = max(len(bins), 1)
    if not bins:
        return {'Q0_per_eta': np.zeros((n_events, n_eta_bins)),
                'delta_eta_per_bin': np.zeros(n_eta_bins)}
    pt = sk.get('pT', (0.1, 10))
    pT_lo, pT_hi = pt[0], min(pt[1], 1e9) if pt[1] else 1e9
    Q0_per_eta = np.zeros((n_events, len(bins)))
    delta_eta_per_bin = np.zeros(len(bins))
    for b, (e_lo, e_hi) in enumerate(bins):
        Q = grid.acceptance(e_lo, e_hi, pT_lo, pT_hi)
        Q0_per_eta[:, b] = Q[:, 0].real / grid.nsamples
        delta_eta_per_bin[b] = abs(e_hi - e_lo)
    return {'Q0_per_eta': Q0_per_eta, 'delta_eta_per_bin': delta_eta_per_bin}


def _wrap_yield_per_eta(data, w):
    """Per-bin dNch/deta(eta_i) = <Q0_per_eta[e, i]> / delta_eta[i].

    Returns a length-n_eta_bins array (the per-bin value, cent-averaged
    over the events in the cent bin).
    """
    return compute_yield_dNch_deta_per_eta(
        data['Q0_per_eta'], data['delta_eta_per_bin'], w)


def _wrap_yield(data, w):
    return compute_yield_dNch_deta(data['Q0'], data['delta_eta'], w)


# ── Spectra prepare + wrappers ──────────────────────────────────────────────

def prepare_spectra_2d(qv, params):
    """Extract Q0 per measurement pT bin for invariant yield.

    dy-mode: when the record specifies ``kinematic_range.y`` and no eta,
    use the flat-y container directly.  The raw counts are normalised by
    the full plateau width ``2*rap_max`` (matching dN/dy), so the
    invariant yield is obtained with ``delta_eta = 2*rap_max``.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    species_idx, charge_factor, species_found = _get_species_selection(
        params.get('particle_id'), qv)
    if not species_found:
        n_meas = len(_extract_2d_pT_bins(params))
        return {'Q0_pt': np.zeros((qv.n_events, n_meas)) if n_meas else None,
                'pt_centers': np.zeros(n_meas) if n_meas else None,
                'pt_widths': np.zeros(n_meas) if n_meas else None,
                'delta_eta': 1.0}
    meas_bins = _extract_2d_pT_bins(params)
    n_meas = len(meas_bins)
    if n_meas == 0:
        return {'Q0_pt': None, 'pt_centers': None, 'pt_widths': None,
                'delta_eta': 1.0}

    pt_edges = qv.pt_bin_edges
    Q0_pt = np.zeros((qv.n_events, n_meas))
    pt_centers = np.zeros(n_meas)
    pt_widths = np.zeros(n_meas)

    y = sk.get('y') or (None, None)
    if sk.get('y_mass') is not None and y[0] is not None and y[1] is not None:
        # Rapidity computed with one assumed mass (e.g. y_pi): per species
        # and pT, the window's true-rapidity extent; yield per unit of it.
        Qn, _ = cut_qvectors_rapidity(qv, y[0], y[1], float(sk['y_mass']))
        if species_idx is not None:
            Qn = Qn[:, species_idx, :, :]
        counts = Qn[:, :, 0, :].real.sum(axis=1) * charge_factor
        for b, (lo, hi) in enumerate(meas_bins):
            Q0_pt[:, b] = _partial_counts(counts, pt_edges, lo, hi)
            pt_centers[b] = (lo + hi) / 2
            pt_widths[b] = hi - lo
        return {'Q0_pt': Q0_pt, 'pt_centers': pt_centers,
                'pt_widths': pt_widths, 'delta_eta': y[1] - y[0]}

    if eta[0] is None or eta[1] is None:
        # dy mode: no Jacobian; counts are per unit rapidity over the
        # full plateau.  Divide by nsamples, then let the wrapper divide
        # by the plateau width.
        counts_raw = qv.Qn[:, :, 0, :].real
        if species_idx is not None:
            counts_raw = counts_raw[:, species_idx, :]
        counts = (counts_raw.sum(axis=1) / qv.nsamples[:, None]
                  * charge_factor)
        for b, (lo, hi) in enumerate(meas_bins):
            Q0_pt[:, b] = _partial_counts(counts, pt_edges, lo, hi)
            pt_centers[b] = (lo + hi) / 2
            pt_widths[b] = hi - lo
        return {'Q0_pt': Q0_pt, 'pt_centers': pt_centers,
                'pt_widths': pt_widths, 'delta_eta': 2.0 * qv.rap_max}

    Qn, _ = cut_qvectors(qv, eta[0], eta[1])
    if species_idx is not None:
        Qn = Qn[:, species_idx, :, :]
    counts = Qn[:, :, 0, :].real.sum(axis=1) * charge_factor
    for b, (lo, hi) in enumerate(meas_bins):
        Q0_pt[:, b] = _partial_counts(counts, pt_edges, lo, hi)
        pt_centers[b] = (lo + hi) / 2
        pt_widths[b] = hi - lo

    delta_eta = abs(eta[1] - eta[0])
    return {'Q0_pt': Q0_pt, 'pt_centers': pt_centers,
            'pt_widths': pt_widths, 'delta_eta': delta_eta}


def prepare_spectra_3d(grid, params):
    """Extract Q0 per pT bin from grid for invariant yield, normalized by nsamples."""
    if getattr(grid, 'pt_integrated', False):
        n_meas = max(len(params.get('differential_bins') or []), 1)
        return {'Q0_pt': None, 'pt_centers': None, 'pt_widths': None,
                'delta_eta': 1.0}
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    y = sk.get('y') or (None, None)
    ymass = sk.get('y_mass')
    if ymass is None and (eta[0] is None or eta[1] is None):
        # A true-rapidity window of an identified species: its own mass
        # (set by the species selection); unknown for mixtures.
        ymass = getattr(grid, 'mass', None)
    if ymass is not None and y[0] is not None and y[1] is not None:
        meas_bins = _extract_2d_pT_bins(params)
        Q0_pt = np.zeros((grid.n_events, len(meas_bins)))
        for b, (lo, hi) in enumerate(meas_bins):
            Q0_pt[:, b] = grid.acceptance_rapidity(y[0], y[1], float(ymass), lo, hi)[:, 0].real / grid.nsamples
        return {'Q0_pt': Q0_pt, 'pt_centers': np.array([(lo + hi) / 2 for lo, hi in meas_bins]),
                'pt_widths': np.array([hi - lo for lo, hi in meas_bins]), 'delta_eta': y[1] - y[0]}
    if eta[0] is None or eta[1] is None:
        return {'Q0_pt': None, 'pt_centers': None, 'pt_widths': None, 'delta_eta': 1.0}
    meas_bins = _extract_2d_pT_bins(params)
    n_meas = len(meas_bins)
    Q0_pt = np.zeros((grid.n_events, n_meas))
    pt_centers = np.zeros(n_meas)
    pt_widths = np.zeros(n_meas)
    for b, (lo, hi) in enumerate(meas_bins):
        Qn = grid.acceptance(eta[0], eta[1], lo, hi)
        Q0_pt[:, b] = Qn[:, 0].real / grid.nsamples
        pt_centers[b] = (lo + hi) / 2
        pt_widths[b] = hi - lo

    delta_eta = abs(eta[1] - eta[0])
    return {'Q0_pt': Q0_pt, 'pt_centers': pt_centers,
            'pt_widths': pt_widths, 'delta_eta': delta_eta}


def _partial_counts(counts, edges, lo, hi):
    """Integrate per-bin counts over ``[lo, hi]`` assuming uniform density.

    ``counts`` has shape (n_events, n_bins); ``edges`` has n_bins+1 entries.
    This correctly handles measurement bins narrower or wider than the
    simulation bins (the isobar campaign's uniform 41-bin grid is coarser
    than the 0.05-wide experimental bins at low pT).
    """
    out = np.zeros(counts.shape[0])
    for b in range(counts.shape[1]):
        e0 = edges[b]
        e1 = edges[b + 1]
        overlap = min(hi, e1) - max(lo, e0)
        if overlap > 0:
            out += counts[:, b] * (overlap / (e1 - e0))
    return out


def _wrap_spectra(data, w):
    if data.get('Q0_pt') is None:
        return np.array([])
    return compute_spectra_invariant_yield(
        data['Q0_pt'], data['pt_centers'], data['pt_widths'],
        data['delta_eta'], w)


# ── Mean-pT prepare + wrapper ───────────────────────────────────────────────

def _mean_pt_moments(qv, counts_raw):
    """Numerator source for mean-pT: exact Sum pT when stored, else the
    bin-midpoint estimate  Sum_b N_b * pT_mid(b)  (error O(width^2 x spectral
    curvature), negligible on fine pT grids)."""
    if getattr(qv, 'sum_pt', None) is not None:
        return qv.sum_pt
    mids = 0.5 * (qv.pt_bin_edges[:-1] + qv.pt_bin_edges[1:])
    return counts_raw * mids[np.newaxis, np.newaxis, :]


def prepare_mean_pt_2d(qv, params):
    """Weighted Sum w*pT and Sum w in the eta/pT acceptance.

    dy-mode: record specifies y (no eta); flat-y selection => uniform
    per-particle weights over the raw container.
    """
    z = np.zeros(qv.n_events)
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    species_idx, _, found = _get_species_selection(params.get('particle_id'), qv)
    if not found:
        return {'num': z, 'den': z}
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, *sk.get('pT', (0.1, 10)))
    counts_raw = qv.Qn[:, :, 0, :].real
    c = counts_raw[:, species_idx, :] if species_idx is not None else counts_raw
    num_src_full = _mean_pt_moments(qv, counts_raw)
    num_raw = num_src_full[:, species_idx, :] if species_idx is not None \
        else num_src_full
    if eta[0] is None or eta[1] is None:
        den = c[:, :, lo:hi].sum(axis=(1, 2)) / qv.nsamples
        num = num_raw[:, :, lo:hi].sum(axis=(1, 2)) / qv.nsamples
        return {'num': num, 'den': den}
    Qn_cut, jac = cut_qvectors(qv, eta[0], eta[1])
    jac_s = jac[species_idx, :] if species_idx is not None else jac
    from analysis.observables_2d import _compute_weighted_moment
    den = _compute_weighted_moment(qv, jac_s, c, lo, hi)
    num = _compute_weighted_moment(qv, jac_s, num_raw, lo, hi)
    return {'num': num, 'den': den}


def prepare_mean_pt_3d(grid, params):
    """Sum w*pT and Sum w from a 3D grid (sum_pt slot or bin midpoints)."""
    z = np.zeros(grid.n_events)
    if getattr(grid, 'pt_integrated', False):
        return {'num': np.full(grid.n_events, np.nan),
                'den': np.full(grid.n_events, np.nan)}
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    if eta[0] is None or eta[1] is None:
        return {'num': z, 'den': z}
    pt = sk.get('pT', (0.1, 10))
    se, sp_ = _grid_slices(grid, eta, pt)
    den = grid.Qn[:, se, sp_, 0].real.sum(axis=(1, 2)) / grid.nsamples
    if getattr(grid, 'sum_pt', None) is not None:
        num_src = grid.sum_pt
    else:
        mids = 0.5 * (grid.pt_edges[:-1] + grid.pt_edges[1:])
        num_src = grid.Qn[:, :, :, 0].real * mids[np.newaxis, np.newaxis, :]
    num = num_src[:, se, sp_].sum(axis=(1, 2)) / grid.nsamples
    return {'num': num, 'den': den}


def _wrap_mean_pt(data, w):
    return compute_mean_pt(data['num'], data['den'], w)


# ── pT-fluctuation + transverse-energy prepares ─────────────────────────────

def prepare_pT_fluct_2d(qv, params):
    """Per-event (s1, s2, n) for the ALICE two-particle pT correlator.

    Only uniform-weight acceptances are exact here (dy-mode / full-acceptance):
    non-uniform Jacobian weights would need second-weight moments for the
    pair self-correlation subtraction.
    """
    z = np.full(qv.n_events, np.nan)
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    species_idx, _, found = _get_species_selection(params.get('particle_id'), qv)
    if not found:
        return {'s1': z, 's2': z, 'n': z}
    if eta[0] is None or eta[1] is None:
        c_full = qv.Qn[:, :, 0, :].real
        mids = 0.5 * (qv.pt_bin_edges[:-1] + qv.pt_bin_edges[1:])
        if getattr(qv, 'sum_pt', None) is not None:
            s1_src = qv.sum_pt
        else:   # bin-midpoint estimate (validated vs Everett <pT> to <0.5%)
            s1_src = c_full * mids[np.newaxis, np.newaxis, :]
        if getattr(qv, 'sum_pt2', None) is not None:
            s2_src = qv.sum_pt2
        else:
            s2_src = c_full * (mids ** 2)[np.newaxis, np.newaxis, :]
        c = c_full[:, species_idx, :] if species_idx is not None else c_full
        s1_src = s1_src[:, species_idx, :] if species_idx is not None \
            else s1_src
        s2_src = s2_src[:, species_idx, :] if species_idx is not None \
            else s2_src
        ns = qv.nsamples
        n = c.sum(axis=(1, 2)) / ns
        s1 = s1_src.sum(axis=(1, 2)) / ns
        s2 = s2_src.sum(axis=(1, 2)) / ns
        return {'s1': s1, 's2': s2, 'n': n}
    # Jacobian-weighted windows need second-weight moments for exact pair
    # self-correlation subtraction -> documented NaN (see AGENTS.md).
    return {'s1': z, 's2': z, 'n': z}


def prepare_pT_fluct_3d(grid, params):
    z = np.full(grid.n_events, np.nan)
    if getattr(grid, 'pt_integrated', False):
        return {'s1': z, 's2': z, 'n': z}
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    if eta[0] is None or eta[1] is None:
        return {'s1': z, 's2': z, 'n': z}
    pt = sk.get('pT', (0.1, 10))
    se, sp_ = _grid_slices(grid, eta, pt)
    ns = grid.nsamples
    n = grid.Qn[:, se, sp_, 0].real.sum(axis=(1, 2)) / ns
    if getattr(grid, 'sum_pt', None) is not None:
        s1_src = grid.sum_pt
    else:
        mids = 0.5 * (grid.pt_edges[:-1] + grid.pt_edges[1:])
        s1_src = grid.Qn[:, :, :, 0].real * mids[np.newaxis, np.newaxis, :]
    if getattr(grid, 'sum_pt2', None) is not None:
        s2_src = grid.sum_pt2
    else:
        mids = 0.5 * (grid.pt_edges[:-1] + grid.pt_edges[1:])
        s2_src = grid.Qn[:, :, :, 0].real \
            * (mids ** 2)[np.newaxis, np.newaxis, :]
    s1 = s1_src[:, se, sp_].sum(axis=(1, 2)) / ns
    s2 = s2_src[:, se, sp_].sum(axis=(1, 2)) / ns
    return {'s1': s1, 's2': s2, 'n': n}


def _wrap_pT_fluct(data, w):
    return compute_pT_fluct(data['s1'], data['s2'], data['n'], w)


# ── Symmetric-cumulant SC(n,m) prepares ─────────────────────────────────────


def prepare_sc_2d(qv, params):
    """Weighted Q^1 vectors in the eta/pT acceptance for SC(n,m).

    Returns {'Q1': (n_events, n_harm) complex, 'n': int, 'm': int}.
    The harmonic indices are taken from params['harmonics'] (set by
    extract_measurement_params from the curated record's `harmonics` list).
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    if eta[0] is None or eta[1] is None:
        z = np.zeros((qv.n_events, 6), dtype=complex)
        return {'Q1': z, 'n': 0, 'm': 0}
    jac = _make_jacobian_list(qv.pt_centers, qv.masses, eta[0], eta[1])
    Q1 = _compute_weighted_qvectors(qv, jac, 1)
    harms = params.get('harmonics') or [0, 0]
    n, m = int(harms[0]), int(harms[1])
    return {'Q1': Q1, 'n': n, 'm': m}


def prepare_sc_3d(grid, params):
    """Q-vectors in (eta, pT) acceptance from a QGridData for SC(n,m)."""
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    if eta[0] is None or eta[1] is None:
        z = np.zeros((grid.n_events, 6), dtype=complex)
        return {'Q1': z, 'n': 0, 'm': 0}
    Q1 = grid.acceptance(eta[0], eta[1], pt[0], pt[1])
    harms = params.get('harmonics') or [0, 0]
    n, m = int(harms[0]), int(harms[1])
    return {'Q1': Q1, 'n': n, 'm': m}


def _wrap_sc(data, w):
    return compute_sc_mn(data['Q1'], w, data['n'], data['m'])


# ── Event-plane correlation NAC(n,m) (subevent) prepares ─────────────────────


def prepare_nac_2d(qv, params):
    """Weighted Q^1 vectors for subevents A and B in their eta/pT acceptances.

    Returns {'Q1_A': (n_events, n_harm) complex,
             'Q1_B': (n_events, n_harm) complex,
             'n': int, 'm': int}.
    Subevent boundaries come from params['subevent_ranges'] (set by
    extract_measurement_params from the record's subevents block).
    """
    sr = params.get('subevent_ranges', {})
    if len(sr) < 2:
        z = np.zeros((qv.n_events, 6), dtype=complex)
        return {'Q1_A': z, 'Q1_B': z, 'n': 0, 'm': 0}
    labels = sorted(sr.keys())
    eA = sr[labels[0]]['eta']; pTA = sr[labels[0]]['pT']
    eB = sr[labels[1]]['eta']; pTB = sr[labels[1]]['pT']
    jac_A = _make_jacobian_list(qv.pt_centers, qv.masses, eA[0], eA[1])
    jac_B = _make_jacobian_list(qv.pt_centers, qv.masses, eB[0], eB[1])
    Q1_A = _compute_weighted_qvectors(qv, jac_A, 1)
    Q1_B = _compute_weighted_qvectors(qv, jac_B, 1)
    harms = params.get('harmonics') or [0, 0]
    return {'Q1_A': Q1_A, 'Q1_B': Q1_B,
            'n': int(harms[0]), 'm': int(harms[1])}


def prepare_nac_3d(grid, params):
    """Q-vectors in two disjoint eta/pT acceptances from a QGridData."""
    sr = params.get('subevent_ranges', {})
    if len(sr) < 2:
        z = np.zeros((grid.n_events, 6), dtype=complex)
        return {'Q1_A': z, 'Q1_B': z, 'n': 0, 'm': 0}
    labels = sorted(sr.keys())
    eA = sr[labels[0]]['eta']; pTA = sr[labels[0]]['pT']
    eB = sr[labels[1]]['eta']; pTB = sr[labels[1]]['pT']
    Q1_A = grid.acceptance(eA[0], eA[1], pTA[0], pTA[1])
    Q1_B = grid.acceptance(eB[0], eB[1], pTB[0], pTB[1])
    harms = params.get('harmonics') or [0, 0]
    return {'Q1_A': Q1_A, 'Q1_B': Q1_B,
            'n': int(harms[0]), 'm': int(harms[1])}


def _wrap_nac(data, w):
    return compute_nac_mn(data['Q1_A'], data['Q1_B'], w, data['n'], data['m'])


def _window_inv_cosh(eta_min, eta_max):
    """Analytic Jac-weighted mean of 1/cosh(eta) over a window.

    dET/deta = < Sum mT / cosh(eta) >; 1/cosh(eta) is species- and
    pT-independent, so with the same dy/deta weighting used everywhere in 2D
    this window factor is exact up to narrow-window pT-direction mix.
    """
    from scipy import integrate
    num, _ = integrate.quad(lambda e: 1.0 / np.cosh(e),
                            eta_min, eta_max, limit=200)
    den = float(eta_max - eta_min)
    return num / den if den > 0 else 1.0


def _sp_reference_2d(qv, eta_win):
    """Charged-summed density-weighted Q-vector + N for a 2D window."""
    Qn_cut, jac = cut_qvectors(qv, eta_win[0], eta_win[1])
    lo, hi = 0, None                      # full pT unless overridden later
    return Qn_cut, jac


def prepare_vn2_sp_diff_2d(qv, params):
    """Per-bin POI vectors + integrated reference for scalar-product diff."""
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    zH = qv.Qn.shape[2]
    nb = max(len(params.get('differential_bins') or []), 1)
    if eta[0] is None or eta[1] is None or nb == 0:
        return {'q_poi': np.full((qv.n_events, zH, nb), np.nan),
                'n_poi': np.full((qv.n_events, nb), np.nan),
                'q_ref': np.full((qv.n_events, zH), np.nan),
                'n_ref': np.full(qv.n_events, np.nan)}
    var = params.get('differential_var', 'pT')
    bins = params['differential_bins']

    Qn_ref, jac_ref = cut_qvectors(qv, eta[0], eta[1])
    q_ref = sum_over_species_pt(Qn_ref)
    n_ref = q_ref[:, 0].real

    # The POI bin is a SUBSET of the reference window (same eta acceptance,
    # restricted in pT), and 2D boost invariance means every eta window is
    # the same particle set anyway -- so the shared set is never empty here.
    # In the weighted picture the diagonal of Re<Q_poi Q_ref*> is
    # sum_i w_i^2 over the POI bin, i.e. exactly dmq.
    from analysis.observables_2d import _compute_dmq_full
    dmq_full = _compute_dmq_full(qv, jac_ref)

    q_poi = np.full((qv.n_events, zH, len(bins)), np.nan, complex)
    n_poi = np.full((qv.n_events, len(bins)), np.nan)
    dq_cross = np.full((qv.n_events, len(bins)), np.nan)
    for b, (lo_b, hi_b) in enumerate(bins):
        if var == 'pT':
            lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, lo_b, hi_b)
            Qn_c = Qn_ref[:, :, :, lo:hi]
            dmq = dmq_full[:, lo:hi].sum(axis=1)
        else:                              # eta-differential: re-cut window
            Qn_c, jac_c = cut_qvectors(qv, lo_b, hi_b)
            plo, phi = _find_pt_slice_edges(qv.pt_bin_edges,
                                            *sk.get('pT', (0.1, 10)))
            Qn_c = Qn_c[:, :, :, plo:phi]
            dmq_c = _compute_dmq_full(qv, jac_c)[:, plo:phi]
            dmq = dmq_c.sum(axis=1)
        s_pt = sum_over_species_pt(Qn_c)
        q_poi[:, :, b] = s_pt
        n_poi[:, b] = s_pt[:, 0].real
        dq_cross[:, b] = dmq
    return {'q_poi': q_poi, 'n_poi': n_poi, 'q_ref': q_ref, 'n_ref': n_ref,
            'dQ_cross': dq_cross}


def prepare_vn2_ep_diff_3d(grid, params):
    """Event-plane v2 per eta bin: q_poi and n_poi from the grid, no q_ref."""
    if getattr(grid, 'pt_integrated', False):
        H = grid.Qn.shape[-1]
        nb = max(len(params.get('differential_bins') or []), 1)
        nan = np.full((grid.n_events, H, nb), np.nan)
        return {'q_poi': nan, 'n_poi': np.full((grid.n_events, nb), np.nan)}
    sk = params.get('single_range', {})
    pt = sk.get('pT', (0.1, 10))
    bins = params.get('differential_bins') or []
    H = grid.Qn.shape[-1]
    nb = max(len(bins), 1)
    if not bins:
        return {'q_poi': np.full((grid.n_events, H, nb), np.nan),
                'n_poi': np.full((grid.n_events, nb), np.nan)}
    q_poi = np.full((grid.n_events, H, len(bins)), np.nan, complex)
    n_poi = np.full((grid.n_events, len(bins)), np.nan)
    for b, (lo_b, hi_b) in enumerate(bins):
        qq = grid.acceptance(lo_b, hi_b, pt[0], pt[1])
        q_poi[:, :, b] = qq
        n_poi[:, b] = qq[:, 0].real
    return {'q_poi': q_poi, 'n_poi': n_poi}


def _wrap_vn2_ep_diff_eta(data, w):
    return compute_vn2_ep_diff_eta(data['q_poi'], data['n_poi'], w)


def prepare_vn2_ep_diff_pT_3d(grid, params):
    """Event-plane v2 per pT bin: q_poi/n_poi per pT bin in the eta window."""
    if getattr(grid, 'pt_integrated', False):
        H = grid.Qn.shape[-1]
        nb = max(len(params.get('differential_bins') or []), 1)
        nan = np.full((grid.n_events, H, nb), np.nan)
        return {'q_poi': nan, 'n_poi': np.full((grid.n_events, nb), np.nan)}
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    bins = params.get('differential_bins') or []
    H = grid.Qn.shape[-1]
    nb = max(len(bins), 1)
    if not bins or eta[0] is None:
        return {'q_poi': np.full((grid.n_events, H, nb), np.nan),
                'n_poi': np.full((grid.n_events, nb), np.nan)}
    q_poi = np.full((grid.n_events, H, len(bins)), np.nan, complex)
    n_poi = np.full((grid.n_events, len(bins)), np.nan)
    for b, (lo_b, hi_b) in enumerate(bins):
        qq = grid.acceptance(eta[0], eta[1], lo_b, hi_b)
        q_poi[:, :, b] = qq
        n_poi[:, b] = qq[:, 0].real
    return {'q_poi': q_poi, 'n_poi': n_poi}


def _wrap_vn2_ep_diff_pT(data, w):
    return compute_vn2_ep_diff_pT(data['q_poi'], data['n_poi'], w)


def prepare_vn3_single_3d_ep(grid, params):
    """Event-plane v3 integrated: just Qn from the grid."""
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    if eta[0] is None or eta[1] is None:
        return {'Qn': np.zeros((grid.n_events, 6), dtype=complex)}
    pt = sk.get('pT', (0.1, 10))
    return {'Qn': grid.acceptance(eta[0], eta[1], pt[0], pt[1])}


def _wrap_vn3_ep(data, w):
    return compute_vn3_ep(data['Qn'], w)


def prepare_vn2_single_3d_ep(grid, params):
    """Event-plane v2 integrated: just Qn from the grid."""
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    if eta[0] is None or eta[1] is None:
        return {'Qn': np.zeros((grid.n_events, 6), dtype=complex)}
    pt = sk.get('pT', (0.1, 10))
    return {'Qn': grid.acceptance(eta[0], eta[1], pt[0], pt[1])}


def _wrap_vn2_ep(data, w):
    return compute_vn2_ep(data['Qn'], w)


def prepare_vn2_sp_diff_3d(grid, params):
    if getattr(grid, 'pt_integrated', False):
        H = grid.Qn.shape[-1]
        nb = max(len(params.get('differential_bins') or []), 1)
        nan = np.full((grid.n_events, H, nb), np.nan)
        return {'q_poi': nan, 'q_ref': np.full((grid.n_events, H), np.nan),
                'n_poi': np.full((grid.n_events, nb), np.nan),
                'n_ref': np.full(grid.n_events, np.nan)}
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    bins = params.get('differential_bins') or []
    var = params.get('differential_var', 'pT')
    H = grid.Qn.shape[-1]
    nb = max(len(bins), 1)
    if not bins or eta[0] is None:
        return {'q_poi': np.full((grid.n_events, H, nb), np.nan),
                'q_ref': np.full((grid.n_events, H), np.nan),
                'n_poi': np.full((grid.n_events, nb), np.nan),
                'n_ref': np.full(grid.n_events, np.nan)}
    # Reference window: record acceptance by default; an explicit
    # params['sp_ref_*'] override (comparison-only, e.g. BBC reference
    # for JETSCAPE) replaces it.  POI always follows the record bins.
    # sp_ref_eta may be one (lo, hi) window or a list of windows (their
    # Q-vectors are summed, e.g. both BBC sides).
    #
    # Otherwise the record's own `flow_reference` block decides.  It can
    # supply a per-bin reference: a mirrored one (the "sub-event on the
    # opposite side of midrapidity" construction, which also guarantees
    # the reference shares no particles with the POI) or a different
    # reference per eta range via `segments`, for publications that merge
    # two analysis methods into one series.
    fr = params.get('flow_reference')
    q_ref, n_ref, per_bin, dq_spec = _sp_reference_for_bins(
        grid, bins, var, eta, pt, params, fr)
    # per_bin -> the builder already returned the (n_ev, n_bins) counts;
    # a uniform reference instead returns a spec tuple to expand per bin.
    if per_bin:
        dq_cross = dq_spec
    elif isinstance(dq_spec, tuple):
        _e, _p, _wins, _rpt = dq_spec
        dq_cross = np.full((grid.n_events, len(bins)), np.nan)
        for _b, (_lo, _hi) in enumerate(bins):
            _pe = (_lo, _hi) if var == 'eta' else (_e[0], _e[1])
            _pp = (_p[0], _p[1]) if var == 'eta' else (_lo, _hi)
            _sh = _shared_count(grid, _pe, _pp, _wins, _rpt)
            dq_cross[:, _b] = 0.0 if _sh is None else _sh
    else:
        dq_cross = None
    q_poi = np.full((grid.n_events, H, len(bins)), np.nan, complex)
    n_poi = np.full((grid.n_events, len(bins)), np.nan)
    # No reference declared: the reference defaults to the record's own
    # acceptance, so the POI bin lies entirely inside the reference and the
    # diagonal is N_poi itself.  Only in that case may it be filled in
    # directly -- with a declared reference the count is geometric and must
    # come from _shared_count, which is ZERO for a disjoint reference.
    same_window_default = (dq_cross is None
                           and params.get('flow_reference') is None
                           and params.get('sp_ref_eta') is None)
    if same_window_default:
        dq_cross = np.full((grid.n_events, len(bins)), np.nan)
    for b, (lo_b, hi_b) in enumerate(bins):
        if var == 'pT':
            qq = grid.acceptance(eta[0], eta[1], lo_b, hi_b)
        else:
            qq = grid.acceptance(lo_b, hi_b, pt[0], pt[1])
        q_poi[:, :, b] = qq
        n_poi[:, b] = qq[:, 0].real
        if same_window_default:
            dq_cross[:, b] = n_poi[:, b]
    return {'q_poi': q_poi, 'n_poi': n_poi, 'q_ref': q_ref, 'n_ref': n_ref,
            'q_ref_per_bin': per_bin, 'dQ_cross': dq_cross}


def _odd_windows(windows):
    """eta-odd weighting of reference windows: (lo, hi, +1 or -1); a window
    across eta = 0 splits there."""
    out = []
    for lo, hi in windows:
        if lo >= 0:
            out.append((lo, hi, 1.0))
        elif hi <= 0:
            out.append((lo, hi, -1.0))
        else:
            out += [(lo, 0.0, -1.0), (0.0, hi, 1.0)]
    return out


def _odd_q(grid, wins, pt):
    """Harmonic-1 Q-vector of signed windows and its self-correlation (count)."""
    q = np.zeros(grid.n_events, complex)
    n = np.zeros(grid.n_events)
    for lo, hi, sg in wins:
        a = grid.acceptance(lo, hi, pt[0], pt[1])
        q += sg * a[:, 1]
        n += a[:, 0].real
    return q, n


def _odd_shared(grid, wins_a, wins_b, pt_a, pt_b):
    """Diagonal of Re<Q_a Q_b*> for signed windows: the particles in both,
    each weighted by the product of its two signs."""
    tot = np.zeros(grid.n_events)
    lo_pt, hi_pt = max(pt_a[0], pt_b[0]), min(pt_a[1], pt_b[1])
    if hi_pt <= lo_pt:
        return tot
    for la, ha, sa in wins_a:
        for lb, hb, sb in wins_b:
            lo, hi = max(la, lb), min(ha, hb)
            if hi > lo:
                tot += sa * sb * grid.acceptance(lo, hi, lo_pt, hi_pt)[:, 0].real
    return tot


def prepare_v1_odd_3d(grid, params):
    """Rapidity-odd v1(eta): per POI bin its eta-odd reference (from
    flow_reference, by |eta| segment), plus the reference of the
    sign-convention window, which every bin is aligned to."""
    fr = params.get('flow_reference') or {}
    bins = params.get('differential_bins') or []
    pt = (params.get('single_range') or {}).get('pT', (0.1, 10))
    n_ev, nb = grid.n_events, max(len(bins), 1)
    if getattr(grid, 'pt_integrated', False) or not bins or not fr.get('sign_convention'):
        nan = np.full((n_ev, nb), np.nan)
        return {'q_poi': nan.astype(complex), 'n_poi': nan, 'q_ref': nan.astype(complex), 'n_ref': nan,
                'dq': nan, 'align_shared': nan, 'q_c': np.full(n_ev, np.nan, complex),
                'n_c': np.full(n_ev, np.nan), 'q_sw': np.full(n_ev, np.nan, complex),
                'n_sw': np.full(n_ev, np.nan), 'dq_sw': np.full(n_ev, np.nan), 'sign': 1.0}

    def ref_for(lo, hi):
        seg = _ref_segment_for(fr, lo, hi)
        return _odd_windows(seg['eta']), (seg.get('pT') or pt)

    (sw_lo, sw_hi), sign = fr['sign_convention']
    wins_c, pt_c = ref_for(sw_lo, sw_hi)
    q_c, n_c = _odd_q(grid, wins_c, pt_c)
    a_sw = grid.acceptance(sw_lo, sw_hi, pt[0], pt[1])
    out = {k: np.zeros((n_ev, nb), complex if k in ('q_poi', 'q_ref') else float)
           for k in ('q_poi', 'n_poi', 'q_ref', 'n_ref', 'dq', 'align_shared')}
    for b, (lo, hi) in enumerate(bins):
        a = grid.acceptance(lo, hi, pt[0], pt[1])
        wins, rpt = ref_for(lo, hi)
        q_r, n_r = _odd_q(grid, wins, rpt)
        out['q_poi'][:, b], out['n_poi'][:, b] = a[:, 1], a[:, 0].real
        out['q_ref'][:, b], out['n_ref'][:, b] = q_r, n_r
        out['dq'][:, b] = _odd_shared(grid, [(lo, hi, 1.0)], wins, pt, rpt)
        out['align_shared'][:, b] = _odd_shared(grid, wins, wins_c, rpt, pt_c)
    out.update(q_c=q_c, n_c=n_c, q_sw=a_sw[:, 1], n_sw=a_sw[:, 0].real,
               dq_sw=_odd_shared(grid, [(sw_lo, sw_hi, 1.0)], wins_c, pt, pt_c), sign=float(sign))
    # (Charge-scaled grids reach this on unit weights: _on_unit_weights.)
    return out


def _wrap_v1_odd(data, w):
    return compute_v1_odd_sp_diff_eta(data, w)


def _abs_eta_range(lo_b, hi_b):
    """Map a signed POI bin to its |eta| interval."""
    if lo_b >= 0:
        return lo_b, hi_b
    if hi_b <= 0:
        return -hi_b, -lo_b
    return 0.0, max(-lo_b, hi_b)      # bin straddles midrapidity


def _ref_segment_for(fr, lo_b, hi_b):
    """Pick the flow_reference definition governing the POI bin (lo_b, hi_b).

    Segment applicability is expressed in |eta| (``applies_to_eta``), so the
    bin is first mapped to its |eta| interval -- comparing signed eta
    against an |eta| range would leave every negative bin matching nothing
    and silently fall through to an empty reference (NaN).

    A differential bin may also straddle the boundary between two reference
    segments (PHOBOS Fig. 5 switches from the track-based to the mirrored
    hit-based reference at |eta| ~ 1.8, and its bins are ~0.1-0.9 wide), so
    containment is too strict: choose the segment with the largest |eta|
    OVERLAP with the bin.  Falls back to the top-level definition when no
    segment overlaps at all.
    """
    if not fr:
        return None
    segs = fr.get('segments') or []
    a_lo, a_hi = _abs_eta_range(lo_b, hi_b)
    best, best_ov = None, 0.0
    for s in segs:
        s_lo, s_hi = s['applies_to_eta']
        ov = min(a_hi, s_hi) - max(a_lo, s_lo)
        if ov > best_ov:
            best, best_ov = s, ov
    return best or fr


def _shared_count(grid, poi_eta, poi_pt, ref_windows, ref_pt):
    """Particles common to the POI box and the reference window(s).

    This is the diagonal of Re<Q_poi Q_ref*>: every particle counted here
    appears once as POI and once as reference, contributing exactly 1 to
    the double sum, so it -- and only it -- is self-correlation.

    It is purely geometric and needs no overlap TEST:
      * disjoint windows      -> empty intersection -> 0
      * POI nested in ref     -> the whole POI -> N_poi
      * partially overlapping  -> the shared piece
      * mirrored reference on the opposite side of midrapidity -> 0
        automatically, because the two eta ranges do not meet.
    """
    lo_pt = max(poi_pt[0], ref_pt[0])
    hi_pt = min(poi_pt[1], ref_pt[1])
    if hi_pt <= lo_pt:
        return None
    acc = None
    for (wlo, whi) in ref_windows or []:
        lo = max(poi_eta[0], wlo)
        hi = min(poi_eta[1], whi)
        if hi <= lo:
            continue
        c = grid.acceptance(lo, hi, lo_pt, hi_pt)[:, 0].real
        acc = c if acc is None else acc + c
    return acc


def _sp_reference_for_bins(grid, bins, var, eta, pt, params, fr):
    """Build the SP reference Q-vectors for a differential record.

    Returns (q_ref, n_ref, per_bin).  When the reference depends on the
    differential bin (mirrored, or segmented by |eta|) the arrays carry a
    trailing bin axis, (n_events, H, n_bins) / (n_events, n_bins), and
    per_bin is True.  A bin for which no reference can be built is left
    as NaN rather than raising, so it is reported as unavailable.
    """
    H = grid.Qn.shape[-1]
    n_ev = grid.n_events
    override = params.get('sp_ref_eta')
    if override is not None:
        if (isinstance(override, (list, tuple)) and len(override) == 2
                and all(isinstance(x, (list, tuple)) for x in override)):
            re_windows = [(float(a), float(b)) for a, b in override]
        else:
            re_windows = [tuple(float(x) for x in override)]
        acc = None
        for (wlo, whi) in re_windows:
            _q = grid.acceptance(wlo, whi, params.get('sp_ref_pt', pt)[0],
                                 params.get('sp_ref_pt', pt)[1])
            acc = _q if acc is None else acc + _q
        # Reference is shared by every bin, so the shared count is computed
        # per bin by the caller; signal "uniform" with per_bin=False.
        return acc, acc[:, 0].real, False, (eta, pt, re_windows,
                                            params.get('sp_ref_pt', pt))

    if not fr:
        acc = grid.acceptance(eta[0], eta[1], pt[0], pt[1])
        return acc, acc[:, 0].real, False, None

    def _acc(windows, pt_win):
        acc = None
        for (wlo, whi) in windows or []:
            _q = grid.acceptance(wlo, whi, pt_win[0], pt_win[1])
            acc = _q if acc is None else acc + _q
        return acc

    needs_per_bin = any(_ref_segment_for(fr, lo, hi).get('mirror')
                        for lo, hi in bins) or bool(fr.get('segments'))
    if not needs_per_bin:
        ref = fr
        acc = _acc(ref['eta'], ref.get('pT') or pt)
        if acc is None:
            z = np.full((n_ev, H), np.nan, complex)
            return z, np.full(n_ev, np.nan), False, None
        return acc, acc[:, 0].real, False, None

    q_ref = np.full((n_ev, H, len(bins)), np.nan, complex)
    n_ref = np.full((n_ev, len(bins)), np.nan)
    dq_cross = np.full((n_ev, len(bins)), np.nan)
    # Two phases on purpose: build every reference window first, then the
    # shared counts.  Keeps the two kinds of grid query separable, and a
    # bin straddling midrapidity really does share particles with a
    # mirrored sub-event (|eta| >= 0.1), which is exactly what phase 2
    # has to account for.
    win_per_bin = []
    for b, (lo_b, hi_b) in enumerate(bins):
        ref = _ref_segment_for(fr, lo_b, hi_b)
        windows = ref['eta']
        if ref.get('mirror'):
            # "the subevent on the opposite side of midrapidity was used":
            # the reference is the SAME sub-event window as declared, but
            # taken on the side opposite the particle of interest.  For a
            # POI bin entirely at eta>0 that is the negated window; for one
            # at eta<0 it is the window as given.  A bin straddling eta=0
            # gets both images.
            lo_w, hi_w = windows[0]
            if lo_b < 0 < hi_b:
                # Bin straddles midrapidity: take both images of the
                # sub-event, so the reference still avoids the bin's own
                # particles on either side.
                windows = [(-hi_w, -lo_w), (lo_w, hi_w)]
            elif hi_b > 0:
                windows = [(-hi_w, -lo_w)]
            else:
                windows = [(lo_w, hi_w)]
        pt_win = ref.get('pT') or pt
        acc = _acc(windows, pt_win)
        if acc is None:
            win_per_bin.append(None)   # leave NaN: bin reported unavailable
            continue
        q_ref[:, :, b] = acc
        n_ref[:, b] = acc[:, 0].real
        win_per_bin.append((windows, pt_win))
    for b, (lo_b, hi_b) in enumerate(bins):
        if win_per_bin[b] is None:
            continue
        windows, pt_win = win_per_bin[b]
        poi_eta = (lo_b, hi_b) if var == 'eta' else (eta[0], eta[1])
        poi_pt = (pt[0], pt[1]) if var == 'eta' else (lo_b, hi_b)
        sh = _shared_count(grid, poi_eta, poi_pt, windows, pt_win)
        # None means the intersection is EMPTY -> a disjoint reference
        # contributes exactly 0, which is not the same as "unknown".
        dq_cross[:, b] = 0.0 if sh is None else sh
    return q_ref, n_ref, True, dq_cross


def _wrap_vn2_sp_diff(data, w):
    # n_ref removes the reference's own auto-correlation from |Q_ref|^2;
    # dQ_cross removes the POI/reference diagonal from the numerator.
    # Both default to the no-correction behaviour when absent, which is
    # only correct for a disjoint reference.
    return _core_vn2_sp_diff(data['q_poi'], data['n_poi'],
                             data['q_ref'], w,
                             dQ_cross=data.get('dQ_cross'),
                             n_ref=data.get('n_ref'))


def _wrap_vn2_sp_diff_means(data, w):
    return compute_vn2_sp_diff_means(data['q_poi'], data['n_poi'],
                                     data['q_ref'], data['n_ref'], w)

def _q_window_2d(qv, eta_lo, eta_hi, pt_lo, pt_hi, species_idx,
                 charge_factor):
    """Q-vectors, counts, and Σw² in an (eta, pT) window (2D container)."""
    Qn, jac = cut_qvectors(qv, eta_lo, eta_hi)
    if species_idx is not None:
        Qn = Qn[:, species_idx, :, :]
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, pt_lo, pt_hi)
    q = Qn[:, :, :, lo:hi].sum(axis=(1, 3)) * charge_factor
    n = Qn[:, :, 0, lo:hi].sum(axis=(1, 2)).real * charge_factor
    w = (jac[np.newaxis, :, :]
         / (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis]))
    if species_idx is not None:
        w = w[:, species_idx, :]
    dq0 = (qv.Qn[:, :, 0, :].real[:, species_idx, :]
           if species_idx is not None else qv.Qn[:, :, 0, :].real)
    dq0 = (dq0[:, :, lo:hi] * w[:, :, lo:hi] ** 2).sum(axis=(1, 2))
    dq0 *= charge_factor ** 2
    return q, n, dq0


def prepare_vn2_sp_2d(qv, params):
    """2D scalar-product / event-plane prepare.

    The POI Q-vectors are taken in the record's acceptance.  If the record
    declares a separate ``flow_reference`` (EPD, ZDC, or the two TPC
    halves), its eta windows are used; otherwise the reference is the POI
    sample itself (``same_window=True``).
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    z = np.full((qv.n_events, qv.Qn.shape[2]), np.nan, complex)
    if eta[0] is None or eta[1] is None:
        return {'q_poi': z, 'q_ref': z, 'n_poi': np.full(qv.n_events, np.nan),
                'n_ref': np.full(qv.n_events, np.nan), 'same_window': True}

    species_idx, cf, found = _get_species_selection(
        params.get('particle_id'), qv)
    if not found:
        return {'q_poi': z, 'q_ref': z, 'n_poi': np.full(qv.n_events, np.nan),
                'n_ref': np.full(qv.n_events, np.nan), 'same_window': True}

    q_poi, n_poi, dq_poi = _q_window_2d(
        qv, eta[0], eta[1], pt[0], pt[1], species_idx, cf)
    fr = params.get('flow_reference')
    ref_eta = fr.get('eta') if isinstance(fr, dict) else None
    if not ref_eta:
        return {'q_poi': q_poi, 'q_ref': q_poi.copy(),
                'n_poi': n_poi, 'n_ref': n_poi.copy(),
                'dQ0_poi': dq_poi, 'dQ0_ref': dq_poi.copy(),
                'same_window': True}

    # Reference is a separate sample (EPD/ZDC or the two TPC halves).
    ref_idx, ref_cf = charged_species_indices(getattr(qv, 'pdg_ids', None))
    q_ref = np.zeros_like(q_poi)
    n_ref = np.zeros_like(n_poi)
    dq_ref = np.zeros_like(dq_poi)
    for lo, hi in ref_eta:
        q, n, dq = _q_window_2d(qv, lo, hi, pt[0], pt[1], ref_idx, ref_cf)
        q_ref += q
        n_ref += n
        dq_ref += dq
    return {'q_poi': q_poi, 'q_ref': q_ref, 'n_poi': n_poi,
            'n_ref': n_ref, 'dQ0_poi': dq_poi, 'dQ0_ref': dq_ref,
            'same_window': False}


def prepare_vn2_sp_3d(grid, params):
    """Density-weighted flow vectors for the scalar-product observable.

    Works for both regular grids and pt_integrated species grids.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    z = np.full((grid.n_events, grid.Qn.shape[-1]), np.nan, complex)
    if eta[0] is None or eta[1] is None:
        return {'q_poi': z, 'q_ref': z, 'n_ref': np.full(grid.n_events,
                                                          np.nan)}
    q = grid.acceptance(eta[0], eta[1], 0.0, 1e9)   # unit weights already
    # The reference defaults to the record's own acceptance, so every POI
    # particle is also a reference particle: the diagonal is N_poi itself.
    # (compute_vn2_sp's same_window=True already encodes this; passing
    # dQ_cross too keeps the two paths consistent.)
    return {'q_poi': q, 'q_ref': q.copy(),
            'n_poi': q[:, 0].real, 'n_ref': q[:, 0].real,
            'dQ_cross': q[:, 0].real}


def _wrap_vn2_sp(data, w):
    # Same-window prepares set q_ref = q_poi; the diagonal is N_poi and
    # same_window subtracts it.  Separate-reference prepares (EPD/ZDC)
    # set same_window=False.
    return compute_vn2_sp(
        data['q_poi'], data['q_ref'], data['n_poi'], data['n_ref'], w,
        same_window=bool(data.get('same_window', True)),
        dQ0_poi=data.get('dQ0_poi'), dQ0_ref=data.get('dQ0_ref'))


def prepare_vn6_single_3d(grid, params):
    """Unit-weight Q-vectors up to harmonic 3n for v_n{6}."""
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    h = int(params.get('harmonic', 2))
    if 3 * h >= grid.Qn.shape[-1]:
        return None
    q = grid.acceptance(eta[0], eta[1], pt[0], pt[1])
    return {'Q1': q, 'Q2': q.copy(), 'Q3': q.copy(), 'Q4': q.copy()}


def prepare_vn8_single_3d(grid, params):
    """Unit-weight Q-vectors up to harmonic 4n for v_n{8}."""
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    h = int(params.get('harmonic', 2))
    if 4 * h >= grid.Qn.shape[-1]:
        return None
    q = grid.acceptance(eta[0], eta[1], pt[0], pt[1])
    return {'Q1': q}


def prepare_vn6_single_2d(qv, params):
    """Build Q-vectors for v_n{6} on the 2D path.

    The Jacobian (correct) particle weights ARE applied via cut_qvectors;
    only the higher-weight-power self-correlation corrections (w^2, w^3, ...
    terms, suppressed by powers of 1/M) are neglected.  Missing harmonics
    are zero-padded with a warning.
    """
    import warnings
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.15, 3.0))
    h = int(params.get('harmonic', 2))
    if 3 * h >= qv.n_harmonics:
        warnings.warn(
            f"v_n{{{h}}}{{6}} needs harmonic {3*h}; 2D data has "
            f"{qv.n_harmonics} slots.  Zero-padding (self-correlation "
            f"negligible at heavy-ion M).", UserWarning, stacklevel=2)
    Qn_cut, jac = cut_qvectors(qv, eta[0], eta[1])
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, *sk.get('pT',(0.1,10)))
    Qn_pt = Qn_cut[:, :, :, lo:hi].sum(axis=3)
    Qn = Qn_pt.sum(axis=1)
    return {'Q1': Qn}


def prepare_vn8_single_2d(qv, params):
    """Build Q-vectors for v_n{8} on the 2D path.

    Correct Jacobian weights are applied via cut_qvectors; only the
    higher-weight-power self-correlation corrections (suppressed by 1/M)
    are neglected.  Missing harmonics are zero-padded with a warning.
    """
    import warnings
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.15, 3.0))
    h = int(params.get('harmonic', 2))
    if 4 * h >= qv.n_harmonics:
        warnings.warn(
            f"v_n{{{h}}}{{8}} needs harmonic {4*h}; 2D data has "
            f"{qv.n_harmonics} slots.  Zero-padding.", UserWarning,
            stacklevel=2)
    Qn_cut, jac = cut_qvectors(qv, eta[0], eta[1])
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, *sk.get('pT',(0.1,10)))
    Qn_pt = Qn_cut[:, :, :, lo:hi].sum(axis=3)
    Qn = Qn_pt.sum(axis=1)
    return {'Q1': Qn}


def prepare_dET_2d(qv, params):
    """dET/deta from a flat-y container.

    E_T = mT / cosh(eta) splits into an exactly-storable eta-independent
    piece (Sum mT per species/pT bin; bin-midpoint fallback mT(mid, mass))
    times the analytic window factor <1/cosh(eta)>.
    """
    z = np.zeros(qv.n_events)
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    if eta[0] is None or eta[1] is None:
        return {'sum_et': z, 'delta_eta': 1.0}
    species_idx, cf, found = _get_species_selection(params.get('particle_id'), qv)
    if not found:
        return {'sum_et': z, 'delta_eta': 1.0}
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, *sk.get('pT', (0.1, 10)))
    counts_raw = qv.Qn[:, :, 0, :].real
    if getattr(qv, 'sum_mT', None) is not None:
        mt_src = qv.sum_mT
    else:
        # counts-carrying numerator: Sum_b N_b * mT(mid_b, mass_species)
        masses = np.asarray(qv.masses, dtype=float)[:, np.newaxis]
        mids = 0.5 * (qv.pt_bin_edges[:-1] + qv.pt_bin_edges[1:])
        mt_cell = np.sqrt(masses ** 2 + mids[np.newaxis, :] ** 2)
        mt_src = counts_raw * np.broadcast_to(mt_cell, counts_raw.shape)
    c = counts_raw[:, species_idx, :] if species_idx is not None else counts_raw
    mt = mt_src[:, species_idx, :] if species_idx is not None else mt_src
    Qn_cut, jac = cut_qvectors(qv, eta[0], eta[1])
    jac_s = jac[species_idx, :] if species_idx is not None else jac
    from analysis.observables_2d import _compute_weighted_moment
    sum_mt = _compute_weighted_moment(qv, jac_s, mt, lo, hi)
    # Charge factor as for yields: sims store positive charges only.
    R = _window_inv_cosh(eta[0], eta[1])
    return {'sum_et': sum_mt * R * (cf or 1.0),
            'delta_eta': abs(eta[1] - eta[0])}


def prepare_dET_3d(grid, params):
    """Exact dET/deta from the sum_ET slot; midpoint-mT fallback otherwise."""
    z = np.zeros(grid.n_events)
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    if eta[0] is None or eta[1] is None:
        return {'sum_et': z, 'delta_eta': 1.0}
    pt = sk.get('pT', (0.1, 10))
    se, sp_ = _grid_slices(grid, eta, pt)
    # NOTE: the incoming grid is already charge-factored by the upstream
    # grid.select() call (sum_ET carries the xCF); do NOT re-apply cf here,
    # otherwise charged dET/deta is double-counted (factor-of-2 bug).
    if getattr(grid, 'sum_ET', None) is not None:
        # exact stored E_T column
        sum_et = grid.sum_ET[:, se, sp_].sum(axis=(1, 2)) / grid.nsamples
    else:
        # no per-cell species on grids: charged-pion mass approximation
        m_pi = 0.13957
        mids = 0.5 * (grid.pt_edges[:-1] + grid.pt_edges[1:])
        ch = 0.5 * (grid.eta_edges[:-1] + grid.eta_edges[1:])
        ET_grid = np.sqrt((mids ** 2)[np.newaxis, np.newaxis, :]
                          + ((m_pi * np.cosh(ch)) ** 2)[np.newaxis, :,
                                                        np.newaxis])
        counts = grid.Qn[:, :, :, 0].real
        sum_et = (counts[:, se, sp_] * ET_grid[:, se, sp_]).sum(axis=(1, 2)) \
            / grid.nsamples
    return {'sum_et': sum_et,
            'delta_eta': abs(eta[1] - eta[0])}


def _wrap_dET(data, w):
    return compute_dET_deta(data['sum_et'], data['delta_eta'], w)


# ── PIPELINE_DISPATCH table ───────────────────────────────────────────

PIPELINE_DISPATCH = {
    "vn2_single": {
        "core_fn": _wrap_vn2_single,
        "prepare_data_2d": prepare_vn2_single_2d,
        "prepare_data_3d": prepare_vn2_single_3d,
        "n_obs_per_cent": 5,
        "description": "v_n{2} single-acceptance integrated",
    },
    "vn2_single_mor": {
        "core_fn": _wrap_vn2_single_mor,
        "prepare_data_2d": prepare_vn2_single_2d,
        "prepare_data_3d": prepare_vn2_single_3d,
        "n_obs_per_cent": 5,
        "description": "v_n{2} single-acceptance, mean-of-ratios (JETSCAPE verbatim)",
    },
    "vn2_gap": {
        "core_fn": _wrap_vn2_gap,
        "prepare_data_2d": prepare_vn2_gap_2d,
        "prepare_data_3d": prepare_vn2_gap_3d,
        "n_obs_per_cent": 5,
        "description": "v_n{2}ab two-subevent integrated",
    },
    "vn2_single_diff_pT": {
        "core_fn": _wrap_vn2_diff_pT,
        "prepare_data_2d": prepare_vn2_diff_pT_2d,
        "prepare_data_3d": _prepare_vn2_diff_3d,
        "n_obs_per_cent": None,  # depends on n_pT bins
        "description": "v_n{2} pT-differential",
    },
    "vn2_single_diff_eta": {
        "core_fn": _wrap_vn2_diff_eta,
        "prepare_data_2d": prepare_vn2_diff_eta_2d,
        "prepare_data_3d": _prepare_vn2_diff_3d,
        "n_obs_per_cent": None,
        "description": "v_n{2} eta-differential",
    },
    "vn2_gap_diff_pT": {
        "core_fn": _wrap_vn2_gap_diff_pT,
        "prepare_data_2d": prepare_vn2_gap_diff_pT_2d,
        "prepare_data_3d": _prepare_vn2_gap_diff_3d,
        "n_obs_per_cent": None,
        "description": "v_n{2}ab pT-differential",
    },
    "vn2_gap_diff_eta": {
        "core_fn": _wrap_vn2_gap_diff_eta,
        "prepare_data_2d": prepare_vn2_gap_diff_eta_2d,
        "prepare_data_3d": _prepare_vn2_gap_diff_3d,
        "n_obs_per_cent": None,
        "description": "v_n{2}ab eta-differential",
    },
    "vn4_single": {
        "core_fn": _wrap_vn4_single,
        "prepare_data_2d": prepare_vn4_single_2d,
        "prepare_data_3d": prepare_vn4_single_3d,
        "n_obs_per_cent": 5,
        "description": "v_n{4} single-acceptance integrated",
    },
    "vn4_single_diff_pT": {
        "core_fn": _wrap_vn4_diff,
        "prepare_data_2d": prepare_vn4_diff_2d,
        "prepare_data_3d": _prepare_vn4_diff_3d,
        "n_obs_per_cent": None,
        "description": "v_n{4} pT-differential",
    },
    "vn4_single_diff_eta": {
        "core_fn": _wrap_vn4_diff,
        "prepare_data_2d": prepare_vn4_diff_2d,
        "prepare_data_3d": _prepare_vn4_diff_3d,
        "n_obs_per_cent": None,
        "description": "v_n{4} eta-differential",
    },
    "vn4_gap": {
        "core_fn": _wrap_vn4_gap,
        "prepare_data_2d": prepare_vn4_gap_2d,
        "prepare_data_3d": prepare_vn4_gap_3d,
        "n_obs_per_cent": 5,
        "description": "v_n{4}AABB two-subevent integrated",
    },
    "yield_dNch_deta": {
        "core_fn": _wrap_yield,
        "prepare_data_2d": prepare_yield_2d,
        "prepare_data_3d": prepare_yield_3d,
        "n_obs_per_cent": 1,
        "description": "dNch/dη, dN/dy, multiplicity (eta-integrated average)",
    },
    "yield_dNch_deta_per_eta": {
        "core_fn": _wrap_yield_per_eta,
        "prepare_data_2d": None,
        "prepare_data_3d": prepare_yield_3d_per_eta,
        "n_obs_per_cent": None,  # depends on n_eta bins
        "description": "dNch/dη(eta) per-eta-bin from 3D grid (matches PHOBOS dN/dη(eta) schema)",
    },
    "spectra_invariant_yield": {
        "core_fn": _wrap_spectra,
        "prepare_data_2d": prepare_spectra_2d,
        "prepare_data_3d": prepare_spectra_3d,
        "n_obs_per_cent": None,
        "description": "Invariant yield dN/(2π pT dpT dη)",
    },
    "mean_pt": {
        "core_fn": _wrap_mean_pt,
        "prepare_data_2d": prepare_mean_pt_2d,
        "prepare_data_3d": prepare_mean_pt_3d,
        "n_obs_per_cent": 1,
        "description": "Mean transverse momentum ⟨pT⟩",
    },
    "vn8_single": {
        "core_fn": lambda data, w: _core_vn8_single(data['Q1'], w),
        "prepare_data_2d": prepare_vn8_single_2d,
        "prepare_data_3d": prepare_vn8_single_3d,
        "n_obs_per_cent": 2,
        "description": "v_n{8} integrated (self-correlation corrections neglected, ~1/M suppressed)",
    },
    "vn6_single": {
        "core_fn": lambda data, w: _core_vn6_single(data['Q1'], w),
        "prepare_data_2d": prepare_vn6_single_2d,
        "prepare_data_3d": prepare_vn6_single_3d,
        "n_obs_per_cent": 2,
        "description": "v_n{6} integrated (self-correlation corrections neglected, ~1/M suppressed)",
    },
    "vn2_sp_diff_pT": {
        "core_fn": _wrap_vn2_sp_diff,
        "prepare_data_2d": prepare_vn2_sp_diff_2d,
        "prepare_data_3d": prepare_vn2_sp_diff_3d,
        "n_obs_per_cent": None,
        "description": "scalar-product differential v_n(pT){SP}",
    },
    "vn2_sp_diff_pT_means": {
        "core_fn": _wrap_vn2_sp_diff_means,
        "prepare_data_2d": prepare_vn2_sp_diff_2d,
        "prepare_data_3d": prepare_vn2_sp_diff_3d,
        "n_obs_per_cent": None,
        "description": "scalar-product differential v_n(pT){SP}, JETSCAPE pair-normalized form",
    },
    "vn2_sp_diff_eta": {
        "core_fn": _wrap_vn2_sp_diff,
        "prepare_data_2d": prepare_vn2_sp_diff_2d,
        "prepare_data_3d": prepare_vn2_sp_diff_3d,
        "n_obs_per_cent": None,
        "description": "scalar-product differential v_n(eta){SP}",
    },
    "vn2_sp_diff_eta_means": {
        "core_fn": _wrap_vn2_sp_diff_means,
        "prepare_data_2d": prepare_vn2_sp_diff_2d,
        "prepare_data_3d": prepare_vn2_sp_diff_3d,
        "n_obs_per_cent": None,
        "description": "scalar-product differential v_n(eta){SP}, JETSCAPE pair-normalized form",
    },
    "v1_odd_sp_diff_eta": {
        "core_fn": _wrap_v1_odd,
        "prepare_data_2d": None,        # boost-invariant 2D has no rapidity-odd flow
        "prepare_data_3d": prepare_v1_odd_3d,
        "n_obs_per_cent": None,
        "description": "rapidity-odd directed flow v1(eta) (eta-odd reference, sign convention)",
    },
    "vn2_ep_diff_eta": {
        "core_fn": _wrap_vn2_ep_diff_eta,
        "prepare_data_2d": None,
        "prepare_data_3d": prepare_vn2_ep_diff_3d,
        "n_obs_per_cent": None,
        "description": "event-plane differential v_n(eta){EP} (PHOBOS v2(eta) exception)",
    },
    "vn2_ep_diff_pT": {
        "core_fn": _wrap_vn2_ep_diff_pT,
        "prepare_data_2d": None,
        "prepare_data_3d": prepare_vn2_ep_diff_pT_3d,
        "n_obs_per_cent": None,
        "description": "event-plane differential v_n(pT){EP}",
    },
    "vn3_ep": {
        "core_fn": _wrap_vn3_ep,
        "prepare_data_2d": None,
        "prepare_data_3d": prepare_vn3_single_3d_ep,
        "n_obs_per_cent": 1,
        "description": "event-plane v3 integrated (STAR 2013qio v3(cent))",
    },
    "vn2_ep": {
        "core_fn": _wrap_vn2_ep,
        "prepare_data_2d": None,
        "prepare_data_3d": prepare_vn2_single_3d_ep,
        "n_obs_per_cent": 1,
        "description": "event-plane v2 integrated (fallback for event_plane records)",
    },
    "vn2_sp": {
        "core_fn": _wrap_vn2_sp,
        "prepare_data_2d": prepare_vn2_sp_2d,
        "prepare_data_3d": prepare_vn2_sp_3d,
        "n_obs_per_cent": 5,
        "description": "scalar-product v_n{SP} (JETSCAPE3D convention)",
    },
    "pT_fluct": {
        "core_fn": _wrap_pT_fluct,
        "prepare_data_2d": prepare_pT_fluct_2d,
        "prepare_data_3d": prepare_pT_fluct_3d,
        "n_obs_per_cent": 1,
        "description": "two-particle pT correlator sqrt(<dpTi dpTj>)/<pT>",
    },
    "dET_deta": {
        "core_fn": _wrap_dET,
        "prepare_data_2d": prepare_dET_2d,
        "prepare_data_3d": prepare_dET_3d,
        "n_obs_per_cent": 1,
        "description": "transverse energy per unit eta (grid only)",
    },
    "sc_mn": {
        "core_fn": _wrap_sc,
        "prepare_data_2d": prepare_sc_2d,
        "prepare_data_3d": prepare_sc_3d,
        "n_obs_per_cent": 1,
        "description": "Symmetric cumulant SC(n,m) single-acceptance, integrated",
    },
    "nac_mn": {
        "core_fn": _wrap_nac,
        "prepare_data_2d": prepare_nac_2d,
        "prepare_data_3d": prepare_nac_3d,
        "n_obs_per_cent": 1,
        "description": "Event-plane correlation NAC(n,m) subevent, integrated",
    },
}


# Quantities that count particles; every other key is normalized.
ABSOLUTE_KEYS = {'yield_dNch_deta', 'yield_dNch_deta_per_eta', 'spectra_invariant_yield', 'dET_deta'}


def _unit_weights(grid):
    """The grid with a charge-symmetry scaling undone.

    A positives-only grid selected as 'charged' (or a 'p' grid as
    'p + pbar') is scaled by grid.charge_factor.  Absolute quantities need
    that; normalized ones -- flow coefficients, correlations, mean pT and
    its fluctuations -- do not, and their self-correlation terms assume
    unit-weighted particles, so on a scaled grid they come out biased
    (v_n{4}, the event-plane amplitudes, the scalar-product v_n(eta) and pT
    fluctuations by factors up to ~6; tests/test_charge_scaled_grids.py).
    """
    cf = getattr(grid, 'charge_factor', 1.0)
    if cf == 1.0 or not isinstance(grid, QGridData):
        return grid
    from analysis.qvector import _scale_grid
    out = _scale_grid(grid, 1.0 / cf)
    out.charge_factor = 1.0
    for nm in ('pt_integrated', 'mass'):
        if hasattr(grid, nm):
            setattr(out, nm, getattr(grid, nm))
    return out


def _on_unit_weights(prepare):
    def wrapped(grid, params):
        return prepare(_unit_weights(grid), params)
    wrapped.__name__ = prepare.__name__
    wrapped.__doc__ = prepare.__doc__
    return wrapped


for _key, _entry in PIPELINE_DISPATCH.items():
    if _key not in ABSOLUTE_KEYS and _entry.get('prepare_data_3d') is not None:
        _entry['prepare_data_3d'] = _on_unit_weights(_entry['prepare_data_3d'])


def _with_charge_average(entry):
    """Absolute quantities of a charge average ((h+ + h-)/2, particle_id.
    charge_average) are half of the sum the species selection returns."""
    def tag(prepare):
        if prepare is None:
            return None

        def wrapped(container, params):
            d = prepare(container, params)
            if isinstance(d, dict):
                d['_charge_average'] = bool(params.get('charge_average'))
            return d
        wrapped.__name__, wrapped.__doc__ = prepare.__name__, prepare.__doc__
        return wrapped
    core = entry['core_fn']

    def core_fn(data, w):
        out = core(data, w)
        return out * 0.5 if isinstance(data, dict) and data.get('_charge_average') else out
    core_fn.__name__, core_fn.__doc__ = core.__name__, core.__doc__
    entry['prepare_data_2d'] = tag(entry.get('prepare_data_2d'))
    entry['prepare_data_3d'] = tag(entry.get('prepare_data_3d'))
    entry['core_fn'] = core_fn


for _key in ABSOLUTE_KEYS:
    _with_charge_average(PIPELINE_DISPATCH[_key])


def get_pipeline_entry(registered_key):
    """Look up a registered key in PIPELINE_DISPATCH.

    Returns the entry dict, or None if not found.
    """
    return PIPELINE_DISPATCH.get(registered_key)
