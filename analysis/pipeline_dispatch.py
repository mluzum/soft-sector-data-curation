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
    compute_vn2_sp_diff_means, _core_vn2_sp_diff_means,
    compute_vn2_single_per_cent_bin,
    compute_vn2_single_mor,
    compute_vn2_subevent_per_cent,
    compute_vn2_gap_poi,
    _core_vn4_gap,
    _core_vn2_gap_diff_pT_all_bins,
    _core_vn2_gap_diff_eta_all_bins,
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


def _poi_and_reference(params):
    """(POI particle_id, POI pT, {label: (reference particle_id, eta, pT)}).

    The POI is the record's particle_id in its top-level pT range; each
    subevent's reference is its own particle_id (default: charged hadrons)
    in its own eta and pT ranges."""
    sr = params.get('subevent_ranges') or {}
    labels = sorted(sr)
    if len(labels) != 2:
        raise ValueError("vn2_gap_poi needs exactly two subevents")
    poi_pt = (params.get('poi_range') or {}).get('pT') or params['single_range']['pT']
    ref = {}
    for lab in labels:
        pid = sr[lab].get('particle_id') or {'species': 'charged'}
        pt = sr[lab]['pT'] if sr[lab]['pT'][0] is not None else poi_pt
        ref[lab] = (pid, sr[lab]['eta'], pt)
    return params.get('particle_id'), poi_pt, labels, ref


def _poi_pt_bins(params, poi_pt):
    """POI pT windows: the differential bins of a pT-differential record,
    else the record's pT range."""
    if params.get('differential_var') == 'pT' and params.get('differential_bins'):
        return [tuple(b) for b in params['differential_bins']], True
    return [tuple(poi_pt)], False


def prepare_vn2_gap_poi_2d(qv, params):
    """Identified particle of interest with a separate reference, two subevents (2D).

    POI Q-vectors (record species, top-level pT) and reference Q-vectors
    (subevent species, default charged; subevent pT) in each subevent's
    eta window.  On the boost-invariant container every particle is in both
    windows, so the POI-reference and reference-reference self-pair terms
    (Sum w_A w_B over the particles common to both samples, with the samples'
    charge factors) are kept for the core to subtract.
    """
    poi_pid, poi_pt, labels, ref = _poi_and_reference(params)
    poi_idx, poi_cf, found = _get_species_selection(poi_pid, qv)
    if not found:
        raise ValueError(f"species {poi_pid!r} not in the container")
    pdg_ids = getattr(qv, 'pdg_ids', None)
    if poi_idx is None:
        poi_idx = list(range(qv.Qn.shape[1]))
    pt_bins, diff = _poi_pt_bins(params, poi_pt)
    centers = qv.pt_centers

    def poi_slice(lo, hi):
        if diff:            # differential bins: the sim cells whose centres fall in the bin
            return int(np.searchsorted(centers, lo)), int(np.searchsorted(centers, hi, side='right'))
        return _find_pt_slice_edges(qv.pt_bin_edges, lo, hi)

    out, jacs, ref_sel, cuts = {}, {}, {}, {}
    for X, lab in zip('AB', labels):
        rpid, eta, rpt = ref[lab]
        if is_charged(rpid.get('species')):
            ridx, rcf = charged_species_indices(pdg_ids)
        else:
            ridx, rcf, ok = _get_species_selection(rpid, qv)
            if not ok:
                raise ValueError(f"reference species {rpid!r} not in the container")
        Qn_cut, jac = cut_qvectors(qv, eta[0], eta[1])
        rlo, rhi = _find_pt_slice_edges(qv.pt_bin_edges, rpt[0], rpt[1])
        out[f'Q_{X}'] = rcf * Qn_cut[:, ridx, :, rlo:rhi].sum(axis=(1, 3))
        cuts[X], jacs[X] = Qn_cut, jac
        ref_sel[X] = (list(ridx), rcf, rlo, rhi)

    def cross(idx, factor, lo, hi):
        if not idx or hi <= lo:
            return np.zeros(qv.n_events)
        return _compute_dq0_cross_species(qv, jacs['A'], jacs['B'], idx,
                                          np.sqrt(factor), lo, hi)

    p = {'A': [], 'B': []}
    d_pQ = []
    for blo, bhi in pt_bins:
        plo, phi_ = poi_slice(blo, bhi)
        for X in 'AB':
            p[X].append(poi_cf * cuts[X][:, poi_idx, :, plo:phi_].sum(axis=(1, 3)))
        # POI in A with the reference in B (and B with A): common particles
        # are the POI species that are also reference species, in the common
        # pT range.  Symmetric in A <-> B.
        terms = []
        for Y in ('B', 'A'):
            ridx, rcf, rlo, rhi = ref_sel[Y]
            common = [i for i in poi_idx if i in ridx]
            terms.append(cross(common, poi_cf * rcf, max(plo, rlo), min(phi_, rhi)))
        d_pQ.append(0.5 * (terms[0] + terms[1]))
    out['p_A'] = np.stack(p['A'], axis=-1)        # (n_events, n_harm, n_bins)
    out['p_B'] = np.stack(p['B'], axis=-1)
    out['d_pQ'] = np.stack(d_pQ, axis=-1)          # (n_events, n_bins)
    ia, cfa, la, ha = ref_sel['A']
    ib, cfb, lb, hb = ref_sel['B']
    common_ref = [i for i in ia if i in ib]
    out['d_QQ'] = cross(common_ref, cfa * cfb, max(la, lb), min(ha, hb))
    return out


def prepare_vn2_gap_poi_3d(container, params):
    """Identified particle of interest with a separate reference, two subevents (3D).

    Receives the unselected container (selects_species): the POI and the
    reference are selected from it separately, on unit weights.  Disjoint
    subevents share no particles, so the self-pair terms vanish; overlapping
    windows are refused.
    """
    poi_pid, poi_pt, labels, ref = _poi_and_reference(params)
    (ea0, ea1), (eb0, eb1) = ref[labels[0]][1], ref[labels[1]][1]
    if min(ea1, eb1) > max(ea0, eb0) + 1e-9:
        raise ValueError("vn2_gap_poi on a 3D grid needs disjoint subevents")
    poi = _unit_weights(container.select(poi_pid))
    pt_bins, _ = _poi_pt_bins(params, poi_pt)
    out = {}
    for X, lab in zip('AB', labels):
        rpid, eta, rpt = ref[lab]
        rgrid = _unit_weights(container.select(rpid))
        out[f'p_{X}'] = np.stack([poi.acceptance(eta[0], eta[1], lo, hi) for lo, hi in pt_bins], axis=-1)
        out[f'Q_{X}'] = rgrid.acceptance(eta[0], eta[1], rpt[0], rpt[1])
    n_ev = out['Q_A'].shape[0]
    out['d_pQ'] = np.zeros((n_ev, len(pt_bins)))
    out['d_QQ'] = np.zeros(n_ev)
    return out


def _wrap_vn2_gap_poi(data, w):
    """(n_harm-1) values for one POI window, or harmonic-major
    (h-1)*n_bins + b for pT bins."""
    n_bins = data['p_A'].shape[-1]
    cols = [compute_vn2_gap_poi(data['p_A'][..., b], data['p_B'][..., b], data['Q_A'], data['Q_B'],
                                data['d_pQ'][:, b], data['d_QQ'], w) for b in range(n_bins)]
    return np.stack(cols, axis=1).ravel()


# ── v_n{2}, v_n{4} of a particle of interest (analysis.correlators) ─────────
#
# The particle of interest (POI) is the record's species in the record's
# acceptance (each differential bin narrows it); the reference is charged
# hadrons in the whole acceptance (pT: flow_reference.pT when given).  For
# a charged record POI and reference coincide and the results are the
# integrated v_n{2}, v_n{4}; identified records get differential flow with
# a charged reference (they used to get the charged-hadron value in 2D, and
# an identified-only reference in 3D).

def _cumulant_windows(params):
    """(reference (eta, pT), [POI (eta, pT) per bin], differential?)."""
    sk = params.get('single_range', {})
    eta = tuple(sk.get('eta', (-1, 1)))
    pt = tuple(sk.get('pT', (0.1, 10)))
    ref_pt = tuple(params['flow_reference_pT']) if params.get('flow_reference_pT') else pt
    var, bins = params.get('differential_var'), params.get('differential_bins') or []
    if var == 'pT' and bins:
        return (eta, ref_pt), [(eta, (lo, hi)) for lo, hi in bins], True
    if var == 'eta' and bins:
        return (eta, ref_pt), [((lo, hi), pt) for lo, hi in bins], True
    return (eta, ref_pt), [(eta, pt)], False


def _poi_species(params):
    pid = params.get('particle_id') or {}
    return pid if pid.get('species') is not None else {'species': 'charged'}


def _stack_bins(per_bin):
    """List of per-bin POI dicts -> arrays with a trailing bin axis."""
    return {k: np.stack([b[k] for b in per_bin], axis=-1) for k in per_bin[0]}


def prepare_cumulant_poi_2d(qv, params, order):
    from analysis.correlators import Container2D, correlator_arrays
    pdg_ids = getattr(qv, 'pdg_ids', None)
    pid = _poi_species(params)
    if is_charged(pid.get('species')):
        poi_idx, _ = charged_species_indices(pdg_ids)
    else:
        poi_idx, _, found = _get_species_selection(pid, qv)
        if not found:
            raise ValueError(f"species {pid.get('species')!r} not in the container")
        if poi_idx is None:
            poi_idx = list(range(qv.Qn.shape[1]))
    ref_idx, _ = charged_species_indices(pdg_ids)
    (r_eta, r_pt), bins, diff = _cumulant_windows(params)
    centers = qv.pt_centers

    def sample(idx, eta, pt, by_centers):
        if by_centers:      # differential bins: the sim cells whose centres fall in the bin
            lo, hi = int(np.searchsorted(centers, pt[0])), int(np.searchsorted(centers, pt[1], side='right'))
        else:
            lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, pt[0], pt[1])
        return {'idx': list(idx), 'pt': (lo, hi),
                'jac': _make_jacobian_list(qv.pt_centers, qv.masses, eta[0], eta[1])}

    ref = sample(ref_idx, r_eta, r_pt, False)
    n_harm = qv.Qn.shape[2]
    out = correlator_arrays(Container2D(qv, ref, ref), n_harm, order, which=('ref',))
    per_bin = [correlator_arrays(Container2D(qv, sample(poi_idx, eta, pt, diff and params.get('differential_var') == 'pT'), ref),
                                 n_harm, order, which=('poi',)) for eta, pt in bins]
    out.update(_stack_bins(per_bin))
    return out


def _poi_within_charged(container, pid):
    """Are the POI particles reference (charged) particles?"""
    from analysis.qvector import MultiSpeciesGrid
    sp = pid.get('species')
    if is_charged(sp) or not isinstance(container, MultiSpeciesGrid):
        return True          # a grid without species returns the combined charged grid
    pdgs = resolve_species(sp)[1]
    if pdgs is None:
        from analysis.species import _ISOSPIN, _PDG_SETS
        key = str(sp).strip().lower()
        pdgs = _PDG_SETS.get(key) or (_ISOSPIN[key][1] if key in _ISOSPIN else [])
    return bool(pdgs) and all(abs(int(p)) in (211, 321, 2212) for p in pdgs)


def prepare_cumulant_poi_3d(container, params, order):
    """3D: receives the unselected container (selects_species)."""
    from analysis.correlators import Container3D, correlator_arrays
    pid = _poi_species(params)
    poi = _unit_weights(container.select(pid))
    ref = _unit_weights(container.select({'species': 'charged'}))
    overlap = _poi_within_charged(container, pid)
    (r_eta, r_pt), bins, _ = _cumulant_windows(params)
    rw = (r_eta[0], r_eta[1], r_pt[0], r_pt[1])
    n_harm = ref.Qn.shape[-1]
    out = correlator_arrays(Container3D(ref, rw, ref, rw, True), n_harm, order, which=('ref',))
    per_bin = [correlator_arrays(Container3D(poi, (eta[0], eta[1], pt[0], pt[1]), ref, rw, overlap),
                                 n_harm, order, which=('poi',)) for eta, pt in bins]
    out.update(_stack_bins(per_bin))
    return out


def _poi_core(order, layout):
    """Core for one centrality class.  layout: 'integrated' -> (n_harm-1,);
    'h_major' -> (h-1)*n_bins + b, the layout run_pipeline reads."""
    from analysis.correlators import vn_poi

    def core(data, w):
        v = vn_poi(data, w, order)                     # (n_harm-1, n_bins)
        if layout == 'integrated':
            return v[:, 0]
        return v.ravel()
    return core


def _poi_prepares(order):
    def p2(qv, params):
        return prepare_cumulant_poi_2d(qv, params, order)

    def p3(container, params):
        return prepare_cumulant_poi_3d(container, params, order)
    p2.__name__, p3.__name__ = f'prepare_vn{order}_poi_2d', f'prepare_vn{order}_poi_3d'
    return p2, p3


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

def _prepare_vn2_gap_diff_3d(grid, params):
    """For vn2_gap_diff_pT/eta: per-bin params narrow subevent's range."""
    return prepare_vn2_gap_3d(grid, params)

# ── core_fn wrappers (uniform interface: data_dict, w) ────────────────

def _wrap_vn2_single_mor(data, w):
    return compute_vn2_single_mor(data['Qn'], data['dQ0'], w)


def _wrap_vn2_gap(data, w):
    return compute_vn2_subevent_per_cent(data['Qn_A'], data['Qn_B'], data['dQ_cross'], w)


def _wrap_vn4_gap(data, w):
    if 'Qn_A' in data and 'Qn' not in data:
        raise ValueError("vn4 gap requires higher-power Q-vectors (Q1_A..Q4_B); "
                         "not available from 3D data")
    return _core_vn4_gap(
        data['Q1_A'], data['Q2_A'], data['Q3_A'], data['Q4_A'],
        data['Q1_B'], data['Q2_B'], data['Q3_B'], data['Q4_B'],
        data['dQ_cross'], w,
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
        # partial cells by overlap fraction: measurement bins need not
        # follow the grid (a 0.2-0.3 bin on 0.15/0.25/0.35 cells)
        Qn = grid.acceptance_fractional(eta[0], eta[1], lo, hi)
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


# ── radial flow: v0, v0(pT), normalized covariance (analysis.radial_flow) ──

def _radial_flow_settings(params):
    sr = params.get('subevent_ranges') or {}
    if sorted(sr) != ['A', 'B']:
        raise ValueError("radial_flow needs subevents A and B")
    spec = params.get('poi_range', {}).get('pT') or (None, None)
    ref = params.get('reference_pT')
    if spec[0] is None or spec[1] is None or not ref:
        raise ValueError("radial_flow needs kinematic_range.pT and reference_pT")
    bins = [tuple(b) for b in (params.get('differential_bins') or [])] \
        if params.get('dimensionality') == 'differential_pT' else []
    mb = params.get('multiplicity_baseline') or None
    return sr, tuple(spec), tuple(ref), bins, mb


def _radial_flow_data(comp, params, mb):
    comp['quantity'] = params.get('quantity')
    comp['use_multiplicity'] = mb is not None
    return comp


def prepare_radial_flow_3d(grid, params):
    """Unit-weight cells of the charged grid: eta-window fractions per
    subevent, pT-range fractions (uniform in pT^2) per range and bin."""
    from analysis.qvector import _overlap_fractions, _pt_squared
    from analysis import radial_flow
    sr, spec, ref, bins, mb = _radial_flow_settings(params)
    if getattr(grid, 'pt_integrated', False):
        raise ValueError("radial_flow needs pT-binned grids")
    n_ev, n_eta, n_pt = grid.Qn.shape[:3]
    c = grid.Qn[..., 0].real
    mids = 0.5 * (grid.pt_edges[:-1] + grid.pt_edges[1:])
    s1 = grid.sum_pt if getattr(grid, 'sum_pt', None) is not None else c * mids
    s2 = grid.sum_pt2 if getattr(grid, 'sum_pt2', None) is not None else c * mids ** 2

    def eta_w(windows):
        fe = sum(_overlap_fractions(grid.eta_edges, lo, hi) for lo, hi in windows)
        return np.repeat(fe, n_pt)[None, :]

    def pt_f(lo, hi):
        return np.tile(_overlap_fractions(grid.pt_edges, lo, hi, _pt_squared), n_eta)

    W = {X: eta_w([sr[X]['eta']]) for X in 'AB'}
    F = {'spec': pt_f(*spec), 'ref': pt_f(*ref)}
    Wm = Fm = None
    if mb:
        Wm = eta_w([(e['min'], e['max']) for e in mb['eta']])
        mpt = mb.get('pT') or {}
        Fm = pt_f(mpt.get('min', spec[0]), mpt.get('max', spec[1]))
    K = n_eta * n_pt
    comp = radial_flow.components(c.reshape(n_ev, K), s1.reshape(n_ev, K), s2.reshape(n_ev, K), W, F,
                                  [pt_f(lo, hi) for lo, hi in bins], Wm, Fm)
    return _radial_flow_data(comp, params, mb)


def prepare_radial_flow_2d(qv, params):
    """Charged species of the boost-invariant container: each subevent
    weights a particle by its window's Jacobian (2 rap_max nsamples per
    event), so every particle is in both subevents and the core removes
    its pairs with itself.  Charge factors cancel (every term is a ratio)."""
    from analysis.qvector import _overlap_fractions, _pt_squared
    from analysis import radial_flow
    sr, spec, ref, bins, mb = _radial_flow_settings(params)
    idx, _cf = charged_species_indices(getattr(qv, 'pdg_ids', None))
    idx = list(idx)
    n_ev, n_pt = qv.n_events, qv.Qn.shape[3]
    counts = qv.Qn[:, :, 0, :].real
    c = counts[:, idx, :]
    mids = 0.5 * (qv.pt_bin_edges[:-1] + qv.pt_bin_edges[1:])
    s1 = (qv.sum_pt if getattr(qv, 'sum_pt', None) is not None else counts * mids)[:, idx, :]
    s2 = (qv.sum_pt2 if getattr(qv, 'sum_pt2', None) is not None else counts * mids ** 2)[:, idx, :]
    norm = (2.0 * qv.rap_max * qv.nsamples)[:, None]
    K = len(idx) * n_pt

    def eta_w(windows):
        jac = sum(_make_jacobian_list(qv.pt_centers, qv.masses, lo, hi) for lo, hi in windows)
        return jac[idx, :].reshape(1, K) / norm

    def pt_f(lo, hi):
        return np.tile(_overlap_fractions(qv.pt_bin_edges, lo, hi, _pt_squared), len(idx))

    W = {X: eta_w([sr[X]['eta']]) for X in 'AB'}
    F = {'spec': pt_f(*spec), 'ref': pt_f(*ref)}
    Wm = Fm = None
    if mb:
        Wm = eta_w([(e['min'], e['max']) for e in mb['eta']])
        mpt = mb.get('pT') or {}
        Fm = pt_f(mpt.get('min', spec[0]), mpt.get('max', spec[1]))
    comp = radial_flow.components(c.reshape(n_ev, K), s1.reshape(n_ev, K), s2.reshape(n_ev, K), W, F,
                                  [pt_f(lo, hi) for lo, hi in bins], Wm, Fm)
    return _radial_flow_data(comp, params, mb)


def _wrap_radial_flow(data, w):
    from analysis import radial_flow
    return radial_flow.core(data, w)


# ── Symmetric-cumulant SC(n,m) prepares ─────────────────────────────────────


def _sc_settings(params):
    """(m, n, eta window, pT window, normalization gap or None) of an SC record."""
    harms = [int(h) for h in (params.get('harmonics') or [])]
    if len(harms) != 2:
        raise ValueError(f"symmetric cumulant of harmonics {harms}: only SC(m, n) is implemented")
    sk = params.get('single_range', {})
    eta = tuple(sk.get('eta', (-0.8, 0.8)))
    pt = tuple(sk.get('pT', (0.2, 5.0)))
    if eta[0] is None or eta[1] is None:
        raise ValueError("symmetric cumulant needs kinematic_range.eta")
    gap = params.get('normalization_eta_gap')
    return harms[0], harms[1], eta, pt, gap


def _sc_check_harmonics(m, n, n_harm):
    if m + n >= n_harm:
        raise ValueError(f"SC({m},{n}) needs Q_{m + n}; the container stores harmonics below {n_harm}")


def _sc_gap_windows(eta, gap):
    mid = 0.5 * (eta[0] + eta[1])
    return (eta[0], mid - gap / 2), (mid + gap / 2, eta[1])


def prepare_sc_2d(qv, params):
    """SC(m, n) on the 2D container: charged particles with the window's
    Jacobian weights; distinct-particle sums (analysis.correlators), so the
    weights' self-correlations are removed exactly.  With a normalization
    gap, <<2>>_m and <<2>>_n of the normalized SC come from the two halves
    of the window separated by the gap."""
    from analysis.correlators import Container2D, sc_arrays, two_particle_gap
    m, n, eta, pt, gap = _sc_settings(params)
    _sc_check_harmonics(m, n, qv.Qn.shape[2])
    idx, _cf = charged_species_indices(getattr(qv, 'pdg_ids', None))
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, pt[0], pt[1])

    def sample(window):
        return {'idx': list(idx), 'pt': (lo, hi), 'jac': _make_jacobian_list(qv.pt_centers, qv.masses, *window)}
    full = sample(eta)
    out = sc_arrays(Container2D(qv, full, full), m, n)
    if gap is not None:
        a, b = _sc_gap_windows(eta, gap)
        c = Container2D(qv, sample(a), sample(b))
        out['G2m'], out['GD'] = two_particle_gap(c, m)
        out['G2n'], _ = two_particle_gap(c, n)
    out['normalized'] = bool(params.get('normalized'))
    return out


def prepare_sc_3d(grid, params):
    """SC(m, n) on a unit-weight charged grid (see prepare_sc_2d)."""
    from analysis.correlators import Container3D, sc_arrays, two_particle_gap
    m, n, eta, pt, gap = _sc_settings(params)
    _sc_check_harmonics(m, n, grid.Qn.shape[-1])
    win = (eta[0], eta[1], pt[0], pt[1])
    out = sc_arrays(Container3D(grid, win, grid, win, True), m, n)
    if gap is not None:
        a, b = _sc_gap_windows(eta, gap)
        c = Container3D(grid, (a[0], a[1], pt[0], pt[1]), grid, (b[0], b[1], pt[0], pt[1]), False)
        out['G2m'], out['GD'] = two_particle_gap(c, m)
        out['G2n'], _ = two_particle_gap(c, n)
    out['normalized'] = bool(params.get('normalized'))
    return out


def _wrap_sc(data, w):
    """SC(m, n) = <<4>> - <<2>>_m <<2>>_n (event weights: the numbers of
    distinct 4-tuples and pairs); normalized: divided by <<2>>_m <<2>>_n,
    from the gapped subevents when the record gives a gap."""
    from analysis.correlators import _avg
    four = _avg(data['N4'], data['D4'], w)
    c2m, c2n = _avg(data['N2m'], data['D2'], w), _avg(data['N2n'], data['D2'], w)
    sc = four - c2m * c2n
    if not data.get('normalized'):
        return np.array([sc])
    if 'G2m' in data:
        c2m, c2n = _avg(data['G2m'], data['GD'], w), _avg(data['G2n'], data['GD'], w)
    return np.array([sc / (c2m * c2n)])


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




# ── scalar-product / event-plane differential v_n of a particle of interest ──
#
# POI: the record's species in the record's acceptance, narrowed by each
# differential bin.  Reference: charged hadrons in the reference windows --
# the comparison override `sp_ref_eta`, else the record's flow_reference
# (uniform, mirrored or segmented per bin), else the record's own window --
# with the reference block's pT, else the record's.  Self-correlations:
# |Q_ref|^2 loses Sum w_ref^2 (`dQ0_ref`), and Re<q_poi Q_ref*> loses the
# particles that are in both samples (`dQ_cross`).
#
# Until November 2026 the 2D prepare ignored the species and the
# flow_reference and subtracted Sum w instead of Sum w^2 from |Q_ref|^2;
# the 3D prepare took the reference from the species-selected grid (an
# identified-only reference) and, for a uniform declared reference that
# overlaps the POI (STAR's TPC halves), kept the POI's own particles in
# the numerator.

def _sp_bins(params):
    """(var, POI bins, record eta, record pT); a record without
    differential bins is one bin, its pT range."""
    sk = params.get('single_range', {})
    eta = tuple(sk.get('eta', (-1, 1)))
    pt = tuple(sk.get('pT', (0.1, 10)))
    var = params.get('differential_var') or 'pT'
    bins = [tuple(b) for b in (params.get('differential_bins') or [])]
    if not bins:
        var, bins = 'pT', [pt]
    return var, bins, eta, pt


def _sp_ref_windows(bins, var, eta, pt, params):
    """Per POI bin: (reference eta windows, reference pT), or None when the
    bin has no reference (left NaN, reported unavailable)."""
    override = params.get('sp_ref_eta')
    if override is not None:
        if (isinstance(override, (list, tuple)) and len(override) == 2
                and all(isinstance(x, (list, tuple)) for x in override)):
            wins = [(float(a), float(b)) for a, b in override]
        else:
            wins = [tuple(float(x) for x in override)]
        return [(wins, tuple(params.get('sp_ref_pt', pt)))] * len(bins)
    fr = params.get('flow_reference')
    if not fr:
        return [([eta], pt)] * len(bins)
    out = []
    for lo_b, hi_b in bins:
        p_lo, p_hi = (lo_b, hi_b) if var == 'eta' else eta
        ref = _ref_segment_for(fr, p_lo, p_hi)
        windows = [tuple(w) for w in (ref.get('eta') or [])]
        if ref.get('mirror') and windows:
            # "the subevent on the opposite side of midrapidity": the
            # declared window taken on the side opposite the POI bin; a bin
            # straddling eta = 0 gets both images.
            lo_w, hi_w = windows[0]
            if p_lo < 0 < p_hi:
                windows = [(-hi_w, -lo_w), (lo_w, hi_w)]
            elif p_hi > 0:
                windows = [(-hi_w, -lo_w)]
            else:
                windows = [(lo_w, hi_w)]
        out.append((windows, tuple(ref.get('pT') or pt)) if windows else None)
    return out


def _sp_poi_box(b, bins, var, eta, pt):
    lo_b, hi_b = bins[b]
    return ((lo_b, hi_b), pt) if var == 'eta' else (eta, (lo_b, hi_b))


def _sp_assemble(specs, ref_of, poi_of, shared_of, n_ev, H):
    """Arrays for _wrap_vn2_sp_diff from per-bin pieces.  ref_of(spec) ->
    (q, n, dq0); poi_of(b) -> (q, n); shared_of(b, spec) -> (n_ev,)."""
    nb = len(specs)
    q_poi = np.full((n_ev, H, nb), np.nan, complex)
    n_poi = np.full((n_ev, nb), np.nan)
    dq_cross = np.full((n_ev, nb), np.nan)
    uniform = all(sp == specs[0] for sp in specs) and specs[0] is not None
    cache = {}

    def ref(spec):
        key = (tuple(spec[0]), spec[1])
        if key not in cache:
            cache[key] = ref_of(spec)
        return cache[key]
    if uniform:
        q_ref, n_ref, dq0_ref = ref(specs[0])
    else:
        q_ref = np.full((n_ev, H, nb), np.nan, complex)
        n_ref = np.full((n_ev, nb), np.nan)
        dq0_ref = np.full((n_ev, nb), np.nan)
    for b, spec in enumerate(specs):
        q_poi[:, :, b], n_poi[:, b] = poi_of(b)
        if spec is None:
            continue
        if not uniform:
            q_ref[:, :, b], n_ref[:, b], dq0_ref[:, b] = ref(spec)
        dq_cross[:, b] = shared_of(b, spec)
    return {'q_poi': q_poi, 'n_poi': n_poi, 'q_ref': q_ref, 'n_ref': n_ref,
            'dQ0_ref': dq0_ref, 'dQ_cross': dq_cross}


def prepare_vn2_sp_diff_2d(qv, params):
    """2D: POI and reference from the species axis, Jacobian weights.

    On the boost-invariant container every particle lies in every eta
    window with the window's weight, so the POI-reference diagonal is
    Sum w_poi w_ref over the particles of both samples (common species,
    common pT cells), with the samples' charge factors."""
    var, bins, eta, pt = _sp_bins(params)
    if eta[0] is None or eta[1] is None:
        nb, H = len(bins), qv.Qn.shape[2]
        return {'q_poi': np.full((qv.n_events, H, nb), np.nan, complex),
                'n_poi': np.full((qv.n_events, nb), np.nan),
                'q_ref': np.full((qv.n_events, H), np.nan, complex),
                'n_ref': np.full(qv.n_events, np.nan)}
    pid = _poi_species(params)
    pdg_ids = getattr(qv, 'pdg_ids', None)
    if is_charged(pid.get('species')):
        poi_idx, poi_cf = charged_species_indices(pdg_ids)
    else:
        poi_idx, poi_cf, found = _get_species_selection(pid, qv)
        if not found:
            raise ValueError(f"species {pid.get('species')!r} not in the container")
        if poi_idx is None:
            poi_idx = list(range(qv.Qn.shape[1]))
    ref_idx, ref_cf = charged_species_indices(pdg_ids)
    poi_idx, ref_idx = list(poi_idx), list(ref_idx)
    common = [i for i in poi_idx if i in ref_idx]
    q0 = qv.Qn[:, :, 0, :].real
    norm = (2.0 * qv.rap_max * qv.nsamples)[:, None, None]
    centers = qv.pt_centers

    def weights(windows):
        jac = sum(_make_jacobian_list(centers, qv.masses, lo, hi) for lo, hi in windows)
        return jac[None, :, :] / norm                      # (n_ev, n_species, n_pt)

    def qsum(idx, w, sl, cf):
        return cf * np.einsum('esp,eshp->eh', w[:, idx, sl], qv.Qn[:, idx, :, sl])

    def poi_slice(lo, hi, differential):
        if differential:    # the sim cells whose centres fall in the bin
            return int(np.searchsorted(centers, lo)), int(np.searchsorted(centers, hi, side='right'))
        return _find_pt_slice_edges(qv.pt_bin_edges, lo, hi)

    boxes = []
    for b in range(len(bins)):
        (e_lo, e_hi), (p_lo, p_hi) = _sp_poi_box(b, bins, var, eta, pt)
        boxes.append((weights([(e_lo, e_hi)]), poi_slice(p_lo, p_hi, var == 'pT')))

    def poi_of(b):
        w, (lo, hi) = boxes[b]
        sl = slice(lo, hi)
        return qsum(poi_idx, w, sl, poi_cf), poi_cf * (w[:, poi_idx, sl] * q0[:, poi_idx, sl]).sum(axis=(1, 2))

    def ref_parts(spec):
        windows, rpt = spec
        w = weights(windows)
        lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, rpt[0], rpt[1])
        return w, lo, hi

    def ref_of(spec):
        w, lo, hi = ref_parts(spec)
        sl = slice(lo, hi)
        n = ref_cf * (w[:, ref_idx, sl] * q0[:, ref_idx, sl]).sum(axis=(1, 2))
        dq0 = ref_cf ** 2 * (w[:, ref_idx, sl] ** 2 * q0[:, ref_idx, sl]).sum(axis=(1, 2))
        return qsum(ref_idx, w, sl, ref_cf), n, dq0

    def shared_of(b, spec):
        wp, (plo, phi_) = boxes[b]
        wr, rlo, rhi = ref_parts(spec)
        lo, hi = max(plo, rlo), min(phi_, rhi)
        if not common or hi <= lo:
            return np.zeros(qv.n_events)
        sl = slice(lo, hi)
        return poi_cf * ref_cf * (wp[:, common, sl] * wr[:, common, sl] * q0[:, common, sl]).sum(axis=(1, 2))

    specs = _sp_ref_windows(bins, var, eta, pt, params)
    return _sp_assemble(specs, ref_of, poi_of, shared_of, qv.n_events, qv.Qn.shape[2])


def prepare_vn2_sp_diff_3d(container, params):
    """3D: receives the unselected container (selects_species); POI and
    reference are selected from it on unit weights.  The diagonal is the
    count of POI particles inside the reference (cells common to the POI
    box and a reference window), or zero for a POI outside the charged
    reference (Lambda, K0_S, ...)."""
    pid = _poi_species(params)
    poi = _unit_weights(container.select(pid))
    ref = _unit_weights(container.select({'species': 'charged'}))
    var, bins, eta, pt = _sp_bins(params)
    H, n_ev, nb = ref.Qn.shape[-1], ref.n_events, len(bins)
    if (getattr(poi, 'pt_integrated', False) or getattr(ref, 'pt_integrated', False)
            or eta[0] is None or eta[1] is None):
        return {'q_poi': np.full((n_ev, H, nb), np.nan, complex),
                'n_poi': np.full((n_ev, nb), np.nan),
                'q_ref': np.full((n_ev, H), np.nan, complex),
                'n_ref': np.full(n_ev, np.nan)}
    overlap = _poi_within_charged(container, pid)

    def poi_of(b):
        (e_lo, e_hi), (p_lo, p_hi) = _sp_poi_box(b, bins, var, eta, pt)
        q = poi.acceptance(e_lo, e_hi, p_lo, p_hi)
        return q, q[:, 0].real

    def ref_of(spec):
        windows, rpt = spec
        q = sum(ref.acceptance(lo, hi, rpt[0], rpt[1]) for lo, hi in windows)
        return q, q[:, 0].real, q[:, 0].real

    def shared_of(b, spec):
        if not overlap:
            return np.zeros(n_ev)
        poi_eta, poi_pt = _sp_poi_box(b, bins, var, eta, pt)
        sh = _shared_count(poi, poi_eta, poi_pt, spec[0], spec[1])
        return np.zeros(n_ev) if sh is None else sh

    specs = _sp_ref_windows(bins, var, eta, pt, params)
    return _sp_assemble(specs, ref_of, poi_of, shared_of, n_ev, H)


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


def _wrap_vn2_sp_diff(data, w):
    # dQ0_ref (Sum w^2; = n_ref on unit weights) removes the reference's own
    # auto-correlation from |Q_ref|^2; dQ_cross removes the POI/reference
    # diagonal from the numerator.
    return _core_vn2_sp_diff(data['q_poi'], data['n_poi'],
                             data['q_ref'], w,
                             dQ_cross=data.get('dQ_cross'),
                             n_ref=data.get('dQ0_ref', data.get('n_ref')))


def _wrap_vn2_sp_diff_means(data, w):
    return _core_vn2_sp_diff_means(data['q_poi'], data['n_poi'],
                                   data['q_ref'], data['n_ref'], w,
                                   dQ_cross=data.get('dQ_cross'),
                                   dQ0_ref=data.get('dQ0_ref'))

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
        "core_fn": _poi_core(2, 'integrated'),
        "prepare_data_2d": _poi_prepares(2)[0],
        "prepare_data_3d": _poi_prepares(2)[1],
        "selects_species": True,
        "n_obs_per_cent": 5,
        "description": "v_n{2} single-acceptance integrated (POI vs charged reference)",
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
    "vn2_gap_poi": {
        "core_fn": _wrap_vn2_gap_poi,
        "prepare_data_2d": prepare_vn2_gap_poi_2d,
        "prepare_data_3d": prepare_vn2_gap_poi_3d,
        # The 3D prepare selects the POI and the reference itself: callers
        # pass the unselected container (see container_for_prepare).
        "selects_species": True,
        "n_obs_per_cent": 5,
        "description": "v_n{2} of identified particles, two subevents, separate reference",
    },
    "vn2_single_diff_pT": {
        "core_fn": _poi_core(2, 'h_major'),
        "prepare_data_2d": _poi_prepares(2)[0],
        "prepare_data_3d": _poi_prepares(2)[1],
        "selects_species": True,
        "n_obs_per_cent": None,  # depends on n_pT bins
        "description": "v_n{2} pT-differential",
    },
    "vn2_single_diff_eta": {
        "core_fn": _poi_core(2, 'h_major'),
        "prepare_data_2d": _poi_prepares(2)[0],
        "prepare_data_3d": _poi_prepares(2)[1],
        "selects_species": True,
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
        "core_fn": _poi_core(4, 'integrated'),
        "prepare_data_2d": _poi_prepares(4)[0],
        "prepare_data_3d": _poi_prepares(4)[1],
        "selects_species": True,
        "n_obs_per_cent": 5,
        "description": "v_n{4} single-acceptance integrated",
    },
    "vn4_single_diff_pT": {
        "core_fn": _poi_core(4, 'h_major'),
        "prepare_data_2d": _poi_prepares(4)[0],
        "prepare_data_3d": _poi_prepares(4)[1],
        "selects_species": True,
        "n_obs_per_cent": None,
        "description": "v_n{4} pT-differential",
    },
    "vn4_single_diff_eta": {
        "core_fn": _poi_core(4, 'h_major'),
        "prepare_data_2d": _poi_prepares(4)[0],
        "prepare_data_3d": _poi_prepares(4)[1],
        "selects_species": True,
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
        "selects_species": True,
    },
    "vn2_sp_diff_pT_means": {
        "core_fn": _wrap_vn2_sp_diff_means,
        "prepare_data_2d": prepare_vn2_sp_diff_2d,
        "prepare_data_3d": prepare_vn2_sp_diff_3d,
        "n_obs_per_cent": None,
        "description": "scalar-product differential v_n(pT){SP}, JETSCAPE pair-normalized form",
        "selects_species": True,
    },
    "vn2_sp_diff_eta": {
        "core_fn": _wrap_vn2_sp_diff,
        "prepare_data_2d": prepare_vn2_sp_diff_2d,
        "prepare_data_3d": prepare_vn2_sp_diff_3d,
        "n_obs_per_cent": None,
        "description": "scalar-product differential v_n(eta){SP}",
        "selects_species": True,
    },
    "vn2_sp_diff_eta_means": {
        "core_fn": _wrap_vn2_sp_diff_means,
        "prepare_data_2d": prepare_vn2_sp_diff_2d,
        "prepare_data_3d": prepare_vn2_sp_diff_3d,
        "n_obs_per_cent": None,
        "description": "scalar-product differential v_n(eta){SP}, JETSCAPE pair-normalized form",
        "selects_species": True,
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
    "radial_flow": {
        "core_fn": _wrap_radial_flow,
        "prepare_data_2d": prepare_radial_flow_2d,
        "prepare_data_3d": prepare_radial_flow_3d,
        "n_obs_per_cent": None,
        "description": "v0, v0(pT), normalized [pT]-spectrum covariance (two subevents; one value per pT bin)",
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


# Cores that return one value per differential bin, not [v_1..v_N] per bin:
# run_pipeline and the toy harness index them by bin alone.
SINGLE_ROW_KEYS = frozenset({
    'vn2_ep', 'vn3_ep', 'mean_pt', 'yield_dNch_deta', 'yield_dNch_deta_per_eta',
    'spectra_invariant_yield', 'dET_deta', 'sc_mn', 'nac_mn', 'radial_flow',
})


def container_for_prepare(container, params, entry):
    """The 3D container a prepare receives: species-selected by the record's
    particle_id, or unselected for entries that select species themselves
    (``selects_species``: a POI and a reference of different species)."""
    if entry.get('selects_species'):
        return container
    return container.select(params.get('particle_id'))


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
