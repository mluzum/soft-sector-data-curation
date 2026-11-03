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


def _window_sum(grid, arr, eta, pt, fractional=False):
    """Sum a per-cell array (n_events, n_axis, n_pt) over an eta-pT window, in
    double precision: whole cells by the snapped edges on eta grids
    (_grid_slices).  On a rapidity-binned grid the window is mapped per pT
    cell (QGridData.window_weights): fractional cell weights for linear sums
    (<pT>, E_T), whole cells by centre where pair terms need unit weights
    (pT fluctuations)."""
    if getattr(grid, 'rapidity_binned', False):
        return np.einsum('ijk,jk->i', arr, grid.window_weights(eta[0], eta[1], pt[0], pt[1],
                                                               fractional=fractional))
    se, sp_ = _grid_slices(grid, eta, pt)
    return arr[:, se, sp_].sum(axis=(1, 2), dtype=np.float64)


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
        out[f'p_{X}'] = np.stack([poi.poi_acceptance(eta[0], eta[1], lo, hi) for lo, hi in pt_bins], axis=-1)
        out[f'Q_{X}'] = rgrid.acceptance(eta[0], eta[1], rpt[0], rpt[1])
    n_ev = out['Q_A'].shape[0]
    out['d_pQ'] = np.zeros((n_ev, len(pt_bins)))
    if getattr(poi, 'rapidity_binned', False) and _poi_within_charged(container, poi_pid):
        # a y cell can straddle both eta windows: the expected POI(X) x
        # reference(other) self-pairs, averaged over the two orderings
        (pa, ea, rpa), (pb, eb, rpb) = ref[labels[0]], ref[labels[1]]
        for b, (lo, hi) in enumerate(pt_bins):
            out['d_pQ'][:, b] = 0.5 * (poi.shared((ea[0], ea[1], lo, hi), (eb[0], eb[1], rpb[0], rpb[1]))[:, 0].real
                                       + poi.shared((eb[0], eb[1], lo, hi), (ea[0], ea[1], rpa[0], rpa[1]))[:, 0].real)
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


def _pair_gap_arrays_3d(poi, ref, overlap, rw, bins, gap):
    """v_n{2} sums over pairs |Delta eta| > gap (eta_gap of a record without
    subevents: the two-particle correlation functions of ATLAS and CMS,
    2 < |Delta eta| < 5; STAR's Delta eta > 0.05 and > 1).  Per eta cell, the
    POI (per bin) and reference Q-vectors; pairs of cells i, j weighted by
    the fraction of their particle pairs beyond the gap
    (QGridData.pair_gap_fractions), self-pairs removed from the diagonal.
    Unit weights.  Same layout as correlator_arrays (N2p, D2p per bin; N2,
    D2), so the v_n{2} cores serve it unchanged."""
    if poi.rapidity_binned:
        raise ValueError('a pseudorapidity pair gap needs an eta-binned POI grid')
    F = ref.pair_gap_fractions(gap)
    fd = np.diag(F)
    Wr = ref.window_weights(*rw)
    Qr = np.einsum('eiph,ip->eih', ref.Qn, Wr).astype(np.complex128)
    Mr, Qr = Qr[..., 0].real, Qr[..., 1:]
    out = {'N2': np.real(np.einsum('eih,ij,ejh->eh', Qr, F, np.conj(Qr))) - (Mr @ fd)[:, None],
           'D2': np.einsum('ei,ij,ej->e', Mr, F, Mr) - Mr @ fd}
    n2p, d2p = [], []
    for eta, pt in bins:
        Wp = poi.window_weights(eta[0], eta[1], pt[0], pt[1])
        P = np.einsum('eiph,ip->eih', poi.Qn, Wp).astype(np.complex128)
        m, P = P[..., 0].real, P[..., 1:]
        shared = (np.einsum('eip,ip->ei', poi.Qn[..., 0].real, Wp * Wr) if overlap
                  else np.zeros_like(m))                     # POI particles that are reference particles
        n2p.append(np.real(np.einsum('eih,ij,ejh->eh', P, F, np.conj(Qr))) - (shared @ fd)[:, None])
        d2p.append(np.einsum('ei,ij,ej->e', m, F, Mr) - shared @ fd)
    out['N2p'] = np.stack(n2p, axis=-1)
    out['D2p'] = np.stack(d2p, axis=-1)
    return out


def prepare_cumulant_poi_3d(container, params, order):
    """3D: receives the unselected container (selects_species).  A record
    with eta_gap and no subevents (order 2) correlates only pairs beyond the
    gap (_pair_gap_arrays_3d; until November 2026 the gap was not applied
    on 3D containers).  On 2D containers the gap removes only the pairs of
    a particle with itself (the correlation is flat in rapidity), which the
    distinct-particle sums do already."""
    from analysis.correlators import Container3D, correlator_arrays
    pid = _poi_species(params)
    poi = _unit_weights(container.select(pid))
    ref = _unit_weights(container.select({'species': 'charged'}))
    overlap = _poi_within_charged(container, pid)
    (r_eta, r_pt), bins, _ = _cumulant_windows(params)
    rw = (r_eta[0], r_eta[1], r_pt[0], r_pt[1])
    gap = float(params.get('eta_gap') or 0.0)
    if order == 2 and gap > 0 and not params.get('subevent_ranges'):
        return _pair_gap_arrays_3d(poi, ref, overlap, rw, bins, gap)
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

    Each subevent sum enters the correlation linearly, so a window that
    cuts a cell takes the cell with its overlap fraction (uniform in eta):
    ALICE's |Delta eta| > 1.4 subevents, -0.8 < eta < -0.7 and 0.7 < eta <
    0.8, are half a cell of a 0.2-wide grid (until November 2026 whole cells
    by edge snapping: the first window took the cell [-0.8, -0.6], the
    second none, and v2 was 0).  The particles of a cell inside both
    windows are expected self-pairs, sum f_A f_B Q0 (dQ_cross; zero for
    disjoint windows).  Windows on cell edges are unchanged.
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
    Qn_A = grid.acceptance_fractional(eA[0], eA[1], pt0, pt1)
    Qn_B = grid.acceptance_fractional(eB[0], eB[1], pt0, pt1)
    WA = grid.window_weights(eA[0], eA[1], pt0, pt1, fractional=True)
    WB = grid.window_weights(eB[0], eB[1], pt0, pt1, fractional=True)
    shared = WA * WB
    if shared.any():
        dQ_cross = np.einsum('ijk,jk->i', grid.Qn[..., 0].real, shared)
    else:
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


def _yield_acceptance(grid, eta_lo, eta_hi, pt_lo, pt_hi):
    """Yield window on a 3D grid: fractional cell overlap when the grid
    provides it (QGridData.acceptance_fractional), else whole cells."""
    frac = getattr(grid, 'acceptance_fractional', None)
    return frac(eta_lo, eta_hi, pt_lo, pt_hi) if frac is not None else grid.acceptance(eta_lo, eta_hi, pt_lo, pt_hi)


def prepare_yield_3d(grid, params):
    """Extract Q0 from grid for dNch/dη, normalized by nsamples.

    The charge-symmetry factor is already baked into ``grid`` by
    ``MultiSpeciesGrid.select`` (grid.charge_factor), so no extra factor is
    applied here.  A raw combined grid carries charge_factor=1.
    """
    sk = params.get('single_range', {})
    eta = sk.get('eta', (-1, 1))
    pt = sk.get('pT', (0.1, 10))
    if eta[0] is None or eta[1] is None:
        # A rapidity window (dN/dy of an identified species): per pT cell,
        # the eta window of the species' mass (grid.mass, set by the species
        # selection).  A mixture (charged hadrons) has no single mass: NaN.
        # (Until November 2026 every rapidity-window yield was 0 on 3D.)
        y = sk.get('y') or (None, None)
        mass = getattr(grid, 'mass', None)
        if y[0] is None or y[1] is None or mass is None:
            return {'Q0': np.full(grid.n_events, np.nan), 'delta_eta': 1.0}
        Q0 = grid.acceptance_rapidity(y[0], y[1], float(mass), pt[0], pt[1])[:, 0].real / grid.nsamples
        return {'Q0': Q0, 'delta_eta': y[1] - y[0]}
    # Cells cut by the window count by overlap (a window off the grid --
    # e.g. a lab-frame window shifted into the container frame -- would
    # otherwise snap to whole cells while dividing by its own width).
    Qn = _yield_acceptance(grid, eta[0], eta[1], pt[0], pt[1])
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
        Q = _yield_acceptance(grid, e_lo, e_hi, pT_lo, pT_hi)
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
        Q0_pt = grid.spectrum_counts(y[0], y[1], meas_bins, var='y', mass=float(ymass)) / grid.nsamples[:, None]
        return {'Q0_pt': Q0_pt, 'pt_centers': np.array([(lo + hi) / 2 for lo, hi in meas_bins]),
                'pt_widths': np.array([hi - lo for lo, hi in meas_bins]), 'delta_eta': y[1] - y[0]}
    if eta[0] is None or eta[1] is None:
        return {'Q0_pt': None, 'pt_centers': None, 'pt_widths': None, 'delta_eta': 1.0}
    meas_bins = _extract_2d_pT_bins(params)
    # partial cells by overlap fraction: measurement bins need not follow the
    # grid (a 0.2-0.3 bin on 0.15/0.25/0.35 cells); one pass for all bins
    Q0_pt = grid.spectrum_counts(eta[0], eta[1], meas_bins) / grid.nsamples[:, None]
    pt_centers = np.array([(lo + hi) / 2 for lo, hi in meas_bins])
    pt_widths = np.array([hi - lo for lo, hi in meas_bins])

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


# Spectrum forms (particle_spectra value_type): the invariant yield
# 1/(2 pi pT) d2N/(dpT dy), or d2N/(dpT dy) itself.
SPECTRUM_FORMS = {'invariant_yield': 'invariant', 'dN_dpT_dy': 'dN_dpT', 'd2N_dpT_dy': 'dN_dpT'}


def _spectrum_form(prepare):
    """Tag the prepared data with the record's spectrum form."""
    def wrapped(container, params):
        d = prepare(container, params)
        if isinstance(d, dict):
            d['_spectrum_form'] = SPECTRUM_FORMS.get(params.get('value_type') or 'invariant_yield', 'invariant')
        return d
    wrapped.__name__, wrapped.__doc__ = prepare.__name__, prepare.__doc__
    return wrapped


def _wrap_spectra(data, w):
    if data.get('Q0_pt') is None:
        return np.array([])
    out = compute_spectra_invariant_yield(
        data['Q0_pt'], data['pt_centers'], data['pt_widths'],
        data['delta_eta'], w)
    if data.get('_spectrum_form') == 'dN_dpT':
        # d2N/(dpT dy) = 2 pi pT_c x the invariant yield of the bin
        # (exactly: the counts over the bin width and the window).
        out = out * 2.0 * np.pi * data['pt_centers']
    return out


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
    pt = sk.get('pT', (0.1, 10))
    if getattr(grid, 'sum_pt', None) is not None:
        num_src = grid.sum_pt
    else:
        mids = 0.5 * (grid.pt_edges[:-1] + grid.pt_edges[1:])
        num_src = grid.Qn[:, :, :, 0].real * mids[np.newaxis, np.newaxis, :]
    if eta[0] is None or eta[1] is None:
        # A rapidity window (<pT> of an identified species): fractional cell
        # weights of the window with the species' mass per pT cell; a mixture
        # has no mass.  (Until November 2026 every rapidity window gave 0/0.)
        y = sk.get('y') or (None, None)
        mass = getattr(grid, 'y_mass', None) if getattr(grid, 'rapidity_binned', False) \
            else getattr(grid, 'mass', None)
        if y[0] is None or y[1] is None or mass is None:
            return {'num': z, 'den': z}
        W = grid.window_weights(y[0], y[1], pt[0], pt[1], var='y', mass=float(mass), fractional=True)
        den = np.einsum('ijk,jk->i', grid.Qn[..., 0].real, W) / grid.nsamples
        num = np.einsum('ijk,jk->i', num_src, W) / grid.nsamples
        return {'num': num, 'den': den}
    den = _window_sum(grid, grid.Qn[..., 0].real, eta, pt, fractional=True) / grid.nsamples
    num = _window_sum(grid, num_src, eta, pt, fractional=True) / grid.nsamples
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
    ns = grid.nsamples
    n = _window_sum(grid, grid.Qn[..., 0].real, eta, pt) / ns
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
    s1 = _window_sum(grid, s1_src, eta, pt) / ns
    s2 = _window_sum(grid, s2_src, eta, pt) / ns
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
    if getattr(grid, 'rapidity_binned', False):
        raise ValueError("radial_flow on a rapidity-binned grid is not implemented")
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
        q = poi.poi_acceptance(e_lo, e_hi, p_lo, p_hi)
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


# ── rapidity-even v1: factorization of the v11 pair matrix (analysis.v1_even) ──

def _v1_even_layout(params):
    """The pT bins of the pair matrix and the roles of their indices:
    {'bins': [(lo, hi)], 'grid', 'targets': index tuples, 'sign_bin', 'sign',
    'absolute'}.  Targets: the record's bins (or its pT window); grid:
    v1_even.fit_pT_bins (default the targets); sign: the sign_convention
    window."""
    ve = params.get('v1_even') or {}
    pt = tuple((params.get('single_range') or {}).get('pT', (0.2, 3.0)))
    targets = [tuple(map(float, b)) for b in (params.get('differential_bins') or [pt])]
    grid = [tuple(map(float, b)) for b in (ve.get('fit_pT_bins') or targets)]
    sc = ve.get('sign_convention') or {}
    sign_win = (float(sc['pT']['min']), float(sc['pT']['max'])) if sc.get('pT') else None
    bins = []

    def index(b):
        if b not in bins:
            bins.append(b)
        return bins.index(b)
    g = tuple(index(b) for b in grid)
    t = tuple(index(b) for b in targets)
    s = index(sign_win) if (sign_win and not ve.get('absolute')) else None
    return {'bins': bins, 'grid': g, 'targets': t, 'sign_bin': s,
            'sign': -1.0 if sc.get('sign') == 'negative' else 1.0, 'absolute': bool(ve.get('absolute'))}


def prepare_v1_even_2d(qv, params):
    """Distinct-pair sums N[e, k, l] = sum cos(phi_i - phi_j), D = sum of pair
    weights, i in pT bin k, j in bin l, on the boost-invariant container:
    the pairs of a particle with itself (bins sharing cells) removed with
    the Jacobian weights.  The eta gap only removes such pairs here, since
    the correlation is flat in rapidity."""
    from analysis.correlators import Container2D
    lay = _v1_even_layout(params)
    pid = _poi_species(params)
    if is_charged(pid.get('species')):
        idx, _ = charged_species_indices(getattr(qv, 'pdg_ids', None))
    else:
        idx, _, found = _get_species_selection(pid, qv)
        if not found:
            raise ValueError(f"species {pid.get('species')!r} not in the container")
        idx = list(range(qv.Qn.shape[1])) if idx is None else idx
    if qv.Qn.shape[2] < 2:
        raise ValueError('rapidity-even v1 needs Q_1 in the container')
    eta = tuple((params.get('single_range') or {}).get('eta', (-1, 1)))
    jac = _make_jacobian_list(qv.pt_centers, qv.masses, eta[0], eta[1])
    centers = qv.pt_centers
    sets = [{'idx': list(idx), 'jac': jac,
             'pt': (int(np.searchsorted(centers, lo)), int(np.searchsorted(centers, hi, side='right')))}
            for lo, hi in lay['bins']]
    K, n_ev = len(sets), qv.Qn.shape[0]
    q = [Container2D(qv, s, s)(1, 1, 0) for s in sets]
    m = [Container2D(qv, s, s)(0, 1, 0).real for s in sets]
    N, D = np.zeros((n_ev, K, K)), np.zeros((n_ev, K, K))
    for k in range(K):
        for l in range(k, K):
            shared = Container2D(qv, sets[k], sets[l])(0, 1, 1).real
            N[:, k, l] = N[:, l, k] = np.real(q[k] * np.conj(q[l])) - shared
            D[:, k, l] = D[:, l, k] = m[k] * m[l] - shared
    return {'N': N, 'D': D, 'layout': lay}


def prepare_v1_even_3d(container, params):
    """Pair sums as prepare_v1_even_2d on unit-weight eta cells, the pairs
    of cells i, j weighted by the fraction of their particle pairs beyond
    eta_gap (QGridData.pair_gap_fractions); pairs of a particle with itself
    removed.  Receives the unselected container (selects_species)."""
    lay = _v1_even_layout(params)
    grid = _unit_weights(container.select(_poi_species(params)))
    if getattr(grid, 'rapidity_binned', False) or getattr(grid, 'pt_integrated', False):
        raise ValueError('rapidity-even v1 needs a pT-binned eta grid')
    if grid.Qn.shape[-1] < 2:
        raise ValueError('rapidity-even v1 needs Q_1 in the container')
    eta = tuple((params.get('single_range') or {}).get('eta', (-1, 1)))
    gap = float(params.get('eta_gap') or 0.0)
    n_ax = len(grid.axis_edges) - 1
    G = grid.pair_gap_fractions(gap) if gap > 0 else np.ones((n_ax, n_ax))
    gd = np.diag(G)
    Q = grid.Qn
    W = [grid.window_weights(eta[0], eta[1], lo, hi) for lo, hi in lay['bins']]
    q = [np.einsum('eip,ip->ei', Q[..., 1].astype(np.complex128), w) for w in W]
    m = [np.einsum('eip,ip->ei', Q[..., 0].real.astype(float), w) for w in W]
    K, n_ev = len(W), Q.shape[0]
    N, D = np.zeros((n_ev, K, K)), np.zeros((n_ev, K, K))
    for k in range(K):
        for l in range(k, K):
            n_kl = np.real(np.einsum('ei,ij,ej->e', q[k], G, np.conj(q[l])))
            d_kl = np.einsum('ei,ij,ej->e', m[k], G, m[l])
            # a particle in both bins is not its own pair (same cell: weight G_ii)
            shared = np.einsum('eip,ip,i->e', Q[..., 0].real.astype(float), W[k] * W[l], gd)
            n_kl, d_kl = n_kl - shared, d_kl - shared
            N[:, k, l] = N[:, l, k] = n_kl
            D[:, k, l] = D[:, l, k] = d_kl
    return {'N': N, 'D': D, 'layout': lay}


def _core_v1_even(data, w):
    from analysis.v1_even import class_matrix, class_pairs, v1_from_matrix
    lay = data['layout']
    V = class_matrix(data['N'], data['D'], w)
    return v1_from_matrix(V, list(lay['grid']), list(lay['targets']), lay['sign_bin'],
                          lay['sign'], lay['absolute'], weights=class_pairs(data['D'], w))


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
        if min(poi_eta[1], whi) <= max(poi_eta[0], wlo):
            continue
        c = grid.shared((poi_eta[0], poi_eta[1], poi_pt[0], poi_pt[1]),
                        (wlo, whi, ref_pt[0], ref_pt[1]))[:, 0].real
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


def _sp_integrated(prepare_diff):
    """Integrated v_n{SP} prepare from the differential one with a single
    bin, the record's pT window (_sp_bins): the POI is the record species
    in the record acceptance, the reference charged hadrons in the
    flow_reference windows with the reference pT (default: the record
    window), the diagonal the particles in both.  compute_vn2_sp keeps its
    normalization, <Q_poi Q_ref*> / (<M_poi> sqrt(<|Q_ref|^2> - Sum w^2)).
    (Until November 2026 the 3D prepare took the record's eta window at all
    pT, ignoring the pT window, species and flow_reference, and the 2D
    reference took the record's pT.)"""
    def prepare(container, params):
        d = prepare_diff(container, params)
        out = {'q_poi': d['q_poi'][:, :, 0], 'n_poi': d['n_poi'][:, 0],
               'q_ref': d['q_ref'], 'n_ref': d['n_ref'], 'same_window': False}
        if d.get('dQ0_ref') is not None:
            out['dQ0_ref'] = d['dQ0_ref']
        if d.get('dQ_cross') is not None:
            out['dQ_cross'] = d['dQ_cross'][:, 0]
        return out
    prepare.__name__ = prepare_diff.__name__.replace('_diff', '')
    prepare.__doc__ = _sp_integrated.__doc__
    return prepare


def _wrap_vn2_sp(data, w):
    # Same-window prepares set q_ref = q_poi; the diagonal is N_poi and
    # same_window subtracts it.  Separate-reference prepares (EPD/ZDC)
    # set same_window=False.
    return compute_vn2_sp(
        data['q_poi'], data['q_ref'], data['n_poi'], data['n_ref'], w,
        same_window=bool(data.get('same_window', True)),
        dQ_cross=data.get('dQ_cross'),
        dQ0_poi=data.get('dQ0_poi'), dQ0_ref=data.get('dQ0_ref'))


# ── v_n{6}, v_n{8}: exact distinct-particle sums (analysis.correlators) ───────
#
# <2k> over distinct particles of one sample from the single-particle sums
# S(h, 0, p) = sum w^p e^{ih phi}: 2D containers with their window weights
# (Jacobians), 3D grids with unit weights.  Harmonics beyond the stored
# ones enter only self-correlation terms of coinciding particles (relative
# order 1/M_eff): they are dropped with a note, as for v_n{4}.  (Until
# November 2026 the 2D cores required an integer multiplicity, which the
# weighted M = sum w never is: <6> and <8> were set to 0, so c{6} was
# 3 v^6 instead of 4 v^6 without fluctuations and of either sign with them;
# the 3D prepares refused, or before that returned None.)

def _ref_sample_2d(qv, params):
    pdg_ids = getattr(qv, 'pdg_ids', None)
    pid = _poi_species(params)
    if is_charged(pid.get('species')):
        idx, _ = charged_species_indices(pdg_ids)
    else:
        idx, _, found = _get_species_selection(pid, qv)
        if not found:
            raise ValueError(f"species {pid.get('species')!r} not in the container")
        if idx is None:
            idx = list(range(qv.Qn.shape[1]))
    (eta, pt), _bins, _diff = _cumulant_windows(params)
    lo, hi = _find_pt_slice_edges(qv.pt_bin_edges, pt[0], pt[1])
    return {'idx': list(idx), 'pt': (lo, hi), 'jac': _make_jacobian_list(qv.pt_centers, qv.masses, eta[0], eta[1])}


def _note_missing_multi(S, order, n_harm, h):
    from analysis.correlators import _NOTED
    missing = [j for j in range(2, order // 2 + 1) if j * h >= n_harm]
    key = ('multi', order, n_harm, h)
    if missing and key not in _NOTED:
        _NOTED.add(key)
        w1, w2 = np.real(S(0, 0, 1)), np.real(S(0, 0, 2))
        ok = w2 > 0
        m_eff = float(np.median(w1[ok] ** 2 / w2[ok])) if ok.any() else float('nan')
        print(f"  note: v_{h}{{{order}}}: Q_{{{', '.join(str(j * h) for j in missing)}}} not stored (harmonics < "
              f"{n_harm}); the self-correlation terms of {', '.join(map(str, missing))} coinciding particles "
              f"(relative order 1/M_eff, median M_eff = {m_eff:.3g}) are dropped")


def _prepare_multi(order, dim):
    def prepare(container, params):
        from analysis.correlators import Container2D, Container3D, reference_cumulant_arrays
        h = int(params.get('harmonic', 2))
        if dim == '2d':
            ref = _ref_sample_2d(container, params)
            S = Container2D(container, ref, ref)
            n_harm = container.Qn.shape[2]
        else:
            (eta, pt), _bins, _diff = _cumulant_windows(params)
            win = (eta[0], eta[1], pt[0], pt[1])
            S = Container3D(container, win, container, win, True)
            n_harm = container.Qn.shape[-1]
        if h >= n_harm:
            raise ValueError(f"v_{h}{{{order}}}: harmonic {h} is not stored")
        _note_missing_multi(S, order, n_harm, h)
        return reference_cumulant_arrays(S, n_harm, order)
    prepare.__name__ = f'prepare_vn{order}_single_{dim}'
    return prepare


def _core_multi(order):
    def core(data, w):
        from analysis.correlators import reference_cumulants, vn_from_cumulant
        return vn_from_cumulant(reference_cumulants(data, w, order)[order], order)
    return core


prepare_vn6_single_2d = _prepare_multi(6, '2d')
prepare_vn6_single_3d = _prepare_multi(6, '3d')
prepare_vn8_single_2d = _prepare_multi(8, '2d')
prepare_vn8_single_3d = _prepare_multi(8, '3d')
_core_vn6_exact = _core_multi(6)
_core_vn8_exact = _core_multi(8)


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
    # NOTE: the incoming grid is already charge-factored by the upstream
    # grid.select() call (sum_ET carries the xCF); do NOT re-apply cf here,
    # otherwise charged dET/deta is double-counted (factor-of-2 bug).
    if getattr(grid, 'sum_ET', None) is not None:
        # exact stored E_T column
        sum_et = _window_sum(grid, grid.sum_ET, eta, pt, fractional=True) / grid.nsamples
    elif getattr(grid, 'rapidity_binned', False):
        raise ValueError("dET/deta on a rapidity-binned grid needs its sum_ET")
    else:
        se, sp_ = _grid_slices(grid, eta, pt)
        # no per-cell species on grids: charged-pion mass approximation
        m_pi = 0.13957
        mids = 0.5 * (grid.pt_edges[:-1] + grid.pt_edges[1:])
        ch = 0.5 * (grid.eta_edges[:-1] + grid.eta_edges[1:])
        ET_grid = np.sqrt((mids ** 2)[np.newaxis, np.newaxis, :]
                          + ((m_pi * np.cosh(ch)) ** 2)[np.newaxis, :,
                                                        np.newaxis])
        counts = grid.Qn[:, :, :, 0].real
        sum_et = (counts[:, se, sp_] * ET_grid[:, se, sp_]).sum(axis=(1, 2), dtype=np.float64) \
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
        "prepare_data_2d": _spectrum_form(prepare_spectra_2d),
        "prepare_data_3d": _spectrum_form(prepare_spectra_3d),
        "n_obs_per_cent": None,
        "description": "Invariant yield dN/(2π pT dpT dη), or d2N/(dpT dy) (value_type dN_dpT_dy)",
    },
    "mean_pt": {
        "core_fn": _wrap_mean_pt,
        "prepare_data_2d": prepare_mean_pt_2d,
        "prepare_data_3d": prepare_mean_pt_3d,
        "n_obs_per_cent": 1,
        "description": "Mean transverse momentum ⟨pT⟩",
    },
    "vn8_single": {
        "core_fn": _core_vn8_exact,
        "prepare_data_2d": prepare_vn8_single_2d,
        "prepare_data_3d": prepare_vn8_single_3d,
        "n_obs_per_cent": 2,
        "description": "v_n{8} integrated (exact distinct-particle sums; missing harmonics: 1/M_eff terms dropped)",
    },
    "vn6_single": {
        "core_fn": _core_vn6_exact,
        "prepare_data_2d": prepare_vn6_single_2d,
        "prepare_data_3d": prepare_vn6_single_3d,
        "n_obs_per_cent": 2,
        "description": "v_n{6} integrated (exact distinct-particle sums; missing harmonics: 1/M_eff terms dropped)",
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
    "v1_even_factorization": {
        "core_fn": _core_v1_even,
        "prepare_data_2d": prepare_v1_even_2d,
        "prepare_data_3d": prepare_v1_even_3d,
        "selects_species": True,
        "n_obs_per_cent": None,
        "description": "rapidity-even directed flow v1^even (factorized v11 pair matrix)",
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
        "prepare_data_2d": _sp_integrated(prepare_vn2_sp_diff_2d),
        "prepare_data_3d": _sp_integrated(prepare_vn2_sp_diff_3d),
        "n_obs_per_cent": 5,
        "description": "scalar-product v_n{SP} (JETSCAPE3D convention)",
        "selects_species": True,
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


# ── combinations of cumulant orders ──────────────────────────────────────────
#
# A record with ``cumulant_combination`` (dispatcher.combination_terms) is
# computed from its terms, each an ordinary v_n{2k} record run through its own
# prepare and core on the same events: the combination is formed inside the
# jackknife, so its error carries the correlation of the orders.

def _combination_value(kind, v, scale):
    if kind == 'ratio':
        out = v[0] / v[1] if v[1] != 0 and np.isfinite(v[1]) else np.nan
    elif kind == 'difference':
        out = v[0] - v[1]
    else:                                  # gamma1: v{2}, v{4}, v{6}
        v2, v4, v6 = v
        den = v2 ** 2 - v4 ** 2
        out = -6.0 * np.sqrt(2.0) * v4 ** 2 * (v4 - v6) / den ** 1.5 if den > 0 else np.nan
    return out * scale


def _prepare_combination(dim):
    def prepare(container, params):
        comb = params['cumulant_combination']
        terms = []
        for t in comb['terms']:
            entry = PIPELINE_DISPATCH[t['key']]
            fn = entry[f'prepare_data_{dim}']
            c = container if dim == '2d' else container_for_prepare(container, t['params'], entry)
            terms.append(fn(c, t['params']))
        # metadata ride along (the jackknife masks only per-event arrays)
        return {'terms': terms, 'keys': [t['key'] for t in comb['terms']], 'kind': comb['kind'],
                'scale': comb['scale'],
                'harmonics': tuple(int(t['params'].get('harmonic', 2)) for t in comb['terms'])}
    prepare.__name__ = f'prepare_cumulant_combination_{dim}'
    return prepare


def _core_cumulant_combination(data, w):
    """One value: the combination of the terms' v_n{2k} (harmonic-major
    cores, each term's harmonic at index n - 1)."""
    vals = []
    for key, d, h in zip(data['keys'], data['terms'], data['harmonics']):
        out = np.atleast_1d(PIPELINE_DISPATCH[key]['core_fn'](d, w))
        vals.append(float(out[h - 1]) if len(out) >= h else np.nan)
    return np.array([_combination_value(data['kind'], vals, data['scale'])])


PIPELINE_DISPATCH['vn_cumulant_combination'] = {
    "core_fn": _core_cumulant_combination,
    "prepare_data_2d": _prepare_combination('2d'),
    "prepare_data_3d": _prepare_combination('3d'),
    # receives the unselected container; each term selects its own
    "selects_species": True,
    "n_obs_per_cent": 1,
    "description": "v_n{2k} combination (ratio, difference, gamma1) of several orders or harmonics",
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
    'vn_cumulant_combination', 'v1_even_factorization',
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
    charge_average) are half of the sum the species selection returns; a
    detector-level one (efficiency_corrected: false with detector_efficiency)
    is efficiency x the corrected quantity."""
    def tag(prepare):
        if prepare is None:
            return None

        def wrapped(container, params):
            d = prepare(container, params)
            if isinstance(d, dict):
                d['_charge_average'] = bool(params.get('charge_average'))
                d['_detector_efficiency'] = float(params.get('detector_efficiency') or 1.0)
            return d
        wrapped.__name__, wrapped.__doc__ = prepare.__name__, prepare.__doc__
        return wrapped
    core = entry['core_fn']

    def core_fn(data, w):
        out = core(data, w)
        if not isinstance(data, dict):
            return out
        # (h+ + h-)/2, and a detector-level quantity: efficiency x N.
        factor = (0.5 if data.get('_charge_average') else 1.0) * data.get('_detector_efficiency', 1.0)
        return out * factor if factor != 1.0 else out
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
