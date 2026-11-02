#!/usr/bin/env python3
"""Compute v_n{2} from simulation for a single curated record.

Usage:
  python scripts/compute_flow_vn.py \\
    --record data/flow_vn/phenix_flow_vn_multi-particle_azimuthal_correlations_200gev.yaml \\
    --value-type v_2{2}ab \\
    --sim-path /path/to/QnsAu/ \\
    --method poisson

All measurement parameters (η/pT acceptance, centrality estimator, subevents,
harmonic) are read from the curated record — never hardcoded.
See scripts/run_pipeline.py for batch processing of all records.
"""

import sys, os
import argparse
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.observables_2d import cut_qvectors, _compute_dq0, _compute_dq0_cross
from analysis.observables import vn2, vn2_gap
from analysis.jackknife import jackknife_error
from analysis.qvector import QVectorData
from analysis.weights import (direct_cent_weights_percentile,
                               poisson_cdf_cent_weights, combine_weights,
                               cent_weights,
                               resolve_anchor)
from analysis.simulation_data import load_events_3d_hdf5
from analysis.dispatcher import load_record, extract_measurement_params, load_experimental
from analysis.species import charged_species_indices

PTCUTS = np.array([
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45,
    1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
    2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
    3., 3.2, 3.4, 3.6, 3.8, 4., 10.
], dtype=float)

DELTA_F = 0


def _pt_slice(qv, pt_min, pt_max):
    """Container-aware pT slice [lo, hi) with float-noise tolerance."""
    edges = np.asarray(qv.pt_bin_edges, dtype=float)
    lo = int(np.searchsorted(edges, pt_min, side='left'))
    hi = int(np.searchsorted(edges, min(pt_max, edges[-1]), side='right'))
    if hi < len(edges) and 0.0 <= edges[hi] - min(pt_max, edges[-1]) <= 1e-9:
        hi += 1
    return max(lo, 0), min(hi, len(edges) - 1)


def compute_charged_mult(qv, eta_min, eta_max, pt_min=0.0, pt_max=None):
    """Charged hadron multiplicity in (η, pT) acceptance.

    Returns accumulated (per-hydro-event) Q0 for charged hadrons.
    Used for centrality sorting + Poisson-fluctuation mean.
    """
    Qn_cut, _ = cut_qvectors(qv, eta_min, eta_max)
    if pt_max is None:
        pt_max = qv.pt_bin_edges[-1]
    lo, hi = _pt_slice(qv, pt_min, pt_max)
    idx, cf = charged_species_indices(getattr(qv, 'pdg_ids', None))
    Qn_pt = Qn_cut[:, :, :, lo:hi].sum(axis=3)
    return cf * Qn_pt[:, idx, 0].real.sum(axis=1)


def compute_charged_qn(qv, eta_min, eta_max, pt_min, pt_max):
    """Full Q-vectors (all harmonics) for charged hadrons in the given window.
    Returns (Qn_ch, dQ0, jac) where jac is the Jacobian matrix (species, pT)."""
    Qn_cut, jac = cut_qvectors(qv, eta_min, eta_max)
    if pt_max is None:
        pt_max = qv.pt_bin_edges[-1]
    lo, hi = _pt_slice(qv, pt_min, pt_max)
    idx, cf = charged_species_indices(getattr(qv, 'pdg_ids', None))
    Qn_pt = Qn_cut[:, :, :, lo:hi].sum(axis=3)
    Qn_ch = cf * Qn_pt[:, idx, :].sum(axis=1)
    w = (jac[np.newaxis, :, :]
         / (2.0 * qv.rap_max * qv.nsamples[:, np.newaxis, np.newaxis]))
    q0 = qv.Qn[:, :, 0, :].real
    q0 = q0[:, idx, :]
    w = w[:, idx, :]
    dQ0 = ((q0[:, :, lo:hi] * w[:, :, lo:hi] ** 2).sum(axis=(1, 2))
           * cf ** 2)
    return Qn_ch, dQ0, jac


def compute_vn2_harmonics(Qn_ch, dQ0, weights):
    """Weighted v_n{2} for all harmonics, from per-event Qn.

    Qn_ch: (n_events, n_harm) complex — charged hadron Q-vectors.
    dQ0: (n_events,) float — double-weighted self-correlation.
    weights: (n_events,) float — centrality weight per event (or None = uniform).
    Returns: (n_harm-1,) float — harmonic-major v1..v_{N-1}.  v2 is index 1.
    """
    n_harm = Qn_ch.shape[1]
    result = np.zeros(n_harm, dtype=np.float64)
    Q0 = Qn_ch[:, 0].real
    for h in range(1, n_harm):
        num = np.abs(Qn_ch[:, h]) ** 2 - dQ0
        den = Q0 * Q0 - dQ0
        if weights is None:
            num_avg = np.mean(num)
            den_avg = np.mean(den)
        else:
            num_avg = np.average(num, weights=weights)
            den_avg = np.average(den, weights=weights)
        r = num_avg / max(den_avg, 1e-60)
        result[h] = np.sign(r) * np.sqrt(np.abs(r))
    return result[1:]


def compute_vn2_gap_harmonics(Qn_A, Qn_B, dQ_cross, weights):
    """Weighted v_n{2}ab for all harmonics (two-subevent).

    dQ_cross: per-event cross-term δ_AB = Σ w_A · w_B.
    """
    n_harm = Qn_A.shape[1]
    result = np.zeros(n_harm, dtype=np.float64)
    MA = Qn_A[:, 0].real
    MB = Qn_B[:, 0].real
    for h in range(1, n_harm):
        num = np.real(Qn_A[:, h] * np.conj(Qn_B[:, h])) - dQ_cross
        den = MA * MB - dQ_cross
        if weights is None:
            num_avg = np.mean(num)
            den_avg = np.mean(den)
        else:
            num_avg = np.average(num, weights=weights)
            den_avg = np.average(den, weights=weights)
        r = num_avg / max(den_avg, 1e-60)
        result[h] = np.sign(r) * np.sqrt(np.abs(r))
    return result[1:]


def jk_vn2_harmonics(Qn_ch, dQ0, weights, n_blocks=20):
    """Block jackknife of v₂{2} through v₆{2} for a single centrality bin."""
    n = len(Qn_ch)
    n_blocks = min(n_blocks, n)
    block_size = n // n_blocks

    theta_hat = compute_vn2_harmonics(Qn_ch, dQ0, weights)
    theta_i = np.zeros((n_blocks, len(theta_hat)), dtype=np.float64)

    for i in range(n_blocks):
        lo = i * block_size
        hi = n if i == n_blocks - 1 else (i + 1) * block_size
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False
        w_sub = weights[mask]
        w_sub /= w_sub.sum()
        theta_i[i] = compute_vn2_harmonics(Qn_ch[mask], dQ0[mask], w_sub)

    n_b = len(theta_i)
    theta_mean = np.mean(theta_i, axis=0)
    var = (n_b - 1) / n_b * np.sum((theta_i - theta_mean) ** 2, axis=0)
    corr = theta_hat          # full-sample estimate (run_pipeline convention)
    err = np.sqrt(np.maximum(var, 0))
    return corr, err


def jk_vn2_gap_harmonics(Qn_A, Qn_B, dQ_cross, weights, n_blocks=20):
    """Block jackknife of v₂{2}ab through v₆{2}ab."""
    n = len(Qn_A)
    n_blocks = min(n_blocks, n)
    block_size = n // n_blocks

    theta_hat = compute_vn2_gap_harmonics(Qn_A, Qn_B, dQ_cross, weights)
    theta_i = np.zeros((n_blocks, len(theta_hat)), dtype=np.float64)

    for i in range(n_blocks):
        lo = i * block_size
        hi = n if i == n_blocks - 1 else (i + 1) * block_size
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False
        w_sub = weights[mask]
        w_sub /= w_sub.sum()
        theta_i[i] = compute_vn2_gap_harmonics(
            Qn_A[mask], Qn_B[mask], dQ_cross[mask], w_sub)

    n_b = len(theta_i)
    theta_mean = np.mean(theta_i, axis=0)
    var = (n_b - 1) / n_b * np.sum((theta_i - theta_mean) ** 2, axis=0)
    corr = theta_hat          # full-sample estimate (run_pipeline convention)
    err = np.sqrt(np.maximum(var, 0))
    return corr, err


def rebin_1pct_to_edges(weights, classes, target_edges):
    """Rebin 1% bins to arbitrary width bins by summing weights.

    weights: (n_events, n_1pct_bins) — fractional weights for 0-1%, 1-2%, ...
    classes: list of 'X-Y%' strings defining current binning
    target_edges: list of (lo_pct, hi_pct) tuples for desired output bins
    Returns: (n_events, n_target_bins)
    """
    n_events = weights.shape[0]
    n_target = len(target_edges)
    out = np.zeros((n_events, n_target))
    for j, (tlo, thi) in enumerate(target_edges):
        for i, cl in enumerate(classes):
            parts = cl.replace("%", "").split("-")
            clo, chi = int(parts[0]), int(parts[1])
            overlap = max(0, min(chi, thi) - max(clo, tlo))
            if overlap > 0:
                out[:, j] += weights[:, i] * (overlap / (chi - clo))
    return out


def run_method(qv, cent_Q0, cent_edges, cent_edges_pct, cent_classes,
               Qn_ch, dQ0, Qn_ch_b, dQ_cross,
               method, poisson_efficiency, event_weights=None,
               cent_anchor=None, min_weight=0.01):
    """Compute v₂{2} and v₂{2}ab for all centrality bins using given method.

    The centrality percentile is anchored to a reference total cross section
    via ``cent_anchor`` (see analysis.weights.resolve_anchor).  Bins with
    total weight below ``min_weight`` (no/insufficient sim events, e.g.
    peripheral bins beyond the anchor) are excluded and returned as NaN.

    Returns: (v2_vals, v2_errs, v2_gap_vals, v2_gap_errs, weight_sum, label,
              valid, partial)
      valid  : (n_cent,) bool — bin has sufficient events
      partial: (n_cent,) bool — bin straddles the anchor edge (coverage shifted)
    """
    n_cent = len(cent_edges)
    n_sim = len(cent_Q0)
    anchor = resolve_anchor(cent_anchor if cent_anchor is not None else 1.0, n_sim)

    # Single-acceptance (no subevents): the gap subevents coincide, so the
    # η-gap estimator must reduce to v_n{2} — set the cross-term to the
    # self-correlation.  (Disjoint 3D subevents pass dQ_cross=0 explicitly.)
    if dQ_cross is None:
        dQ_cross = dQ0

    if method == "simple":
        weights = cent_weights(
            "simple", cent_Q0, cent_edges, n_ref=anchor['n_ref'])
        label = f"simple percentile (anchor={anchor['anchor_cent']:.1f}%)"
    else:
        weights = cent_weights(
            "poisson", cent_Q0, cent_edges,
            efficiency=poisson_efficiency, anchor_cent=anchor['anchor_cent'])
        label = (f"Poisson (eff={poisson_efficiency}, "
                 f"anchor={anchor['anchor_cent']:.1f}%)")

    # Optional per-event reweighting (None ⇒ equal weighting).  Same combine
    # used by the batch pipeline, so 2D/3D parity is preserved.
    weights = combine_weights(weights, event_weights)

    ws = weights.sum(axis=0)
    total_w = ws.sum()

    v2_vals = np.full((n_cent, 5), np.nan)
    v2_errs = np.full((n_cent, 5), np.nan)
    v2_gap_vals = np.full((n_cent, 5), np.nan)
    v2_gap_errs = np.full((n_cent, 5), np.nan)
    valid = np.zeros(n_cent, dtype=bool)
    partial = np.zeros(n_cent, dtype=bool)

    # Bins that straddle the anchor edge have shifted (partial) coverage.
    for i in range(n_cent):
        cmin, cmax = cent_edges[i]
        if cmin < anchor['anchor_cent'] < cmax:
            partial[i] = True

    for ic in range(n_cent):
        w = weights[:, ic]
        if w.sum() < min_weight:
            continue
        valid[ic] = True
        val, err = jk_vn2_harmonics(Qn_ch, dQ0, w, n_blocks=20)
        v2_vals[ic] = val
        v2_errs[ic] = err
        val_g, err_g = jk_vn2_gap_harmonics(
            Qn_ch, Qn_ch_b, dQ_cross, w, n_blocks=20)
        v2_gap_vals[ic] = val_g
        v2_gap_errs[ic] = err_g

    return v2_vals, v2_errs, v2_gap_vals, v2_gap_errs, total_w, label, valid, partial


def _summarize_exclusions(cent_edges, cent_classes, valid, partial, label):
    """Log which centrality bins were excluded or only partially covered."""
    def _name(i):
        return cent_classes[i] if i < len(cent_classes) else f"bin{i}"
    excluded = [_name(i) for i in range(len(valid)) if not valid[i]]
    part = [_name(i) for i in range(len(partial)) if partial[i] and valid[i]]
    if excluded:
        print(f"  Excluded bins (no/insufficient sim events): "
              f"{', '.join(excluded)}  [{label}]")
    if part:
        print(f"  Partial-coverage bins (kept; centrality shifted): "
              f"{', '.join(part)}  [{label}]")


def main():
    parser = argparse.ArgumentParser(
        description="Compute v_n{2} from simulation for a single curated record")
    parser.add_argument("--record", required=True,
                        help="Path to curated record YAML file")
    parser.add_argument("--value-type", default="v_2{2}ab",
                        help="Value type substring to match in the record")
    parser.add_argument("--sim-path", default=None,
                        help="2D simulation data directory")
    parser.add_argument("--sim-3d-path", default=None,
                        help="3D HDF5 path (overrides --sim-path if set)")
    parser.add_argument("--species-grids", action="store_true",
                        help="3D: load per-PDG identified grids (MultiSpeciesGrid) "
                             "and apply record species selection (else charged-only "
                             "particle_9999, as before)")
    parser.add_argument("--method", choices=["simple", "poisson", "both"],
                        default="poisson",
                        help="Centrality weight method (default: poisson)")
    parser.add_argument("--poisson-efficiency", type=float, default=1.0,
                        help="Effective Poisson mean multiplier (default: 1.0)")
    parser.add_argument("--cent-anchor", default=None,
                        help="Centrality anchor (reference total cross section). "
                             "Coverage fraction f in (0,1], reference total event "
                             "count N_ref (>1), or 'bmax:VALUE,sigtot:VALUE' "
                             "(f = pi*b_max^2/sigma_tot). Default = full sim set.")
    parser.add_argument("--min-cent-weight", type=float, default=0.01,
                        help="Minimum total centrality weight for a bin to be "
                             "kept; bins below are excluded (default 0.01 = empty).")
    parser.add_argument("--rebin", type=int, default=None,
                        help="Rebin to N%% bin widths (e.g. 10 for 0-10%%, 10-20%%, ...)")
    parser.add_argument("--design-point", type=int, default=0,
                        help="Design point index (default: 0)")
    parser.add_argument("--delta-f", type=int, default=DELTA_F,
                        help="Delta_f viscous model index (default: 0)")
    parser.add_argument("--hepmc", type=int, default=None,
                        help="HepMC run index (for future use)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load record and print parameters without loading sim data")
    parser.add_argument("--dimensionality", default=None,
                        help="Filter by dimensionality (integrated, differential_pT, differential_eta)")
    args = parser.parse_args()
    if args.sim_path is None and args.sim_3d_path is None and not args.dry_run:
        parser.error("give --sim-path (2D Qns_*.npy directory) or --sim-3d-path")

    # 1. Load the curated record
    exp_rec = load_record(args.record, args.value_type,
                           dimensionality=args.dimensionality)
    if exp_rec is None:
        print(f"Could not find record with value_type containing '{args.value_type}'")
        return
    print(f"Loaded: {exp_rec.get('hepdata_file','?')} "
          f"→ {exp_rec.get('value_type','?')}")

    # 2. Extract ALL measurement parameters from the record
    p = extract_measurement_params(exp_rec)
    if p.get('cent_normalization'):
        raise SystemExit("this record defines its centrality normalization (centrality.normalization); "
                         "compute it with scripts/run_pipeline.py, which applies it")
    harmonic = p['harmonic']
    is_subevent = p['particles'] is not None
    sr = p['subevent_ranges']

    if is_subevent:
        labels = sorted(sr.keys())
    single_kr = p['single_range']
    single_eta = single_kr['eta']
    single_pT = single_kr['pT']

    dim = p['dimensionality']
    is_diff = dim in ('differential_pT', 'differential_eta')
    diff_var = p['differential_var']
    diff_bins = p['differential_bins']

    cent_eta_ranges = p['cent_eta_ranges']
    cent_classes = p['cent_classes']
    cent_edges = p['cent_edges']

    print(f"\nHarmonic: n={harmonic}")
    print(f"Particles: {p['particles'] or 'single-acceptance'}")
    if is_subevent:
        for lab in labels:
            e = sr[lab]['eta']
            pT = sr[lab]['pT']
            print(f"  Subevent {lab}: η ∈ [{e[0]}, {e[1]}], pT ∈ [{pT[0]}, {pT[1]}]")
        if p['eta_gap']:
            print(f"  η-gap: |Δη| > {p['eta_gap']}")
    else:
        print(f"  Acceptance: η ∈ [{single_eta[0]}, {single_eta[1]}], "
              f"pT ∈ [{single_pT[0]}, {single_pT[1]}]")

    if is_diff:
        print(f"\nDifferential: {diff_var} bins ({len(diff_bins)})")
        for lo, hi in diff_bins[:5]:
            print(f"  {lo:.2f}-{hi:.2f}")
        if len(diff_bins) > 5:
            print(f"  ... and {len(diff_bins) - 5} more")

    print(f"\nCentrality estimator:")
    print(f"  Detector: {p['cent_estimator'].get('detector','?')}")
    print(f"  Observable: {p['cent_estimator'].get('observable','Nch')}")
    for emin, emax in cent_eta_ranges:
        print(f"  η ∈ [{emin}, {emax}]")
    if cent_classes:
        print(f"  Classes: {len(cent_classes)} bins ({cent_classes[0]} .. {cent_classes[-1]})")
    else:
        print(f"  Classes: (none)")

    if args.dry_run:
        print("\n--dry-run: validation complete, no sim data loaded")
        return

    # 3. Load simulation data
    use_3d = args.sim_3d_path is not None
    if use_3d:
        print(f"\nLoading 3D HDF5 simulation data...")
        cent_eta_ranges_flat = [(e[0], e[1]) for e in cent_eta_ranges]
        if args.species_grids:
            # Per-PDG identified grids → record species selection via the same
            # shared resolve_species logic as run_pipeline (charge_factor baked
            # into the selected grid).  Yields the full-fidelity combined grid
            # for 'charged', and the pt-integrated identified grid for single
            # species (pT-resolved identified flow is not supported by the
            # legacy 20-col writer -- documented limitation).
            from analysis.simulation_data import load_3d_species_grids
            msg = load_3d_species_grids(args.sim_3d_path,
                                        n_harmonics=9,
                                        event_weights=None)
            sel = msg.select(p['particle_id'])
            # Centrality from the combined charged grid (positive-only), as the
            # charged-only path does.
            cent_Q0 = np.zeros(msg.combined.n_events)
            for (emin, emax) in cent_eta_ranges_flat:
                cent_Q0 += (msg.combined.acceptance(emin, emax, 0.0, 1e9)[:, 0].real
                            / msg.combined.nsamples)
            event_weights = msg.event_weights

            def _acc(eta, pT):
                return sel.acceptance(eta[0], eta[1],
                                      pT[0], min(pT[1], 1e9) if pT[1] else 1e9)

            if is_subevent:
                Qn_ch = _acc(sr[labels[0]]['eta'], single_pT)
                Qn_ch_b = (_acc(sr[labels[1]]['eta'], single_pT)
                           if len(labels) > 1 else Qn_ch)
                # Disjoint 3D subevents share no particles → no self-correlation
                dQ0 = Qn_ch[:, 0].real * sel.charge_factor
                dQ_cross = np.zeros(len(Qn_ch))
            else:
                Qn_ch = _acc(single_eta, single_pT)
                Qn_ch_b = Qn_ch
                dQ0 = Qn_ch[:, 0].real * sel.charge_factor
                dQ_cross = None
            qv = None
            print(f"  Events: {msg.combined.n_events}, species-selected: "
                  f"{p['particle_id'] or 'charged'} (cf={sel.charge_factor})")
        else:
            meas_eta_ranges = {}
            if is_subevent:
                for lab in labels:
                    meas_eta_ranges[lab] = sr[lab]['eta']
            else:
                meas_eta_ranges['A'] = single_eta
            d3 = load_events_3d_hdf5(args.sim_3d_path, PTCUTS, meas_eta_ranges,
                                      cent_eta_ranges_flat, pt_min=0.0, pt_max=None)
            Qn_ch = d3.get(f"Qn_{labels[0]}", d3['Qn_A']) if is_subevent else d3['Qn_A']
            Qn_ch_b = d3.get(f"Qn_{labels[1]}", Qn_ch) if (is_subevent and len(labels) > 1) else Qn_ch
            cent_Q0 = d3['cent_mult']
            dQ0 = np.zeros(len(Qn_ch))
            dQ0_b = np.zeros(len(Qn_ch_b))
            # 3D subevents are disjoint → no shared particles → no
            # self-correlation cross-term.  Single-acceptance leaves dQ_cross
            # None (run_method reduces the gap estimator to v_n{2}).
            dQ_cross = np.zeros(len(Qn_ch)) if is_subevent else None
            event_weights = d3.get('event_weights')
            qv = None
            print(f"  Events: {d3['n_events']}")
    else:
        qn_path = os.path.join(args.sim_path, f"Qns_{args.design_point}.npy")
        ns_path = os.path.join(args.sim_path, f"Nsamples_{args.design_point}.npy")
        print(f"\nLoading design point {args.design_point}...")
        qv = QVectorData.from_2d_npy(qn_path, ns_path, PTCUTS,
                                      delta_f_index=args.delta_f)
        print(f"  Events: {qv.n_events}, species: {qv.n_species}, "
              f"harmonics: {qv.n_harmonics}, pt_bins: {qv.n_pt}")

        cent_Q0 = np.zeros(qv.n_events)
        for emin, emax in cent_eta_ranges:
            cent_Q0 += compute_charged_mult(qv, emin, emax,
                                             pt_min=0.0, pt_max=None)
        print(f"\nCentrality multiplicity (BBC range): mean {cent_Q0.mean():.1f}")

        # Optional per-event reweighting (None ⇒ equal weighting).
        event_weights = (qv.event_weights if qv is not None
                         else d3.get('event_weights'))

        pt_min_sim = single_pT[0]
        pt_max_sim = min(single_pT[1], PTCUTS[-1]) if single_pT[1] else PTCUTS[-1]
        if is_subevent and len(labels) >= 2:
            eA = sr[labels[0]]['eta']
            eB = sr[labels[1]]['eta']
        else:
            eA = single_eta
            eB = single_eta
        lo = int(np.argwhere(np.isclose(PTCUTS, pt_min_sim))[0, 0])
        hi = int(np.argwhere(np.isclose(PTCUTS, pt_max_sim))[0, 0])
        Qn_ch, dQ0, jac_A = compute_charged_qn(qv, eA[0], eA[1], pt_min_sim, pt_max_sim)
        Qn_ch_b, dQ0_b, jac_B = compute_charged_qn(qv, eB[0], eB[1], pt_min_sim, pt_max_sim)
        if is_subevent:
            dQ_cross = (_compute_dq0_cross(qv, jac_A, jac_B, lo, hi)
                        if not np.allclose(jac_A, jac_B) else dQ0.copy())
        else:
            dQ_cross = None

    print(f"\nAfter measurement cuts:")
    if is_subevent:
        lab0 = labels[0]
        lab1 = labels[1] if len(labels) > 1 else labels[0]
        print(f"  Subevent {lab0}: mean Q0 = {Qn_ch[:, 0].real.mean():.1f}")
        print(f"  Subevent {lab1}: mean Q0 = {Qn_ch_b[:, 0].real.mean():.1f}")
    else:
        print(f"  Mean Q0 = {Qn_ch[:, 0].real.mean():.1f}")

    # 4. Handle rebinning
    n_cent = len(cent_edges)
    display_edges = cent_edges
    display_classes = cent_classes

    if args.rebin is not None:
        rb = args.rebin
        max_cent = max(hi for (_, hi) in cent_edges) if cent_edges else 90
        new_edges = []
        new_classes = []
        for start in range(0, max_cent, rb):
            end = start + rb
            new_edges.append((start, end))
            new_classes.append(f"{start}-{end}%")
        display_edges = new_edges
        display_classes = new_classes
        print(f"\nRebinning from {n_cent}×1% bins to {len(new_edges)}×{rb}% bins")

    cent_edges_pct = np.array([0.0] + [float(hi) for (_, hi) in display_edges])

    # 5. Load experimental values
    exp_vals, exp_errs = load_experimental(p)
    if exp_vals is not None:
        print(f"  Loaded {len(exp_vals)} experimental values")
        if args.rebin is not None:
            rb = args.rebin
            rb_exp_vals = []
            rb_exp_errs = []
            for i in range(len(display_classes)):
                lo, hi = display_edges[i]
                lo_i = lo
                hi_i = hi - 1 if hi <= 90 else 90
                if lo_i < len(exp_vals) and hi_i <= len(exp_vals):
                    vs = exp_vals[lo_i:hi_i]
                    es = exp_errs[lo_i:hi_i]
                    rb_exp_vals.append(np.mean(vs))
                    rb_exp_errs.append(np.sqrt(np.sum(es**2)) / len(es) if len(es) > 0 else 0)
                else:
                    rb_exp_vals.append(0)
                    rb_exp_errs.append(0)
            exp_vals, exp_errs = np.array(rb_exp_vals), np.array(rb_exp_errs)
            print(f"  Rebinned to {len(exp_vals)} experimental values")

    # 6. Compute — handle differential vs integrated
    methods_to_run = ["simple", "poisson"] if args.method == "both" else [args.method]

    if is_diff and not use_3d:
        # Differential: compute per-bin, store in list of dicts
        all_results = []
        for bin_idx, (bin_lo, bin_hi) in enumerate(diff_bins):
            if diff_var == 'pT':
                Qn_ch_bin, dQ0_bin, _ = compute_charged_qn(qv, eA[0], eA[1], bin_lo, bin_hi)
                if is_subevent:
                    Qn_ch_b_bin, dQ0_b_bin, jac_B_bin = compute_charged_qn(
                        qv, eB[0], eB[1], bin_lo, bin_hi)
                    lo_bin = int(np.argwhere(np.isclose(PTCUTS, bin_lo))[0, 0])
                    hi_bin = int(np.argwhere(np.isclose(PTCUTS, bin_hi))[0, 0])
                    dQ_cross_bin = (_compute_dq0_cross(qv, jac_A, jac_B_bin, lo_bin, hi_bin)
                                    if not np.allclose(jac_A, jac_B_bin) else dQ0_bin.copy())
                else:
                    Qn_ch_b_bin = Qn_ch_bin
                    dQ0_b_bin = dQ0_bin
                    dQ_cross_bin = None
                bin_label = f"pT {bin_lo:.2f}-{bin_hi:.2f}"
            elif diff_var == 'eta':
                if is_subevent:
                    e_eta_A = (sr[labels[0]]['eta'][0], sr[labels[0]]['eta'][1])
                    e_eta_B = (bin_lo, bin_hi)
                else:
                    e_eta_A = (bin_lo, bin_hi)
                    e_eta_B = e_eta_A
                Qn_ch_bin, dQ0_bin, jac_A_bin = compute_charged_qn(
                    qv, e_eta_A[0], e_eta_A[1], pt_min_sim, pt_max_sim)
                Qn_ch_b_bin, dQ0_b_bin, jac_B_bin = compute_charged_qn(
                    qv, e_eta_B[0], e_eta_B[1], pt_min_sim, pt_max_sim)
                if is_subevent and not np.allclose(jac_A_bin, jac_B_bin):
                    lo_bin = int(np.argwhere(np.isclose(PTCUTS, pt_min_sim))[0, 0])
                    hi_bin = int(np.argwhere(np.isclose(PTCUTS, pt_max_sim))[0, 0])
                    dQ_cross_bin = _compute_dq0_cross(qv, jac_A_bin, jac_B_bin, lo_bin, hi_bin)
                elif is_subevent:
                    dQ_cross_bin = dQ0_bin.copy()
                else:
                    dQ_cross_bin = None
                bin_label = f"η {bin_lo:.2f}-{bin_hi:.2f}"

            bin_results = {}
            for m in methods_to_run:
                v2v, v2e, v2gv, v2ge, tw, lbl, valid, partial = run_method(
                    qv, cent_Q0, display_edges,
                    cent_edges_pct, cent_classes,
                    Qn_ch_bin, dQ0_bin, Qn_ch_b_bin, dQ_cross_bin,
                    m, args.poisson_efficiency,
                    event_weights=event_weights,
                    cent_anchor=args.cent_anchor,
                    min_weight=args.min_cent_weight)
                bin_results[m] = {
                    "v2": v2v, "v2_err": v2e,
                    "v2_gap": v2gv, "v2_gap_err": v2ge,
                    "total_weight": tw, "label": lbl,
                    "valid": valid, "partial": partial,
                }
                _summarize_exclusions(display_edges, display_classes, valid, partial, lbl)
            all_results.append((bin_label, bin_results))
            print(f"  [{bin_label}] {bin_results[methods_to_run[0]]['label']}: "
                  f"{bin_results[methods_to_run[0]]['total_weight']:.0f} total weight")

        # 7. Print comparison table per differential bin
        col_label = f"v{harmonic}{{2}}"
        col_gap = f"v{harmonic}{{2}}ab" if is_subevent else col_label

        for bin_label, rdict in all_results:
            print(f"\n── {bin_label}: {col_gap} comparison ──")
            if args.method == "both":
                header = (f"{'Cent':>6}  "
                          f"{f'{col_label} simple':>11}  {'err':>7}  "
                          f"{f'{col_label} Poisson':>11}  {'err':>7}  "
                          f"{'diff':>8}  "
                          f"{f'{col_gap} exp':>10}  {'err':>7}  {'pull':>7}")
                print(header)
                print("-" * 85)
            else:
                header = (f"{'Cent':>6}  "
                          f"{f'{col_label} sim':>9}  {'err':>7}  "
                          f"{f'{col_gap} sim':>9}  {'err':>7}  "
                          f"{f'{col_gap} exp':>10}  {'err':>7}  {'pull':>7}")
                print(header)
                print("-" * 75)

            h_idx = harmonic - 1
            for ibin in range(len(display_classes)):
                cl = display_classes[ibin][:6]
                if args.method == "both":
                    r_s = rdict["simple"]
                    r_p = rdict["poisson"]
                    v2s = r_s["v2"][ibin, h_idx] if ibin < len(r_s["v2"]) else 0
                    e2s = r_s["v2_err"][ibin, h_idx] if ibin < len(r_s["v2_err"]) else 0
                    v2p = r_p["v2"][ibin, h_idx] if ibin < len(r_p["v2"]) else 0
                    e2p = r_p["v2_err"][ibin, h_idx] if ibin < len(r_p["v2_err"]) else 0
                    diff = v2p - v2s
                    ev = exp_vals[ibin] if exp_vals is not None and ibin < len(exp_vals) else 0
                    ee = exp_errs[ibin] if exp_errs is not None and ibin < len(exp_errs) else 0
                    pull = (v2p - ev) / np.sqrt(e2p**2 + ee**2) if e2p > 0 and ee > 0 else 0
                    print(f"  {cl:>5}  {v2s:11.6f}  {e2s:7.6f}  "
                          f"{v2p:11.6f}  {e2p:7.6f}  {diff:8.4f}  "
                          f"{ev:10.6f}  {ee:7.6f}  {pull:7.3f}")
                else:
                    r = rdict[args.method]
                    v2v = r["v2"][ibin, h_idx] if ibin < len(r["v2"]) else 0
                    v2e = r["v2_err"][ibin, h_idx] if ibin < len(r["v2_err"]) else 0
                    vgv = r["v2_gap"][ibin, h_idx] if ibin < len(r["v2_gap"]) else 0
                    vge = r["v2_gap_err"][ibin, h_idx] if ibin < len(r["v2_gap_err"]) else 0
                    if exp_vals is not None and ibin < len(exp_vals):
                        ev = exp_vals[ibin]
                        ee = exp_errs[ibin]
                        pull = (vgv - ev) / np.sqrt(vge**2 + ee**2) if vge > 0 and ee > 0 else 0
                        print(f"  {cl:>5}  {v2v:9.6f}  {v2e:7.6f}  {vgv:9.6f}  "
                              f"{vge:7.6f}  {ev:10.6f}  {ee:7.6f}  {pull:7.3f}")
                    else:
                        print(f"  {cl:>5}  {v2v:9.6f}  {v2e:7.6f}  {vgv:9.6f}  "
                              f"{vge:7.6f}")
        print()
    else:
        # Integrated (or 3D differential — fallback to integrated extraction)
        if is_diff and use_3d:
            print(f"\n  Note: 3D differential uses full-acceptance Q-vectors "
                  f"(not per-bin)")

        results = {}
        for m in methods_to_run:
            v2v, v2e, v2gv, v2ge, tw, lbl, valid, partial = run_method(
                qv, cent_Q0, display_edges,
                cent_edges_pct, cent_classes,
                Qn_ch, dQ0, Qn_ch_b, dQ_cross,
                m, args.poisson_efficiency,
                event_weights=event_weights,
                cent_anchor=args.cent_anchor,
                min_weight=args.min_cent_weight)
            results[m] = {
                "v2": v2v, "v2_err": v2e,
                "v2_gap": v2gv, "v2_gap_err": v2ge,
                "total_weight": tw, "label": lbl,
                "valid": valid, "partial": partial,
            }
            _summarize_exclusions(display_edges, display_classes, valid, partial, lbl)
            print(f"\n  {lbl}: {tw:.0f} total weight across {len(display_edges)} bins")

        # 7. Print comparison table
        col_label = f"v{harmonic}{{2}}"
        if is_subevent:
            col_gap = f"v{harmonic}{{2}}ab"
        else:
            col_gap = f"v{harmonic}{{2}}"

        print()
        print(f"── Comparison table ({col_gap}) ──")
        if args.method == "both":
            header = (f"{'Cent':>6}  "
                      f"{f'{col_label} simple':>11}  {'err':>7}  "
                      f"{f'{col_label} Poisson':>11}  {'err':>7}  "
                      f"{'diff':>8}  "
                      f"{f'{col_gap} exp':>10}  {'err':>7}  {'pull':>7}")
            print(header)
            print("-" * 85)
        else:
            header = (f"{'Cent':>6}  "
                      f"{f'{col_label} sim':>9}  {'err':>7}  "
                      f"{f'{col_gap} sim':>9}  {'err':>7}  "
                      f"{f'{col_gap} exp':>10}  {'err':>7}  {'pull':>7}")
            print(header)
            print("-" * 75)

        h_idx = harmonic - 1
        for ibin in range(len(display_classes)):
            label = display_classes[ibin][:6]

            if args.method == "both":
                r_s = results["simple"]
                r_p = results["poisson"]
                v2s = r_s["v2"][ibin, h_idx] if ibin < len(r_s["v2"]) else 0
                e2s = r_s["v2_err"][ibin, h_idx] if ibin < len(r_s["v2_err"]) else 0
                v2p = r_p["v2"][ibin, h_idx] if ibin < len(r_p["v2"]) else 0
                e2p = r_p["v2_err"][ibin, h_idx] if ibin < len(r_p["v2_err"]) else 0
                diff = v2p - v2s
                ev = exp_vals[ibin] if exp_vals is not None and ibin < len(exp_vals) else 0
                ee = exp_errs[ibin] if exp_errs is not None and ibin < len(exp_errs) else 0
                pull = (v2p - ev) / np.sqrt(e2p**2 + ee**2) if e2p > 0 and ee > 0 else 0
                print(f"  {label:>5}  {v2s:11.6f}  {e2s:7.6f}  "
                      f"{v2p:11.6f}  {e2p:7.6f}  {diff:8.4f}  "
                      f"{ev:10.6f}  {ee:7.6f}  {pull:7.3f}")
            else:
                r = results[args.method]
                v2v = r["v2"][ibin, h_idx] if ibin < len(r["v2"]) else 0
                v2e = r["v2_err"][ibin, h_idx] if ibin < len(r["v2_err"]) else 0
                vgv = r["v2_gap"][ibin, h_idx] if ibin < len(r["v2_gap"]) else 0
                vge = r["v2_gap_err"][ibin, h_idx] if ibin < len(r["v2_gap_err"]) else 0
                if exp_vals is not None and ibin < len(exp_vals):
                    ev = exp_vals[ibin]
                    ee = exp_errs[ibin]
                    pull = (vgv - ev) / np.sqrt(vge**2 + ee**2) if vge > 0 and ee > 0 else 0
                    print(f"  {label:>5}  {v2v:9.6f}  {v2e:7.6f}  {vgv:9.6f}  "
                          f"{vge:7.6f}  {ev:10.6f}  {ee:7.6f}  {pull:7.3f}")
                else:
                    print(f"  {label:>5}  {v2v:9.6f}  {v2e:7.6f}  {vgv:9.6f}  "
                          f"{vge:7.6f}")
        print()


if __name__ == "__main__":
    main()
