"""3D computations on charge-scaled grids.

A positives-only grid selected as 'charged' (or a 'p' grid as 'p + pbar')
is scaled by the charge factor (QGridData.charge_factor = 2).  Absolute
quantities -- yields, spectra, dE_T/deta -- must scale by that factor;
normalized ones -- flow coefficients, correlations, mean pT and its
fluctuations -- must not change.  A self-correlation term taken from the
scaled counts carries the factor once where it needs it squared, and
biased v_n{4}, the scalar-product v_n(eta) and pT fluctuations by factors
up to ~6 on such grids (found Nov 2026).
Run: python3 tests/test_charge_scaled_grids.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.toy_mc import (ToyGenConfig, generate_events, make_qgrid_data, record_flow_vn2,  # noqa: E402
                             record_flow_vn4, record_flow_ep, record_flow_ep_diff_eta, record_flow_gap,
                             record_yield, record_spectra,
                             record_pT_fluct, record_dET, record_mean_pt)
from analysis.dispatcher import resolve  # noqa: E402
from analysis.pipeline_dispatch import get_pipeline_entry  # noqa: E402
from analysis.toy_mc import evaluate  # noqa: E402

results = []
C = ('0-100%',)


def records():
    # The single-acceptance differential keys run per bin through vn2_single /
    # vn4_single in run_pipeline (not mirrored by toy_mc.evaluate); those two
    # keys are covered here.
    sp_diff = dict(record_flow_vn2(eta=(-1, 1), pt=(0.2, 2.0), cent_classes=C), dimensionality='differential_eta',
                   method='event_plane',
                   kinematic_range=dict(eta=dict(min=-1, max=1, bin_ranges=[[-1, -0.5], [0.5, 1]]), pT=dict(min=0.2, max=2.0)),
                   flow_reference=dict(eta=[dict(min=-3.5, max=-2.0), dict(min=2.0, max=3.5)]))
    sp_mirror = dict(sp_diff, flow_reference=dict(mirror=True, eta=[dict(min=0.1, max=2.0)]))
    v1 = dict(observable_type='flow_vn', value_type='v1', harmonic=1, method='event_plane',
              dimensionality='differential_eta', hepdata_file='t', inspire_id='0', version=1,
              particle_id=dict(species='charged'),
              kinematic_range=dict(eta=dict(min=-2, max=2, bin_ranges=[[-1.0, -0.5], [0.5, 1.0]]), pT=dict(min=0.2, max=2.0)),
              centrality=dict(classes=['0-100%'], eta=[dict(min=-3.9, max=-3.0), dict(min=3.0, max=3.9)]),
              flow_reference=dict(odd=True, eta=[dict(min=-3.5, max=-2.0), dict(min=2.0, max=3.5)],
                                  sign_convention=dict(eta=dict(min=0.5, max=1.0), sign='negative')))
    return {
        'vn2 single': (record_flow_vn2(cent_classes=C), 'norm'),
        'vn4 single': (record_flow_vn4(cent_classes=C), 'norm'),
        'vn2 EP': (record_flow_ep(cent_classes=C), 'norm'),
        'vn2 EP diff eta': (record_flow_ep_diff_eta([(-1, -0.5), (0.5, 1)], cent_classes=C), 'norm'),
        'vn2 gap': (record_flow_gap(cent_classes=C), 'norm'),
        'vn2 SP diff eta': (sp_diff, 'norm'),
        'vn2 SP diff eta (mirrored)': (sp_mirror, 'norm'),
        'v1 odd': (v1, 'norm'),
        'mean pT': (record_mean_pt(cent_classes=C), 'norm'),
        'pT fluct': (record_pT_fluct(), 'norm'),
        'yield': (record_yield(cent_classes=C), 'abs'),
        'spectra': (record_spectra(cent_classes=C), 'abs'),
        'dET/deta': (record_dET(), 'abs'),
    }


def run_core(key, grid, params):
    e = get_pipeline_entry(key)
    return np.atleast_1d(e['core_fn'](e['prepare_data_3d'](grid, params), np.ones(grid.n_events)))


def main():
    cfg = ToyGenConfig(n_events=600, mult_mean=600, mult_sigma=60, rap_max=4.0, seed=8,
                       flow={1: dict(kind='const', value=1.0), 2: dict(kind='const', value=0.06),
                             3: dict(kind='const', value=0.03)},
                       fluctuate={2: 0.02}, pt_rescale_sigma=0.03, eta_shape={1: lambda eta: -0.01 * eta})
    events, _ = generate_events(cfg)
    # The same particles, read once as all charged hadrons (no scaling) and
    # once as positives only (selected as 'charged': x2).
    pos = make_qgrid_data(events, {'cfg': cfg}, n_eta=80, n_pt=20)
    allch = make_qgrid_data(events, {'cfg': cfg}, n_eta=80, n_pt=20)
    allch.positives_only = False
    cf = pos.select({'species': 'charged'}).charge_factor
    assert cf == 2.0 and getattr(allch.select({'species': 'charged'}), 'charge_factor', 1.0) == 1.0
    for name, (rec, kind) in records().items():
        # Real records name their species; the toy flow records leave it
        # out, which (unscaled) never exercised the charge factor.
        rec = dict(rec, particle_id=rec.get('particle_id') or {'species': 'charged'})
        key = resolve(rec)['compute']
        a, _ = evaluate(rec, grid=allch, cent_method='simple', with_jackknife=False)
        b, _ = evaluate(rec, grid=pos, cent_method='simple', with_jackknife=False)
        want = a * (cf if kind == 'abs' else 1.0)
        ok = np.allclose(b, want, rtol=1e-6, atol=1e-12, equal_nan=True) and not np.all(np.isnan(a))
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'} {name:28s} {key:22s} unscaled {np.round(a[:2], 5)} scaled {np.round(b[:2], 5)}"
              f" ({'x' + str(cf) if kind == 'abs' else 'invariant'})")
    for key, harms in (('sc_mn', [2, 3]),):
        params = {'single_range': {'eta': (-1, 1), 'pT': (0.2, 2.0)}, 'harmonics': harms}
        a = run_core(key, allch.select({'species': 'charged'}), params)
        b = run_core(key, pos.select({'species': 'charged'}), params)
        ok = np.allclose(a, b, rtol=1e-6, equal_nan=True)
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'} {key:28s} unscaled {np.round(a[:2], 7)} scaled {np.round(b[:2], 7)} (invariant)")
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
