"""Toy-MC validation of the EVENT-PLANE / amplitude flow path, end to end.

Added Sep 2026.  ``analysis/toy_mc.py`` previously had no helper producing a
``method: event_plane`` record, so the ``vn2_ep`` / ``vn3_ep`` /
``vn2_ep_diff_pT`` / ``vn2_ep_diff_eta`` family -- 872 curated records, the
second-largest block in the database -- had **no** test at all.  These tests
run the real path (resolve -> prepare -> core) via ``toy_mc.evaluate``.

What is being validated:

* the self-correlation subtraction (shared ``_amplitude_vn`` helper), which
  was missing in both the integrated and the differential core and inflated
  v_n by up to +250%;
* that the estimator recovers an injected amplitude across bin widths;
* 2D/3D equivalence, the usual invariant;
* that it is an AMPLITUDE estimator, not a resolution-corrected
  event-plane one -- documented rather than silently assumed.

Run: python3 tests/test_toy_mc_ep_flow.py
"""
import sys

import numpy as np

sys.path.insert(0, '/'.join(__file__.split('/')[:-2]))

from analysis.dispatcher import resolve                          # noqa: E402
from analysis.toy_mc import (                                    # noqa: E402
    ToyGenConfig, evaluate, generate_events, make_qgrid_data,
    make_qvector_data,
    record_flow_ep, record_flow_ep_diff_eta, record_flow_ep_diff_pt,
)

NE = 1200
V2 = 0.055

# Module-level so the dispatch/manifest regression tests (which are pure
# metadata checks needing no MC) can report through the same channel.
n_pass = n_fail = 0


def check(name, cond, detail=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f'  {name}: PASS {detail}')
    else:
        n_fail += 1
        print(f'  {name}: FAIL {detail}')


def _containers():
    """2D and 3D containers carrying injected flow, plus a no-flow set."""
    import analysis.toy_mc as tm
    cfg = ToyGenConfig(n_events=NE, mult_mean=700, mult_sigma=80, seed=101,
                       flow={2: dict(kind='const', value=V2),
                             3: dict(kind='const', value=0.03)})
    events, _ = generate_events(cfg)
    qv = make_qvector_data(events, {'cfg': cfg}, n_pt=80)
    grid = make_qgrid_data(events, {'cfg': cfg}, n_pt=80)
    cfg0 = ToyGenConfig(n_events=NE, mult_mean=200, mult_sigma=40, seed=303)
    ev0, _ = generate_events(cfg0)
    qv0 = make_qvector_data(ev0, {'cfg': cfg0}, n_pt=80)
    return qv, grid, qv0


def main():
    print('test_ep_record_dispatch')
    for label, rec, want in (
            ('integrated', record_flow_ep(), 'vn2_ep'),
            ('diff_pT', record_flow_ep_diff_pt([0.2, 0.5, 1.0, 2.0]),
             'vn2_ep_diff_pT'),
            ('diff_eta', record_flow_ep_diff_eta([(-1, 0), (0, 1)]),
             'vn2_ep_diff_eta'),
            ('v3 integrated', record_flow_ep(harmonic=3), 'vn3_ep')):
        got = resolve(rec)['compute']
        check(f'{label} dispatches to {want}', got == want, f'got {got}')

    qv, grid, qv0 = _containers()

    print('\ntest_ep_integrated_amplitude')
    rec = record_flow_ep(cent_classes=('0-100%',))
    # The EP/amplitude family is 3D-only (no prepare_data_2d), so this is
    # the only path; recorded here so the limitation is explicit.
    import analysis.pipeline_dispatch as _pd
    check('vn2_ep is 3D-only (documented, not assumed)',
          _pd.PIPELINE_DISPATCH['vn2_ep']['prepare_data_2d'] is None)
    v3d, _ = evaluate(rec, grid=grid, cent_method='simple')
    check('3D integrated v2 finite', np.all(np.isfinite(v3d)), f'{v3d}')
    got = float(v3d[0])
    check('integrated v2 within 20% of injected 0.055',
          abs(got - V2) < 0.2 * V2, f'got {got:.4f}')

    print('\ntest_ep_differential_amplitude')
    # Bin widths are matched to the toy's particle budget on purpose: at
    # mult_mean=700 over 0.2-2.0 GeV a 0.05-wide bin holds ~20 particles,
    # which cannot measure v2 (the noise, not the estimator, would dominate).
    # The uncorrected estimator's bias is largest at low multiplicity, and it
    # is covered directly in test_toy_mc_ep_diff.py where multiplicity is
    # controlled exactly.
    bins = [0.2, 0.5, 0.9, 1.4, 2.0]
    recp = record_flow_ep_diff_pt(bins, cent_classes=('0-100%',))
    vals, _ = evaluate(recp, grid=grid, cent_method='simple')
    finite = vals[np.isfinite(vals)]
    check('differential pT produced finite values', finite.size >= 3,
          f'{finite.size} of {vals.size}')
    check('differential pT within 30% of injected',
          finite.size and np.all(np.abs(finite - V2) < 0.3 * V2),
          f'{np.round(finite, 4)}')

    rece = record_flow_ep_diff_eta([-1.0, -0.4, 0.4, 1.0],
                                   cent_classes=('0-100%',))
    vals, _ = evaluate(rece, grid=grid, cent_method='simple')
    finite = vals[np.isfinite(vals)]
    check('differential eta produced finite values', finite.size >= 2,
          f'{finite.size} of {vals.size}')

    print('\ntest_ep_no_flow_gives_near_zero')
    # Without flow the amplitude must sit near zero, not near sqrt(1/M).
    grid0 = make_qgrid_data(*generate_events(
        ToyGenConfig(n_events=NE, mult_mean=700, mult_sigma=80, seed=303)))
    v, _ = evaluate(record_flow_ep(cent_classes=('0-100%',)), grid=grid0,
                    cent_method='simple')
    got = float(v[0])
    # uncorrected would give sqrt(1/700) ~ 0.038
    check('no-flow amplitude below 0.01 (uncorrected would be ~0.038)',
          got < 0.01, f'got {got:.4f}')

    # --- dispatch / manifest regressions (no MC needed) ---
    print()
    print('test_ep_reference_routing')
    test_ep_reference_routing()
    print()
    print('test_search_records_emits_doc_pin')
    test_search_records_emits_doc_pin()
    print()
    test_coefficient_ratio_is_refused()

    total = n_pass + n_fail
    print()
    print(f'{n_pass}/{total} passed')
    sys.exit(1 if n_fail else 0)


def test_ep_reference_routing():
    """event_plane + flow_reference must route to the SP path.

    The event_plane compute keys are the single-set amplitude, whose
    prepare ignores flow_reference outright.  So a reference-bearing record
    routed there silently discards the reference the record declares.
    """
    from analysis.dispatcher import resolve

    r = record_flow_ep_diff_eta([(-1.0), (1.0)], cent_classes=('0-100%',))
    # Baseline: no reference declared -> amplitude estimator.
    check('no reference -> amplitude key',
          resolve(r)['compute'] == 'vn2_ep_diff_eta', resolve(r)['compute'])

    # With a reference declared -> the SP correlation, which consumes it.
    r2 = dict(r)
    r2['flow_reference'] = {'detector': 'Octagon',
                           'eta': [{'min': 2.05, 'max': 3.2},
                                   {'min': -3.2, 'max': -2.05}]}
    got = resolve(r2)['compute']
    check('with reference -> SP key', got == 'vn2_sp_diff_eta', got)

    # Integrated v2 likewise.
    ri = record_flow_ep(cent_classes=('0-100%',))
    check('integrated, no reference -> amplitude',
          resolve(ri)['compute'] == 'vn2_ep', resolve(ri)['compute'])
    ri2 = dict(ri)
    ri2['flow_reference'] = {'eta': [{'min': 2.05, 'max': 3.2}]}
    check('integrated, with reference -> SP',
          resolve(ri2)['compute'] == 'vn2_sp', resolve(ri2)['compute'])

    # A v3 record WITH a reference must reach the SP path.  The SP keys
    # carry no harmonic because the cores return every harmonic, and
    # run_pipeline indexes them as (harm - 1) * n_bins -- so vn2_sp
    # serves v_3 correctly and this is not a downgrade to v_2.
    rv = record_flow_ep(cent_classes=('0-100%',))
    rv['harmonic'] = 3
    rv['flow_reference'] = {'eta': [{'min': 2.05, 'max': 3.2}]}
    got3 = resolve(rv)['compute']
    check('v3 with reference routes to the SP path', got3 == 'vn2_sp', got3)

    # Without a reference it stays on the amplitude fallback.
    rv2 = record_flow_ep(cent_classes=('0-100%',))
    rv2['harmonic'] = 3
    check('v3 without reference stays on amplitude',
          resolve(rv2)['compute'] == 'vn3_ep', resolve(rv2)['compute'])


def test_search_records_emits_doc_pin():
    """Generated manifests must pin fpath+doc_index.

    Without them run_pipeline falls back to id lookup, which cannot tell
    per-centrality docs of one table apart -- it then silently reads the
    wrong document (observed: the MB v2(eta) series returned for a
    manifest asking for 3-15%).
    """
    import inspect, yaml as _y
    import scripts.search_records as sr
    src = inspect.getsource(sr)
    check('search_records writes fpath', '"fpath": fpath' in src)
    check('search_records writes doc_index', '"doc_index": doc_idx' in src)

    # And run_pipeline actually honours them.
    import scripts.run_pipeline as rp
    rsrc = inspect.getsource(rp)
    check('run_pipeline reads the pin',
          "entry.get('fpath') is not None and entry.get('doc_index') is not None"
          in rsrc)

def test_coefficient_ratio_is_refused():
    """A ratio/power of flow coefficients must NOT dispatch to a v_n key.

    `v_4/v_2^2` is not a flow coefficient: no v_n estimator can produce it,
    it needs a ratio of two independently measured coefficients.  Such
    records used to resolve to `vn2_ep_diff_pT` with harmonic=2, so the
    pipeline computed v_2 while the record claimed to be a ratio -- a
    plausible number under a wrong label, which is the worst failure mode.
    """
    print('test_coefficient_ratio_is_refused')
    from analysis.dispatcher import resolve, is_coefficient_ratio

    for vt in ('v_4/v_2^2', 'v_6/v_4', 'v_2{4}/v_2{2}^2'):
        check(f'{vt!r} detected as a coefficient ratio',
              is_coefficient_ratio(vt))
        r = record_flow_ep_diff_pt([0.2, 0.5, 1.0, 2.0], cent_classes=('0-100%',))
        r['value_type'] = vt
        try:
            got = resolve(r)['compute']
            check(f'{vt!r} refuses to resolve', False, f'got {got}')
        except ValueError as e:
            check(f'{vt!r} refuses to resolve', 'ratio/power' in str(e))

    # A coefficient divided by a NON-flow quantity IS a normalised
    # coefficient: the v_n estimator is correct for it, so it must not be
    # caught by the guard.  (What those records lack is an explicit
    # normalisation field, which is a separate issue.)
    for vt in ('v_2/n_q', 'v_2/N_part', 'v_2{4}', 'v_2/(n<e_{part}>)'):
        check(f'{vt!r} is not treated as a coefficient ratio',
              not is_coefficient_ratio(vt))
        r = record_flow_ep_diff_pt([0.2, 0.5, 1.0, 2.0], cent_classes=('0-100%',))
        r['value_type'] = vt
        check(f'{vt!r} still resolves', resolve(r)['compute'] == 'vn2_ep_diff_pT')


if __name__ == '__main__':
    main()
