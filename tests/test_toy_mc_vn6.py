"""Toy-MC validation of v_n{6} (integrated cumulant, unit weights).

Checks:
  - fixed injected v2 -> v2{6} recovers it (within JK/statistical tolerance)
  - event-wise fluctuating v2 -> ordering v2{6} <= v2{4} <= v2{2}
  - no flow -> a signed value at the statistical floor
Run: python3 tests/test_toy_mc_vn6.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.toy_mc import ToyGenConfig, generate_events, make_qgrid_data, \
    record_flow_vn2
from analysis.dispatcher import extract_measurement_params, resolve
from analysis.pipeline_dispatch import get_pipeline_entry


def _run(cfg):
    events, _ = generate_events(cfg)
    grid = make_qgrid_data(events, {'cfg': cfg}, n_eta=30, n_pt=30,
                           n_harm=10)
    rec = record_flow_vn2(harmonic=2, cent_classes=('0-100%',))
    rec.update(method='cumulant', cumulant_order=6)
    assert resolve(rec)['compute'] == 'vn6_single'
    params = extract_measurement_params(rec)
    entry = get_pipeline_entry('vn6_single')
    d = entry['prepare_data_3d'](grid, params)
    if d is None:
        return np.array([np.nan])
    w = np.ones(grid.n_events)
    raw = np.atleast_1d(entry['core_fn'](d, w))
    return raw[1:2] if len(raw) > 1 else np.array([np.nan])


def test_fixed_flow_recovery():
    cfg = ToyGenConfig(n_events=500, mult_mean=1200, mult_sigma=100,
                       seed=11, flow={2: dict(kind='const', value=0.05)})
    got = float(_run(cfg)[0])
    ok = abs(got - 0.05) < 0.01
    print(f"  {'PASS' if ok else 'FAIL'} fixed-v v2{{6}}={got:.4f} "
          f"(injected 0.0500)")
    assert ok


def test_fluctuating_ordering():
    # v{6} = v{4} for these fluctuations: the ordering test needs the
    # statistical spread of v{6} - v{4} below its 1e-3 tolerance (600 events
    # gave +2e-3 once every order was weighted by its tuple count)
    cfg = ToyGenConfig(n_events=3000, mult_mean=1200, mult_sigma=100,
                       seed=12, flow={2: dict(kind='const', value=0.06)},
                       fluctuate={2: 0.02})
    events, truth = generate_events(cfg)
    grid = make_qgrid_data(events, {'cfg': cfg}, n_eta=30, n_pt=30, n_harm=10)
    from analysis.dispatcher import extract_measurement_params as emp
    from analysis.pipeline_dispatch import get_pipeline_entry as gpe
    out = {}
    for order, key in ((2, 'vn2_single'), (4, 'vn4_single'), (6, 'vn6_single')):
        rec = record_flow_vn2(harmonic=2, cent_classes=('0-100%',))
        rec.update(method='cumulant', cumulant_order=order)
        params = emp(rec)
        d = gpe(key)['prepare_data_3d'](grid, params)
        if key == 'vn6_single':
            raw = np.atleast_1d(gpe(key)['core_fn'](d, np.ones(grid.n_events)))
            out[order] = float(raw[1])
        else:
            raw = np.atleast_1d(gpe(key)['core_fn'](d, np.ones(grid.n_events)))
            out[order] = float(raw[1])
    print(f"  info: v2{{2}}={out[2]:.4f} v2{{4}}={out[4]:.4f} "
          f"v2{{6}}={out[6]:.4f}")
    ok = (np.isfinite(out[6]) and out[6] <= out[4] + 1e-3
          and out[4] <= out[2] + 1e-3)
    print(f"  {'PASS' if ok else 'FAIL'} ordering v{{6}}<=v{{4}}<=v{{2}}")
    assert ok


def test_noflow_signed_floor():
    """No flow: sign-preserving root gives a finite value at the noise
    floor, magnitude well below any physical signal."""
    cfg = ToyGenConfig(n_events=300, mult_mean=1200, mult_sigma=100,
                       seed=13)
    got = float(_run(cfg)[0])
    ok = np.isfinite(got) and abs(got) < 0.05
    print(f"  {'PASS' if ok else 'FAIL'} no-flow signed floor (got {got:+.4f})")
    assert ok


if __name__ == '__main__':
    tests = [fn for fn in sorted(dir()) if fn.startswith('test_')]
    n_pass = n_fail = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"  PASS  {name}\n")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}\n")
            n_fail += 1
    print(f"{n_pass}/{n_pass + n_fail} passed")
    sys.exit(1 if n_fail else 0)
