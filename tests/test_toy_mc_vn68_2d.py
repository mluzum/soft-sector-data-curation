"""Robustness tests: limited-harmonic and unit-weight 2D paths for v{6}/{8}.

User requirement: scripts must be robust regardless of simulation source.
  - Missing high harmonics (needed only for self-correlation subtraction)
    default to zero with a warning (suppressed by 1/M, negligible at
    heavy-ion multiplicity).
  - Missing weighted-cumulant formulas default to the unit-weight
    simplified expressions with a warning.
Both defaults are exercised here.
Run: python3 tests/test_toy_mc_vn68_2d.py
"""
import os
import sys
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.toy_mc import ToyGenConfig, generate_events
from analysis.converters import particles_to_qvector_data
from analysis.dispatcher import extract_measurement_params, resolve
from analysis.pipeline_dispatch import get_pipeline_entry


def _make_qv(n_harm, n_events=2000):
    cfg = ToyGenConfig(n_events=n_events, mult_mean=800, mult_sigma=80,
                       seed=42, flow={2: dict(kind='const', value=0.05)})
    events, _ = generate_events(cfg)
    return events, particles_to_qvector_data(
        events, pt_edges=np.linspace(0.05, 3.0, 31), rap_max=1.0,
        species_names=['pion', 'kaon', 'proton'],
        masses=[0.14, 0.49, 0.94], n_harm=n_harm)


def _record(order):
    return {'observable_type': 'flow_vn', 'method': 'cumulant',
            'cumulant_order': order, 'harmonic': 2,
            'dimensionality': 'integrated',
            'value_type': f'v2{{{order}}}',
            'kinematic_range': {'eta': {'min': -1, 'max': 1},
                                'pT': {'min': 0.2, 'max': 3}},
            'centrality': {'classes': ['0-100%'],
                           'eta': [{'min': -3.9, 'max': -3.0},
                                   {'min': 3.0, 'max': 3.9}]}}


def test_vn6_2d_limited_harmonics():
    """Production-like 2D data (6 slots, Q0..Q5) -> v{6} via zero-padding."""
    events, qv = _make_qv(n_harm=6)
    rec = _record(6)
    assert resolve(rec)['compute'] == 'vn6_single'
    params = extract_measurement_params(rec)
    entry = get_pipeline_entry('vn6_single')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = entry['prepare_data_2d'](qv, params)
        raw = np.atleast_1d(entry['core_fn'](d, np.ones(qv.n_events)))
    got = raw[1] if len(raw) > 1 else np.nan
    warned = any('padding' in str(w.message) or 'self-corr' in str(w.message)
                 for w in caught)
    print(f"  info: v2{{6}} (2D, 6 slots) = {got:.4f} (injected 0.05), "
          f"warning issued: {warned}")
    # Self-correlation omission is O(1/M^5)-suppressed at high order --
    # generous tolerance, but must be a sane finite number near truth.
    assert np.isfinite(got) and abs(got - 0.05) < 0.02
    assert warned, "expected a warning about missing harmonics"
    print("  PASS  vn6 2D limited-harmonic + warning")


def test_vn8_2d_full_harmonics():
    """2D with 9 slots -> v{8} works exactly (no padding needed)."""
    events, qv = _make_qv(n_harm=9)
    rec = _record(8)
    assert resolve(rec)['compute'] == 'vn8_single'
    params = extract_measurement_params(rec)
    entry = get_pipeline_entry('vn8_single')
    d = entry['prepare_data_2d'](qv, params)
    raw = np.atleast_1d(entry['core_fn'](d, np.ones(qv.n_events)))
    got = raw[1] if len(raw) > 1 else np.nan
    print(f"  info: v2{{8}} (2D, 9 slots) = {got:.4f} (injected 0.05)")
    assert np.isfinite(got) and abs(got - 0.05) < 0.02
    print("  PASS  vn8 2D full-harmonic")


def test_vn8_2d_limited_harmonics_graceful():
    """2D with 6 slots -> v{8} may return NaN (insufficient info) but must
    NOT crash; the missing-harmonic warning fires."""
    events, qv = _make_qv(n_harm=6, n_events=1500)
    rec = _record(8)
    params = extract_measurement_params(rec)
    entry = get_pipeline_entry('vn8_single')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = entry['prepare_data_2d'](qv, params)
        raw = np.atleast_1d(entry['core_fn'](d, np.ones(qv.n_events)))
    warned = any('padding' in str(w.message) or 'self-corr' in str(w.message)
                 for w in caught)
    print(f"  info: v2{{8}} (2D, 6 slots) = {raw[1] if len(raw)>1 else 'n/a'} "
          f"(NaN acceptable: insufficient harmonics), warned={warned}")
    assert warned
    # No crash = pass; value may be NaN by design.
    print("  PASS  vn8 2D limited-harmonic graceful")


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
