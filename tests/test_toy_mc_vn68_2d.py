"""v{6}/v{8} on 2D containers: exact weighted distinct-particle sums.

User requirement: scripts must be robust regardless of simulation source.
  - The 2D window weights (Jacobians) enter exactly (analysis.correlators
    reference_cumulant_arrays); until November 2026 the cores required an
    integer multiplicity and so dropped <6> and <8> on every weighted
    container (v{6} = (3/4)^(1/6) v without fluctuations).
  - Missing high harmonics enter only self-correlation terms of coinciding
    particles (relative order 1/M_eff): dropped with a printed note.
Run: python3 tests/test_toy_mc_vn68_2d.py
"""
import contextlib
import io
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


def _run(n_harm, order, n_events=2000):
    events, qv = _make_qv(n_harm=n_harm, n_events=n_events)
    rec = _record(order)
    assert resolve(rec)['compute'] == f'vn{order}_single'
    params = extract_measurement_params(rec)
    entry = get_pipeline_entry(f'vn{order}_single')
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        d = entry['prepare_data_2d'](qv, params)
    raw = np.atleast_1d(entry['core_fn'](d, np.ones(qv.n_events)))
    return (raw[1] if len(raw) > 1 else np.nan), out.getvalue()


def test_vn6_2d_limited_harmonics():
    """Production-like 2D data (6 slots, Q0..Q5): Q6 missing -> note."""
    got, printed = _run(6, 6)
    print(f"  info: v2{{6}} (2D, 6 slots) = {got:.4f} (injected 0.05)")
    assert np.isfinite(got) and abs(got - 0.05) < 0.004
    assert 'not stored' in printed, "expected a note about missing harmonics"
    print("  PASS  vn6 2D limited-harmonic + note")


def test_vn8_2d_full_harmonics():
    """2D with 9 slots -> v{8} exact."""
    got, printed = _run(9, 8)
    print(f"  info: v2{{8}} (2D, 9 slots) = {got:.4f} (injected 0.05)")
    assert np.isfinite(got) and abs(got - 0.05) < 0.004
    assert 'not stored' not in printed
    print("  PASS  vn8 2D full-harmonic")


def test_vn8_2d_limited_harmonics():
    """2D with 6 slots: Q6, Q8 missing -> note, value still near truth."""
    got, printed = _run(6, 8, n_events=1500)
    print(f"  info: v2{{8}} (2D, 6 slots) = {got:.4f} (injected 0.05)")
    assert np.isfinite(got) and abs(got - 0.05) < 0.005
    assert 'not stored' in printed
    print("  PASS  vn8 2D limited-harmonic + note")


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
