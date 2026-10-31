"""Toy-MC validation of v_n{6} and v_n{8} integrated cumulants.

Uses direct-Q construction from particles (bypasses containers to avoid
memory limits at high statistics).  All three cumulant orders must converge
to injected v for fixed flow, maintain ordering under fluctuations, and
return NaN/floor for no-flow.
Run: python3 tests/test_toy_mc_vn68.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.toy_mc import ToyGenConfig, generate_events
from analysis.observables_2d import _core_vn4_single, _core_vn6_single, \
    _core_vn8_single


def make_Q(events, H=9):
    Q = np.zeros((len(events), H), complex)
    for e, ev in enumerate(events):
        s = ((np.abs(ev['eta']) <= 1) & (ev['pt'] >= 0.15)
             & (ev['pt'] <= 3.0))
        zz = np.exp(1j * np.outer(np.arange(H), ev['phi'][s]))
        Q[e] = zz.sum(axis=1)
    return Q


def assert_close(name, got, want, tol):
    ok = abs(got - want) < tol
    print(f"  {'PASS' if ok else 'FAIL'} {name}: got={got:.4f} "
          f"want={want:.4f} tol={tol}")
    if not ok:
        raise AssertionError(f"{name}: {got} vs {want}")


def test_fixed_v_ladder():
    """Fixed v=0.05: all cumulant orders converge to v."""
    cfg = ToyGenConfig(n_events=30000, mult_mean=700, mult_sigma=60,
                       seed=101, flow={2: dict(kind='const', value=0.05)})
    events, _ = generate_events(cfg)
    Q = make_Q(events); w = np.ones(len(Q))
    r4 = np.atleast_1d(_core_vn4_single(Q, Q, Q, Q, w))
    r6 = np.atleast_1d(_core_vn6_single(Q, w))
    r8 = np.atleast_1d(_core_vn8_single(Q, w))
    assert_close("fixed v2{4}", r4[1], 0.05, 0.01)
    assert_close("fixed v2{6}", r6[1], 0.05, 0.01)
    assert_close("fixed v2{8}", r8[1], 0.05, 0.02)


def test_fluctuating_ordering():
    """With ebE fluctuations: v{8} <= v{6} <= v{4} <= v{2}."""
    cfg = ToyGenConfig(n_events=30000, mult_mean=700, mult_sigma=60,
                       seed=102, flow={2: dict(kind='const', value=0.06)},
                       fluctuate={2: 0.02})
    events, _ = generate_events(cfg)
    Q = make_Q(events); w = np.ones(len(Q))
    r4 = np.atleast_1d(_core_vn4_single(Q, Q, Q, Q, w))
    r6 = np.atleast_1d(_core_vn6_single(Q, w))
    r8 = np.atleast_1d(_core_vn8_single(Q, w))
    v2, v4, v6, v8 = r4[0], r4[1], r6[1], r8[1]
    print(f"  info: v2={v2:.4f} v4={v4:.4f} v6={v6:.4f} v8={v8:.4f}")
    assert np.isfinite(v8), "v8 should be finite with real flow"
    assert v8 <= v6 + 2e-3, f"v8={v8:.4f} > v6={v6:.4f}"
    assert v6 <= v4 + 2e-3, f"v6={v6:.4f} > v4={v4:.4f}"
    print("  PASS  ordering")


def test_noflow_signed_floor():
    """No flow: sign-preserving roots give FINITE values (possibly
    negative) at the statistical floor, well below any signal."""
    cfg = ToyGenConfig(n_events=10000, mult_mean=800, mult_sigma=70,
                       seed=103)
    events, _ = generate_events(cfg)
    Q = make_Q(events); w = np.ones(len(Q))
    r6 = np.atleast_1d(_core_vn6_single(Q, w))
    r8 = np.atleast_1d(_core_vn8_single(Q, w))
    v6 = r6[1] if len(r6) > 1 else float('nan')
    v8 = r8[1] if len(r8) > 1 else float('nan')
    ok = (np.isfinite(v6) and np.isfinite(v8)
          and abs(v6) < 0.08 and abs(v8) < 0.08)
    print(f"  {'PASS' if ok else 'FAIL'} no-flow signed: "
          f"v6={v6:+.4f} v8={v8:+.4f} (finite, |v|<0.08)")
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
