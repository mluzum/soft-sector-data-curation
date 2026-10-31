"""Per-event reweighting parity (2D/3D) via combine_weights.

Run: python3 tests/test_event_weights.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.weights import combine_weights


def test_combine_weights_shapes_and_product():
    n = 50
    cent_w = np.random.rand(n, 3) + 0.1
    ew = np.linspace(0.5, 1.5, n)
    out = combine_weights(cent_w, ew)
    assert out.shape == cent_w.shape
    # elementwise product across cent bins
    assert np.allclose(out, cent_w * ew[:, None])
    # None event weights => unchanged
    assert np.allclose(combine_weights(cent_w, None), cent_w)
    # None cent weights => event weights broadcast to column vector
    assert np.allclose(combine_weights(None, ew), ew[:, None])


def test_event_reweight_recovers_weighted_mean():
    """A weighted-mean-style observable with w_event must equal the plain
    weighted average, i.e. event_weights genuinely enter the estimator."""
    n = 200
    x = np.random.RandomState(0).randn(n) + 1.0
    w = np.random.RandomState(1).rand(n) + 0.5
    # weighted mean with combine_weights (single cent bin)
    cw = combine_weights(np.ones((n, 1)), w)
    weighted_mean = float(np.sum(cw[:, 0] * x) / np.sum(cw[:, 0]))
    ref = float(np.sum(w * x) / np.sum(w))
    assert abs(weighted_mean - ref) < 1e-12


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
