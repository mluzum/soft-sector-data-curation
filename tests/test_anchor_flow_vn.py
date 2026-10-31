import sys
import warnings

import numpy as np

sys.path.insert(0, '.')
warnings.simplefilter('ignore')
from scripts.compute_flow_vn import run_method


def _make(n=200):
    mult = (n - np.arange(n)).astype(float)  # event 0 = most central
    Qn = np.zeros((n, 6), complex)
    Qn[:, 0] = mult
    for h in range(1, 6):
        Qn[:, h] = (0.3 * mult) * np.exp(1j * np.linspace(0, 6.28, n))
    dQ0 = mult.copy()
    return mult, Qn, dQ0


def _edges():
    # continuous (as real records), covering 0-100%
    cent_edges = [(0, 5), (5, 10), (10, 90), (90, 95), (95, 100)]
    cent_pct = [0, 5, 10, 90, 95, 100]
    classes = ['0-5%', '5-10%', '10-90%', '90-95%', '95-100%']
    return cent_edges, cent_pct, classes


def test_default_no_exclusions():
    mult, Qn, dQ0 = _make()
    ce, cp, cl = _edges()
    valid, partial = run_method(None, mult, ce, cp, cl, Qn, dQ0, Qn, dQ0,
                                'simple', 1.0, cent_anchor=None)[6:8]
    assert valid.all(), valid
    assert not partial.any(), partial


def test_anchor_excludes_peripheral():
    mult, Qn, dQ0 = _make()
    ce, cp, cl = _edges()
    valid, partial = run_method(None, mult, ce, cp, cl, Qn, dQ0, Qn, dQ0,
                                'simple', 1.0, cent_anchor=0.8)[6:8]
    # f=0.8 -> anchor_cent=80; 90-95% and 95-100% beyond coverage -> excluded
    assert valid.tolist() == [True, True, True, False, False], valid.tolist()
    # 10-90% straddles 80 -> partial
    assert partial.tolist() == [False, False, True, False, False], partial.tolist()


def test_anchor_partial_straddle():
    mult, Qn, dQ0 = _make()
    ce, cp, cl = _edges()
    valid, partial = run_method(None, mult, ce, cp, cl, Qn, dQ0, Qn, dQ0,
                                'simple', 1.0, cent_anchor=0.92)[6:8]
    # 95-100% excluded; 90-95% straddles 92 -> partial
    assert valid.tolist() == [True, True, True, True, False], valid.tolist()
    assert partial.tolist() == [False, False, False, True, False], partial.tolist()


def test_poisson_anchor_excludes_peripheral():
    mult, Qn, dQ0 = _make()
    ce, cp, cl = _edges()
    valid, partial = run_method(None, mult, ce, cp, cl, Qn, dQ0, Qn, dQ0,
                                'poisson', 1.0, cent_anchor=0.8)[6:8]
    assert valid.tolist() == [True, True, True, False, False], valid.tolist()


def test_excluded_bins_are_nan():
    mult, Qn, dQ0 = _make()
    ce, cp, cl = _edges()
    v2 = run_method(None, mult, ce, cp, cl, Qn, dQ0, Qn, dQ0,
                    'simple', 1.0, cent_anchor=0.8)[0]
    assert np.isnan(v2[3, 0]) and np.isnan(v2[4, 0])
    assert not np.isnan(v2[0, 0])


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
