"""Every v_n from a cumulant or a reference power is a signed root.

v_n{2} = sign(c2) sqrt|c2|, v_n{4} = sign(-c4) |c4|^(1/4), v{6}, v{8} likewise
(correlators.vn_from_cumulant): the map is continuous and invertible
(c = sign(v) |v|^k x norm), gives no NaN, and keeps a jackknife replicate
whose cumulant crosses zero instead of replacing it.  A POI flow divides by
the signed reference: v'{2} = d2 / v{2}, v'{4} = -d4 / v{4}^3; a
scalar-product reference amplitude is sign(P) sqrt|P| of its power P; the
no-reference amplitude estimator likewise.  Until November 2026 v'{4} dropped
the sign of -c4, and a non-positive reference power or amplitude gave NaN.
Checks on constructed sums (one event, unit class weight unless stated).
Run: python3 tests/test_signed_roots.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.correlators import vn_from_cumulant, vn_poi  # noqa: E402
from analysis.observables_2d import (_amplitude_vn, _core_vn2_sp_diff,  # noqa: E402
                                     _core_vn2_sp_diff_means, compute_vn2_sp)

results = []


def check(name, cond, detail=''):
    results.append((name, bool(cond)))
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def test_reference_cumulants():
    for order, norm in ((2, 1.0), (4, -1.0), (6, 4.0), (8, -33.0)):
        c = np.array([-3e-4, 2e-5, 0.0])
        v = vn_from_cumulant(c, order)
        back = np.sign(v) * np.abs(v) ** order * norm
        check(f'v{{{order}}}: signed root, invertible, no NaN', np.all(np.isfinite(v)) and np.allclose(back, c, atol=1e-20),
              str(v))


def _poi_arrays(c2, d2, four, four_p):
    one = np.ones(1)
    return {'N2': np.array([[c2]]), 'D2': one, 'N2p': np.array([[d2]]), 'D2p': one,
            'N4': np.array([[four]]), 'D4': one, 'N4p': np.array([[four_p]]), 'D4p': one}


def test_poi():
    w = np.ones(1)
    c2, d2 = 0.01, 0.012
    for four, label in ((0.0001, 'c4 < 0'), (0.0003, 'c4 > 0')):
        a = _poi_arrays(c2, d2, four, 0.0005)
        c4 = four - 2 * c2 ** 2
        d4 = 0.0005 - 2 * d2 * c2
        v4 = vn_from_cumulant(np.array([c4]), 4)[0]
        vp4 = vn_poi(a, w, 4)[0]
        check(f"v'{{4}} = -d4 / v{{4}}^3 with the signed v{{4}} ({label})", np.isclose(vp4 * v4 ** 3, -d4), f'{vp4:.4f}')
    a = _poi_arrays(-c2, d2, 0.0001, 0.0005)
    check("v'{2} = d2 / v{2} with c2 < 0", np.isclose(vn_poi(a, w, 2)[0], d2 / vn_from_cumulant(np.array([-c2]), 2)[0]))


def test_scalar_product():
    # reference power P = <|Q_ref|^2> - <n_ref> < 0: the amplitude is -sqrt|P|
    q_ref = np.array([[0, 2.0 + 0j, 0], [0, 1.0 + 0j, 0]])          # (n_events, n_harm)
    n_ref = np.array([10.0, 10.0])
    P = np.mean(np.abs(q_ref[:, 1]) ** 2) - 10.0
    q_poi = np.array([[[0], [0.6 + 0j], [0]], [[0], [0.3 + 0j], [0]]])  # (n_events, n_harm, 1 bin)
    n_poi = np.array([[3.0], [3.0]])
    w = np.ones(2)
    v = _core_vn2_sp_diff(q_poi, n_poi, q_ref, w, n_ref=n_ref).reshape(2, 1)
    num = np.mean(np.real(q_poi[:, 1, 0] / 3.0 * np.conj(q_ref[:, 1])))
    check('SP differential: a negative reference power flips the sign (was NaN)',
          P < 0 and np.isclose(v[0, 0], num / (-np.sqrt(-P))), f'{v[0, 0]:.4f}')
    vi = compute_vn2_sp(q_poi[:, :, 0], q_ref, n_poi[:, 0], n_ref, w)
    check('SP integrated: signed reference amplitude (was NaN)', np.isclose(vi[0], num * 3.0 / (-np.sqrt(-P) * 3.0)),
          f'{vi[0]:.4f}')
    vm = _core_vn2_sp_diff_means(q_poi, n_poi, q_ref, n_ref, w).reshape(2, 1)
    check('SP JETSCAPE form: a negative reference c2 gives a finite, negative-denominator value (was NaN)',
          np.isfinite(vm[0, 0]) and np.sign(vm[0, 0]) == -np.sign(num), f'{vm[0, 0]:.4f}')
    empty = _core_vn2_sp_diff(q_poi, n_poi, np.zeros_like(q_ref), w, n_ref=np.zeros(2)).reshape(2, 1)
    check('SP: an empty reference (zero power) stays NaN', np.isnan(empty[0, 0]))


def test_amplitude():
    q = np.array([1.0 + 0j, 2.0 + 0j])
    n = np.array([10.0, 10.0])
    v = _amplitude_vn(q, n, np.ones(2))
    want = -np.sqrt((10.0 - np.mean(np.abs(q) ** 2)) / 90.0)
    check('amplitude estimator: <|Q|^2> < <M> gives a negative signed root (was NaN)', np.isclose(v, want), f'{v:.4f}')


if __name__ == '__main__':
    for t in (test_reference_cumulants, test_poi, test_scalar_product, test_amplitude):
        print(t.__name__)
        t()
    failed = [n for n, ok in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} passed')
    if failed:
        raise SystemExit('FAILED: ' + ', '.join(failed))
