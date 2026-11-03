"""Observable selection for emulator training (analysis/observable_selection.py).

Whole observables (columns of a runs x observables table) are selected by
the model's precision across the design; values are never changed.  Checks
on a synthetic table:
  - the summary (undefined fraction, median relative error, median model /
    experimental error ratio, data present);
  - each criterion rejects the observable built to fail it, and the first
    failing criterion is reported;
  - with no thresholds only undefined columns and columns without data go;
  - the inputs are not modified.
Run: python3 tests/test_observable_selection.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.observable_selection import quality, select  # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")


def table():
    """6 runs x 5 observables: good | undefined in one run | no data | noisy vs data | noisy relative."""
    v = np.tile([0.06, 0.03, 100.0, 0.05, 0.002], (6, 1)) * (1 + 0.01 * np.arange(6)[:, None])
    e = np.tile([0.001, 0.001, 1.0, 0.02, 0.002], (6, 1))
    v[2, 1] = np.nan
    e[2, 1] = np.nan
    ev = np.array([0.06, 0.03, np.nan, 0.05, 0.002])
    ee = np.array([0.002, 0.002, np.nan, 0.004, 0.01])
    return v, e, ev, ee


def test_summary():
    v, e, ev, ee = table()
    q = quality(v, e, ev, ee)
    check('undefined fraction', np.allclose(q['undefined_fraction'], [0, 1 / 6, 0, 0, 0]))
    check('median model / experimental error ratio',
          np.allclose(q['median_error_ratio'][[0, 1, 3, 4]], [0.5, 0.5, 5.0, 0.2]) and np.isnan(q['median_error_ratio'][2]))
    check('median relative error', abs(q['median_rel_error'][4] - np.median(0.002 / v[:, 4])) < 1e-12)
    check('data present', list(q['has_data']) == [True, True, False, True, True])


def test_select():
    v, e, ev, ee = table()
    before = [x.copy() for x in (v, e, ev, ee)]
    q = quality(v, e, ev, ee)
    sel, why = select(q)
    check('defaults: undefined and no-data columns rejected',
          list(sel) == [True, False, False, True, True] and list(why) == ['', 'undefined', 'no_data', '', ''])
    sel, why = select(q, max_error_ratio=1.0, max_rel_error=0.5)
    check('noise (model error > data error) and relative noise rejected',
          list(why) == ['', 'undefined', 'no_data', 'noise', 'rel_noise'], f'{list(why)}')
    sel, why = select(q, max_undefined=0.5, require_data=False)
    check('looser: a column undefined in 1/6 of the runs and one without data kept', sel.all())
    check('inputs unchanged', all(np.array_equal(a, b, equal_nan=True) for a, b in zip((v, e, ev, ee), before)))


if __name__ == '__main__':
    for t in (test_summary, test_select):
        print(t.__name__)
        t()
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
