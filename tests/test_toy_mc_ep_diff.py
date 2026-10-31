"""Toy-MC validation of the EVENT-PLANE DIFFERENTIAL estimator.

The event-plane differential core computes, per differential bin,

    v_n^2 = ( <|Q_n|^2> - <M> ) / <M(M-1)>

The <M> self-correlation subtraction is what makes this unbiased.  It was
missing until Sep 2026, and because 1/M overtakes v_n^2 once
M <~ 1/v_n^2 (~400 particles at v_n = 0.05), the omission inflated
differential flow by tens to hundreds of percent in exactly the narrow
bins that differential measurements use -- while integrated flow, computed
by a different core, was unaffected.  That asymmetry is what hid it.

This test pins both halves: the corrected estimator must recover v_true at
every multiplicity, and the naive one must be shown to fail (so a future
"simplification" that drops the subtraction is caught).

Run: python3 tests/test_toy_mc_ep_diff.py
"""
import sys

import numpy as np

sys.path.insert(0, '/'.join(__file__.split('/')[:-2]))

from analysis.observables_2d import _core_vn2_ep_diff_eta  # noqa: E402

NEV = 4000


def toy_q(M, v, phi0=0.7, seed=0, harmonics=(2,)):
    """Per-event Q_n for M particles; flow v injected ONLY into n=2."""
    r = np.random.default_rng(seed)
    phi = r.uniform(0, 2 * np.pi, (NEV, M))
    out = {}
    for n in range(1, max(harmonics) + 1):
        ang = n * phi
        wgt = (1 + 2 * v * np.cos(ang - n * phi0)) if n == 2 else 1.0
        out[n] = (np.exp(1j * ang) * wgt).sum(axis=1)
    return out


def run(M, v, harmonics=(2,), seed=0):
    """Return the flattened core output; row for harmonic n starts at n-1."""
    q = toy_q(M, v, seed=seed, harmonics=harmonics)
    nh = max(harmonics) + 1
    qp = np.zeros((NEV, nh, 1), dtype=complex)
    for n, arr in q.items():
        qp[:, n, 0] = arr
    npoi = np.full((NEV, 1), float(M))
    w = np.ones(NEV)
    return _core_vn2_ep_diff_eta(qp, npoi, w)


def at(flat, n, nbins=1):
    """Harmonic n, all bins, from the harmonic-major flattened layout."""
    return flat[(n - 1) * nbins: n * nbins]


def naive(M, v, seed=0):
    """The pre-fix estimator (no <M> subtraction), for the guard test."""
    q = toy_q(M, v, seed=seed)
    return np.sqrt(np.mean(np.abs(q[2]) ** 2) / M ** 2)


def main():
    n_pass = n_fail = 0

    def check(name, cond, detail=''):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
            print(f'  {name}: PASS {detail}')
        else:
            n_fail += 1
            print(f'  {name}: FAIL {detail}')

    print('test_ep_diff_self_correlation_subtraction')
    print(f"{'M/bin':>6s} {'v_true':>7s} {'corrected':>10s} {'dev%':>7s} "
          f"{'naive':>8s} {'naive dev%':>11s}")
    for M in (2000, 400, 100, 30):
        for v in (0.05, 0.02):
            got = float(at(run(M, v), 2)[0])
            nv = naive(M, v)
            print(f'{M:6d} {v:7.3f} {got:10.4f} {100*(got-v)/v:6.1f}% '
                  f'{nv:8.4f} {100*(nv-v)/v:10.0f}%')
            # Tolerance is 3 sigma on the estimator's intrinsic
            # statistical precision, which scales as ~0.3/sqrt(M) in
            # relative terms here (the residual non-flow noise after the
            # <M> subtraction is ~sqrt(1/M) in v^2, i.e. ~0.3/sqrt(M) in
            # v).  Floored at 10%.  The systematic part is covered
            # separately by test_ep_diff_multiplicity_scaling.
            tol = max(0.10, 0.9 / np.sqrt(M))
            check(f'M={M:<5d} v={v:.2f} recovers v_true within 3-sigma',
                  abs(got - v) <= tol * v, f'got {got:.4f}, tol {tol:.0%}')

    print('\ntest_ep_diff_multiplicity_scaling')
    # The naive estimator inflates as M falls; the corrected one must not.
    devs = [100 * (float(at(run(M, 0.05), 2)[0]) - 0.05) / 0.05
            for M in (2000, 400, 100, 30)]
    check('corrected deviation does not grow as M falls',
          max(devs) < 0.15 * max(devs[0], 1e-9) + 8, f'{[round(d, 1) for d in devs]}')
    ndevs = [100 * (naive(M, 0.05) - 0.05) / 0.05 for M in (2000, 400, 100, 30)]
    check('naive estimator DOES inflate as M falls (guards the fix)',
          ndevs[-1] > 100, f'{[round(d) for d in ndevs]}')

    print('\ntest_ep_diff_harmonics')
    flat = run(400, 0.05, harmonics=(2, 3, 4))
    # Flow is injected ONLY into n=2; n=1,3,4 must sit near the noise floor.
    check('injected n=2 recovered', abs(float(at(flat, 2)[0]) - 0.05) < 0.005,
          f'{float(at(flat, 2)[0]):.4f}')
    check('no-flow harmonics near zero',
          float(at(flat, 3)[0]) < 0.02 and float(at(flat, 4)[0]) < 0.02,
          f'n3={float(at(flat, 3)[0]):.4f} n4={float(at(flat, 4)[0]):.4f}')

    print('\ntest_ep_diff_degenerate_inputs')
    w = np.ones(NEV)
    qp = np.zeros((NEV, 3, 2), dtype=complex)
    npoi = np.zeros((NEV, 2))
    out = _core_vn2_ep_diff_eta(qp, npoi, w)
    check('empty bins give NaN, not a finite number',
          bool(np.all(~np.isfinite(out))), f'{out}')
    qp[:, 1, :] = 1.0
    out = _core_vn2_ep_diff_eta(qp, npoi, w)
    check('still NaN when |Q|=0 and M=0',
          bool(np.all(~np.isfinite(out))), f'{out}')

    total = n_pass + n_fail
    print()
    print(f'{n_pass}/{total} passed')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()