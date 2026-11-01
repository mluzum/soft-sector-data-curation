"""Toy-MC validation of rapidity-odd directed flow v1(eta) (v1_odd_sp_diff_eta).

Truth: v1(eta) = -0.015 eta + 0.0012 eta^3 per particle (plus v2 = 0.05),
negative around midrapidity and positive at large |eta|, like PHOBOS at
19.6 GeV.  The record follows PHOBOS: the octagon-like region
(|eta| < 3) uses an eta-odd reference at 3 < |eta| < 5, the ring-like
region an eta-odd reference at 1.5 < |eta| < 3, and v1 is negative over
1.5 < eta < 3 by convention.  The two references carry directed flow of
opposite sign, so the segments must be aligned for one consistent curve.
Run: python3 tests/test_toy_mc_v1_odd.py
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.toy_mc import ToyGenConfig, generate_events, make_qgrid_data, evaluate  # noqa: E402

BINS = [(-4.5, -3.85), (-3.0, -2.5), (-2.0, -1.5), (-1.0, -0.5), (-0.5, -0.1), (0.1, 0.5),
        (0.5, 1.0), (1.5, 2.0), (2.5, 3.0), (3.13, 3.72), (3.85, 4.52), (4.6, 5.0)]


def shape(eta):
    return -0.015 * eta + 0.0012 * eta ** 3


RECORD = dict(
    observable_type='flow_vn', value_type='v1', harmonic=1, method='event_plane',
    dimensionality='differential_eta', hepdata_file='toy.yaml', inspire_id='0', version=1,
    particle_id=dict(species='charged'),
    kinematic_range=dict(eta=dict(min=-4.5, max=5.0, bin_ranges=[list(b) for b in BINS]),
                         pT=dict(min=0.05, max=3.0)),
    centrality=dict(classes=['0-100%'], eta=[dict(min=-5.5, max=-5.0), dict(min=5.0, max=5.5)]),
    flow_reference=dict(
        odd=True,
        segments=[dict(applies_to_eta=dict(min=0.0, max=3.0), eta=[dict(min=-5.0, max=-3.0), dict(min=3.0, max=5.0)]),
                  dict(applies_to_eta=dict(min=3.0, max=5.5), eta=[dict(min=-3.0, max=-1.5), dict(min=1.5, max=3.0)])],
        sign_convention=dict(eta=dict(min=1.5, max=3.0), sign='negative')))

results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + ('' if ok else f'  {detail}'))


def fixture():
    cfg = ToyGenConfig(n_events=3000, mult_mean=1500, mult_sigma=100, rap_max=6.0, seed=91, pt_min=0.05, pt_max=3.0,
                       flow={1: dict(kind='const', value=1.0), 2: dict(kind='const', value=0.05)},
                       eta_shape={1: shape})
    events, _ = generate_events(cfg)
    grid = make_qgrid_data(events, {'cfg': cfg}, n_eta=120, n_pt=10)
    truth = []
    for lo, hi in BINS:
        vals = np.concatenate([shape(ev['eta'][(ev['eta'] >= lo) & (ev['eta'] < hi)]) for ev in events])
        truth.append(vals.mean())
    return grid, np.array(truth)


def main():
    from analysis.dispatcher import resolve
    grid, truth = fixture()
    vals, errs = evaluate(RECORD, grid=grid, cent_method='simple', n_blocks=20)
    for (lo, hi), v, e, t in zip(BINS, vals, errs, truth):
        print(f"    eta {lo:+.2f}..{hi:+.2f}: v1 = {v:+.4f} +- {e:.4f}   truth {t:+.4f}")
    pull = np.abs(vals - truth) / np.maximum(errs, 1e-6)
    check('every bin agrees with the input v1 within 4 sigma', np.all(pull < 4), f'pulls {pull.round(1)}')
    # One overall scale between result and input: a normalization error (e.g.
    # a self-correlation term off by the charge factor gave ~0.55) shows
    # here even when every bin is within a few sigma.
    w2 = truth ** 2 / errs ** 2
    scale, s_err = np.sum(vals * truth / errs ** 2) / w2.sum(), 1 / np.sqrt(w2.sum())
    print(f"    fitted scale result/input = {scale:.3f} +- {s_err:.3f}")
    check('overall scale consistent with 1 and known to 0.1', abs(scale - 1) < 3 * s_err and s_err < 0.1,
          f'{scale:.3f} +- {s_err:.3f}')
    check('sign pattern: negative at 1.5 < eta < 3, positive at eta > 4', vals[7] < 0 and vals[8] < 0 and vals[10] > 0)
    check('odd in eta: v1(-0.75) = -v1(0.75) within errors',
          abs(vals[3] + vals[6]) < 4 * np.hypot(errs[3], errs[6]))

    flipped = copy.deepcopy(RECORD)
    flipped['flow_reference']['sign_convention']['sign'] = 'positive'
    v2, _ = evaluate(flipped, grid=grid, cent_method='simple', with_jackknife=False)
    v1n, _ = evaluate(RECORD, grid=grid, cent_method='simple', with_jackknife=False)
    check('the sign convention flips the whole curve', np.allclose(v2, -v1n))

    for name, rec in (('without an odd reference', {k: v for k, v in RECORD.items() if k != 'flow_reference'}),
                      ('without a sign convention', dict(RECORD, flow_reference=dict(RECORD['flow_reference'], sign_convention=None)))):
        try:
            resolve(rec)
            ok = False
        except ValueError:
            ok = True
        check(f'harmonic-1 record {name} is refused', ok)
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
