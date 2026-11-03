"""Jackknife blocks by event id (common-random-number campaigns).

Checks:
  - event-id blocks partition the events (block b: id % n_blocks == b);
  - two design points sharing seeds, with different events missing and a
    different container order, delete the same seeds in every block;
  - with correlated per-seed values at the two points, the event-id
    replicates give the cross-point noise correlation; contiguous blocks
    of differently ordered containers do not;
  - run_pipeline.jk_event_ids: auto uses meta['event_ids'] when there is one
    id per event, contiguous ignores it, event_id requires it;
  - run_block_jackknife with event ids: the delete-block jackknife of a mean;
  - analysis.jackknife.covariance_from_replicates (run_pipeline --covariance
    replicates) gives the covariance exactly.
Run: python3 tests/test_jackknife_blocks.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.jackknife import jackknife_blocks  # noqa: E402
from scripts.run_pipeline import jk_event_ids, run_block_jackknife  # noqa: E402

B = 20


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _point(rng, seeds, missing_frac):
    """A design point's container order: some seeds missing (no fluid), the
    rest shuffled, the missing ones appended at the end (as the iebe_music
    adapter pads them)."""
    lost = rng.random(len(seeds)) < missing_frac
    kept = rng.permutation(seeds[~lost])
    return np.concatenate([kept, seeds[lost]]), lost


def test_partition():
    print('test_partition')
    ids = np.random.default_rng(1).permutation(1000) + 7
    blocks = jackknife_blocks(len(ids), B, ids)
    deleted = np.array([~m for m in blocks])
    check('every event deleted in exactly one block', np.all(deleted.sum(axis=0) == 1))
    check('block b deletes id % B == b', all(np.all(ids[~m] % B == b) for b, m in enumerate(blocks)))
    try:
        jackknife_blocks(10, B, np.arange(9))
        check('wrong-length ids refused', False)
    except ValueError:
        check('wrong-length ids refused', True)


def test_same_seeds_across_points():
    print('test_same_seeds_across_points')
    rng = np.random.default_rng(2)
    seeds = np.arange(2000)
    order1, _ = _point(rng, seeds, 0.03)
    order2, _ = _point(rng, seeds, 0.08)
    b1, b2 = jackknife_blocks(len(order1), B, order1), jackknife_blocks(len(order2), B, order2)
    same = all(set(order1[~m1]) == set(order2[~m2]) for m1, m2 in zip(b1, b2))
    check('block b deletes the same seeds at both points', same)
    c1, c2 = jackknife_blocks(len(order1), B), jackknife_blocks(len(order2), B)
    overlap = np.mean([len(set(order1[~m1]) & set(order2[~m2])) / len(order1[~m1]) for m1, m2 in zip(c1, c2)])
    check('contiguous blocks of different orders share few seeds', overlap < 0.2, f'{overlap:.3f}')


def test_cross_point_covariance():
    print('test_cross_point_covariance')
    rng = np.random.default_rng(3)
    seeds = np.arange(4000)
    common = rng.normal(0, 1, len(seeds))                 # the seed's own fluctuation (CRN)
    x = {1: common + 0.3 * rng.normal(0, 1, len(seeds)), 2: common + 0.3 * rng.normal(0, 1, len(seeds))}
    reps = {}
    for rule in ('event_id', 'contiguous'):
        r = {}
        for k, frac in ((1, 0.03), (2, 0.08)):
            order, _ = _point(rng, seeds, frac)
            vals = x[k][order]
            blocks = jackknife_blocks(len(order), B, order if rule == 'event_id' else None)
            r[k] = np.array([vals[m].mean() for m in blocks])
        reps[rule] = np.corrcoef(r[1], r[2])[0, 1]
    want = 1 / (1 + 0.3 ** 2)                               # correlation of the per-seed values
    check('event-id replicates carry the cross-point correlation',
          abs(reps['event_id'] - want) < 0.1, f"{reps['event_id']:.3f} (expected ~{want:.2f})")
    check('contiguous replicates of reordered points do not', abs(reps['contiguous']) < 0.6,
          f"{reps['contiguous']:.3f}")


def test_jk_event_ids():
    print('test_jk_event_ids')
    ids = np.arange(50) * 3
    c = SimpleNamespace(n_events=50, meta={'event_ids': ids})
    check('auto uses meta event_ids', np.array_equal(jk_event_ids(c, 'auto'), ids))
    check('contiguous ignores them', jk_event_ids(c, 'contiguous') is None)
    check('auto without meta -> contiguous', jk_event_ids(SimpleNamespace(n_events=50), 'auto') is None)
    check('auto with a wrong-length list -> contiguous',
          jk_event_ids(SimpleNamespace(n_events=49, meta={'event_ids': ids}), 'auto') is None)
    try:
        jk_event_ids(SimpleNamespace(n_events=50), 'event_id')
        check('event_id without ids refused', False)
    except SystemExit:
        check('event_id without ids refused', True)


def test_run_block_jackknife():
    print('test_run_block_jackknife')
    rng = np.random.default_rng(4)
    vals = rng.normal(2.0, 1.0, 3000)
    ids = rng.permutation(3000) + 100
    corr, err, cov = run_block_jackknife(lambda m: np.array([vals[m].mean()]), len(vals), B, event_ids=ids)
    reps = np.array([vals[m].mean() for m in jackknife_blocks(len(vals), B, ids)])
    want = np.sqrt((B - 1) / B * np.sum((reps - reps.mean()) ** 2))
    check('error = delete-block jackknife of the mean', np.isclose(err[0], want), f'{err[0]:.5f} vs {want:.5f}')
    # unbiased: over 50 samples the error averages sigma/sqrt(N) (one sample
    # scatters by ~1/sqrt(2 (B - 1)) = 16%)
    ratios = []
    for seed in range(50):
        r = np.random.default_rng(100 + seed)
        v, i = r.normal(2.0, 1.0, 3000), r.permutation(3000)
        ratios.append(run_block_jackknife(lambda m: np.array([v[m].mean()]), 3000, B, event_ids=i)[1][0]
                      * np.sqrt(3000))
    check('error averages sigma/sqrt(N)', abs(np.mean(ratios) - 1) < 0.08, f'{np.mean(ratios):.3f}')
    corr, err, cov = run_block_jackknife(lambda m: np.array([vals[m].mean()]), len(vals), B, event_ids=ids)
    full, rep = run_block_jackknife.last
    check('replicates kept for --save-jackknife-replicates', rep.shape == (B, 1) and np.isclose(full[0], vals.mean()))
    # --covariance replicates: the replicates give the covariance exactly
    from analysis.jackknife import covariance_from_replicates
    x = rng.normal(0, 1, (3000, 5)) @ rng.normal(0, 1, (5, 5))
    corr, err, cov = run_block_jackknife(lambda m: x[m].mean(axis=0), len(x), B, event_ids=ids)
    check('covariance_from_replicates = the stored covariance',
          np.allclose(covariance_from_replicates(run_block_jackknife.last[1]), cov, rtol=1e-12, atol=0))


if __name__ == '__main__':
    tests = [test_partition, test_same_seeds_across_points, test_cross_point_covariance, test_jk_event_ids,
             test_run_block_jackknife]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
            print(f'  PASS  {t.__name__}\n')
        except AssertionError as ex:
            print(f'  FAIL  {t.__name__}: {ex}\n')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
