"""Centrality classes in the jackknife: edges rebuilt in each replicate.

The class edges are percentiles of the simulated events, and a record's
normalization (threshold_anchor) is a fraction of them: both are estimated
from the sample, so each jackknife replicate rebuilds them from the events
it keeps (analysis.weights.CentralityWeights).  With weights fixed from the
full sample, <N> per class came out with 2-4x too small an error (zr96c2).
Checks:
  - the full sample's weights equal cent_weights exactly (simple, Poisson,
    a threshold normalization); a replicate's equal cent_weights on the kept
    events (the Poisson search started from the full sample's crossings);
    the arrays are read-only and shared within a replicate;
  - independent samples through the production path (toy_mc.evaluate, 2D
    containers): the spread of dN/deta per class over the samples equals the
    mean jackknife error; with the weights fixed, the dN/deta errors are 2x
    and more too small.  v2{2}, a quantity whose per-event noise is
    uncorrelated with the estimator: calibrated with Poisson weights (the
    default); with sharp percentile edges ('simple') conservative by ~1.2:
    a replicate re-ranks a few kept events across each edge, whose noise
    enters (n_blocks - 1) times;
  - run_pipeline (toy package): the errors of the toy harness on the same
    container, so its weights are rebuilt per replicate too.
Run: python3 tests/test_jackknife_centrality.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis import weights as W  # noqa: E402
from analysis.pipeline_dispatch import PTCUTS  # noqa: E402
from analysis.sim_package import build_container  # noqa: E402
from analysis.toy_mc import evaluate, record_flow_vn2, record_yield  # noqa: E402

CLASSES = ('0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%')
results = []


def check(name, cond, detail=''):
    results.append((name, bool(cond)))
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def test_weights():
    rng = np.random.default_rng(5)
    x = rng.gamma(2.0, 40.0, 3000)
    edges = [(0.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, 60.0), (70.0, 80.0)]
    anchor = W.resolve_anchor(0.9, len(x))
    norm = {'estimator_threshold': 30, 'fraction_above': 0.8}
    mask = np.ones(len(x), bool)
    mask[rng.choice(len(x), 150, replace=False)] = False
    for method in ('simple', 'poisson'):
        cw = W.CentralityWeights(method, x, anchor, efficiency=0.9)
        full = W.cent_weights(method, x, edges, n_ref=anchor['n_ref'], efficiency=0.9,
                              anchor_cent=anchor['anchor_cent'])
        check(f'{method}: full sample = cent_weights', np.array_equal(cw.weights(edges), full))
        rep = W.cent_weights(method, x[mask], edges, n_ref=anchor['n_ref'] * mask.sum() / len(x),
                             efficiency=0.9, anchor_cent=anchor['anchor_cent'])
        check(f'{method}: replicate = cent_weights of the kept events (same f)',
              np.allclose(cw.weights(edges, mask), rep, rtol=0, atol=1e-15))
        cn = W.CentralityWeights(method, x, anchor, efficiency=0.87, normalization=norm)
        a = W.threshold_anchor(x[mask], method, 30, 0.8, efficiency=0.87)
        rep = W.cent_weights(method, x[mask], edges, n_ref=a['n_ref'], efficiency=0.87,
                             anchor_cent=a['anchor_cent'])
        check(f'{method}: normalization rebuilt from the kept events',
              cn.anchor(mask)['f'] == a['f'] and np.allclose(cn.weights(edges, mask), rep, rtol=0, atol=1e-15)
              and cn.anchor()['f'] != a['f'])
    cw = W.CentralityWeights('poisson', x, anchor)
    r1 = cw.weights(edges, mask)
    check('one replicate: built once, read-only', cw.weights(edges, mask) is r1 and not r1.flags.writeable)
    check('a mask keeping every event = the full sample',
          cw.weights(edges, np.ones(len(x), bool)) is cw.weights(edges))
    ew = rng.uniform(0.5, 2.0, len(x))
    ce = W.CentralityWeights('simple', x, anchor, event_weights=ew)
    check('event weights multiply the class weights',
          np.array_equal(ce.weights(edges, mask), ce.weights(edges, mask, raw=True) * ew[mask][:, None]))


def container(rng, n_ev=1500):
    """A 2D container (JETSCAPE layout) with impact-parameter-driven
    multiplicity, fluctuations, and v2 rising with b."""
    n_pt = len(PTCUTS) - 1
    pc = 0.5 * (PTCUTS[:-1] + PTCUTS[1:])
    dpt = np.diff(PTCUTS)
    b2 = rng.uniform(0, 1, n_ev)
    lam = 600.0 * np.clip(1 - b2, 0, None) ** 1.5 * rng.gamma(4, 0.25, n_ev)
    v2 = 0.02 + 0.1 * np.sqrt(b2)
    qn = np.zeros((n_ev, 5, 6, n_pt), complex)
    psi = rng.uniform(0, 2 * np.pi, n_ev)
    for s, (frac, temp) in enumerate(((1.0, 0.45), (0.15, 0.6), (0.06, 0.7), (0.01, 0.8), (0.004, 0.9))):
        n = rng.poisson(np.outer(lam * frac, pc * np.exp(-pc / temp) * dpt))
        qn[:, s, 0, :] = n
        for h in range(1, 6):
            v = (v2[:, None] * np.minimum(pc / 1.5, 1.0)) if h == 2 else 0.0
            noise = rng.normal(size=n.shape) + 1j * rng.normal(size=n.shape)
            qn[:, s, h, :] = n * v * np.exp(1j * h * psi[:, None]) + np.sqrt(n / 2) * noise
    return build_container({'Qn': qn, 'pt_bin_edges': PTCUTS, 'masses': [0.13957, 0.49368, 0.93827, 1.18937, 1.32132],
                            'species_names': ['pion', 'kaon', 'proton', 'Sigma', 'Xi'],
                            'nsamples': np.ones(n_ev), 'rap_max': 1.0}, '2d')


class _Fixed(W.CentralityWeights):
    """The weights before November 2026: the full sample's, masked."""
    def weights(self, edges, mask=None, raw=False):
        full = super().weights(edges, None, raw)
        return full if mask is None or mask.all() else full[mask]


def _spread(records, n_samples, method, seed):
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in records}
    errs = {k: [] for k in records}
    fixed = []
    for _ in range(n_samples):
        qv = container(rng)
        for k, rec in records.items():
            v, e = evaluate(rec, qv=qv, cent_method=method)
            vals[k].append(v)
            errs[k].append(e)
        orig, W.CentralityWeights = W.CentralityWeights, _Fixed
        try:
            fixed.append(evaluate(records['dN/deta'], qv=qv, cent_method=method)[1])
        finally:
            W.CentralityWeights = orig
    return ({k: np.std(vals[k], axis=0, ddof=1) / np.mean(errs[k], axis=0) for k in records},
            np.mean(errs['dN/deta'], axis=0) / np.mean(fixed, axis=0))


def test_spread():
    """Independent samples: the jackknife error is the spread of the values
    (ratio 1 within the sampling error of the spread, ~1/sqrt(2 n_samples)
    per class; checked on the mean over classes and on each class)."""
    records = {'dN/deta': record_yield(eta=(-0.5, 0.5), pt=(0.0, 3.0), cent_classes=CLASSES),
               'v2{2}': record_flow_vn2(eta=(-0.8, 0.8), pt=(0.2, 3.0), cent_classes=CLASSES)}
    for method, n, seed in (('simple', 80, 11), ('poisson', 50, 12)):
        ratio, gain = _spread(records, n, method, seed)
        tol = 3.5 / np.sqrt(2 * n)
        for k, r in ratio.items():
            if method == 'simple' and k == 'v2{2}':
                check(f'{method} {k}: spread / jackknife error <= 1 (conservative)',
                      0.55 < np.mean(r) < 1 + tol and np.all(r < 1 + 3 * tol), f'{np.round(r, 2)}')
                continue
            check(f'{method} {k}: spread / jackknife error = 1',
                  abs(np.mean(r) - 1) < tol and np.all(np.abs(r - 1) < 3 * tol), f'{np.round(r, 2)}')
        check(f'{method}: dN/deta errors with fixed weights 2x and more too small (5-60%)',
              np.all(gain[1:] > 2.0), f'rebuilt / fixed {np.round(gain, 2)}')


def test_run_pipeline():
    import subprocess
    import tempfile
    import h5py
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    import test_sim_package as tsp
    from analysis.dispatcher import resolve
    from scripts.search_records import generate_manifest, load_records
    rec = next((f, i, d) for f, i, d in load_records('data/particle_yields')
               if (d.get('collision_system') or {}).get('reaction') == 'AU AU --> CHARGED X'
               and abs(float(d['collision_system'].get('energy_gev') or 0) - 200) < 1
               and resolve(d)['compute'] == 'yield_dNch_deta'
               and len((d.get('centrality') or {}).get('classes') or []) >= 4)
    with tempfile.TemporaryDirectory() as tmp:
        pkg, data, qn, ns = tsp._package(tmp, 'toy_adapter_jkcent')
        mf, out = os.path.join(tmp, 'm.yaml'), os.path.join(tmp, 'o.h5')
        generate_manifest([rec], mf)
        r = subprocess.run([sys.executable, 'scripts/run_pipeline.py', '--sim-package', str(pkg), '--variant', 'plain',
                            '--design-point', '0', '--type', 'particle_yields', '--manifest', mf,
                            '--cent-method', 'poisson', '--cent-anchor', '1.0', '--n-jk-blocks', '20',
                            '--output', out], cwd=ROOT, capture_output=True, text=True)
        check('run_pipeline runs', r.returncode == 0, r.stderr[-500:] if r.returncode else '')
        if r.returncode:
            return
        with h5py.File(out, 'r') as f:
            g = next(g for g in f.values() if 'values' in g)
            v_rp, e_rp = g['values'][()], g['errors'][()]
        qv = build_container({'Qn': qn, 'pt_bin_edges': PTCUTS, 'masses': [0.13957, 0.49368, 0.93827, 1.18937, 1.32132],
                              'species_names': ['pion', 'kaon', 'proton', 'Sigma', 'Xi'],
                              'nsamples': ns.astype(float), 'rap_max': 1.0}, '2d')
        v, e = evaluate(rec[2], qv=qv, cent_method='poisson')
        ok = np.isfinite(v)
        same = (ok.sum() == len(v_rp) and np.allclose(v[ok], v_rp, rtol=1e-10)
                and np.allclose(e[ok], e_rp, rtol=1e-10))
        check('run_pipeline = toy harness: values and errors (edges rebuilt per replicate)',
              same, '' if same else f'{e[ok]} vs {e_rp}')


if __name__ == '__main__':
    for t in (test_weights, test_run_pipeline, test_spread):
        print(t.__name__)
        t()
    failed = [n for n, ok in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} passed')
    if failed:
        raise SystemExit('FAILED: ' + ', '.join(failed))
