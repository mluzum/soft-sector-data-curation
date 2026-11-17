"""Importance-sampled simulations: per-event weights w = 1/P_accept.

A campaign may simulate only a subset of its drawn events, each accepted with
a probability P that depends on an initial-state centrality estimate, and
weighted by 1/P (iEBE-MUSIC docs/importance_sampling.md).  The weights enter
the event averages (combine_weights) and the centrality: the percentile edges
count each event w times, the total is the campaign's exact count of
collided draws (anchor n_ref), and minimum-statistics floors count
simulated events.

1. Weights level, exact: an event of integer weight k is k copies of it
   (Poisson and sharp edges, an anchor, an experiment-defined normalization,
   a jackknife replicate's anchor).
2. Combined classes: the floor of a unit-multiplicity cell (two events)
   counts simulated events, not summed weight.
3. Toy MC, production path (toy_mc.evaluate): a weighted subset of a sample
   whose multiplicity and v2 vary with centrality reproduces the full
   sample's dN/deta and v2{2} per class within errors; the same subset
   without its weights does not; with a hard cut (P = 0 beyond 50%) the
   classes inside the cut still agree and those beyond it are empty.

Run: python3 tests/test_toy_mc_importance_sampling.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.weights import (CentralityWeights, resolve_anchor,  # noqa: E402
                              threshold_anchor)

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("  PASS " if cond else "  FAIL ") + name)


# ── 1. weights level: weight k == k copies ────────────────────────────────
def _duplicated(x, k):
    k = k.astype(int)
    idx = np.repeat(np.arange(len(x)), k)
    first = np.concatenate([[0], np.cumsum(k)[:-1]])
    return x[idx], first


def test_weights_equal_copies():
    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(5, 300, n)
    k = rng.integers(1, 4, n).astype(float)
    xd, first = _duplicated(x, k)
    edges = [(0, 5), (5, 10), (10, 30), (30, 60), (60, 90), (2.5, 7.5)]
    for f in (1.0, 0.8):
        a_w = resolve_anchor(f, n, sum_w=k.sum())
        a_d = resolve_anchor(f, len(xd))
        check(f"anchor f={f}: weighted n_ref = copies' n_ref",
              abs(a_w['n_ref'] - a_d['n_ref']) < 1e-9 * a_d['n_ref'] and abs(a_w['f'] - a_d['f']) < 1e-12)
        cw = CentralityWeights('poisson', x, a_w, efficiency=0.9, event_weights=k)
        cd = CentralityWeights('poisson', xd, a_d, efficiency=0.9)
        Ww, Wd = cw.weights(edges, raw=True), cd.weights(edges)
        check(f"poisson f={f}: memberships = the copies'", np.allclose(Ww, Wd[first], atol=1e-12))
        check(f"poisson f={f}: class sums (x w) = the copies'",
              np.allclose(cw.weights(edges).sum(axis=0), Wd.sum(axis=0), rtol=1e-12))
    # sharp edges, on whole copies: sum k = 1000, edges at multiples of 0.1%
    k = np.repeat([3.0, 2.0, 1.0], [250, 100, 50])
    rng.shuffle(k)
    assert k.sum() == 1000
    xd, first = _duplicated(x, k)
    a_w, a_d = resolve_anchor(1.0, n, sum_w=k.sum()), resolve_anchor(1.0, len(xd))
    sharp = [(0, 5), (5, 10.3), (10.3, 30), (30, 60), (60, 100)]
    Ww = CentralityWeights('simple', x, a_w, event_weights=k).weights(sharp, raw=True)
    Wd = CentralityWeights('simple', xd, a_d).weights(sharp)
    frac = np.add.reduceat(Wd, first, axis=0) / k[:, None]       # share of an event's copies in each class
    check("simple: membership = share of the copies in the class", np.allclose(Ww, frac, atol=1e-12))
    check("simple: classes hold their width in weight",
          np.allclose((Ww * k[:, None]).sum(axis=0), [50, 53, 197, 300, 400], atol=1e-9))
    # experiment-defined normalization (STAR's F50-type anchor)
    t_w = threshold_anchor(x, 'poisson', 120, 0.4, efficiency=0.9, event_weights=k)
    t_d = threshold_anchor(xd, 'poisson', 120, 0.4, efficiency=0.9)
    check("threshold normalization: weighted = copies",
          abs(t_w['n_ref'] - t_d['n_ref']) < 1e-9 * t_d['n_ref'] and abs(t_w['f'] - t_d['f']) < 1e-12)


def test_anchor_forms():
    a = resolve_anchor({'n_ref': 500}, 40, sum_w=400.0)
    check("n_ref given: coverage = sum w / n_ref", abs(a['f'] - 0.8) < 1e-12 and a['n_ref'] == 500)
    a = resolve_anchor(0.8, 40, sum_w=400.0)
    check("f given: n_ref = sum w / f", abs(a['n_ref'] - 500) < 1e-9 and abs(a['f'] - 0.8) < 1e-12)
    a0, a1 = resolve_anchor(0.8, 1000), resolve_anchor(0.8, 1000, sum_w=1000.0)
    check("unit weights: the unweighted anchor", a0 == a1)
    # a replicate keeps the share of n_ref of the events it keeps; its
    # coverage is its own sum of weights over that
    rng = np.random.default_rng(2)
    x, w = rng.uniform(1, 100, 200), rng.choice([1.0, 4.0], 200)
    a = resolve_anchor({'n_ref': 600}, 200, sum_w=w.sum())
    cw = CentralityWeights('poisson', x, a, event_weights=w)
    m = np.arange(200) % 10 != 3
    r = cw.anchor(m)
    check("replicate anchor: n_ref x kept share, f = kept weight / n_ref",
          abs(r['n_ref'] - 600 * m.sum() / 200) < 1e-9 and abs(r['f'] - w[m].sum() / r['n_ref']) < 1e-12)
    # unit weights on Poisson edges: the unweighted weights
    W0 = CentralityWeights('poisson', x, resolve_anchor(0.9, 200)).weights([(0, 10), (10, 50)])
    W1 = CentralityWeights('poisson', x, resolve_anchor(0.9, 200, sum_w=200.0),
                           event_weights=np.ones(200)).weights([(0, 10), (10, 50)])
    check("unit weights: the unweighted Poisson weights", np.allclose(W0, W1, atol=1e-13))


# ── 2. combined classes: the floor counts simulated events ────────────────
def test_combined_floor_counts_events():
    from analysis.cent_combine import wrap_core
    core = wrap_core(lambda d, w: np.array([np.sum(w * d['v']) / np.sum(w)]), 0.01,
                     {'min_sub_weight': 2.0})
    v = np.array([1.0, 2.0, 3.0])
    ew = np.array([5.0, 1.0, 1.0])
    W = np.array([[5.0, 0.0], [0.0, 1.0], [0.0, 1.0]])       # cell 0: one event of weight 5
    out = core({'obs': {'v': v}, 'yield': None, 'event_weights': ew}, W)
    check("cell of one weight-5 event left out (one simulated event)", np.isclose(out[0], 2.5))
    out = core({'obs': {'v': v}, 'yield': None, 'event_weights': None}, W)
    check("without event weights the summed weight decides (as before)",
          np.isclose(out[0], (5 * 1.0 + 2 * 2.5) / 7))


# ── 3. toy MC through the production path ─────────────────────────────────
def _toy_sample():
    """Sub-samples of falling multiplicity and rising v2: centrality matters."""
    import dataclasses
    from analysis.toy_mc import ToyGenConfig, generate_events, make_qvector_data
    parts = []
    for j, (mult, v2) in enumerate([(450, 0.03), (360, 0.04), (280, 0.05), (210, 0.06),
                                    (150, 0.07), (100, 0.08), (60, 0.09), (30, 0.09)]):
        cfg = ToyGenConfig(n_events=900, flow={2: dict(kind='const', value=v2)}, fluctuate={2: 0.02},
                           mult_mean=mult, mult_sigma=0.15 * mult, seed=100 + j)
        ev, tr = generate_events(cfg)
        parts.append(make_qvector_data(ev, tr))
    qv = parts[0]
    # events in random order, as the ids of a campaign (the jackknife's
    # contiguous blocks would otherwise each hold one sub-sample)
    order = np.random.default_rng(3).permutation(sum(p.n_events for p in parts))
    cat = {f: np.concatenate([getattr(p, f) for p in parts])[order]
           for f in ('Qn', 'nsamples', 'sum_pt', 'sum_pt2', 'sum_mT') if getattr(qv, f) is not None}
    return dataclasses.replace(qv, **cat)


def _subset(qv, keep, weights):
    import dataclasses
    kw = {f: getattr(qv, f)[keep] for f in ('Qn', 'nsamples', 'sum_pt', 'sum_pt2', 'sum_mT')
          if getattr(qv, f) is not None}
    return dataclasses.replace(qv, event_weights=None if weights is None else weights[keep], **kw)


def test_importance_sampled_subset():
    from analysis.toy_mc import evaluate, record_flow_vn2, record_yield
    qv = _toy_sample()
    n = qv.n_events
    rng = np.random.default_rng(11)
    # initial-state estimate: the event's multiplicity with 5% noise (initial
    # and final orderings differ), its percentile among all draws
    est = qv.Qn[:, :, 0, :].real.sum(axis=(1, 2)) * (1 + 0.05 * rng.normal(size=n))
    c = 100.0 * (np.argsort(np.argsort(-est)) + 0.5) / n
    u = rng.uniform(size=n)
    classes = ('0-5%', '5-10%', '10-20%', '20-30%', '30-50%', '50-70%')
    recs = {'dN/deta': record_yield(cent_classes=classes), 'v2{2}': record_flow_vn2(cent_classes=classes)}
    full = {k: evaluate(r, qv=qv, n_blocks=20) for k, r in recs.items()}

    P = np.where(c < 10, 1.0, np.where(c < 40, 0.3, 0.08))
    keep = u < P
    sub = _subset(qv, keep, 1.0 / P)
    print(f"    simulated {keep.sum()} of {n} events")
    anchor = {'n_ref': float(n)}              # the campaign's collided draws
    for name, r in recs.items():
        vw, ew = evaluate(r, qv=sub, cent_anchor=anchor, n_blocks=20)
        vf, ef = full[name]
        print(f"    {name} full    " + " ".join(f"{v:.4g}+-{e:.2g}" for v, e in zip(vf, ef)))
        print(f"    {name} weighted" + " ".join(f"{v:.4g}+-{e:.2g}" for v, e in zip(vw, ew)))
        pull = np.abs(vw - vf) / ew
        check(f"{name}: weighted subset = full sample within 3.5 sigma per class "
              f"(max {np.max(pull):.2f})", np.all(pull < 3.5))
        if name == 'dN/deta':
            vu, _ = evaluate(r, qv=_subset(qv, keep, None), cent_anchor=1.0, n_blocks=20)
            bad = np.abs(vu - vf) / ew
            check(f"{name}: the subset without its weights is far off (max {np.max(bad):.0f} sigma)",
                  np.max(bad) > 10)
            check(f"{name}: the error grows where few events are simulated "
                  f"(50-70%, P = 0.08: x{ew[5] / ef[5]:.1f})", ew[5] > 2 * ef[5])

    # hard cut: no events accepted beyond 50% (initial-state)
    P = np.where(c < 10, 1.0, np.where(c < 50, 0.3, 0.0))
    keep = u < P
    sub = _subset(qv, keep, 1.0 / np.where(P > 0, P, 1.0))
    cut = dict(anchor, covered_cent=50.0)
    a = resolve_anchor(cut, keep.sum(), sum_w=float(sub.event_weights.sum()))
    check(f"hard cut at 50%: coverage {a['anchor_cent']:.1f}% (within 3 of 50)", abs(a['anchor_cent'] - 50) < 3)
    vw, ew = evaluate(recs['dN/deta'], qv=sub, cent_anchor=cut, n_blocks=20)
    vf, _ = full['dN/deta']
    inside = np.abs(vw[:4] - vf[:4]) / ew[:4]
    check(f"hard cut: classes to 30% agree (max {np.max(inside):.2f} sigma)", np.all(inside < 3.5))
    raw, _ = evaluate(recs['dN/deta'], qv=sub, cent_anchor=cut, with_jackknife=False)
    check("hard cut: the class beyond the cut (50-70%) is empty", not np.isfinite(raw[5]))


if __name__ == '__main__':
    tests = [fn for fn in sorted(dir()) if fn.startswith('test_')]
    for name in tests:
        print(name)
        try:
            globals()[name]()
        except Exception as e:                 # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append((f"{name} raised {type(e).__name__}: {e}", False))
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
    sys.exit(1 if failed else 0)
