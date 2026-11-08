"""Classes in the estimator's track count (centrality.multiplicity_bins).

A record's classes are ranges of N_trk in the estimator acceptance (STAR's
isobar ratios vs N_trk(|eta| < 0.5), ATLAS N_ch^rec, CMS N_trk^offline): a
model event of charged multiplicity N there counts N_trk = Poisson(eff N)
and enters each class with that probability
(analysis.weights.MultiplicityBinWeights).

Checks, on toy events with a broad multiplicity distribution:
- the charged dN/deta at 0.5 < eta < 1 per N_trk class, on a 3D grid of
  positive particles (the charged count is twice the stored one) and on a
  2D container, against brute force: every event's N_trk drawn 300 times
  from Poisson(eff N), the observable averaged over the draws in each
  class;
- 'simple' (eff N in [lo, hi)): exactly the events so selected;
- a jackknife replicate's weights are the full sample's rows of the events
  it keeps; nothing is re-estimated;
- 2D (oversampled containers): the estimator count is the event's expected
  count in the window, so the reference draws use expected counts (with
  realized ones the top class differs by 8%: binomial window noise);
- run_pipeline on a 2D simulation package gives the toy harness's values
  (its own code paths: the weights factory, no anchor);
- refusals: percent labels, overlapping ranges, actual_classes.

Cells of one track (`combined_from: unit_multiplicity`, STAR's isobar
procedure: every observable in unit N_trk bins, averaged into the wider
bins with the number of events), on toy events whose <pT> rises with the
multiplicity:
- <dpT_i dpT_j> per class equals explicit particle sums: each cell k with
  event weights P(N_trk = k | eff N), its own <<pT>>, the cells averaged
  with their expected events (cells under two events left out);
- the cell average removes the spread of <pT> across the class: below the
  whole-class value in every class;
- for a yield (linear in the event weights) it equals the whole class;
- run_pipeline on a 2D package gives the toy harness's values (mean pT);
- refusals: percentile classes, analysis_classes or a class width with
  cells, percentile sub-classes of track-count classes.
Run: python3 tests/test_toy_mc_multiplicity_bins.py
"""
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from analysis.toy_mc import ToyGenConfig, generate_events, make_qgrid_data, make_qvector_data, evaluate  # noqa: E402

EFF = 0.8
CLASSES = ['60-100', '100-140', '140-180', '180-220', '220-300']
OBS_ETA, OBS_PT = (0.5, 1.0), (0.2, 3.0)
EST_ETA, EST_PT = (-0.5, 0.5), 0.2
RECORD = dict(
    observable_type='particle_yields', value_type='dNch_deta', observable='dNch_deta',
    hepdata_file='toy.yaml', inspire_id='0', version=1, particle_id=dict(species='charged'),
    kinematic_range=dict(eta=dict(min=OBS_ETA[0], max=OBS_ETA[1]), pT=dict(min=OBS_PT[0], max=OBS_PT[1])),
    centrality=dict(classes=list(CLASSES), estimator=dict(name='Ntrk', observable='Nch', detector='TPC'),
                    eta=[dict(min=EST_ETA[0], max=EST_ETA[1])], pT=dict(min=EST_PT, max=100.0),
                    multiplicity_bins=dict(efficiency=EFF)))
results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + ('' if ok else f'  {detail}'))


def edges():
    return [tuple(float(x) for x in c.split('-')) for c in CLASSES]


def truth(events, n_draw=300, seed=5):
    """Class means of dNch/deta at 0.5 < eta < 1 from explicit Poisson draws."""
    rng = np.random.default_rng(seed)
    N = np.array([2 * np.count_nonzero((np.abs(ev['eta']) < 0.5) & (ev['pt'] >= EST_PT)) for ev in events])
    y = np.array([2 * np.count_nonzero((ev['eta'] >= OBS_ETA[0]) & (ev['eta'] < OBS_ETA[1])
                                       & (ev['pt'] >= OBS_PT[0]) & (ev['pt'] < OBS_PT[1]))
                  for ev in events]) / (OBS_ETA[1] - OBS_ETA[0])
    draws = rng.poisson(EFF * N[:, None], size=(len(N), n_draw))
    out = []
    for lo, hi in edges():
        f = ((draws >= lo) & (draws < hi)).mean(axis=1)
        out.append(np.sum(f * y) / np.sum(f))
    simple = []
    for lo, hi in edges():
        sel = (EFF * N >= lo) & (EFF * N < hi)
        simple.append(y[sel].mean() if sel.any() else np.nan)
    return np.array(out), np.array(simple), N, y


def truth_2d(events, n_draw=300, seed=6):
    """The same on a boost-invariant container: each particle of |y| < 1 counts with its
    probability to fall in the eta window (y uniform over |y| < 1, eta(y) at its pT and
    mass), so N and the observable are expected counts; N_trk drawn from Poisson(eff N)."""
    rng = np.random.default_rng(seed)

    def p_in(ev, lo, hi, sel):
        pt, m = ev['pt'][sel], ev['mass'][sel]
        mt = np.sqrt(pt ** 2 + m ** 2)
        y = lambda e: np.arcsinh(pt * np.sinh(e) / mt)
        return (y(hi) - y(lo)) / 2.0
    N, yv = [], []
    for ev in events:
        a = np.abs(ev['y']) < 1.0
        N.append(2 * p_in(ev, *EST_ETA, a & (ev['pt'] >= EST_PT)).sum())
        yv.append(2 * p_in(ev, *OBS_ETA, a & (ev['pt'] >= OBS_PT[0]) & (ev['pt'] < OBS_PT[1])).sum()
                  / (OBS_ETA[1] - OBS_ETA[0]))
    N, yv = np.array(N), np.array(yv)
    draws = rng.poisson(EFF * N[:, None], size=(len(N), n_draw))
    out = []
    for lo, hi in edges():
        f = ((draws >= lo) & (draws < hi)).mean(axis=1)
        out.append(np.sum(f * yv) / np.sum(f))
    return np.array(out)


def fixture():
    cfg = ToyGenConfig(n_events=3000, mult_mean=260, mult_sigma=110, rap_max=1.6, seed=77, pt_min=0.05,
                       pt_max=3.0, flow={2: dict(kind='const', value=0.05)})
    events, _ = generate_events(cfg)
    grid = make_qgrid_data(events, {'cfg': cfg}, n_eta=64, n_pt=59, eta_half=1.6)     # 0.05 cells: edges on cells
    return cfg, events, grid


def main():
    from analysis.dispatcher import resolve
    from analysis.weights import MultiplicityBinWeights
    cfg, events, grid = fixture()
    t_pois, t_simple, N, y = truth(events)
    print(f"    charged N in |eta| < 0.5: mean {N.mean():.0f}, 5-95% {np.percentile(N, 5):.0f}-"
          f"{np.percentile(N, 95):.0f}; truth per class {t_pois.round(2)}")

    v3, e3 = evaluate(RECORD, grid=grid, cent_method='poisson', n_blocks=20)
    print(f"    3D: {v3.round(2)} +- {e3.round(2)}")
    rel = np.abs(v3 / t_pois - 1)
    check('3D, Poisson: each class within 1% of explicit Poisson draws (charged = 2 x stored)',
          np.all(rel < 0.01), f'{rel.round(4)}')
    vs, _ = evaluate(RECORD, grid=grid, cent_method='simple', with_jackknife=False)
    check("3D, 'simple': exactly the events with eff N in [lo, hi)", np.allclose(vs, t_simple, rtol=1e-10),
          f'{vs} vs {t_simple}')
    check('jackknife errors finite and positive', np.all(np.isfinite(e3) & (e3 > 0)))

    # 2D: the estimator count is the event's expected count in the eta window, not a
    # realized one (oversampled containers), so its classes select on that
    cfg2 = copy.copy(cfg)
    cfg2.rap_max = 1.0                  # the 2D plateau: the events' |y| < 1, flat as generated
    qv = make_qvector_data([{k: v[np.abs(ev['y']) < 1.0] for k, v in ev.items()} for ev in events],
                           {'cfg': cfg2}, n_pt=59)
    v2, e2 = evaluate(RECORD, qv=qv, cent_method='poisson', n_blocks=20)
    print(f"    2D: {v2.round(2)} +- {e2.round(2)}")
    t2 = truth_2d(events)
    rel2 = np.abs(v2 / t2 - 1)
    print(f"    2D truth (expected counts) {t2.round(2)}; realized-count truth differs by "
          f"{np.abs(t2 / t_pois - 1).max():.1%} at most (no binomial window noise)")
    check('2D, Poisson: each class within 1% of draws on the expected counts (charged = 2 x stored)',
          np.all(rel2 < 0.01), f'{rel2.round(4)}')

    W = MultiplicityBinWeights('poisson', np.array([50., 100., 150., 200.]), efficiency=EFF)
    full = W.weights(edges(), raw=True)
    m = np.array([True, False, True, True])
    check('a replicate keeps the full rows of its events', np.array_equal(W.weights(edges(), m, raw=True), full[m]))
    check('rows sum to the probability of the listed range', np.allclose(
        full.sum(axis=1), [__import__('scipy').stats.poisson.cdf(299, EFF * x) - __import__('scipy').stats.poisson.cdf(59, EFF * x)
                           for x in (50., 100., 150., 200.)]))

    def refused(mod):
        rec = copy.deepcopy(RECORD)
        mod(rec['centrality'])
        try:
            resolve(rec)
        except ValueError:
            return True
        return False
    check("percent labels refused", refused(lambda c: c.update(classes=[x + '%' for x in CLASSES])))
    check("overlapping ranges refused", refused(lambda c: c.update(classes=['60-100', '90-140'])))
    check("actual_classes refused", refused(lambda c: c.update(actual_classes=list(CLASSES))))

    # run_pipeline on a 2D simulation package: the same values as the toy harness
    import test_sim_package as tsp
    from scripts.run_pipeline import load_2d_group
    with tempfile.TemporaryDirectory() as tmp:
        pkg, data, *_ = tsp._package(tmp, "toy_adapter_mbins")
        rec = copy.deepcopy(RECORD)
        rec.update(collision_system={'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
                   centrality=dict(rec['centrality'], classes=['0-10', '10-20', '20-40', '40-80']),
                   kinematic_range={'eta': {'min': 0.5, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 3.0}})
        rdir = Path(tmp) / "records" / "data" / "particle_yields"
        rdir.mkdir(parents=True)
        yaml.safe_dump(rec, open(rdir / "mbins.yaml", "w"))
        out = Path(tmp) / "out.h5"
        r = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--sim-package", str(pkg),
                            "--variant", "plain", "--design-point", "0", "--n-jk-blocks", "10",
                            "--extra-records", str(Path(tmp) / "records" / "data"), "--type", "particle_yields",
                            "--value-type-regex", "^dNch_deta$", "--reaction", "AU AU --> CHARGED X",
                            "--energy", "200", "--output", str(out)], cwd=ROOT, capture_output=True, text=True)
        check("run_pipeline succeeds", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])
        if r.returncode == 0:
            import h5py
            got = None
            with h5py.File(out, "r") as f:
                for g in f.values():
                    if "labels" in g:
                        labs = [x.decode() for x in g["labels"][:]]
                        idx = [i for i, l in enumerate(labs) if any(l.endswith(' ' + c) or (' ' + c + ' ') in l
                                                                     for c in rec['centrality']['classes'])]
                        if idx:
                            got = np.asarray(g["values"][:])[idx]
            qv2 = load_2d_group(str(data), 0)
            want, _ = evaluate(rec, qv=qv2, cent_method='poisson', with_jackknife=False)
            print(f"    run_pipeline {None if got is None else got.round(3)}  toy harness {want.round(3)}")
            check("run_pipeline equals the toy harness (classes in track count)",
                  got is not None and len(got) == len(want) and np.allclose(got, want, rtol=1e-9))


UNIT_CLASSES = ['80-120', '120-160', '160-200']


def unit_cells():
    """Classes in track count averaged over cells of one track."""
    import scipy.stats
    from analysis.dispatcher import resolve
    from analysis.cent_combine import MIN_CELL_EVENTS
    print("\n  cells of one track (combined_from: unit_multiplicity)")
    cfg = ToyGenConfig(n_events=6000, mult_mean=260, mult_sigma=80, rap_max=1.6, seed=91, pt_min=0.05,
                       pt_max=3.0, flow={2: dict(kind='const', value=0.05)})
    events, _ = generate_events(cfg)
    from analysis.converters import y_to_eta
    for ev in events:              # <pT> rising with the multiplicity; the grid ends at 3 GeV
        ev['pt'] = ev['pt'] * (1.0 + 0.3 * (len(ev['pt']) - 260.0) / 260.0)
        ev['eta'] = y_to_eta(ev['y'], ev['pt'], ev['mass'])          # the grid bins eta from y
        keep = ev['pt'] < 3.0
        for k in list(ev):
            if isinstance(ev[k], np.ndarray) and len(ev[k]) == len(keep):
                ev[k] = ev[k][keep]
    grid = make_qgrid_data(events, {'cfg': cfg}, n_eta=64, n_pt=59, eta_half=1.6)
    cent = dict(RECORD['centrality'], classes=list(UNIT_CLASSES))
    fl = dict(observable_type='mean_pT_fluctuations', value_type='<dpt1_dpt2>', observable='mean_pT_fluctuations',
              hepdata_file='toy.yaml', inspire_id='0', version=1, particle_id=dict(species='charged'),
              kinematic_range=dict(eta=dict(min=OBS_ETA[0], max=OBS_ETA[1]), pT=dict(min=0.2, max=2.0)),
              centrality=cent)
    cells = copy.deepcopy(fl)
    cells['centrality']['combined_from'] = 'unit_multiplicity'

    # explicit: the estimator count (charged = 2 x stored) and the window sums per event
    N = np.array([2 * np.count_nonzero((np.abs(ev['eta']) < 0.5) & (ev['pt'] >= EST_PT)) for ev in events])
    sums = []
    for ev in events:
        p = ev['pt'][(ev['eta'] >= OBS_ETA[0]) & (ev['eta'] < OBS_ETA[1]) & (ev['pt'] >= 0.2) & (ev['pt'] < 2.0)]
        sums.append((len(p), p.sum(), (p ** 2).sum()))
    S0, S1, S2 = (np.array(x, float) for x in zip(*sums))

    def cov(w):                    # pairs pooled, deviations from the mean pT of the weighted events
        M = np.sum(w * S1) / np.sum(w * S0)
        P = (S1 - M * S0) ** 2 - (S2 - 2 * M * S1 + M ** 2 * S0)
        return np.sum(w * P) / np.sum(w * (S0 * S0 - S0))
    want, whole_x, short = [], [], []
    for c in UNIT_CLASSES:
        lo, hi = (int(x) for x in c.split('-'))
        f = np.array([scipy.stats.poisson.pmf(k, EFF * N) for k in range(lo, hi)])     # cells x events
        nk = f.sum(axis=1)
        short.append(int(np.count_nonzero(nk < MIN_CELL_EVENTS)))
        ok = nk >= MIN_CELL_EVENTS
        want.append(np.sum([n * cov(fk) for n, fk in zip(nk[ok], f[ok])]) / nk[ok].sum())
        whole_x.append(cov(f.sum(axis=0)))
    want, whole_x = np.array(want), np.array(whole_x)
    got, err = evaluate(cells, grid=grid, cent_method='poisson', n_blocks=20)
    whole, _ = evaluate(fl, grid=grid, cent_method='poisson', with_jackknife=False)
    print(f"    <dpT dpT> (GeV^2): cells {got} +- {err}\n      explicit {want}; whole classes {whole} "
          f"(explicit {whole_x}); cells under {MIN_CELL_EVENTS:g} events per class {short}")
    check('cells: <dpT_i dpT_j> equals explicit particle sums per cell, averaged with the events',
          np.allclose(got, want, rtol=1e-9, atol=0), f'{got} vs {want}')
    check('whole classes equal explicit particle sums (reference)', np.allclose(whole, whole_x, rtol=1e-9))
    check('the cell average removes the spread of <pT> across the class (below the whole class)',
          np.all(got < whole), f'{got} vs {whole}')
    check('jackknife errors finite and positive', np.all(np.isfinite(err) & (err > 0)))

    ych = dict(RECORD, centrality=copy.deepcopy(cells['centrality']))
    yw, _ = evaluate(dict(RECORD, centrality=cent), grid=grid, with_jackknife=False, min_weight=0.0)
    yc, _ = evaluate(ych, grid=grid, with_jackknife=False, min_weight=0.0)
    check('a yield: the cell average equals the whole class (no cell under two events here)',
          sum(short) == 0 and np.allclose(yc, yw, rtol=1e-9), f'{yc} vs {yw}')

    def refused(mod, base=cells):
        rec = copy.deepcopy(base)
        mod(rec['centrality'])
        try:
            resolve(rec)
        except ValueError:
            return True
        return False
    check('cells of percentile classes refused', refused(lambda c: (
        c.pop('multiplicity_bins'), c.update(classes=['0-10%', '10-20%']))))
    check('cells with analysis_class_width refused', refused(lambda c: c.update(analysis_class_width=2.0)))
    check('percentile sub-classes of track-count classes refused', refused(
        lambda c: c.update(combined_from='analysis_classes', analysis_class_width=10.0)))

    # run_pipeline on a 2D simulation package: mean pT from cells, as the toy harness
    import test_sim_package as tsp
    from scripts.run_pipeline import load_2d_group
    with tempfile.TemporaryDirectory() as tmp:
        pkg, data, *_ = tsp._package(tmp, "toy_adapter_mbins_cells")
        rec = dict(observable_type='mean_pT', value_type='mean_pT', observable='mean_pT', hepdata_file='cells_toy.yaml',
                   inspire_id='0', version=1, particle_id=dict(species='charged'),
                   collision_system={'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
                   kinematic_range={'eta': {'min': 0.5, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 3.0}},
                   centrality=dict(RECORD['centrality'], classes=['10-20', '20-40', '40-60'],
                                   combined_from='unit_multiplicity'))
        rdir = Path(tmp) / "records" / "data" / "mean_pT"
        rdir.mkdir(parents=True)
        yaml.safe_dump(rec, open(rdir / "cells.yaml", "w"))
        out = Path(tmp) / "out.h5"
        r = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--sim-package", str(pkg),
                            "--variant", "plain", "--design-point", "0", "--n-jk-blocks", "10",
                            "--extra-records", str(Path(tmp) / "records" / "data"), "--type", "mean_pT",
                            "--reaction", "AU AU --> CHARGED X", "--energy", "200", "--output", str(out)],
                           cwd=ROOT, capture_output=True, text=True)
        check("run_pipeline succeeds (cells of one track)", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])
        if r.returncode == 0:
            import h5py
            got = None
            with h5py.File(out, "r") as f:
                for g in f.values():
                    if "labels" in g:
                        labs = [x.decode() for x in g["labels"][:]]
                        idx = [i for i, l in enumerate(labs) if 'cells_toy.yaml' in l]
                        if idx:
                            got = np.asarray(g["values"][:])[idx]
            qv2 = load_2d_group(str(data), 0)
            want, _ = evaluate(rec, qv=qv2, cent_method='poisson', with_jackknife=False)
            whole, _ = evaluate(dict(rec, centrality=dict(rec['centrality'], combined_from=None)), qv=qv2,
                                cent_method='poisson', with_jackknife=False)
            print(f"    mean pT: run_pipeline {None if got is None else got.round(5)}  toy harness "
                  f"{want.round(5)}  whole classes {whole.round(5)}")
            check("run_pipeline equals the toy harness (cells of one track; 10-20: every cell under "
                  "two events, NaN)", got is not None and len(got) == len(want) and np.isnan(want[0])
                  and np.all(np.isfinite(want[1:])) and np.allclose(got, want, rtol=1e-9, equal_nan=True))


if __name__ == '__main__':
    main()
    unit_cells()
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
