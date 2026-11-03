"""Rapidity-binned species grids (QGridData.y_edges) against particle-level truth.

The same toy particles are binned twice: identified grids in eta (as the
JETSCAPE 3D grids) and in rapidity y (as the O+O JETSCAPE-framework
QnVector output), with the charged combined grid in eta.  Checks:
  - a y window of the grid's own mass slices directly (dN/dy, spectra,
    <pT>) and matches the particle counts exactly when the edges align;
  - an eta window on a y grid is mapped per pT cell with the species mass
    (dN/deta, <pT> in eta), within the uniform-in-cell approximation;
  - identified v2{2} (POI on the y grid, charged reference on the eta
    grid) agrees with the eta-binned result;
  - y grids of different masses are not summed; build_container accepts
    y_edges for species grids only;
  - <pT> in a rapidity window (until November 2026 0/0 on 3D) on eta grids.
Run: python3 tests/test_toy_mc_rapidity_grid.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.qvector import MultiSpeciesGrid, QGridData, SPECIES_MASS  # noqa: E402
from analysis.sim_package import SimPackageError, build_container  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, evaluate, record_yield, record_spectra, record_mean_pt,
    record_flow_vn2, SPECIES_MASSES,
)

PROTON, PION = 2, 0
PT = (0.3, 2.4)
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _y_grid(evs, y_edges, pt_edges, mass, mult):
    """Bin particle lists in rapidity: the converter with y in the eta slot."""
    # the converter bins its 'eta'; give it y there and no 'y' (it would derive eta from y)
    yev = [{**{k: v for k, v in ev.items() if k not in ('y', 'eta')}, 'eta': ev['y']} for ev in evs]
    g = particles_to_qgrid_data(yev, y_edges, pt_edges, n_harm=6, cent_mult=mult, masses=SPECIES_MASSES)
    return QGridData(Qn=g.Qn, eta_edges=None, y_edges=g.eta_edges, y_mass=mass, pt_edges=g.pt_edges,
                     cent_mult=g.cent_mult, nsamples=g.nsamples, sum_pt=g.sum_pt)


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=800, mult_mean=600, mult_sigma=60, seed=4711,
                           species_mix={'pion': 0.8, 'kaon': 0.08, 'proton': 0.12},
                           temp={'pion': 0.22, 'kaon': 0.30, 'proton': 0.35},
                           flow={2: dict(kind='linear', value=0.06, pt0=1.0)}, fluctuate={2: 0.01})
        events, truth = generate_events(cfg)
        edges = np.linspace(-5, 5, 101)                  # 0.1 cells in eta and in y
        pt_edges = np.linspace(0.0, 3.0, 61)             # 0.05 cells
        mult = np.array([len(e['pt']) for e in events], float)

        def sel(slots):
            return [{k: v[np.isin(e['species_idx'], slots)] for k, v in e.items()} for e in events]

        def eta_grid(slots):
            return particles_to_qgrid_data(sel(slots), edges, pt_edges, n_harm=6, masses=SPECIES_MASSES,
                                           cent_mult=mult)
        combined = eta_grid([0, 1, 2])
        ms_eta = MultiSpeciesGrid({211: eta_grid([0]), 321: eta_grid([1]), 2212: eta_grid([2])},
                                  combined=combined)
        ms_y = MultiSpeciesGrid({pdg: _y_grid(sel([s]), edges, pt_edges, SPECIES_MASS[pdg], mult)
                                 for pdg, s in ((211, 0), (321, 1), (2212, 2))}, combined=combined)
        _CACHE['fx'] = dict(events=events, ms_eta=ms_eta, ms_y=ms_y)
    return _CACHE['fx']


def _with_y(rec, y):
    kr = dict(rec['kinematic_range'])
    kr.pop('eta')
    kr['y'] = {'min': y[0], 'max': y[1]}
    return dict(rec, kinematic_range=kr)


def _one(rec, ms):
    rec = dict(rec, centrality=dict(rec['centrality'], classes=['0-100%']))
    vals, _ = evaluate(rec, grid=ms, cent_method='simple', with_jackknife=False)
    return np.asarray(vals)


def truth_counts(var, lo, hi, pt=PT, slot=PROTON):
    ev = _fx()['events']
    n = np.mean([np.sum((e['species_idx'] == slot) & (e[var] > lo) & (e[var] < hi)
                        & (e['pt'] > pt[0]) & (e['pt'] < pt[1])) for e in ev])
    return n / (hi - lo)


def truth_mean_pt(var, lo, hi, pt=PT, slot=PROTON):
    s = n = 0.0
    for e in _fx()['events']:
        m = (e['species_idx'] == slot) & (e[var] > lo) & (e[var] < hi) & (e['pt'] > pt[0]) & (e['pt'] < pt[1])
        s += e['pt'][m].sum()
        n += m.sum()
    return s / n


def test_y_window_direct():
    fx = _fx()
    rec = _with_y(record_yield('p', pt=PT), (-0.5, 0.5))
    got, want = _one(rec, fx['ms_y'])[0], truth_counts('y', -0.5, 0.5)
    check('dN/dy |y|<0.5 on the y grid (edges aligned: exact)', abs(got / want - 1) < 1e-9, f'{got:.5f} vs {want:.5f}')
    got_eta = _one(rec, fx['ms_eta'])[0]
    check('dN/dy |y|<0.5 on the eta grid (mass-mapped)', abs(got_eta / want - 1) < 0.02,
          f'{got_eta:.4f} vs {want:.4f}')


def test_eta_window_mapped():
    fx = _fx()
    rec = record_yield('p', eta=(-0.5, 0.5), pt=PT)
    want = truth_counts('eta', -0.5, 0.5)
    got = _one(rec, fx['ms_y'])[0]
    check('dN/deta |eta|<0.5 of protons on the y grid (mapped per pT cell)', abs(got / want - 1) < 0.02,
          f'{got:.4f} vs {want:.4f}')
    check('... and on the eta grid (exact)', abs(_one(rec, fx['ms_eta'])[0] / want - 1) < 1e-9)


def test_spectra_y_window():
    fx = _fx()
    bins = [(0.3, 0.6), (0.6, 1.0), (1.0, 1.6), (1.6, 2.4)]
    rec = _with_y(record_spectra('p', bins=bins), (-0.5, 0.5))
    got = _one(rec, fx['ms_y'])
    ev = fx['events']
    for b, (lo, hi) in enumerate(bins):
        n = np.mean([np.sum((e['species_idx'] == PROTON) & (np.abs(e['y']) < 0.5) & (e['pt'] > lo)
                            & (e['pt'] < hi)) for e in ev])
        want = n / (2 * np.pi * 0.5 * (lo + hi) * (hi - lo) * 1.0)
        # the core divides by 2 pi <pT> or the bin centre; compare shapes via the count
        check(f'spectrum bin {lo}-{hi} (y grid, direct)', abs(got[b] / want - 1) < 0.05,
              f'{got[b]:.4g} vs {want:.4g}')


def test_mean_pt_windows():
    fx = _fx()
    rec_y = _with_y(record_mean_pt('p', pt=PT), (-0.5, 0.5))
    want = truth_mean_pt('y', -0.5, 0.5)
    for name, ms, tol in (('y grid', fx['ms_y'], 1e-3), ('eta grid', fx['ms_eta'], 0.01)):
        got = _one(rec_y, ms)[0]
        check(f'<pT> |y|<0.5 of protons on the {name}', abs(got / want - 1) < tol, f'{got:.4f} vs {want:.4f}')
    rec_eta = record_mean_pt('p', eta=(-0.5, 0.5), pt=PT)
    want = truth_mean_pt('eta', -0.5, 0.5)
    got = _one(rec_eta, fx['ms_y'])[0]
    check('<pT> |eta|<0.5 of protons on the y grid (mapped, fractional)', abs(got / want - 1) < 0.003,
          f'{got:.4f} vs {want:.4f}')


def test_identified_v2_agrees():
    fx = _fx()
    rec = record_flow_vn2(eta=(-0.8, 0.8), pt=PT, cent_classes=('0-100%',))
    rec['method'] = 'cumulant'
    rec['particle_id'] = {'species': 'p'}
    a, b = _one(rec, fx['ms_eta'])[0], _one(rec, fx['ms_y'])[0]
    check('v2{2} of protons: y grid vs eta grid', abs(b / a - 1) < 0.03, f'{b:.5f} vs {a:.5f}')
    rec['particle_id'] = {'species': 'p + pbar'}
    check('p + pbar (equal masses) is summed', np.isfinite(_one(rec, fx['ms_y'])[0]))


def test_identified_v2_two_subevents():
    """POI in one eta subevent, charged reference in the other; on the y grid
    the cells straddling both windows get the expected self-pair term."""
    fx = _fx()
    sub = {lab: {'kinematic_range': {'eta': {'min': e[0], 'max': e[1]}, 'pT': {'min': 0.2, 'max': 3.0}}}
           for lab, e in (('A', (-0.8, -0.1)), ('B', (0.1, 0.8)))}
    rec = dict(observable_type='flow_vn', method='cumulant', cumulant_order=2, particles='AB',
               dimensionality='integrated', harmonic=2, value_type='v2{2}', hepdata_file='toy.yaml',
               inspire_id='0', version=1, particle_id={'species': 'p'},
               kinematic_range={'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': PT[0], 'max': PT[1]}},
               subevents=sub, centrality={'classes': ['0-100%'], 'eta': [{'min': -3.9, 'max': -3.0},
                                                                          {'min': 3.0, 'max': 3.9}]})
    a, b = _one(rec, fx['ms_eta'])[0], _one(rec, fx['ms_y'])[0]
    check('two-subevent v2{2} of protons: y grid vs eta grid', abs(b / a - 1) < 0.04, f'{b:.5f} vs {a:.5f}')


def test_unequal_masses_refused():
    fx = _fx()
    g = fx['ms_y'].grids
    try:
        g[211] + g[2212]
        ok = False
    except ValueError:
        ok = True
    check('pi + p rapidity grids are not summed', ok)


def test_build_container_y_edges():
    g = _fx()['ms_y'].grids[2212]
    d = dict(Qn=g.Qn, y_edges=g.y_edges, pt_edges=g.pt_edges, cent_mult=g.cent_mult, nsamples=g.nsamples,
             positives_only=True)
    c = _fx()['ms_y'].combined
    comb = dict(Qn=c.Qn, eta_edges=c.eta_edges, pt_edges=c.pt_edges, cent_mult=c.cent_mult,
                nsamples=c.nsamples, positives_only=True)
    ms = build_container({'species': {2212: d}, 'combined': comb}, '3d', 'toy')
    check('build_container: y_edges species grid', ms.grids[2212].rapidity_binned
          and abs(ms.grids[2212].y_mass - SPECIES_MASS[2212]) < 1e-12)
    for bad, why in (({'species': {2212: dict(d, eta_edges=g.y_edges)}}, 'both axes'),
                     (dict(comb, y_edges=comb.pop('eta_edges')), 'y_edges on a combined grid')):
        try:
            build_container(bad, '3d', 'toy')
            ok = False
        except SimPackageError:
            ok = True
        check(f'build_container refuses {why}', ok)


def test_complex64_storage():
    g = _fx()['ms_y'].grids[2212]
    g32 = QGridData(Qn=g.Qn.astype(np.complex64), eta_edges=None, y_edges=g.y_edges, y_mass=g.y_mass,
                    pt_edges=g.pt_edges, cent_mult=g.cent_mult, nsamples=g.nsamples)
    a, b = g.acceptance(-0.8, 0.8, *PT), g32.acceptance(-0.8, 0.8, *PT)
    check('complex64 grid: window sums in complex128', b.dtype == np.complex128
          and np.allclose(a, b, rtol=1e-5, atol=1e-3), str(b.dtype))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    n_ok = 0
    for t in tests:
        print(t.__name__)
        try:
            t()
            n_ok += 1
        except AssertionError as e:
            print(f'  FAILED: {e}')
    print(f'\n{n_ok}/{len(tests)} passed')
    sys.exit(0 if n_ok == len(tests) else 1)
