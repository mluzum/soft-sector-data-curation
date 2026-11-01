"""Scalar-product / event-plane differential v_n of a particle of interest.

The SP keys (vn2_sp_diff_pT, vn2_sp_diff_eta and their _means forms) take
the particle of interest (POI) from the record's species, per differential
bin, and the reference from charged hadrons in the record's flow_reference
windows (default: the record's own window).  Self-correlations: Sum w^2 of
the reference comes out of |Q_ref|^2, and the particles in both samples
out of Re<q_poi Q_ref*>.

Toy: pions and kaons with v2 = 0.05, protons with v2 = 0.12 (same event
plane), and a slot outside the charged reference (Lambda) with v2 = 0.12.

Checks:
  - 3D species grids: the production prepare + core equals the particle-
    level evaluation of the same estimator on the same particles (machine
    precision) for protons, charged hadrons and Lambda, with no reference
    (same window), STAR-like TPC halves overlapping the POI, a disjoint
    forward reference, and a mirrored reference (eta-differential);
  - 2D boost-invariant container: production equals the particle-level
    weighted estimator (Jacobian weights; machine precision);
  - physics: identified v2(pT) recovers the injected 0.12 (2D and 3D),
    charged hadrons the pT-dependent mixture; the _means form agrees.
Until November 2026 the 2D path ignored the species and the reference and
subtracted Sum w instead of Sum w^2 (here: NaN or ~1e29), and the 3D path
used an identified-only reference and kept the POI's own particles in the
numerator for overlapping references (charged v2 = 0.18 instead of 0.06).
Run: python3 tests/test_toy_mc_sp_poi.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.observables_2d import _make_jacobian_list  # noqa: E402
from analysis.pipeline_dispatch import (  # noqa: E402
    _find_pt_slice_edges, container_for_prepare, get_pipeline_entry,
)
from analysis.qvector import MultiSpeciesGrid  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    SPECIES_MASSES, ToyGenConfig, generate_events, make_qvector_data,
)

NE = 1500
ETA = (-0.8, 0.8)
PT_BINS = [0.3, 0.8, 1.4, 2.4]
ETA_BINS = [-0.8, -0.4, 0.0, 0.4, 0.8]
V_PI, V_P = 0.05, 0.12
PROTON, LAMBDA = 2, 3                 # toy slots; slot 3 plays a species outside the reference
REFS = {
    'none': None,
    'tpc halves': {'eta': [{'min': -0.8, 'max': -0.1}, {'min': 0.1, 'max': 0.8}],
                   'pT': {'min': 0.15, 'max': 2.0}},
    'forward': {'eta': [{'min': 2.0, 'max': 3.0}]},
    'mirrored': {'eta': [{'min': 1.0, 'max': 3.0}], 'mirror': True},
}
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fx():
    if 'fx' in _CACHE:
        return _CACHE['fx']
    a_cfg = ToyGenConfig(n_events=NE, mult_mean=1500, mult_sigma=60, rap_max=3.0, seed=11,
                         species_mix={'pion': 0.85, 'kaon': 0.15}, flow={2: dict(kind='const', value=V_PI)})
    b_cfg = ToyGenConfig(n_events=NE, mult_mean=400, mult_sigma=20, rap_max=3.0, seed=12,
                         species_mix={'proton': 0.75, 'Sigma': 0.25},
                         temp={'proton': 0.35, 'Sigma': 0.38}, flow={2: dict(kind='const', value=V_P)})
    A, ta = generate_events(a_cfg)
    B, tb = generate_events(b_cfg)
    events = []
    for e in range(NE):
        b = dict(B[e])
        b['phi'] = (b['phi'] + ta['psi'][2][e] - tb['psi'][2][e]) % (2 * np.pi)   # common event plane
        b['species_idx'] = np.where(b['species_idx'] == 0, PROTON, LAMBDA)          # local -> toy slots
        events.append({k: np.concatenate([A[e][k], b[k]]) for k in A[e]})
    mult = np.array([len(e['pt']) for e in events], float)
    eta_edges = np.round(np.linspace(-6, 6, 121), 10)     # 0.1 cells: every window edge on a cell edge
    pt_edges = np.round(np.linspace(0.0, 3.0, 61), 10)    # 0.05 cells

    def grid(slots):
        evs = [{k: v[np.isin(e['species_idx'], slots)] for k, v in e.items()} for e in events]
        return particles_to_qgrid_data(evs, eta_edges, pt_edges, n_harm=6, masses=SPECIES_MASSES, cent_mult=mult)
    ms = MultiSpeciesGrid({211: grid([0]), 321: grid([1]), 2212: grid([PROTON]), 3122: grid([LAMBDA])},
                          combined=grid([0, 1, PROTON]))
    charged = [{k: v[e['species_idx'] < 3] for k, v in e.items()} for e in events]
    qv = make_qvector_data(charged, {'cfg': a_cfg}, n_pt=60)
    _CACHE['fx'] = dict(events=events, charged=charged, ms=ms, qv=qv)
    return _CACHE['fx']


def record(species, ref, var='pT'):
    if var == 'pT':
        kr = {'eta': {'min': ETA[0], 'max': ETA[1]},
              'pT': {'min': PT_BINS[0], 'max': PT_BINS[-1], 'bins': PT_BINS}}
    else:
        kr = {'eta': {'min': ETA_BINS[0], 'max': ETA_BINS[-1], 'bins': ETA_BINS}, 'pT': {'min': 0.3, 'max': 2.4}}
    r = dict(observable_type='flow_vn', method='event_plane', dimensionality=f'differential_{var}', harmonic=2,
             value_type='v2', hepdata_file='t.yaml', inspire_id='0', version=1, kinematic_range=kr,
             centrality={'classes': ['0-100%'], 'eta': [{'min': -0.5, 'max': 0.5}]},
             particle_id={'species': species})
    if ref is not None:
        r['flow_reference'] = ref
    return r


def production(rec, dim, key=None):
    fx = _fx()
    params = extract_measurement_params(rec)
    ent = get_pipeline_entry(key or resolve(rec)['compute'])
    if dim == '2D':
        data = ent['prepare_data_2d'](fx['qv'], params)
    else:
        data = ent['prepare_data_3d'](container_for_prepare(fx['ms'], params, ent), params)
    nb = len(params['differential_bins'])
    return np.asarray(ent['core_fn'](data, np.ones(NE)))[nb:2 * nb]          # harmonic 2 (harmonic-major)


def _windows(rec, b_lo, b_hi, var):
    """POI box and reference (windows, pT) of one bin, from the record."""
    kr = rec['kinematic_range']
    eta = (kr['eta']['min'], kr['eta']['max'])
    pt = (kr['pT']['min'], kr['pT']['max'])
    poi = ((b_lo, b_hi), pt) if var == 'eta' else (eta, (b_lo, b_hi))
    fr = rec.get('flow_reference')
    if fr is None:
        return poi, ([eta], pt)
    wins = [(w['min'], w['max']) for w in fr['eta']]
    if fr.get('mirror'):
        lo_w, hi_w = wins[0]
        c_lo, c_hi = poi[0]
        wins = ([(-hi_w, -lo_w), (lo_w, hi_w)] if c_lo < 0 < c_hi else
                [(-hi_w, -lo_w)] if c_hi > 0 else [(lo_w, hi_w)])
    rpt = (fr['pT']['min'], fr['pT']['max']) if fr.get('pT') else pt
    return poi, (wins, rpt)


def particle_level_3d(rec, poi_slots, var='pT', with_err=False):
    """<(Re q_poi.Q_ref* - shared)/n_poi> / sqrt(<|Q_ref|^2 - N_ref>), unit weights, per bin
    (and its delete-block jackknife error, 20 blocks)."""
    edges = PT_BINS if var == 'pT' else ETA_BINS
    out, errs = [], []
    for b_lo, b_hi in zip(edges[:-1], edges[1:]):
        (pe, pp), (wins, rpt) = _windows(rec, b_lo, b_hi, var)
        num, den = [], []
        for e in _fx()['events']:
            def inside(eta_lo, eta_hi, pt_lo, pt_hi):
                return (e['eta'] >= eta_lo) & (e['eta'] < eta_hi) & (e['pt'] >= pt_lo) & (e['pt'] < pt_hi)
            poi = inside(pe[0], pe[1], pp[0], pp[1]) & np.isin(e['species_idx'], poi_slots)
            ref = np.zeros(len(e['pt']), bool)
            for lo, hi in wins:
                ref |= inside(lo, hi, rpt[0], rpt[1])
            ref &= e['species_idx'] < 3
            u = np.exp(2j * e['phi'])
            Q, n = u[ref].sum(), poi.sum()
            num.append((np.real(u[poi].sum() * np.conj(Q)) - (poi & ref).sum()) / n if n else 0.0)
            den.append(abs(Q) ** 2 - ref.sum())
        num, den = np.array(num), np.array(den)
        out.append(num.mean() / np.sqrt(den.mean()))
        jk = np.array([np.delete(num, k).mean() / np.sqrt(np.delete(den, k).mean())
                       for k in np.array_split(np.arange(len(num)), 20)])
        errs.append(np.sqrt(19 / 20 * np.sum((jk - jk.mean()) ** 2)))
    return (np.array(out), np.array(errs)) if with_err else np.array(out)


def particle_level_2d(rec, poi_slots):
    """The weighted estimator on the 2D container's particles: each particle
    carries the window's Jacobian weight at its pT cell (flat in y)."""
    qv = _fx()['qv']
    cen, edges = qv.pt_centers, qv.pt_bin_edges
    rap = qv.rap_max
    out = []
    for b_lo, b_hi in zip(PT_BINS[:-1], PT_BINS[1:]):
        (pe, _), (wins, rpt) = _windows(rec, b_lo, b_hi, 'pT')
        jp = _make_jacobian_list(cen, qv.masses, pe[0], pe[1]) / (2 * rap)
        jr = sum(_make_jacobian_list(cen, qv.masses, lo, hi) for lo, hi in wins) / (2 * rap)
        plo, phi_ = int(np.searchsorted(cen, b_lo)), int(np.searchsorted(cen, b_hi, side='right'))
        rlo, rhi = _find_pt_slice_edges(edges, rpt[0], rpt[1])
        num, den = [], []
        for e in _fx()['charged']:
            cell = np.searchsorted(edges, e['pt'], side='right') - 1
            sp = e['species_idx']
            ok = (cell >= 0) & (cell < len(cen))
            wp = np.where(ok & (cell >= plo) & (cell < phi_) & np.isin(sp, poi_slots),
                          jp[sp, np.clip(cell, 0, len(cen) - 1)], 0.0)
            wr = np.where(ok & (cell >= rlo) & (cell < rhi), jr[sp, np.clip(cell, 0, len(cen) - 1)], 0.0)
            u = np.exp(2j * e['phi'])
            Q = (wr * u).sum()
            n = wp.sum()
            num.append((np.real((wp * u).sum() * np.conj(Q)) - (wp * wr).sum()) / n if n else 0.0)
            den.append(abs(Q) ** 2 - (wr ** 2).sum())
        out.append(np.mean(num) / np.sqrt(np.mean(den)))
    return np.array(out)


def test_routing():
    print('test_routing')
    for name, ref in REFS.items():
        if name == 'none':
            continue
        var = 'eta' if name == 'mirrored' else 'pT'
        key = resolve(record('p + pbar', ref, var))['compute']
        check(f'identified EP with {name} reference -> SP key', key == f'vn2_sp_diff_{var}', key)
        check(f'{key} selects species itself', get_pipeline_entry(key).get('selects_species') is True)


def test_3d_equals_particle_level():
    print('test_3d_equals_particle_level')
    for species, slots in (('p + pbar', [PROTON]), ('charged', [0, 1, PROTON]), ('Lambda', [LAMBDA])):
        for name, ref in REFS.items():
            var = 'eta' if name == 'mirrored' else 'pT'
            rec = record(species, ref if name != 'none' else None, var)
            if name == 'none':
                rec['method'] = 'scalar_product'       # no reference: the record's own window
            got, want = production(rec, '3D'), particle_level_3d(rec, slots, var)
            check(f'{species:8s} {name:10s}', np.allclose(got, want, rtol=1e-9, atol=1e-12),
                  f'{np.round(got, 4)} vs {np.round(want, 4)}')


def test_2d_equals_particle_level():
    print('test_2d_equals_particle_level')
    for species, slots in (('p + pbar', [PROTON]), ('charged', [0, 1, PROTON])):
        for name in ('tpc halves', 'forward'):
            rec = record(species, REFS[name])
            got, want = production(rec, '2D'), particle_level_2d(rec, slots)
            check(f'{species:8s} {name:10s}', np.allclose(got, want, rtol=1e-9, atol=1e-12),
                  f'{np.round(got, 4)} vs {np.round(want, 4)}')


def test_physics():
    print('test_physics')
    fx = _fx()
    # charged mixture per pT bin: v = (V_PI N_piK + V_P N_p) / N in |eta| < 0.8
    mix = []
    for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:]):
        n_p = n_all = 0
        for e in fx['charged']:
            s = (np.abs(e['eta']) < ETA[1]) & (e['pt'] >= lo) & (e['pt'] < hi)
            n_all += s.sum()
            n_p += (s & (e['species_idx'] == PROTON)).sum()
        mix.append(V_PI + (V_P - V_PI) * n_p / n_all)
    mix = np.array(mix)
    # 2D uses every particle (boost invariance): 15%.  3D only those in the
    # windows: within 3 sigma (jackknife of the identical particle-level
    # estimator, which production equals).
    for name in ('tpc halves', 'forward'):
        for species, slots, want in (('p + pbar', [PROTON], V_P), ('charged', [0, 1, PROTON], mix),
                                     ('Lambda', [LAMBDA], V_P)):
            rec = record(species, REFS[name])
            if species != 'Lambda':
                v = production(rec, '2D')
                check(f'2D {species:8s} {name:10s}', np.all(np.abs(v - want) < 0.15 * want),
                      f'{np.round(v, 4)} vs {np.round(want, 4)}')
            v = production(rec, '3D')
            _, err = particle_level_3d(rec, slots, with_err=True)
            check(f'3D {species:8s} {name:10s}', np.all(np.abs(v - want) < 3 * err),
                  f'{np.round(v, 4)} vs {np.round(want, 4)} (3 sigma: {np.round(3 * err, 4)})')
    # the JETSCAPE pair-normalized form weights events differently: same
    # value within the statistical error
    rec = record('p + pbar', REFS['tpc halves'])
    _, err = particle_level_3d(rec, [PROTON], with_err=True)
    for dim in ('2D', '3D'):
        a, b = production(rec, dim), production(rec, dim, 'vn2_sp_diff_pT_means')
        check(f'{dim} _means form agrees', np.all(np.abs(a - b) < 2 * err), f'{np.round(a, 4)} vs {np.round(b, 4)}')


if __name__ == '__main__':
    tests = [test_routing, test_3d_equals_particle_level, test_2d_equals_particle_level, test_physics]
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
