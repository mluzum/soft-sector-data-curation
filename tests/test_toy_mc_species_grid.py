"""Species-resolved 3D grids via the converter pattern (source-independent).

Everything here runs on dummy Monte Carlo particle lists through
analysis.converters + MultiSpeciesGrid -- no production file formats.
Validates:
  - per-species grids built by filtering particle lists;
  - additive closure: sum of species grids == all-particle grid exactly;
  - MultiSpeciesGrid.select() routes record species strings correctly;
  - yields / mean-pT per species through the standard prepares;
  - scalar-product v_n{SP} recovers injected flow on the charged grid.
Run: python3 tests/test_toy_mc_species_grid.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.pipeline_dispatch import get_pipeline_entry
from analysis.dispatcher import extract_measurement_params

from analysis.toy_mc import (
    ToyGenConfig, generate_events, ref_yield, ref_mean_pt,
    SPECIES_INDEX,
)
from analysis.qvector import MultiSpeciesGrid
from analysis.converters import particles_to_qgrid_data


NE = 1200
ETA = (-0.5, 0.5)
PTW = (0.15, 3.0)
_CACHE = {}


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=NE, mult_mean=250, mult_sigma=50,
                           seed=31337,
                           flow={2: dict(kind='const', value=0.05)})
        events, truth = generate_events(cfg)
        eta_edges = np.linspace(-6, 6, 61)
        pt_edges = np.linspace(0.05, 3.0, 60)

        def build(sel=None):
            evs = events if sel is None else [
                {k: v[sel_i] for k, v in ev.items()}
                for ev, sel_i in ((ev, sel(ev)) for ev in events)]
            return particles_to_qgrid_data(
                evs, eta_edges, pt_edges, n_harm=6, masses=SPECIES_MASSES)

        # per-PDG-style species grids from toy slots:
        #   toy species_idx: 0=pion(+), 1=kaon(+), 2=proton(+) (positives-only)
        pdg_of_slot = {211: 0, 321: 1, 2212: 2}
        grids = {}
        for pdg, slot in pdg_of_slot.items():
            grids[pdg] = build(lambda ev, s=slot: ev['species_idx'] == s)
        combined = build()
        ms = MultiSpeciesGrid(grids, combined=combined)
        _CACHE['fx'] = dict(cfg=cfg, events=events, ms=ms,
                            combined=combined)
    return _CACHE['fx']


SPECIES_MASSES = np.array([0.13957, 0.49368, 0.93827])


def assert_close(name, got, want, tol):
    ok = abs(got - want) <= tol * max(abs(want), 1e-12) + 1e-6
    print(f"  {'PASS' if ok else 'FAIL'} {name}: got={got:.4f} "
          f"want={want:.4f} tol={tol}")
    if not ok:
        raise AssertionError(f"{name}: {got} vs {want}")


def test_additive_closure_exact():
    fx = _fx()
    s = fx['ms'].grids[211] + fx['ms'].grids[321] + fx['ms'].grids[2212]
    assert np.allclose(s.Qn, fx['combined'].Qn), "species sum != combined"
    assert_close("additive closure |dQ|", 0.0,
                 float(np.abs(s.Qn - fx['combined'].Qn).max()), 0.0)


def test_select_routes_species():
    fx = _fx()
    kaon_via_select = fx['ms'].select({'species': 'K+'})
    assert np.allclose(kaon_via_select.Qn, fx['ms'].grids[321].Qn)
    charged_via_select = fx['ms'].select({'species': 'charged'})
    # combined grid is positives-only; 'charged' applies charge symmetry x2
    assert np.allclose(charged_via_select.Qn, 2.0 * fx['combined'].Qn)


def test_string_pdg_keys_normalized():
    """MultiSpeciesGrid must accept string PDG keys (as from YAML-derived
    specs) and still match resolve_species() integer PDG codes."""
    fx = _fx()
    # rebuild with string keys
    grids = {str(pdg): g for pdg, g in fx['ms'].grids.items()}
    ms = MultiSpeciesGrid(grids, combined=fx['combined'])
    for name, pdg in [('K+', 321), ('p', 2212), ('pi+', 211)]:
        sel = ms.select({'species': name})
        assert np.allclose(sel.Qn, fx['ms'].grids[pdg].Qn), name
    assert np.allclose(ms.select({'species': 'charged'}).Qn,
                       2.0 * fx['combined'].Qn)


def _yield(grid):
    rec = dict(
        observable_type='particle_yields', value_type='dN_dy',
        hepdata_file='t.yaml', inspire_id='0', version=1,
        particle_id={'species': 'charged'},
        kinematic_range=dict(eta=dict(min=ETA[0], max=ETA[1]),
                             pT=dict(min=PTW[0], max=PTW[1])),
        centrality=dict(classes=['0-100%'],
                        eta=[{'min': -3.9, 'max': -3.0},
                             {'min': 3.0, 'max': 3.9}]),
    )
    e = get_pipeline_entry('yield_dNch_deta')
    d = e['prepare_data_3d'](grid, extract_measurement_params(rec))
    return float(e['core_fn'](d, np.ones(grid.n_events))[0])


def test_species_yields_vs_truth():
    fx = _fx()
    # Single-sign records compare to positive-only truth (charge_factor = 1.0):
    # the sim stores positives only, so no doubling is applied.  The combined
    # 'charged' grid (used for 'charged' records) is scaled x2 by select.
    sel_map = [('pi+', 0), ('K+', 1), ('p', 2)]
    for name, slot in sel_map:
        gg = fx['ms'].select({'species': name})
        got = _yield(gg)
        want = ref_yield(fx['events'], ETA, PTW, species_sel=[slot])
        assert_close(f"yield {name}", got, want, 0.03)


def test_mean_pt_species():
    fx = _fx()
    rec = dict(
        observable_type='mean_pT', value_type='mean_pT',
        hepdata_file='t.yaml', inspire_id='0', version=1,
        kinematic_range=dict(eta=dict(min=ETA[0], max=ETA[1]),
                             pT=dict(min=PTW[0], max=PTW[1])),
        centrality=dict(classes=['0-100%'],
                        eta=[{'min': -3.9, 'max': -3.0},
                             {'min': 3.0, 'max': 3.9}]),
    )
    e = get_pipeline_entry('mean_pt')
    w = np.ones(fx['combined'].n_events)
    for name, slot, mass in [('pi+', 0, 0.13957), ('p', 2, 0.93827)]:
        gg = fx['ms'].select({'species': name})
        d = e['prepare_data_3d'](gg, extract_measurement_params(rec))
        got = float(e['core_fn'](d, w)[0])
        want = ref_mean_pt(fx['events'], ETA, PTW, species_sel=[slot])
        assert_close(f"mean-pT {name}", got, want, 0.02)


def test_scalar_product_recovers_flow():
    """SP estimator recovers the injected v2 (0.05), same-window (tests
    the overlap subtraction) and disjoint POI/reference (tests the App. M
    <MA> normalization: the old <MB> form would be ~6x off here)."""
    fx = _fx()

    def run(poi_eta, ref_eta, same_window):
        base = dict(
            observable_type='flow_vn', method='scalar_product',
            harmonic=2, dimensionality='integrated', value_type='v2{SP}',
            hepdata_file='t.yaml', inspire_id='0', version=1,
            kinematic_range=dict(eta=dict(min=poi_eta[0], max=poi_eta[1]),
                                 pT=dict(min=0.2, max=3.0)),
            centrality=dict(classes=['0-100%'],
                            eta=[{'min': -3.9, 'max': -3.0},
                                 {'min': 3.0, 'max': 3.9}]),
        )
        e = get_pipeline_entry('vn2_sp')
        d_poi = e['prepare_data_3d'](fx['combined'],
                                     extract_measurement_params(base))
        ref_rec = dict(base)
        ref_rec['kinematic_range']['eta'] = {'min': ref_eta[0],
                                             'max': ref_eta[1]}
        d_ref = e['prepare_data_3d'](fx['combined'],
                                     extract_measurement_params(ref_rec))
        from analysis.observables_2d import compute_vn2_sp
        w = np.ones(fx['combined'].n_events)
        got = float(compute_vn2_sp(d_poi['q_poi'], d_ref['q_ref'],
                                   d_poi['n_poi'], d_ref['n_ref'], w,
                                   same_window=same_window)[1])
        return got

    assert_close("v2{SP} same-window (injected 0.05)",
                 run((-0.4, 0.4), (-0.4, 0.4), True), 0.05, 0.15)
    assert_close("v2{SP} disjoint (injected 0.05)",
                 run((-0.4, 0.4), (0.5, 1.0), False), 0.05, 0.25)


if __name__ == '__main__':
    tests = [fn for fn in sorted(dir()) if fn.startswith('test_')]
    n_pass = n_fail = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"  PASS  {name}\n")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}\n")
            n_fail += 1
    print(f"{n_pass}/{n_pass + n_fail} passed")
    sys.exit(1 if n_fail else 0)
