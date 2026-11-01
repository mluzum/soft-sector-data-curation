"""Asymmetric-system frames (analysis.frames).

- mapping: a lab-frame record (shift -0.465, Pb toward +eta_lab) onto a
  container with p toward +eta: eta_c = -(eta_lab + 0.465); bins keep their
  row order (bin_ranges); estimator and subevent windows move too; a
  mirrored reference is refused; an undeclared container is refused; a
  record without frame fields is unchanged;
- production: a toy 3D grid whose dN/deta falls linearly toward the p side
  (+eta), and the lab-frame dN/deta_lab record computed through the
  production prepare/core equals the toy density at the mapped eta.
Run: python3 tests/test_frames.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.frames import to_container_frame  # noqa: E402
from analysis.pipeline_dispatch import get_pipeline_entry  # noqa: E402

CONTAINER = {'frame': 'nn_cms', 'forward_beam': 'p'}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def lab_record(**extra):
    r = {'observable_type': 'particle_yields', 'inspire_id': '0', 'version': 1, 'hepdata_file': 't.yaml',
         'value_type': 'dNch/deta',
         'collision_system': {'energy_gev': 8160.0, 'reaction': 'P PB --> CHARGED X', 'frame': 'lab',
                              'rapidity_shift': -0.465, 'forward_beam': 'Pb'},
         'centrality': {'classes': ['0-100%'], 'estimator': {'name': 'V0A', 'observable': 'amplitude'},
                        'eta': [{'min': 2.8, 'max': 5.1}]},
         'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0, 'bins': [-1.0, 0.0, 1.0]},
                             'pT': {'min': 0.0, 'max': 100.0}},
         'dimensionality': 'differential_eta'}
    r.update(extra)
    return r


def test_mapping():
    print('test_mapping')
    r, why = to_container_frame(lab_record(), CONTAINER)
    check('mapped', why is None, str(why))
    kr = r['kinematic_range']['eta']
    check('window', (kr['min'], kr['max']) == (-1.465, 0.535), str((kr['min'], kr['max'])))
    check('bins keep row order', kr['bin_ranges'] == [[-0.465, 0.535], [-1.465, -0.465]], str(kr['bin_ranges']))
    check('estimator window', r['centrality']['eta'] == [{'min': -5.565, 'max': -3.265}], str(r['centrality']['eta']))
    check('marked as container frame', r['collision_system']['forward_beam'] == 'p'
          and r['collision_system']['rapidity_shift'] == 0.0)
    same = dict(lab_record(), collision_system={'energy_gev': 8160.0, 'reaction': 'P PB --> X', 'frame': 'nn_cms',
                                                'forward_beam': 'p'})
    check('same frame unchanged', to_container_frame(same, CONTAINER)[0] is same)
    plain = {k: v for k, v in lab_record().items() if k != 'collision_system'}
    plain['collision_system'] = {'energy_gev': 200.0, 'reaction': 'AU AU --> X'}
    check('symmetric record unchanged', to_container_frame(plain, CONTAINER)[0] is plain)
    check('undeclared container refused', to_container_frame(lab_record(), None)[1] is not None)
    mir = lab_record(flow_reference={'detector': 'FCal', 'mirror': True, 'eta': [{'min': 3.2, 'max': 4.9}]})
    check('mirrored reference refused', to_container_frame(mir, CONTAINER)[1] is not None)


def test_production_dndeta():
    print('test_production_dndeta')
    from analysis.qvector import QGridData
    eta_edges = np.round(np.linspace(-6, 6, 121), 10)
    pt_edges = np.array([0.0, 1.0, 100.0])
    centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    dens = 20.0 - 2.0 * centers                     # per unit eta, larger at -eta (Pb side)
    n_ev = 4
    Qn = np.zeros((n_ev, len(centers), 2, 2), complex)
    Qn[:, :, 0, 0] = dens * 0.1                     # counts per 0.1-wide cell, all in the first pT cell
    grid = QGridData(Qn=Qn, eta_edges=eta_edges, pt_edges=pt_edges, cent_mult=np.ones(n_ev), nsamples=np.ones(n_ev),
                     positives_only=False)
    rec, why = to_container_frame(lab_record(), CONTAINER)
    p = extract_measurement_params(rec)
    ent = get_pipeline_entry(resolve(rec)['compute'])
    out = np.asarray(ent['core_fn'](ent['prepare_data_3d'](grid, p), np.ones(n_ev)))
    # lab bins [-1,0], [0,1] -> container [-0.465,0.535], [-1.465,-0.465]: mean density at their centres
    want = [20.0 - 2.0 * 0.035, 20.0 - 2.0 * (-0.965)]
    check('dN/deta_lab via the container frame', np.allclose(out[:2], want, rtol=1e-6), f'{out[:2]} vs {want}')


if __name__ == '__main__':
    tests = [test_mapping, test_production_dndeta]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
            print(f'  PASS  {t.__name__}\n')
        except Exception as ex:
            print(f'  FAIL  {t.__name__}: {ex}\n')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
