#!/usr/bin/env python3
"""Curate STAR pT correlations in Cu+Cu at 62.4 and 200 GeV, INSPIRE 1216565.

L. Adamczyk et al. (STAR), "System-size dependence of transverse momentum
correlations at sqrt(sNN) = 62.4 and 200 GeV at the BNL Relativistic
Heavy Ion Collider", PRC 87 (2013) 064902, arXiv:1301.6633.

From the paper (Sec. II-IV):
- TPC tracks with |eta| < 1.0 and 0.15 < pT < 2.0 GeV/c (DCA < 1 cm, at
  least 21 fit points);
- <Delta pT_i Delta pT_j> = (1/N_event) sum_k C_k / (N_k (N_k - 1)) (Eq.
  4-6): the event average of each event's pair average, deviations from
  <<pT>> = the mean of the event means;
- centrality from the uncorrected charged tracks in the TPC with |eta| <
  0.5, classes 0-10, 10-20, ..., 50-60%; <<pT>> is computed as a function
  of that multiplicity N_ch to remove the spread across a class
  (multiplicity_baseline);
- the figures' errors are stat and sys in quadrature (the statistical
  ones are small); HEPData labels them 'stat error';
- Au+Au points are those of PRC 72 (2005) 044902 (INSPIRE 681688).

Records (public, 42):
- Fig. 3(c): sqrt(<Delta pT_i Delta pT_j>)/<<pT>> vs class at 200 and
  62.4 GeV (value_scale 100: percent);
- Fig. 7: <Delta pT_i Delta pT_j> ((MeV/c)^2, value_type '<dpt1_dpt2>')
  vs class in eight pT ranges (0.15-0.35 ... 0.8-2.0 GeV/c) per energy;
- Fig. 5: <Delta pT_i Delta pT_j> in |eta| < 0.25 and < 0.5, one record
  per class, energy and window (the |eta| < 1 row is Fig. 3).
Not curated: Fig. 3(a, b, d) and the 0.15-2.0 GeV/c column of Fig. 7 (the
same correlations), the Au+Au columns (INSPIRE 681688), Fig. 2 (sigma_dyn
from mixed events), Fig. 6 (azimuthal acceptance: no azimuthal windows
here), Fig. 4 and 8 (models), Fig. 1 and 9 (<pT> distributions, an
uncorrected spectrum and model-scaled correlations).

Run from the repository root:  python scripts/curate_star_pt_correlations_cucu.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1216565/HEPData-ins1216565-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'L. Adamczyk et al. (STAR), PRC 87 (2013) 064902, arXiv:1301.6633'
CLASSES = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%']
NCH = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': -0.5, 'max': 0.5}],
       'notes': 'uncorrected charged tracks in the TPC with |eta| < 0.5.'}
BASELINE = {'observable': 'Nch', 'eta': [{'min': -0.5, 'max': 0.5}]}
FILES = {200.0: ('fig3_CuCu200GeV.yaml', 'fig7_CuCu200GeV.yaml'),
         62.4: ('fig3_CuCu62GeV.yaml', 'fig7_CuCu62GeV.yaml')}
OUT = 'data/mean_pT_fluctuations/star_mean_pT_fluctuations_pT_correlations_CuCu_62.4_200gev_ins1216565.yaml'
METHOD = ('event average of the pair averages (Eq. 4), deviations from <<pT>> as a function of the centrality '
          'multiplicity N_ch (|eta| < 0.5; multiplicity_baseline)')
ERRORS = 'Errors: stat and sys in quadrature (HEPData label: stat error).'


def load(f):
    return yaml.safe_load(open(os.path.join(RAW, f)))


def record(hf, k, energy, label, eta, pt, value_type, classes, rows=None):
    d = {'observable_type': 'mean_pT_fluctuations', 'inspire_id': '1216565', 'version': VERSION,
         'hepdata_file': hf, 'dv_index': k, 'hepdata_table_label': f'{label}, Cu+Cu {energy:g} GeV ({hf})',
         'collision_system': {'energy_gev': energy, 'reaction': 'CU CU --> CHARGED X'},
         'centrality': {'classes': list(classes), **NCH},
         'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -eta, 'max': eta}, 'pT': {'min': pt[0], 'max': pt[1]}}}
    if value_type == 'relative':
        d.update(observable='sqrt_dpt_dpt_over_mean_pT', value_type='sqrt_<dpt1_dpt2>_over_<pT>')
    else:
        d.update(observable='dpt_dpt', value_type='<dpt1_dpt2>')
    d.update(method='two_particle', averaging='events', multiplicity_baseline=dict(BASELINE))
    if value_type == 'relative':
        d['value_scale'] = 100.0
    if rows is not None:
        d['table_rows'] = rows
    return d


def npart_rows(t):
    npart = [float(v['value']) for v in t['independent_variables'][0]['values']]
    assert len(npart) == len(CLASSES) and npart == sorted(npart, reverse=True), npart
    return npart


def main():
    docs = []
    for energy, (f3, f7) in FILES.items():
        t = load(f3)
        npart_rows(t)
        dv = t['dependent_variables'][2]
        assert dv['qualifiers'][0]['value'] == 'sqrt(correlation)' and '(\\%)' in dv['header']['name'], dv
        d = record(f3, 2, energy, 'sqrt(<Delta pT_i Delta pT_j>)/<<pT>> vs centrality', 1.0, (0.15, 2.0),
                   'relative', CLASSES)
        d['notes'] = {'value': f'{CITE}, Fig. 3(c). sqrt(<Delta pT_i Delta pT_j>)/<<pT>> of charged particles, '
                               f'|eta| < 1, 0.15 < pT < 2 GeV/c, Cu+Cu {energy:g} GeV, in percent (value_scale '
                               f'100); {METHOD}. {ERRORS}'}
        docs.append(d)
        t = load(f7)
        npart_rows(t)
        for k, dv in enumerate(t['dependent_variables']):
            m = re.fullmatch(r'pt: ([\d.]+)-([\d.]+)\s+GeV/c', dv['qualifiers'][0]['value'])
            lo, hi = float(m.group(1)), float(m.group(2))
            assert '(MeV/c)^{2}' in dv['header']['name'], dv['header']
            if (lo, hi) == (0.15, 2.0):
                continue
            d = record(f7, k, energy, f'<Delta pT_i Delta pT_j> vs centrality, {lo:g} < pT < {hi:g} GeV/c', 1.0,
                       (lo, hi), 'covariance', CLASSES)
            d['notes'] = {'value': f'{CITE}, Fig. 7. <Delta pT_i Delta pT_j> of charged particles, |eta| < 1, '
                                   f'{lo:g} < pT < {hi:g} GeV/c, Cu+Cu {energy:g} GeV, in (MeV/c)^2 (loaded as '
                                   f'GeV^2); {METHOD}. {ERRORS}'}
            docs.append(d)
    t = load('fig5.yaml')
    windows = [float(v['value']) for v in t['independent_variables'][0]['values']]
    assert windows == [0.25, 0.5, 1.0], windows
    for k, dv in enumerate(t['dependent_variables']):
        m = re.fullmatch(r'([\d.]+) GeV, (\d+-\d+%)', dv['qualifiers'][0]['value'])
        energy, cls = float(m.group(1)), m.group(2)
        assert cls in CLASSES and '(MeV/c)^{2}' in dv['header']['name'], dv
        for j, eta in enumerate(windows[:2]):
            d = record('fig5.yaml', k, energy, f'<Delta pT_i Delta pT_j>, |eta| < {eta:g}, {cls}', eta, (0.15, 2.0),
                       'covariance', [cls], rows=[j])
            d['notes'] = {'value': f'{CITE}, Fig. 5. <Delta pT_i Delta pT_j> of charged particles, |eta| < {eta:g}, '
                                   f'0.15 < pT < 2 GeV/c, Cu+Cu {energy:g} GeV, {cls}, in (MeV/c)^2 (loaded as '
                                   f'GeV^2); {METHOD}. {ERRORS}'}
            docs.append(d)
    assert len(docs) == 2 + 16 + 24, len(docs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()
