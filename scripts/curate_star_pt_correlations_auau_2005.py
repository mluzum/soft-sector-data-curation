#!/usr/bin/env python3
"""Curate STAR pT correlations in Au+Au at 19.6-200 GeV, INSPIRE 681688.

J. Adams et al. (STAR), "Incident energy dependence of pt correlations at
RHIC", PRC 72 (2005) 044902, nucl-ex/0504031.

From the paper (Sec. II):
- tracks from the TPC with 0.15 <= pT <= 2.0 GeV/c and |eta| < 1.0;
- <Delta pT_i Delta pT_j> = (1/N_event) sum_k C_k / (N_k (N_k - 1)) (Eq.
  1-2): the event average of each event's pair average, deviations from
  <<pT>> = the mean of the event means (Eq. 3-4);
- centrality from the multiplicity of all charged particles in the TPC
  with |eta| < 0.5, classes 0-5, 5-10, 10-20, ..., 70-80%;
- <<pT>> is fitted as a function of that multiplicity N_ch and the fit
  used event by event in Eq. 1-4 (removes the spread of <<pT>> across a
  class; lowers the correlations by ~3%): multiplicity_baseline;
- errors: statistical; systematic 2% relative on <Delta pT_i Delta pT_j>
  (pT-dependent efficiency 1.2%, track merging and splitting 1.4%), not
  in the tables;
- energies 20 (the 19.6 GeV run), 62 (62.4), 130 and 200 GeV.

Records (public, 4): Fig. 4, sqrt(<Delta pT_i Delta pT_j>)/<<pT>> vs class
per energy (value_scale 100: percent).  Not curated: Fig. 2-3 (the same
correlations as <Delta pT_i Delta pT_j> and times dN/deta), the HIJING
columns, the Fig. 4 inset (0-5% vs energy with CERES points) and Fig. 1
(<pT> distributions).

Run from the repository root:  python scripts/curate_star_pt_correlations_auau_2005.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins681688/HEPData-ins681688-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRC 72 (2005) 044902, nucl-ex/0504031'
CLASSES = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%']
NCH = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': -0.5, 'max': 0.5}],
       'notes': 'all charged particles measured in the TPC with |eta| < 0.5.'}
# HEPData column label -> energy (GeV); HEPData and the paper round 19.6 and 62.4
ENERGIES = {'20 GeV': 19.6, '62 GeV': 62.4, '130 GeV': 130.0, '200 GeV': 200.0}
OUT = 'data/mean_pT_fluctuations/star_mean_pT_fluctuations_pT_correlations_AuAu_19.6_200gev_ins681688.yaml'


def main():
    t = yaml.safe_load(open(os.path.join(RAW, 'Fig4.yaml')))
    rows = [str(v['value']) for v in t['independent_variables'][0]['values']]
    assert rows == CLASSES, rows
    docs = []
    for k, dv in enumerate(t['dependent_variables']):
        label = dv['qualifiers'][0]['value']
        if label not in ENERGIES:
            assert k == 0 or 'HIJING' in label, label
            continue
        assert '(\\%)' in dv['header']['name'], dv['header']
        energy = ENERGIES[label]
        d = {'observable_type': 'mean_pT_fluctuations', 'inspire_id': '681688', 'version': VERSION,
             'hepdata_file': 'Fig4.yaml', 'dv_index': k,
             'hepdata_table_label': f'sqrt(<Delta pT_i Delta pT_j>)/<<pT>> vs centrality, Au+Au {energy:g} GeV '
                                    '(Fig4.yaml)',
             'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
             'centrality': {'classes': list(CLASSES), **NCH},
             'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.15, 'max': 2.0}},
             'observable': 'sqrt_dpt_dpt_over_mean_pT', 'value_type': 'sqrt_<dpt1_dpt2>_over_<pT>',
             'method': 'two_particle', 'averaging': 'events',
             'multiplicity_baseline': {'observable': 'Nch', 'eta': [{'min': -0.5, 'max': 0.5}]},
             'value_scale': 100.0}
        if energy in (19.6, 62.4):
            d['hepdata_corrections'] = {'energy': f'HEPData labels the column {label} (the paper rounds too); the '
                                                  f'run was at {energy:g} GeV'}
        d['notes'] = {'value': f'{CITE}, Fig. 4. sqrt(<Delta pT_i Delta pT_j>)/<<pT>> of charged particles, |eta| '
                               f'< 1, 0.15 < pT < 2 GeV/c, Au+Au {energy:g} GeV, in percent (value_scale 100). '
                               'Event average of the pair averages (Eq. 1), deviations from <<pT>> fitted vs the '
                               'centrality multiplicity N_ch (|eta| < 0.5), event by event (multiplicity_baseline). '
                               'Errors: stat; sys 2% relative on the covariance (Sec. II), not tabulated.'}
        docs.append(d)
    assert len(docs) == 4, len(docs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()
