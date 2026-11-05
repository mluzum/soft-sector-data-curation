#!/usr/bin/env python3
"""Curate STAR relative dynamical pT correlations in Au+Au at 7.7-200 GeV, INSPIRE 1712047.

J. Adam et al. (STAR), "Collision-energy dependence of pt correlations in
Au+Au collisions at energies available at the BNL Relativistic Heavy Ion
Collider", PRC 99 (2019) 044918, arXiv:1901.00837.

From the paper (Sec. II, III):
- charged tracks with 0.2 <= pT <= 2.0 GeV/c and |eta| < 0.5;
- centrality from the reference multiplicity in 0.5 < |eta| < 1.0 (so the
  tracks of the correlation are not those of the centrality), classes 0-5,
  5-10, 10-20, ..., 70-80% of the inelastic cross section; HEPData gives
  each class by its Glauber <N_part> (the nine rows, in that order);
- the relative dynamical correlation sqrt(<Delta pT_i Delta pT_j>) /
  <<pT>> (Eq. 1-4: the event average of each event's pair average,
  deviations from <<pT>> = the mean of the event means), in percent;
  errors stat and sys;
- <<pT>> and <Delta pT_i Delta pT_j> are calculated as a function of the
  reference multiplicity and then averaged over the centrality bin (Sec.
  II): here deviations from <<pT>> at the event's reference multiplicity
  (multiplicity_baseline, a linear regression within the class);
- energies 7.7, 11.5, 14.5 (HEPData: 14.6), 19.6, 27, 39, 62.4, 200 GeV.

Records (public, 8): the Fig. 1 data per energy vs class (value_scale 100:
percent), computed by the pT_fluct core with `averaging: events` (until
November 2026 ALICE's pooled pairs and the class mean pT).  Not curated: the UrQMD and power-law tables (model
curves and data/model ratios), the efficiency studies (Fig. 2), the
Fig. 4-5 replots (the same STAR points, CERES and ALICE data of other
papers, models).

Run from the repository root:  python scripts/curate_star_pt_correlations_bes.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1712047/HEPData-ins1712047-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (STAR), PRC 99 (2019) 044918, arXiv:1901.00837'
CLASSES = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%']
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -1.0, 'max': -0.5}, {'min': 0.5, 'max': 1.0}],
           'notes': 'reference multiplicity in 0.5 < |eta| < 1.0, outside the correlation window.'}
FILES = {7.7: 'Fig1_Data_7.7GeV.yaml', 11.5: 'Fig1_Data_11.5GeV.yaml', 14.5: 'Fig1_Data_14.6GeV.yaml',
         19.6: 'Fig1_Data_19.6GeV.yaml', 27.0: 'Fig1_Data_27.0GeV.yaml', 39.0: 'Fig1_Data_39.0GeV.yaml',
         62.4: 'Fig1_Data_62.4GeV.yaml', 200.0: 'Fig1_Data_200.0GeV.yaml'}
OUT = 'data/mean_pT_fluctuations/star_mean_pT_fluctuations_pT_correlations_AuAu_7_200gev_ins1712047.yaml'


def main():
    docs = []
    for energy, hf in FILES.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        npart = [float(v['value']) for v in t['independent_variables'][0]['values']]
        assert len(npart) == len(CLASSES) and npart == sorted(npart, reverse=True), (hf, npart)
        dv = t['dependent_variables'][0]
        assert '(\\%)' in str(dv['qualifiers'][0]['value']), hf
        assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
        docs.append({
            'observable_type': 'mean_pT_fluctuations', 'inspire_id': '1712047', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': f'sqrt(<Delta pT_i Delta pT_j>)/<<pT>> vs centrality, Au+Au '
                                                  f'{energy:g} GeV ({hf})',
            'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
            'centrality': {'classes': list(CLASSES), **REFMULT},
            'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.2, 'max': 2.0}},
            'observable': 'sqrt_dpt_dpt_over_mean_pT', 'value_type': 'sqrt_<dpt1_dpt2>_over_<pT>', 'method': 'two_particle',
            'averaging': 'events', 'multiplicity_baseline': {'observable': 'Nch', 'eta': [dict(e) for e in REFMULT['eta']]},
            'value_scale': 100.0,
            'notes': {'value': f'{CITE}, Fig. 1. Relative dynamical pT correlation sqrt(<Delta pT_i Delta pT_j>)/<<pT>> '
                               f'of charged particles, |eta| < 0.5, 0.2 < pT < 2.0 GeV/c, Au+Au {energy:g} GeV, in '
                               'percent (value_scale 100); the rows are the classes 0-5 ... 70-80% (Sec. II), given '
                               'in HEPData by their <N_part>' + ('; HEPData labels the energy 14.6 GeV'
                                                                if energy == 14.5 else '')
                               + '. Event average of the pair averages, deviations from <<pT>> at the event\'s '
                               'reference multiplicity (multiplicity_baseline). Errors: stat and sys.'}})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()
