#!/usr/bin/env python3
"""Curate BRAHMS identified-hadron spectra at y = 0 and y ~ 1 in Au+Au at 200 GeV, INSPIRE 678407.

I. Arsene et al. (BRAHMS), "Centrality dependent particle production at
y = 0 and y ~ 1 in Au+Au collisions at sqrt(sNN) = 200 GeV", PRC 72,
014908 (2005), nucl-ex/0503010.

From the paper:
- the mid-rapidity spectrometer at 90 degrees, -0.1 < y < 0.1, and at
  45-35 degrees, 0.7 < y < 1.1 (RHIC Run II);
- centrality from the multiplicity array (Si strips and tiles averaged,
  -2.2 < eta < 2.2): 0-10, 10-20, 20-40, 40-60%;
- p and pbar corrected for Lambda (Lambda-bar) feed-down; pions not
  corrected for weak decays;
- errors: statistical on the points; overall systematic 10-15%.

Records (public): invariant yields 1/(2 pi pT) d2N/(dpT dy) of pi+-, K+-,
p, pbar in four classes at y = 0 and y ~ 1 (Table 5, 48 records).  Staged
(--staged OUT): <pT> of each species (Table 6: no uncertainties in
HEPData).  Not curated: blast-wave fit parameters (Table 7), dN/dy per
participant pair (Table 9, N_part normalized), the particle ratios vs pT
and at 1.2 < pT < 2 GeV/c (Tables 10-12: ratios of the Table 5 spectra).
The old record read Table 9 (per participant pair) as pbar dN/dy in 0-1%.

Run from the repository root:
    python scripts/curate_brahms_identified_spectra_y0_y1_200.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins678407/HEPData-ins678407-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'I. Arsene et al. (BRAHMS), PRC 72, 014908 (2005), nucl-ex/0503010'
MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'Multiplicity array'},
      'eta': [{'min': -2.2, 'max': 2.2}]}
SPECIES = {'PI+': 'pi+', 'PI-': 'pi-', 'K+': 'K+', 'K-': 'K-', 'P': 'p', 'PBAR': 'pbar'}
Y = {'0': {'min': -0.1, 'max': 0.1}, '1': {'min': 0.7, 'max': 1.1}}
CLASSES = ['0-10%', '10-20%', '20-40%', '40-60%']
OLD = 'data/particle_yields/brahms_particle_yields_centrality_dependent_particle_production_AuAu_200gev.yaml'
OUT = 'data/particle_spectra/brahms_particle_spectra_identified_y0_y1_AuAu_200gev_ins678407.yaml'
BLANK = ('', '-', '--', 'none', 'nan')


def feed(sp):
    return ('p and pbar corrected for Lambda feed-down' if sp in ('p', 'pbar')
            else 'not corrected for weak-decay feed-down' if sp.startswith('pi') else '')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    t = yaml.safe_load(open(os.path.join(RAW, 'tab5.yaml')))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        q = {x['name']: str(x['value']) for x in dv['qualifiers']}
        sp, cls, y = SPECIES[q['PARTICLE']], q['CENTRALITY'] + '%', Y[q['YRAP']]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        where = f"{y['min']:g} < y < {y['max']:g}"
        d = {'observable_type': 'particle_spectra', 'inspire_id': '678407', 'version': VERSION,
             'hepdata_file': 'tab5.yaml', 'dv_index': i, 'hepdata_table_label': f'{sp} invariant yield, {cls}, {where} (tab5.yaml)',
             'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
             'centrality': {'classes': [cls], **MA}, 'particle_id': {'species': sp},
             'kinematic_range': {'y': dict(y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}, Fig. 5. 1/(2 pi pT) d2N/(dpT dy) of {sp}, {cls}, {where}'
                                + (f'; {feed(sp)}' if feed(sp) else '')
                                + '. Errors: statistical (overall systematic 10-15% in the paper).'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        public.append(d)
    t = yaml.safe_load(open(os.path.join(RAW, 'tab6.yaml')))
    for i, dv in enumerate(t['dependent_variables']):
        q = {x['name']: str(x['value']) for x in dv['qualifiers']}
        sp, y = SPECIES[q['PARTICLE']], Y[q['YRAP']]
        staged.append({'observable_type': 'mean_pT', 'inspire_id': '678407', 'version': VERSION, 'hepdata_file': 'tab6.yaml',
                       'dv_index': i, 'hepdata_table_label': f"<pT> of {sp} vs centrality, {y['min']:g} < y < {y['max']:g} "
                                                            '(tab6.yaml)',
                       'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
                       'centrality': {'classes': list(CLASSES), **MA}, 'particle_id': {'species': sp},
                       'kinematic_range': {'y': dict(y), 'pT': {'min': 0.0, 'max': 100.0}},
                       'value_type': 'mean_pT', 'observable': 'mean_pT',
                       'notes': {'value': f"{CITE}, Fig. 6. <pT> of {sp} from fits to the spectra, "
                                          f"{y['min']:g} < y < {y['max']:g}, vs centrality (the N_part of each class "
                                          'in HEPData).'},
                       'staging_reason': 'no uncertainties in HEPData'})
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '678407' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
