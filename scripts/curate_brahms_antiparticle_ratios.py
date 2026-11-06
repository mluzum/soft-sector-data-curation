#!/usr/bin/env python3
"""Curate BRAHMS antiparticle/particle ratios vs rapidity in Au+Au at 200 GeV, INSPIRE 590481.

I. G. Bearden et al. (BRAHMS), "Rapidity dependence of charged
antiparticle-to-particle ratios in Au+Au collisions at sqrt(sNN) = 200
GeV", PRL 90 (2003) 102301, arXiv:nucl-ex/0207006.

From the paper:
- pi-/pi+, K-/K+, pbar/p from the two spectrometers, 0 < y < 3;
- centrality from the tile multiplicity array (|eta| < 2.2); the ratios vs
  y are integrated over 0-20% and over the measured pT;
- not corrected for the hyperon feed-down (estimated < 5% on the ratios);
- Fig. 3 (Tables 3a-c): the ratios in rapidity bins, stat and sys.

Records (public): the ratios vs y (3), particle_ratio records linked to
side records that carry no values (experimental_values: false; BRAHMS
publishes only the ratios), dN/dy per rapidity bin of each species, all pT
(the measured pT acceptance is not tabulated; the ratios are flat in pT,
Fig. 2).  Staged (--staged OUT): the ratios vs pT and centrality at y = 0
and y ~ 2 (Fig. 2; the windows not stated: 'y = 2.05 for protons to 2.55
for pions'), and the statistical-model K-/K+ vs pbar/p and mu_B (Fig. 4).

Run from the repository root:  python scripts/curate_brahms_antiparticle_ratios.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins590481/HEPData-ins590481-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'I. G. Bearden et al. (BRAHMS), PRL 90 (2003) 102301, arXiv:nucl-ex/0207006'
OUT_R = 'data/observable_ratios/brahms_observable_ratios_antiparticle_vs_y_AuAu_200gev_ins590481.yaml'
OUT_S = 'data/particle_yields/brahms_particle_yields_ratio_sides_vs_y_AuAu_200gev_ins590481.yaml'
MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'TMA'},
      'eta': [{'min': -2.2, 'max': 2.2}], 'notes': 'tile multiplicity array, |eta| < 2.2.'}
TABLES = {'tab3a.yaml': ('PI-/PI+', 'pi-', 'pi+', 'PI-', 'PI+'), 'tab3b.yaml': ('K-/K+', 'K-', 'K+', 'K-', 'K+'),
          'tab3c.yaml': ('PBAR/P', 'pbar', 'p', 'PBAR', 'P')}
FEED = 'Not corrected for the hyperon feed-down (< 5% on the ratios).'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sides, ratios, staged = [], [], []
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if hf not in TABLES:
            for i, dv in enumerate(t['dependent_variables']):
                staged.append({'observable_type': 'observable_ratios', 'inspire_id': '590481', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc.strip()}'},
                               'staging_reason': ('statistical-model K-/K+ vs pbar/p and mu_B' if hf == 'tab4.yaml'
                                                  else "Fig. 2: ratios at y = 0 and y ~ 2, windows not stated "
                                                       "('y = 2.05 for protons to 2.55 for pions')")})
            continue
        name, num, den, pn, pd = TABLES[hf]
        dv = t['dependent_variables'][0]
        assert len(t['dependent_variables']) == 1 and dv['header']['name'] == name, hf
        ybins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        assert all(a[1] <= b[0] for a, b in zip(ybins, ybins[1:])), ybins
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat', 'sys')}
        refs = []
        for species, prod in ((num, pn), (den, pd)):
            refs.append(len(sides))
            sides.append({
                'observable_type': 'particle_yields', 'inspire_id': '590481', 'version': VERSION,
                'experimental_values': False,
                'hepdata_table_label': f'{species} dN/dy per rapidity bin, Au+Au 200 GeV 0-20% (the side of the {hf} '
                                       'ratio; no values published)',
                'collision_system': {'energy_gev': 200.0, 'reaction': f'AU AU --> {prod} X'},
                'centrality': {'classes': ['0-20%'], **MA}, 'particle_id': {'species': species},
                'kinematic_range': {'y': {'min': ybins[0][0], 'max': ybins[-1][1], 'bin_ranges': ybins},
                                    'pT': {'min': 0.0, 'max': 100.0}},
                'observable': 'dN_dy', 'value_type': 'dN_dy', 'dimensionality': 'differential_y',
                'notes': {'value': f'{CITE}. dN/dy of {species} per rapidity bin, Au+Au 200 GeV 0-20%: the side of '
                                   'the published antiparticle/particle ratio (BRAHMS gives only the ratio; the ratio '
                                   f'record holds the values). {FEED}'}})
        ratios.append({
            'observable_type': 'observable_ratios', 'inspire_id': '590481', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': f'{num}/{den} vs y, Au+Au 200 GeV 0-20% ({hf})',
            'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X X'},
            'centrality': {'classes': ['0-20%'], **MA},
            'kinematic_range': {'y': {'min': ybins[0][0], 'max': ybins[-1][1], 'bin_ranges': ybins},
                                'pT': {'min': 0.0, 'max': 100.0}},
            'observable': 'dN_dy', 'value_type': f'{num}/{den}', 'ratio_type': 'particle_ratio',
            'dimensionality': 'differential_y',
            'numerator': num, 'denominator': den, 'numerator_particle': num, 'denominator_particle': den,
            'numerator_system': 'Au+Au', 'denominator_system': 'Au+Au',
            'numerator_record': f'{OUT_S}#{refs[0]}', 'denominator_record': f'{OUT_S}#{refs[1]}',
            'hepdata_corrections': {'centrality': 'the table states no class; the paper integrates over 0-20%'},
            'notes': {'value': f'{CITE}, Fig. 3. {num}/{den} per rapidity bin, Au+Au 200 GeV 0-20%, integrated over '
                               f'the measured pT (all pT here; the ratios are flat in pT). {FEED} Errors: stat and '
                               'sys.'}})
    for out, docs in ((OUT_S, sides), (OUT_R, ratios)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
