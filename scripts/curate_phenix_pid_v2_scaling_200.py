#!/usr/bin/env python3
"""Curate PHENIX v2 of pi, K, p in Au+Au at 200 GeV, INSPIRE 1093596.

A. Adare et al. (PHENIX), "Deviation from quark-number scaling of the
anisotropy parameter v2 of pions, kaons, and protons in Au+Au collisions
at sqrt(sNN) = 200 GeV", PRC 85, 064914 (2012), arXiv:1203.2644.

From the paper (Sec. II):
- pi+-, K+-, p + pbar (charges combined), |eta| < 0.35, PID with TOF
  (TOFw, ACC); 2007 data;
- event plane from the RxNP, inner 1.0 < |eta| < 1.5 and outer 1.5 <
  |eta| < 2.8, north and south (scintillators: every particle);
- centrality classes 0-10 ... 40-60% and wider ones (0-20, 20-60, 10-40%);
  the centrality detector is not restated here: the BBC charge (3.0 <
  |eta| < 3.9), PHENIX's estimator for this run, is assumed;
- errors: statistical and systematic.

Records: v2(pT) of each species and class (the second column of each
table).  The first column, <pT> of the bin, is not a measurement: the 21
old records were those columns, filed as mean_pT with species 'p' for all.

Run from the repository root:  python scripts/curate_phenix_pid_v2_scaling_200.py
"""
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1093596/HEPData-ins1093596-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRC 85, 064914 (2012), arXiv:1203.2644'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge assumed (PHENIX standard; not restated in the paper).'}
RXNP = {'detector': 'RxNP', 'eta': [{'min': -2.8, 'max': -1.0}, {'min': 1.0, 'max': 2.8}],
        'pT': {'min': 0.0, 'max': 100.0},
        'notes': 'RxNP event plane (inner 1.0-1.5, outer 1.5-2.8 in |eta|, north and south). Reference pT: all '
                 'particles (scintillator; its weighting not modelled).'}
SPECIES = {'pi': 'pi+ + pi-', 'K': 'K+ + K-', 'p': 'p + pbar'}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        i, dv = 1, t['dependent_variables'][1]
        assert dv['header']['name'] == '$v_2$', (hf, dv['header'])
        q = [str(qq.get('value')) for qq in dv['qualifiers'] if qq['name'] not in ('RE', 'SQRT(S)/NUCLEON')]
        cls = f'{q[0]}%'
        sp = SPECIES[re.search(r'\\?(pi|K|p)\b', q[1].replace('$', ' ').replace('^', ' ')).group(1)]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        if rows == list(range(rows[0], rows[-1] + 1)):
            pt = {'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [pairs[0][0]] + [p[1] for p in pairs]}
        else:
            pt = {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}
        d = {'observable_type': 'flow_vn', 'inspire_id': '1093596', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': i, 'hepdata_table_label': f"{' '.join(x['description'].split())[:140]} | {sp} {cls}",
             'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
             'centrality': {'classes': [cls], **BBC}, 'particle_id': {'species': sp},
             'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35}, 'pT': pt},
             'harmonic': 2, 'method': 'event_plane', 'value_type': 'v_2', 'dimensionality': 'differential_pT',
             'flow_reference': dict(RXNP),
             'notes': {'value': f'{CITE}. v2(pT) of {sp}, {cls}, |eta| < 0.35, RxNP event plane; computed with the '
                                'scalar-product estimator. Errors: statistical and systematic.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        docs.append(d)
    old = 'data/mean_pT/phenix_mean_pT_deviation_from_quark-number_scaling_of_AuAu_200gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/flow_vn/phenix_flow_vn_pid_v2_AuAu_200gev_ins1093596.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(
        (d['particle_id']['species'], d['centrality']['classes'][0]) for d in docs)))


if __name__ == '__main__':
    main()
