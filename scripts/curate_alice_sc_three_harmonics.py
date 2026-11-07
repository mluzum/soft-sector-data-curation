#!/usr/bin/env python3
"""Curate ALICE symmetric cumulants of three flow amplitudes, INSPIRE 1839720.

S. Acharya et al. (ALICE), "Event-by-event multiharmonic correlations of
different flow amplitudes in Pb-Pb collisions at sqrt(sNN) = 2.76 TeV",
PRL 127 (2021) 092302, arXiv:2101.02579.

From the paper: SC(k,l,m) = <v_k^2 v_l^2 v_m^2> - <v_k^2 v_l^2><v_m^2> -
<v_k^2 v_m^2><v_l^2> - <v_l^2 v_m^2><v_k^2> + 2<v_k^2><v_l^2><v_m^2> (Eq. 2),
estimated by the six-, four- and two-particle correlators of Eq. 5 over
all distinct particles of one window (generic framework, no eta gap);
NSC(k,l,m) = SC(k,l,m) / (<v_k^2><v_l^2><v_m^2>) (Eq. 3); TPC-only tracks
|eta| < 0.8, 0.2 < pT < 5.0 GeV/c; centrality from the SPD (the default;
the V0 is the systematic check), curated as CL1 (outer-layer clusters,
|eta| < 1.4), 0-50%.

Records (public): SC(2,3,4), SC(2,3,5), SC(2,4,6), SC(3,4,5), NSC(2,3,4),
NSC(2,3,5), correlation_type symmetric_cumulant with three harmonics
(key sc_mn; computed where the container stores harmonics up to k+l+m,
the correlators' coincidence terms).  Nothing is staged; the six
auto-staged records are replaced.

Run from the repository root:  python scripts/curate_alice_sc_three_harmonics.py
"""
import glob
import os
import re

import yaml

IID = '1839720'
RAW = glob.glob(f'hepdata_raw/records/ins{IID}/HEPData-ins{IID}-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PRL 127 (2021) 092302, arXiv:2101.02579'
OUT = f'data/flow_harmonic_correlations/alice_flow_harmonic_correlations_sc_three_harmonics_PbPb_2760gev_ins{IID}.yaml'
CL1 = {'estimator': {'name': 'CL1', 'observable': 'Nch', 'detector': 'SPD'}, 'eta': [{'min': -1.4, 'max': 1.4}],
       'notes': 'the SPD (default), curated as CL1: clusters in the outer SPD layer, |eta_lab| < 1.4.'}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv, iv = t['dependent_variables'][0], t['independent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        assert q['ETARAP'] == '-0.8 TO 0.8' and q['PT'] == '0.2 TO 5.0' and q['SQRT(S)/NUCLEON'] == '2760', (hf, q)
        assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
        name = dv['header']['name']
        m = re.fullmatch(r'(N?SC)\((\d),(\d),(\d)\)', name)
        kind, harms = m.group(1), [int(m.group(i)) for i in (2, 3, 4)]
        classes = [f"{float(b['low']):g}-{float(b['high']):g}%" for b in iv['values']]
        what = (f'NSC{tuple(harms)} = SC / (<v_{harms[0]}^2><v_{harms[1]}^2><v_{harms[2]}^2>) (Eq. 3)'
                if kind == 'NSC' else f'SC{tuple(harms)}, the third cumulant of v_k^2, v_l^2, v_m^2 (Eq. 2)')
        docs.append({
            'observable_type': 'flow_harmonic_correlations', 'inspire_id': IID, 'version': VERSION,
            'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': f"{x['name']}: {' '.join(x['description'].split())}",
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
            'centrality': {'classes': classes, **CL1}, 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.2, 'max': 5.0}},
            'observable': f"{kind}_{''.join(map(str, harms))}", 'correlation_type': 'symmetric_cumulant',
            'harmonics': harms, 'normalized': kind == 'NSC', 'dimensionality': 'integrated', 'value_type': name,
            'notes': {'value': f'{CITE}. {what}, from the six-, four- and two-particle correlators of Eq. 5 over all '
                               'distinct charged particles |eta| < 0.8, 0.2 < pT < 5.0 GeV/c (no eta gap), Pb-Pb '
                               '2.76 TeV. Errors: stat and sys.'}})
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records;', [d['value_type'] for d in docs])


if __name__ == '__main__':
    main()
