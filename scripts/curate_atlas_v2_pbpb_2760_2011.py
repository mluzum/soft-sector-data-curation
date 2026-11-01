#!/usr/bin/env python3
"""Curate ATLAS v2(pT) and v2(eta) in Pb+Pb at 2.76 TeV, INSPIRE 925720.

G. Aad et al. (ATLAS), "Measurement of the pseudorapidity and transverse
momentum dependence of the elliptic flow of charged particles in lead-lead
collisions at sqrt(sNN) = 2.76 TeV with the ATLAS detector", PLB 707 (2012)
330, arXiv:1108.6018.

From the paper:
- tracks |eta| < 2.5; centrality from the FCal Sigma E_T (3.2 < |eta| <
  4.9), 0-80% in 10% classes;
- event plane from the transverse energy of the FCal first sampling layer,
  taken in the eta hemisphere opposite each track (|Delta eta| > 3.2),
  resolution-corrected with the subevent technique (Eq. 4);
- v2(pT) in |eta| < 1, 1 < |eta| < 2, 2 < |eta| < 2.5 and v2(eta) in five pT
  ranges, per class;
- errors: statistical and systematic.

Records: one per table (pT range of Table 59 from its description: its
qualifier reads '9.0 TO GeV').  event_plane with the FCal half opposite the track
(flow_reference mirror, 3.2 < |eta| < 4.9; charged particles with unit
weights here, E_T weights in the paper), computed with the scalar-product
estimator; the |eta| ranges of v2(pT) are computed at eta > 0.

Run from the repository root:  python scripts/curate_atlas_v2_pbpb_2760_2011.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins925720/HEPData-ins925720-v1-yaml'
CITE = 'G. Aad et al. (ATLAS), PLB 707 (2012) 330, arXiv:1108.6018'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
REF = {'detector': 'FCal', 'mirror': True, 'eta': [{'min': 3.2, 'max': 4.9}],
       'notes': 'Event plane of the FCal (first sampling layer, E_T weighted) in the eta hemisphere opposite the '
                'track; charged particles with unit weights here.'}


def rng(text):
    lo, hi = re.match(r'\s*(-?[0-9.]+)\s*-\s*(-?[0-9.]+)', str(text)).groups()
    return float(lo), float(hi)


def axis(values):
    pairs = [[float(v['low']), float(v['high'])] for v in values]
    ax = {'min': pairs[0][0], 'max': pairs[-1][1]}
    if all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs, pairs[1:])):
        ax['bins'] = [pairs[0][0]] + [p[1] for p in pairs]
    else:
        ax['bin_ranges'] = pairs
    return ax


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        desc = ' '.join(x['description'].split())
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert len(t['dependent_variables']) == 1, hf
        dv = t['dependent_variables'][0]
        q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
        iv = t['independent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert rows == list(range(rows[0], rows[-1] + 1)), hf
        vals = [iv['values'][j] for j in rows]
        cls = '%g-%g%%' % rng(q['CENTRALITY'])
        if iv['header']['name'] == 'PT':
            elo, ehi = rng(q['ABS(ETARAP)'])
            kr = {'eta': {'min': elo, 'max': ehi}, 'pT': axis(vals)}
            dim, what = 'differential_pT', f'v2(pT), {elo:g} < |eta| < {ehi:g} (computed at eta > 0)'
        else:
            assert iv['header']['name'] == 'ETARAP', hf
            # Table 59's qualifier reads '9.0 TO GeV': the description has the range
            m = re.search(r'and ([0-9.]+) ?< ?pT ?< ?([0-9.]+) GeV', desc)
            plo, phi_ = float(m.group(1)), float(m.group(2))
            if re.match(r'\s*[0-9.]+\s*-\s*[0-9.]+', str(q['PT'])):
                assert rng(q['PT']) == (plo, phi_), (hf, q['PT'], desc)
            kr = {'eta': axis(vals), 'pT': {'min': plo, 'max': phi_}}
            dim, what = 'differential_eta', f'v2(eta), {plo:g} < pT < {phi_:g} GeV'
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '925720', 'version': 1, 'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': desc,
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
            'centrality': {'classes': [cls], **FCAL},
            'particle_id': {'species': 'charged'},
            'kinematic_range': kr,
            'harmonic': 2, 'method': 'event_plane', 'value_type': 'V2 (ELLIPTIC-FLOW)', 'dimensionality': dim,
            'flow_reference': dict(REF),
            'notes': {'value': (f'{CITE}. {what}; event plane of the FCal in the hemisphere opposite the track '
                                '(|Delta eta| > 3.2), computed with the scalar-product estimator. Errors: statistical '
                                'and systematic.')},
        })
    out = 'data/flow_vn/atlas_flow_vn_pseudorapidity_and_transverse_momentum_dependence_PbPb_2760gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(d['dimensionality'] for d in docs)))


if __name__ == '__main__':
    main()
