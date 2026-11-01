#!/usr/bin/env python3
"""Curate STAR symmetric cumulants in Au+Au at 200 and 39 GeV, INSPIRE 1662057.

J. Adam et al. (STAR), "Correlation measurements between flow harmonics in
Au+Au collisions at RHIC", PLB 783, 459 (2018), arXiv:1803.03876.

From the paper (Sec. 2):
- charged tracks |eta| < 1.0, 0.2 < pT < 2 GeV/c; centrality from the TPC
  multiplicity in |eta| < 0.5 (RefMult);
- SC(m,n) from the Q-cumulant four- and two-particle correlations of all
  tracks (no gap); NSC(m,n) = SC(m,n) / (<v_m^2><v_n^2>), the denominator
  from two-particle correlations with |Delta eta| > 1 between sub-events A
  and B (Eq. 11);
- SC measured in multiplicity bins of one unit and combined into 10%
  classes (the centrality-bin-width correction), at both energies: the
  "Narrow Mult. Bins" results;
- 10% classes, given in HEPData only by <N_part> (0-10% ... 50-60% at 200
  GeV, 0-10% ... 40-50% at 39 GeV; SC(2,4) at 39 GeV in 0-20%, 20-40%);
- errors: statistical and systematic.

Records: SC(2,3), SC(2,4) (Tables 1-3) and NSC(2,3), NSC(2,4) (Tables 4-6)
with `centrality.combined_from: unit_multiplicity`.  The pipeline computes
over the whole class, so the dispatcher refuses them: they are written to
OUT (a staging collection) with a staging_reason, not to data/.

Run from the repository root:  python scripts/curate_star_sc_auau.py OUT.yaml
"""
import argparse
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1662057/HEPData-ins1662057-v1-yaml'
CITE = 'J. Adam et al. (STAR), PLB 783, 459 (2018), arXiv:1803.03876'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'combined_from': 'unit_multiplicity'}
KR = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 2.0}}
# <N_part> of the classes (STAR Glauber), to check the class assignment.
NPART = {200.0: {'0-10%': 325.5, '10-20%': 237.3, '20-30%': 169.0, '30-40%': 116.7, '40-50%': 76.9,
                 '50-60%': 47.8},
         39.0: {'0-10%': 317.3, '10-20%': 229.8, '20-30%': 162.4, '30-40%': 111.4, '40-50%': 73.5,
                '0-20%': 273.2, '20-40%': 137.0}}
STAGING = ('STAR combines SC measured in unit RefMult bins into 10% classes (the centrality-bin-width '
           'correction, centrality.combined_from); the pipeline computes over the whole class.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out', help='output YAML (a staging collection: data/flow_harmonic_correlations/<file>)')
    out = ap.parse_args().out
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        npart = [float(v['value']) for v in t['independent_variables'][0]['values']]
        desc = ' '.join(x['description'].split())
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
            energy = float(q['SQRT(sNN)'])
            classes = []
            for npt in npart:
                c = [k for k, v in NPART[energy].items() if abs(v - npt) < 0.5]
                assert len(c) == 1, (hf, npt)
                classes.append(c[0])
            m = re.fullmatch(r'(Normalized )?Symmetric Cumulant (N?SC)\((\d),(\d)\)', dv['header']['name'])
            kind, a, b = m.group(2), int(m.group(3)), int(m.group(4))
            norm = kind == 'NSC'
            d = {'observable_type': 'flow_harmonic_correlations', 'inspire_id': '1662057', 'version': 1,
                 'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f"{desc} | {dv['header']['name']}",
                 'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
                 'centrality': {'classes': classes, **REFMULT},
                 'kinematic_range': {k: dict(v) for k, v in KR.items()},
                 'observable': f'{kind}_{a}_{b}', 'correlation_type': 'symmetric_cumulant', 'harmonics': [a, b],
                 'normalized': norm}
            if norm:
                d['normalization_eta_gap'] = 1.0
            what = (f'NSC({a},{b}) = SC({a},{b}) / (<v{a}^2><v{b}^2>), the denominator from two-particle '
                    'correlations with |Delta eta| > 1' if norm else
                    f'SC({a},{b}), Q-cumulant four- and two-particle correlations without gap')
            d.update({'value_type': f'{kind}({a},{b})', 'dimensionality': 'integrated', 'estimator': 'standard',
                      'notes': {'value': f'{CITE}. {what}; charged, |eta| < 1, 0.2 < pT < 2 GeV/c; measured in '
                                         'unit RefMult bins and combined into the classes (given by <N_part>). '
                                         'Errors: statistical and systematic.'},
                      'staging_reason': STAGING})
            docs.append(d)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(
        (d['value_type'], d['collision_system']['energy_gev']) for d in docs)))


if __name__ == '__main__':
    main()
