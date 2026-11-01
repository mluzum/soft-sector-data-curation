#!/usr/bin/env python3
"""Curate ALICE symmetric cumulants SC(m,n), NSC(m,n) in Pb-Pb: INSPIRE
1452590, 2654313, 2865181 (1621591 has its own script,
curate_alice_sc_pbpb_2760.py).

The three analyses share one setup (each paper, Sec. 2-3):
- charged tracks |eta| < 0.8, 0.2 < pT < 5 GeV/c; centrality from V0M;
- SC(m,n) from four- and two-particle cumulants of all particles in the
  window, no eta gap;
- NSC(m,n) = SC(m,n) / (<v_m^2><v_n^2>), the denominator from
  two-particle correlations with |Delta eta| > 1 (normalization_eta_gap);
- errors: statistical and systematic.

Papers and records:
- 1452590: J. Adam et al. (ALICE), "Correlated event-by-event fluctuations
  of flow harmonics in Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", PRL 117,
  182301 (2016), arXiv:1604.07663.  SC(3,2), SC(4,2) (Table 1) and NSC
  (Table 2) vs centrality; the same in 1% classes, 0-10% (Tables 3, 4).
- 2654313: S. Acharya et al. (ALICE), "Higher-order correlations between
  different moments of two flow amplitudes in Pb-Pb collisions at sqrt(sNN)
  = 5.02 TeV", PRC 108, 055203 (2023), arXiv:2303.13414.  SC(2,3), SC(2,4),
  SC(3,4) vs centrality (data1, 12, 22; not normalized).  The AC_{a,b}(m,n)
  and NAC of higher moments are not curated: no computation.
- 2865181: S. Acharya et al. (ALICE), "First measurement of symmetric
  cumulants of hexagonal flow harmonics in Pb-Pb collisions", PRC (2025),
  arXiv:2412.15873.  NSC(5,2), NSC(5,3), NSC(5,4), NSC(6,2), NSC(6,3) vs
  centrality at 5.02 TeV.

Run from the repository root:  python scripts/curate_alice_symmetric_cumulants.py
"""
import collections
import glob
import os
import re

import yaml

V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
KR = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.2, 'max': 5.0}}
PAPERS = {
    '1452590': {'cite': 'J. Adam et al. (ALICE), PRL 117, 182301 (2016), arXiv:1604.07663', 'energy': 2760.0,
                'tables': ['Table1.yaml', 'Table2.yaml', 'Table3.yaml', 'Table4.yaml'],
                'out': 'alice_flow_harmonic_correlations_correlated_event-by-event_fluctuations_of_flow_PbPb_2760gev.yaml'},
    '2654313': {'cite': 'S. Acharya et al. (ALICE), PRC 108, 055203 (2023), arXiv:2303.13414', 'energy': 5020.0,
                'tables': ['data1.yaml', 'data12.yaml', 'data22.yaml'],
                'out': 'alice_flow_harmonic_correlations_higher-order_correlations_between_different_moments_'
                       'PbPb_5020gev.yaml'},
    '2865181': {'cite': 'S. Acharya et al. (ALICE), arXiv:2412.15873', 'energy': 5020.0,
                'tables': ['hepdata_52.yaml', 'hepdata_53.yaml', 'hepdata_54.yaml', 'hepdata_62.yaml',
                           'hepdata_63.yaml'],
                'out': 'alice_flow_harmonic_correlations_first_measurement_of_symmetric_cumulants_PbPb_5020gev.yaml'},
}


def kind_of(header):
    """('SC' | 'NSC', m, n) of a column header."""
    h = header.replace(' ', '')
    m = re.fullmatch(r'(N?SC)\((\d),(\d)\)', h)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r'SC\((\d),(\d)\)/\(V\d\{2,\|DELTAETA\|>1\}\^2V\d\{2,\|DELTAETA\|>1\}\^2\)', h)
    if m:                                     # 1452590 Table 2, 4: NSC spelled out
        return 'NSC', int(m.group(1)), int(m.group(2))
    raise ValueError(header)


def records(inspire_id, cfg):
    raw = glob.glob(f'hepdata_raw/records/ins{inspire_id}/HEPData-ins{inspire_id}-v*-yaml')[0]
    version = int(re.search(r'-v(\d+)-yaml', raw).group(1))
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(raw, 'submission.yaml')))
           if x and 'data_file' in x}
    docs = []
    for hf in cfg['tables']:
        t = yaml.safe_load(open(os.path.join(raw, hf)))
        iv = t['independent_variables'][0]
        assert iv['header']['name'].lower() == 'centrality', (hf, iv['header'])
        desc = ' '.join(sub[hf]['description'].split())
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
            assert q.get('ETARAP', '-0.8 TO 0.8').replace('TO', '-').replace(' ', '') == '-0.8-0.8', (hf, q)
            assert q.get('PT', '0.2 TO 5.0').replace('TO', '-').replace(' ', '') == '0.2-5.0', (hf, q)
            kind, m, n = kind_of(dv['header']['name'])
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            classes = ['%g-%g%%' % (float(iv['values'][j]['low']), float(iv['values'][j]['high'])) for j in rows]
            norm = kind == 'NSC'
            d = {'observable_type': 'flow_harmonic_correlations', 'inspire_id': inspire_id, 'version': version,
                 'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f"{desc} | {dv['header']['name']}",
                 'collision_system': {'energy_gev': cfg['energy'], 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': classes, **V0M},
                 'kinematic_range': {k: dict(v) for k, v in KR.items()},
                 'observable': f'{kind}_{m}_{n}', 'correlation_type': 'symmetric_cumulant', 'harmonics': [m, n],
                 'normalized': norm}
            if norm:
                d['normalization_eta_gap'] = 1.0
            d.update({'value_type': f'{kind}({m},{n})', 'dimensionality': 'integrated', 'estimator': 'standard'})
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            what = (f'NSC({m},{n}) = SC({m},{n}) / (<v{m}^2><v{n}^2>), the denominator from two-particle '
                    'correlations with |Delta eta| > 1' if norm else
                    f'SC({m},{n}), four- and two-particle cumulants without gap')
            d['notes'] = {'value': f"{cfg['cite']}. {what}; charged, |eta| < 0.8, 0.2 < pT < 5 GeV/c. "
                                   'Errors: statistical and systematic.'}
            docs.append(d)
    return docs


def main():
    for inspire_id, cfg in PAPERS.items():
        docs = records(inspire_id, cfg)
        out = os.path.join('data/flow_harmonic_correlations', cfg['out'])
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records;', dict(collections.Counter(d['value_type'] for d in docs)))


if __name__ == '__main__':
    main()
