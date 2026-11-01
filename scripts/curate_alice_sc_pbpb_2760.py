#!/usr/bin/env python3
"""Curate ALICE symmetric cumulants in Pb-Pb at 2.76 TeV, INSPIRE 1621591.

S. Acharya et al. (ALICE), "Systematic studies of correlations between
different order flow harmonics in Pb-Pb collisions at sqrt(sNN) = 2.76
TeV", PRC 97, 024906 (2018), arXiv:1709.01127.

From the paper (Secs. 2-5):
- charged tracks |eta| < 0.8, 0.2 < pT < 5 GeV/c; centrality from V0M;
- SC(m,n) = <<cos(m phi1 + n phi2 - m phi3 - n phi4)>> - <<cos m(phi1 -
  phi2)>> <<cos n(phi1 - phi2)>>, all particles of the window, no gap;
- NSC(m,n) = SC(m,n) / (<v_m^2> <v_n^2>), the denominator from
  two-particle correlations with |Delta eta| > 1 (v_n{2, |Delta eta| > 1},
  Fig. 3);
- pT dependence: the lower pT cut pT_min = 0.2 ... 1.5 GeV/c, upper cut 5
  GeV/c, for SC, NSC and v_n alike;
- errors: statistical and systematic.

Records:
- SC and NSC of (5,2), (5,3), (4,3) vs centrality (Tables 1-6);
- SC(3,2), SC(4,2), NSC(3,2), NSC(4,2) vs pT_min, per class (Tables
  7-30): one record per pT_min point (an integrated record with pT
  [pT_min, 5] and table_rows), since each point is its own acceptance;
- v_2, v_3, v_4 {2, |Delta eta| > 1} vs pT_min (Tables 31-48), same
  layout, and v_5 vs centrality (Table 49): two subevents -0.8 < eta <
  -0.5 and 0.5 < eta < 0.8 (ALICE's |Delta eta| > 1 within |eta| < 0.8).
HEPData describes Tables 19-30 as "centrality dependence"; they are the
pT_min dependence of NSC(3,2) and NSC(4,2) (their axis and the paper's
Figs. 2b, 2d).

Run from the repository root:  python scripts/curate_alice_sc_pbpb_2760.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1621591/HEPData-ins1621591-v2-yaml'
CITE = 'S. Acharya et al. (ALICE), PRC 97, 024906 (2018), arXiv:1709.01127'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
REACTION = 'PB PB --> CHARGED X'
PT_MAX = 5.0
GAP = 1.0


def subevents(pt_min):
    return {k: {'kinematic_range': {'eta': {'min': lo, 'max': hi}, 'pT': {'min': pt_min, 'max': PT_MAX}}}
            for k, (lo, hi) in (('A', (-0.8, -0.5)), ('B', (0.5, 0.8)))}


def base(otype, hf, label, classes, pt_min):
    return {'observable_type': otype, 'inspire_id': '1621591', 'version': 2, 'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': label,
            'collision_system': {'energy_gev': 2760.0, 'reaction': REACTION},
            'centrality': {'classes': classes, **V0M},
            'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': pt_min, 'max': PT_MAX}}}


def sc_record(hf, label, classes, pt_min, kind, m, n, where):
    d = base('flow_harmonic_correlations', hf, label, classes, pt_min)
    norm = kind == 'NSC'
    d.update({'observable': f'{kind}_{m}_{n}', 'correlation_type': 'symmetric_cumulant', 'harmonics': [m, n],
              'normalized': norm, 'value_type': f'{kind}({m},{n})', 'dimensionality': 'integrated',
              'estimator': 'standard'})
    if norm:
        d['normalization_eta_gap'] = GAP
    what = (f'NSC({m},{n}) = SC({m},{n}) / (<v{m}^2><v{n}^2>), the denominator from two-particle correlations '
            f'with |Delta eta| > 1' if norm else f'SC({m},{n}), four- and two-particle cumulants without gap')
    d['notes'] = {'value': f'{CITE}. {what}; charged, |eta| < 0.8, {pt_min:g} < pT < 5 GeV/c; {where}. '
                           'Errors: statistical and systematic.'}
    return d


def vn_record(hf, label, classes, pt_min, n, where):
    d = base('flow_vn', hf, label, classes, pt_min)
    d.update({'particle_id': {'species': 'charged'}, 'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2,
              'particles': 'AB', 'subevents': subevents(pt_min), 'eta_gap': GAP,
              'value_type': f'v_{n}', 'dimensionality': 'integrated',
              'notes': {'value': f'{CITE}. v{n}{{2, |Delta eta| > 1}}: two subevents -0.8 < eta < -0.5 and 0.5 < eta < '
                                 f'0.8; charged, {pt_min:g} < pT < 5 GeV/c; {where} (the NSC denominators). Errors: '
                                 'statistical and systematic.'}})
    return d


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = {'sc': [], 'vn': []}
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        assert len(t['dependent_variables']) == 1, hf
        dv = t['dependent_variables'][0]
        h = dv['header']['name']
        q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
        assert q['ETARAP'] == '-0.8 TO 0.8' and q['RE'] == REACTION, (hf, q)
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert rows == list(range(len(iv['values']))), hf
        desc = ' '.join(x['description'].split())
        sc = re.fullmatch(r'(N?SC)\((\d),(\d)\)', h)
        vn = re.fullmatch(r'\$v_(\d)\$', h)
        if iv['header']['name'] == 'CENTRALITY':
            assert q['PT'] == '0.2 TO 5.0', (hf, q)
            classes = ['%g-%g%%' % (float(v['low']), float(v['high'])) for v in iv['values']]
            if sc:
                docs['sc'].append(sc_record(hf, desc, classes, 0.2, sc.group(1), int(sc.group(2)),
                                            int(sc.group(3)), 'vs centrality'))
            else:
                docs['vn'].append(vn_record(hf, desc, classes, 0.2, int(vn.group(1)), 'vs centrality'))
            continue
        assert iv['header']['name'] == '$p_{T,min}$', hf
        cls = q['CENTRALITY']
        assert re.fullmatch(r'\d+-\d+%', cls), (hf, cls)
        desc = desc.replace('Centrality dependence', 'Transverse momentum dependence')
        for j, v in enumerate(iv['values']):
            pt_min = float(v['value'] if 'value' in v else v['low'])
            label = f'{desc} {cls}, pT_min = {pt_min:g} GeV/c'
            where = f'{cls}, the pT_min = {pt_min:g} GeV/c point of the pT_min dependence'
            if sc:
                d = sc_record(hf, label, [cls], pt_min, sc.group(1), int(sc.group(2)), int(sc.group(3)), where)
                docs['sc'].append(d)
            else:
                d = vn_record(hf, label, [cls], pt_min, int(vn.group(1)), where)
                docs['vn'].append(d)
            d['table_rows'] = [j]
    outs = {'sc': 'data/flow_harmonic_correlations/'
                  'alice_flow_harmonic_correlations_systematic_studies_of_correlations_between_PbPb_2760gev.yaml',
            'vn': 'data/flow_vn/alice_flow_vn_systematic_studies_of_correlations_between_PbPb_2760gev.yaml'}
    for k, out in outs.items():
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs[k], fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs[k])} records;',
              dict(collections.Counter(d['value_type'] for d in docs[k])))


if __name__ == '__main__':
    main()
