#!/usr/bin/env python3
"""Curate ATLAS v_n from multi-particle cumulants in Pb+Pb at 2.76 TeV, INSPIRE 1311487.

G. Aad et al. (ATLAS), "Measurement of flow harmonics with multi-particle
cumulants in Pb+Pb collisions at sqrt(sNN) = 2.76 TeV with the ATLAS
detector", EPJC 74 (2014) 3157, arXiv:1408.4342.

From the paper:
- charged particles |eta| < 2.5, 0.5 < pT < 20 GeV; centrality from the
  FCal Sigma E_T (3.2 < |eta| < 4.9), 0-80% in 14 classes (0-2% ... 60-80%);
- generating-function cumulants (no eta gap): reference cumulants c_n{2k}
  from all charged particles |eta| < 2.5, 0.5 < pT < 5 GeV; differential
  d_n{2k} in pT and eta bins, each particle correlated with 2k-1 reference
  particles: v_n{2} = d_n{2}/sqrt(c_n{2}), v_n{4} = -d_n{4}/(-c_n{4})^(3/4)
  (Eqs. 7-8); integrated values are the multiplicity-weighted averages of
  the differential ones over 0.5 < pT < 20 GeV and the eta range;
- v_n{EP}: the event plane of the full FCal (3.2 < |eta| < 4.9), the
  measurement of PRC 86, 014907 reanalysed with this paper's selections;
- the <N_part> of each class from a Glauber model;
- errors: statistical and systematic.

Records (one per table):
- v_n{2}(pT), v_n{4}(pT), v_n{2}(eta), v_n{4}(eta) and the integrated
  v_n{2k} vs <N_part> (|eta| < 2.5 and |eta| < 1): cumulant order 2k with
  the 0.5-5 GeV reference (flow_reference.pT); reference eta = the record
  window (|eta| < 2.5 in the paper; |eta| < 0.8 and < 1 records use their
  own window here);
- v2{6}, v2{8} vs <N_part>: integrated six- and eight-particle cumulants
  of the record window (the pipeline has no POI form for them);
- v_n{EP}(pT) and vs <N_part>: event_plane with the FCal (both sides),
  computed with the scalar-product estimator; v_n{EP}(eta): the FCal half
  opposite the bin (as in PRC 86, 014907), computed with the scalar-product
  estimator;
- <N_part> -> class: the 14 classes in order (396.0 for 0-2% to 22.6 for
  60-80%);
- harmonic and order from the column header (the descriptions of Tables
  167 and 192 say two-particle for v3{4} columns).
Not curated: v2{6}(pT, eta) and v2{8}(pT, eta) (no differential six- and
eight-particle computation), the ratios v2{6}/v2{4} and v2{8}/v2{4}, and
the flow-fluctuation estimates F(v_n).

Run from the repository root:  python scripts/curate_atlas_cumulants_pbpb_2760.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1311487/HEPData-ins1311487-v1-yaml'
CITE = 'G. Aad et al. (ATLAS), EPJC 74 (2014) 3157, arXiv:1408.4342'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
FCAL_BOTH = [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]
NPART = {396.0: '0-2%', 372.533: '2-5%', 330.3: '5-10%', 281.9: '10-15%', 239.5: '15-20%', 203.0: '20-25%',
         170.0: '25-30%', 142.0: '30-35%', 117.0: '35-40%', 95.0: '40-45%', 76.1: '45-50%', 59.9: '50-55%',
         46.1: '55-60%', 22.6: '60-80%'}
HARM = {'second': 2, 'triangular': 3, 'quadrangular': 4}
ORDER = {'two': 2, 'four': 4, 'six': 6, 'eight': 8}
DESC = re.compile(r'The (second|triangular|quadrangular) flow harmonic measured with the '
                  r'(?:(two|four|six|eight)-particle cumula\w+|(Event Plane) method) as a function of '
                  r'(transverse momentum|pseudorapidity|<Npart>)')


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
    docs, skipped = [], collections.Counter()
    for x in sub:
        desc = ' '.join(x['description'].split())
        hf = x['data_file']
        m = DESC.match(desc)
        if not m:
            skipped[desc.split(' as a function')[0][:60]] += 1
            continue
        n, order, ep, var = HARM[m.group(1)], ORDER.get(m.group(2)), bool(m.group(3)), m.group(4)
        if order in (6, 8) and var != '<Npart>':
            skipped[f'v{n}{{{order}}}({var})'] += 1
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert len(t['dependent_variables']) == 1, hf
        dv = t['dependent_variables'][0]
        q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
        iv = t['independent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert rows == list(range(len(iv['values']))), hf
        # the column header is authoritative: the descriptions of Tables 167
        # and 192 say two-particle for v3{4} columns (their <N_part> lists
        # match the four-particle tables)
        hm = re.fullmatch(r'v(\d)\{(\d|EP)\}', dv['header']['name'])
        if (int(hm.group(1)), hm.group(2)) != (n, 'EP' if ep else str(order)):
            print(f'  {hf}: description says v{n}{{{"EP" if ep else order}}}, column {dv["header"]["name"]}')
            n, ep = int(hm.group(1)), hm.group(2) == 'EP'
            order = None if ep else int(hm.group(2))
        label = dv['header']['name']
        d = {'observable_type': 'flow_vn', 'inspire_id': '1311487', 'version': 1, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': desc,
             'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
             'centrality': dict(FCAL), 'particle_id': {'species': 'charged'}, 'harmonic': n, 'value_type': label}
        if var == 'transverse momentum':
            elo, ehi = rng(q['ETARAP'])
            d['centrality'] = {'classes': ['%g-%g%%' % rng(q['CENTRALITY'])], **FCAL}
            kr = {'eta': {'min': elo, 'max': ehi}, 'pT': axis(iv['values'])}
            dim, what = 'differential_pT', f'{label}(pT), |eta| < {ehi:g}'
        elif var == 'pseudorapidity':
            plo, phi_ = rng(q['PT'])
            d['centrality'] = {'classes': ['%g-%g%%' % rng(q['CENTRALITY'])], **FCAL}
            kr = {'eta': axis(iv['values']), 'pT': {'min': plo, 'max': phi_}}
            dim, what = 'differential_eta', f'{label}(eta), {plo:g} < pT < {phi_:g} GeV'
        else:
            elo, ehi = rng(q['ETARAP'])
            plo, phi_ = rng(q['PT'])
            classes = [NPART[float(v['value'])] for v in iv['values']]
            d['centrality'] = {'classes': classes, **FCAL}
            kr = {'eta': {'min': elo, 'max': ehi}, 'pT': {'min': plo, 'max': phi_}}
            dim, what = 'integrated', (f'{label} vs <N_part>, |eta| < {ehi:g}, {plo:g} < pT < {phi_:g} GeV; classes '
                                       'from <N_part> (396.0 for 0-2% to 22.6 for 60-80%)')
        d.update({'kinematic_range': kr, 'dimensionality': dim})
        if ep:
            d['method'] = 'event_plane'
            if dim == 'differential_eta':
                d['flow_reference'] = {'detector': 'FCal', 'mirror': True, 'eta': [{'min': 3.2, 'max': 4.9}],
                                       'notes': 'Event plane of the FCal half opposite the track (PRC 86, 014907).'}
            else:
                d['flow_reference'] = {'detector': 'FCal', 'eta': FCAL_BOTH,
                                       'notes': 'Event plane of the full FCal, 3.2 < |eta| < 4.9.'}
            how = 'FCal event plane (PRC 86, 014907 reanalysed); computed with the scalar-product estimator'
        else:
            d.update({'method': 'cumulant', 'cumulant_order': order})
            if order in (2, 4):
                d['flow_reference'] = {'pT': {'min': 0.5, 'max': 5.0},
                                       'notes': 'Reference cumulants c_n{2k}: charged particles |eta| < 2.5, 0.5 < '
                                                'pT < 5 GeV (here in the record eta window).'}
                how = (f'{order}-particle generating-function cumulant, no eta gap, reference 0.5 < pT < 5 GeV'
                       + ('' if dim != 'integrated' else '; the paper integrates v_n(pT, eta) weighted by the '
                          'multiplicity'))
            else:
                how = (f'{order}-particle generating-function cumulant; computed as the integrated v2{{{order}}} of '
                       'the record window')
        d['notes'] = {'value': f'{CITE}. {what}; {how}. Errors: statistical and systematic.'}
        docs.append(d)
    out = 'data/flow_vn/atlas_flow_vn_flow_harmonics_with_multi-particle_cumulants_PbPb_2760gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;',
          dict(collections.Counter((d['value_type'], d['dimensionality']) for d in docs)))
    print('not curated:', dict(skipped))


if __name__ == '__main__':
    main()
