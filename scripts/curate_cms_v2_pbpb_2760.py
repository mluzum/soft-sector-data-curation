#!/usr/bin/env python3
"""Curate CMS v2 of charged particles in PbPb at 2.76 TeV, INSPIRE 1107659.

S. Chatrchyan et al. (CMS), "Measurement of the elliptic anisotropy of
charged particles produced in PbPb collisions at sqrt(sNN) = 2.76 TeV",
PRC 87, 014902 (2013), arXiv:1204.1409.

From the paper (Sec. 2):
- tracks |eta| < 2.4, 0.3 < pT < 20 GeV/c; centrality from the HF energy
  (3 < |eta| < 5), 12 classes 0-5% ... 70-80%;
- event plane: from the HF- (-5 < eta < -3) or HF+ (3 < eta < 5) -- the
  one farther in eta from the track (tracks at eta > 0 use HF-), minimum
  gap 3 units; three-subevent resolution;
- generating-function cumulants v2{2}, v2{4}: reference flow 0.3 < pT < 3
  GeV/c in |eta| < 0.8 (v2(pT)) or |eta| < 2.4 (v2(eta)); the particles of
  the differential bin are left out of the reference; Lee-Yang zeros;
- integrated v2: |eta| < 0.8, 0.3 < pT < 3 GeV/c (Sec. 5);
- errors: statistical and systematic.

Records:
- v2{EP}, v2{2}, v2{4} vs pT (|eta| < 0.8; Tables 1-34, 80-83 the 10%
  classes 0-40%), v2{2}, v2{4} vs pT in 20-60% (Table 88): pT points as
  tabulated (mean pT, 0.1 GeV/c windows);
- integrated v2{EP}, v2{2}, v2{4} vs centrality (Table 43; classes from the
  tabulated centres 2.5, 7.5, ... 45, 55, 65);
- v2{EP}(eta), 0.3 < pT < 3 GeV/c (Tables 44-55; 0.4-wide eta bins from
  -2.4 to 2.4, the tabulated mean eta inside each);
- v2{EP}(pT) in |eta| ranges 0-0.4 ... 2.0-2.4 (Tables 56-79), computed at
  eta > 0.
event_plane: the HF on the opposite side (flow_reference mirror, 3 <
|eta| < 5), computed with the scalar-product estimator; cumulants:
cumulant order 2 or 4 with the 0.3-3 GeV/c reference (flow_reference.pT),
reference eta = the record window.  Not curated: Lee-Yang zeros (no
computation), Tables 84-87 (the same values as Tables 9-12), the v2
extrapolated to pT = 0, v2/epsilon and v2 vs eta - y_beam.

Run from the repository root:  python scripts/curate_cms_v2_pbpb_2760.py
"""
import collections
import os
import re

import numpy as np
import yaml

# Forward-detector references (calorimeter/scintillator) see every particle: reference pT 0-100 GeV/c.
ALL_PT_REF = {'min': 0.0, 'max': 100.0}
FWD_NOTE = (' Reference pT: all particles (a calorimeter/scintillator reference has no pT selection; its E_T or '
            'hit weighting is not modelled).')
RAW = 'hepdata_raw/records/ins1107659/HEPData-ins1107659-v1-yaml'
CITE = 'S. Chatrchyan et al. (CMS), PRC 87, 014902 (2013), arXiv:1204.1409'
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.0, 'max': -3.0}, {'min': 3.0, 'max': 5.0}]}
EP_REF = {'detector': 'HF', 'mirror': True, 'eta': [{'min': 3.0, 'max': 5.0}],
          'notes': 'Event plane of the HF on the side opposite the track (HF- for eta > 0), gap >= 3.' + FWD_NOTE,
          'pT': dict(ALL_PT_REF)}
CUM_REF = {'pT': {'min': 0.3, 'max': 3.0},
           'notes': 'Reference flow 0.3 < pT < 3 GeV/c in the record eta window (generating-function cumulants).'}
CENTRES = {2.5: '0-5', 7.5: '5-10', 12.5: '10-15', 17.5: '15-20', 22.5: '20-25', 27.5: '25-30', 32.5: '30-35',
           37.5: '35-40', 45.0: '40-50', 55.0: '50-60', 65.0: '60-70'}
ETA_EDGES = [round(float(x), 1) for x in np.linspace(-2.4, 2.4, 13)]
WINDOW = 0.1
METHOD = {'EP': 'event plane', 'C2': 'v2{2}', 'C4': 'v2{4}'}


def rng(text):
    lo, hi = re.match(r'\s*(-?[0-9.]+)\s*-\s*(-?[0-9.]+)', str(text)).groups()
    return float(lo), float(hi)


def method_fields(m, dim):
    if m == 'EP':
        return {'method': 'event_plane', 'flow_reference': dict(EP_REF)}, 'HF event plane (opposite side), ' \
            'computed with the scalar-product estimator'
    order = 2 if m == 'C2' else 4
    return {'method': 'cumulant', 'cumulant_order': order, 'flow_reference': dict(CUM_REF)}, \
        f'{order}-particle generating-function cumulant, reference 0.3 < pT < 3 GeV/c'


def record(hf, i, m, classes, kr, dim, label, what, table_rows=None):
    fields, how = method_fields(m, dim)
    d = {'observable_type': 'flow_vn', 'inspire_id': '1107659', 'version': 1, 'hepdata_file': hf, 'dv_index': i,
         'hepdata_table_label': label,
         'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
         'centrality': {'classes': classes, **HF},
         'particle_id': {'species': 'charged'}, 'kinematic_range': kr, 'harmonic': 2,
         'value_type': f'V2({m})', 'dimensionality': dim, **fields,
         'notes': {'value': f'{CITE}. {what}; {how}. Errors: statistical and systematic.'}}
    if table_rows is not None:
        d['table_rows'] = table_rows
    return d


def pt_points(iv, rows):
    pts = [float(iv['values'][j]['value']) for j in rows]
    return {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4), 'points': pts,
            'point_window': WINDOW}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs, skipped = [], collections.Counter()
    for x in sub:
        hf = x['data_file']
        n = int(re.fullmatch(r'Table(\d+)\.yaml', hf).group(1))
        desc = ' '.join(x['description'].split())
        if (35 <= n <= 42) or (84 <= n <= 87) or n >= 89:
            skipped[n] += 1
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            m = re.fullmatch(r'V2\([CV]=(EP|C2|C4|LYZ)\)', dv['header']['name'].strip()).group(1)
            if m == 'LYZ':
                continue
            q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            if not rows:
                continue
            label = f'{desc} ({dv["header"]["name"].strip()})'
            if n == 43:
                cents = [float(v['value']) for v in iv['values']]
                classes = [f'{CENTRES[c]}%' for j, c in enumerate(cents) if j in rows]
                kr = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.3, 'max': 3.0}}
                docs.append(record(hf, i, m, classes, kr, 'integrated', label,
                                   'integrated v2, |eta| < 0.8, 0.3 < pT < 3 GeV/c, vs centrality (classes from the '
                                   'tabulated centres)', rows if rows != list(range(len(cents))) else None))
                continue
            assert rows == list(range(rows[0], rows[-1] + 1)), (hf, i)
            if 44 <= n <= 55:
                pts = [float(iv['values'][j]['value']) for j in rows]
                b0 = int(np.searchsorted(ETA_EDGES, pts[0])) - 1
                edges = ETA_EDGES[b0:b0 + len(pts) + 1]
                assert all(lo < p < hi for p, lo, hi in zip(pts, edges[:-1], edges[1:])), (hf, pts)
                plo, phi_ = rng(q['PT'])
                kr = {'eta': {'min': edges[0], 'max': edges[-1], 'bins': edges}, 'pT': {'min': plo, 'max': phi_}}
                cls = '%g-%g%%' % rng(q['CENTRALITY'])
                docs.append(record(hf, i, m, [cls], kr, 'differential_eta', label,
                                   f'v2(eta), {plo:g} < pT < {phi_:g} GeV/c, 0.4-wide eta bins'))
            elif 56 <= n <= 79:
                cls = re.search(r'centrality ([0-9.]+-[0-9.]+)%', desc).group(1) + '%'
                elo, ehi = rng(q['ABS(ETARAP)'])
                kr = {'eta': {'min': elo, 'max': ehi}, 'pT': pt_points(iv, rows)}
                docs.append(record(hf, i, m, [cls], kr, 'differential_pT', label,
                                   f'v2(pT), {elo:g} < |eta| < {ehi:g} (computed at eta > 0)'))
            else:
                cls = '%g-%g%%' % rng(q['CENTRALITY'])
                assert q['ABS(ETARAP)'] == '< 0.8', (hf, q)
                kr = {'eta': {'min': -0.8, 'max': 0.8}, 'pT': pt_points(iv, rows)}
                docs.append(record(hf, i, m, [cls], kr, 'differential_pT', label,
                                   f'v2(pT), |eta| < 0.8; pT points (mean pT, {WINDOW} GeV/c windows)'))
    out = 'data/flow_vn/cms_flow_vn_elliptic_anisotropy_of_charged_particles_PbPb_2760gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter((d['value_type'], d['dimensionality']) for d in docs)))
    print('tables not curated:', sorted(skipped))


if __name__ == '__main__':
    main()
