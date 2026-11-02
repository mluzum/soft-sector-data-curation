#!/usr/bin/env python3
"""Curate STAR v2-v4 in Cu+Au at 200 GeV, INSPIRE 1641113.

L. Adamczyk et al. (STAR), "Azimuthal anisotropy in Cu+Au collisions at
sqrt(sNN) = 200 GeV", PRC 98, 014915 (2018), arXiv:1712.01332.

From the paper (Secs. II-III):
- tracks 0.15 < pT < 5 GeV/c, |eta| < 1; pi, K, p (charges combined) with
  TPC dE/dx and TOF; centrality from RefMult (|eta| < 0.5);
- Cu (projectile) flies toward +eta (west), Au toward -eta: frame nn_cms,
  forward_beam Cu;
- v_n{Psi_n}: TPC eta subevents -1 < eta < -0.4, |eta| < 0.2, 0.4 < eta <
  1 (event-plane tracks pT < 2 GeV/c, pT-weighted), "Delta eta > 0.4"
  between the particles and the plane: the subevent on the opposite side
  (mirrored reference 0.4 < |eta| < 1; unit weights here); resolution
  from three subevents;
- v1 relative to the ZDC-SMD spectator planes; <p_x> shifts;
- errors: statistical and systematic.

Records (public): charged v2, v3, v4 (pT) of six classes (Fig. 13),
identified v2, v3 (pT) of four or five classes (Fig. 17) and v2-v4 of
0-40% (Fig. 18), pT points (0.2 GeV/c windows).  Not curated: v1 and
<p_x> (spectator planes, three-point correlator: refused), their slopes,
the scalar-product comparison of Fig. 15 (its four columns repeat their
labels), v_n vs N_part (Fig. 14).  The 16 old records filed <p_x> and
slopes as v2, called the pi and K tables p or charged, and had harmonic 2
throughout.

Run from the repository root:  python scripts/curate_star_flow_cuau200.py
"""
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1641113/HEPData-ins1641113-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'L. Adamczyk et al. (STAR), PRC 98, 014915 (2018), arXiv:1712.01332'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
REF = {'detector': 'TPC', 'mirror': True, 'eta': [{'min': 0.4, 'max': 1.0}], 'pT': {'min': 0.15, 'max': 2.0},
       'notes': 'TPC eta subevent on the opposite side (-1 < eta < -0.4 or 0.4 < eta < 1, Delta eta > 0.4); '
                'event-plane tracks pT < 2 GeV/c, pT-weighted in the paper (unit weights here).'}
SYSTEM = {'energy_gev': 200.0, 'frame': 'nn_cms', 'rapidity_shift': 0.0, 'forward_beam': 'Cu'}
SPECIES = {'pi': 'pi+ + pi-', 'K': 'K+ + K-', 'p': 'p + pbar'}
TABLES = ['fig13a', 'fig13b', 'fig13a2', 'fig17a', 'fig17a2', 'fig17b', 'fig17b2', 'fig17c',
          'fig18a', 'fig18b', 'fig18b2', 'fig18c', 'fig18d', 'fig18d2', 'fig18f', 'fig18f2']
WINDOW = 0.2


def main():
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    docs = []
    for stem in TABLES:
        hf = f'{stem}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            label = [str(q.get('value')) for q in dv['qualifiers'] if q['name'] not in ('RE', 'SQRT(S)/NUCLEON')][0]
            text = re.sub(r'[${}\\]', '', label)
            n = int(re.search(r'v_(\d)', text).group(1))
            cls = re.search(r'(\d+)%?-(\d+)%', text)
            cls = f'{cls.group(1)}-{cls.group(2)}%'
            m = re.search(r'\b(pi|K|p)\b', text.replace('v_', ' '))
            sp = SPECIES[m.group(1)] if stem.startswith(('fig17', 'fig18')) else 'charged'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pts = [float(iv['values'][j]['value']) for j in rows]
            d = {'observable_type': 'flow_vn', 'inspire_id': '1641113', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f"{' '.join(sub[hf]['description'].split())[:140]} | {label}",
                 'collision_system': dict(SYSTEM, reaction='CU AU --> CHARGED X' if sp == 'charged'
                                          else 'CU AU --> X'),
                 'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': sp},
                 'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                     'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                            'points': pts, 'point_window': WINDOW}},
                 'harmonic': n, 'method': 'event_plane', 'value_type': f'v_{n}{{Psi_{n}}}',
                 'dimensionality': 'differential_pT', 'flow_reference': dict(REF),
                 'notes': {'value': f'{CITE}. v{n}{{Psi_{n}}}(pT) of {sp}, {cls}, |eta| < 1, TPC eta-subevent plane '
                                    'from the opposite side; computed with the scalar-product estimator; pT points '
                                    f'({WINDOW} GeV/c windows). Errors: statistical and systematic.'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            docs.append(d)
    old = 'data/flow_vn/star_flow_vn_azimuthal_anisotropy_200gev_1641113.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/flow_vn/star_flow_vn_v2_v4_CuAu_200gev_ins1641113.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(
        (d['harmonic'], d['particle_id']['species']) for d in docs)))


if __name__ == '__main__':
    main()
