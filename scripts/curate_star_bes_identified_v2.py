#!/usr/bin/env python3
"""Curate STAR identified-hadron v2(pT) at 7.7-62.4 GeV, INSPIRE 1210464.

L. Adamczyk et al. (STAR), "Elliptic flow of identified hadrons in Au+Au
collisions at sqrt(sNN) = 7.7-62.4 GeV", PRC 88, 014902 (2013),
arXiv:1301.2348.

From the paper:
- v2 from the eta-sub event plane: the second-harmonic plane of TPC tracks
  (0.15 < pT < 5 GeV/c, |eta| < 1, pT weighted) in the hemisphere opposite
  the particle, with an extra gap of +-0.05; resolution-corrected;
- pi, K, p by dE/dx and time of flight in |eta| < 1; K0_S, Lambda, Xi,
  Omega, phi by invariant mass;
- centrality from the uncorrected multiplicity in |eta| < 0.5 (RefMult);
  0-80% for all species, 0-10, 10-40, 40-80% for p and pbar;
- errors: statistical, systematic (asymmetric) and a global systematic
  (non-flow).

Records: one per particle species, energy and centrality, from the table
with the most pT points; pT points as tabulated (mean pT of each bin,
0.1 GeV/c windows).  method event_plane with the eta-sub subevents
(particles AB: -1 < eta < -0.05 and 0.05 < eta < 1, reference tracks 0.15
< pT < 5 GeV/c, unit weights here), computed as the two-subevent v2 of
the particle against the charged reference in the opposite half.  Not
curated: the
same data re-plotted for selected particles (truncated pT ranges, Figs. 13,
19), v2 vs m_T - m, particle-antiparticle differences, the energy
dependence of those differences, and the (+,-) event-plane variants.

Run from the repository root:  python scripts/curate_star_bes_identified_v2.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1210464/HEPData-ins1210464-v1-yaml'
CITE = 'L. Adamczyk et al. (STAR), PRC 88, 014902 (2013), arXiv:1301.2348'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
# qualifier text, stripped of $ { } \\ ^ and spaces -> species
SPECIES = {'pi+': 'pi+', 'pi-': 'pi-', 'K+': 'K+', 'K-': 'K-', 'K+-': 'K+ + K-', 'K0_s': 'K0_S', 'p': 'p',
           'overlinep': 'pbar', 'barp': 'pbar', 'Lambda': 'Lambda', 'barLambda': 'Lambdabar', 'Xi-': 'Xi-',
           'barXi+': 'Xibar+', 'Omega-': 'Omega-', 'barOmega+': 'Omegabar+', 'phi': 'phi'}


def species(text):
    return SPECIES.get(re.sub(r'[${}\\^ ]', '', str(text)))


SKIP = re.compile(r'selected (anti-)?particles|reduced transverse mass|difference|\(\+,-\) EP', re.I)
WINDOW = 0.1


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    best, unknown = {}, collections.Counter()
    for x in sub:
        desc = ' '.join(x['description'].split())
        if SKIP.search(desc):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, x['data_file'])))
        if 'p_T' not in t['independent_variables'][0]['header']['name']:
            continue
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            if 'Delta' in dv['header']['name']:
                continue
            q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
            sp = species(q.get('Au+Au'))
            if sp is None:
                unknown[str(q.get('Au+Au'))] += 1
                continue
            nonblank = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            # HEPData copy errors: a 'value' equal to its own pT, without errors
            # (Data43,a: the Lambdabar pT points sit in the Lambda column)
            rows = [j for j in nonblank if dv['values'][j].get('errors')
                    or abs(float(dv['values'][j]['value']) - pts[j]) > 1e-9]
            key = (float(q['SQRT(S)']), sp, str(q['Centrality']))
            cand = (len(rows), x['data_file'], i, [pts[j] for j in rows], desc, rows if rows != nonblank else None)
            if key not in best or cand[0] > best[key][0]:
                best[key] = cand
    print('qualifiers not a species (skipped):', dict(unknown))
    docs = []
    for (e, sp, cls), (_n, hf, i, pts, desc, table_rows) in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1], int(kv[0][2].split('-')[0]))):
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '1210464', 'version': 1, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'v2(pT) of {sp}, {cls}% Au+Au at {e:g} GeV',
            'collision_system': {'energy_gev': e, 'reaction': 'AU AU --> X'},
            'centrality': {'classes': [f'{cls}%'], **REFMULT},
            'particle_id': {'species': sp},
            'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                       'points': pts, 'point_window': WINDOW}},
            'harmonic': 2, 'method': 'event_plane', 'value_type': 'v2', 'dimensionality': 'differential_pT',
            'particles': 'AB', 'eta_gap': 0.1,
            'subevents': {'A': {'kinematic_range': {'eta': {'min': -1.0, 'max': -0.05}, 'pT': {'min': 0.15, 'max': 5.0}}},
                          'B': {'kinematic_range': {'eta': {'min': 0.05, 'max': 1.0}, 'pT': {'min': 0.15, 'max': 5.0}}}},
            'notes': {'value': (f'{CITE}. v2(pT) of {sp} from the eta-sub event plane (the particle in one TPC '
                                'half, the plane of tracks with 0.15 < pT < 5 GeV/c from the other, gap 0.1, pT '
                                'weighted), |eta| < 1; pT points (mean pT of each bin, '
                                f'{WINDOW} GeV/c windows). Computed as the two-subevent v2 (unit weights). Errors: '
                                'statistical, systematic and a global (non-flow) systematic.'
                                + (' Table rows equal to their own pT, without errors (a HEPData copy error), '
                                   'left out.' if table_rows else ''))},
        })
        if table_rows:
            docs[-1]['table_rows'] = table_rows
    out = 'data/flow_vn/star_flow_vn_identified_hadrons_multigev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter(d['particle_id']['species'] for d in docs)))


if __name__ == '__main__':
    main()
