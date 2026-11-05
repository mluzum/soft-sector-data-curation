#!/usr/bin/env python3
"""Curate PHENIX pi0 v2(pT) in Cu+Au at 200 GeV and U+U at 193 GeV, INSPIRE 3187176.

N. J. Abdulameer et al. (PHENIX), "Elliptic flow of pi0 mesons in Cu+Au
collisions at sqrt(sNN) = 200 GeV and U+U collisions at sqrt(sNN) = 193
GeV", arXiv:2608.02787.

From the paper (Secs. II, III):
- pi0 -> gamma gamma in the EMCal, |eta| < 0.35, pT 1-10 GeV/c; centrality
  from the BBC charge (3.0 < |eta| < 3.9), 10% classes 0-60% and 0-20,
  20-60%;
- event-plane v2 corrected by the resolution (three-subevent method,
  Table I): Cu+Au with the FVTX (both arms, 1 < |eta| < 3) for pT < 5
  GeV/c and the BBC (3.0 < |eta| < 3.9) for pT >= 5 GeV/c; U+U with the MPC
  (-3.7 < eta < -3.1 and 3.1 < eta < 3.9) for all pT;
- errors: stat and sys (types B and C combined in the tables).

Records (public): v2(pT) of pi0 (computed by isospin, (pi+ + pi-)/2) per
class -- Cu+Au as two records per class, the pT < 5 GeV/c rows with the
FVTX plane and the pT >= 5 rows with the BBC plane (table_rows); U+U one
record per class with the MPC plane.  Cu+Au: frame nn_cms, Cu toward +eta
(as PHENIX 1394897; the windows are symmetric, so the orientation does not
enter).  The event-plane values are computed with the scalar-product
estimator (the low-resolution limit).  Staged (--staged-out): v2/eps2 and
v2/(eps2 N_part^(1/3)) (Glauber eccentricity and N_part scaling, Figs.
6-7) and R_AB(Delta phi), S_loss (high-pT nuclear modification, Figs.
8-10).

Run from the repository root:
    python scripts/curate_phenix_pi0_v2_cuau_uu.py [--staged-out <collection root>]
"""
import argparse
import copy
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins3187176/HEPData-ins3187176-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'N. J. Abdulameer et al. (PHENIX), arXiv:2608.02787'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
REFS = {'FVTX': {'detector': 'FVTX', 'eta': [{'min': -3.0, 'max': -1.0}, {'min': 1.0, 'max': 3.0}],
                 'pT': {'min': 0.0, 'max': 100.0},
                 'notes': 'Second-order event plane of both FVTX arms (tracks: all pT).'},
        'BBC': {'detector': 'BBC', 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
                'pT': {'min': 0.0, 'max': 100.0},
                'notes': 'Event plane of the combined South and North BBCs (all particles; the charge weighting is '
                         'not modelled).'},
        'MPC': {'detector': 'MPC', 'eta': [{'min': -3.7, 'max': -3.1}, {'min': 3.1, 'max': 3.9}],
                'pT': {'min': 0.0, 'max': 100.0},
                'notes': 'Event plane of both MPC arms (charged and neutral particles: all; the energy weighting is '
                         'not modelled).'}}
SYSTEMS = {'Cu + Au': ('CU AU --> PI0 X', 200.0, 'Cu+Au'), 'U + U': ('U U --> PI0 X', 193.0, 'U+U')}
V2_TABLES = ('Figure_4a.yaml', 'Figure_4b.yaml', 'Figure_4c.yaml', 'Figure_4d.yaml', 'Figure_4e.yaml',
             'Figure_5a.yaml', 'Figure_5b.yaml')
STAGED_SCALED = 'v2 divided by the Glauber eccentricity (and N_part^(1/3)): model-dependent scaling, not computed'
STAGED_RAB = 'R_AB(Delta phi) and S_loss: high-pT nuclear modification relative to the event plane, not computed'


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def quals(dv):
    return {q['name']: str(q['value']) for q in dv.get('qualifiers', [])}


def collision_system(sysname):
    reaction, energy, _ = SYSTEMS[sysname]
    cs = {'energy_gev': energy, 'reaction': reaction}
    if sysname == 'Cu + Au':
        cs.update(frame='nn_cms', rapidity_shift=0.0, forward_beam='Cu')
    return cs


def record(hf, i, dv, pairs, rows, ref, desc):
    q = quals(dv)
    sysname, c = q['Collision system'], q['Centrality']
    name = SYSTEMS[sysname][2]
    sel = [pairs[j] for j in rows]
    where = f'{sel[0][0]:g} < pT < {sel[-1][1]:g} GeV/c'
    d = {'observable_type': 'flow_vn', 'inspire_id': '3187176', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
         'hepdata_table_label': f'pi0 v2(pT), {c} {name}, {where} ({hf})',
         'collision_system': collision_system(sysname),
         'centrality': {'classes': [c], **copy.deepcopy(BBC)},
         'particle_id': {'species': 'pi0'},
         'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35},
                             'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': sel}},
         'harmonic': 2, 'method': 'event_plane', 'value_type': 'v_2', 'dimensionality': 'differential_pT',
         'flow_reference': copy.deepcopy(REFS[ref])}
    if rows != list(range(len(pairs))):
        d['table_rows'] = rows
    d['notes'] = {'value': f'{CITE}. {desc} v2(pT) of pi0 (gamma gamma in the EMCal, |eta| < 0.35), {c} {name}, '
                           f'{where}, {ref} event plane, resolution-corrected; pi0 computed by isospin '
                           '((pi+ + pi-)/2), with the scalar-product estimator. Errors: stat and sys.'}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged-out', help='collection root for the staged records (data/<type>/ below it)')
    args = ap.parse_args()
    sub = {x['data_file']: ' '.join(str(x.get('description', '')).split())
           for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x}
    public, staged = [], []
    for hf in V2_TABLES:
        t = load(hf)
        pairs = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            assert 'v_2' in dv['header']['name'] and 'varepsilon' not in dv['header']['name'], hf
            assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
            sysname = quals(dv)['Collision system']
            if sysname == 'Cu + Au':
                low = [j for j, (lo, hi) in enumerate(pairs) if hi <= 5.0 + 1e-9]
                high = [j for j, (lo, hi) in enumerate(pairs) if lo >= 5.0 - 1e-9]
                assert len(low) + len(high) == len(pairs), hf
                public.append(record(hf, i, dv, pairs, low, 'FVTX', 'Below 5 GeV/c:'))
                if high:
                    public.append(record(hf, i, dv, pairs, high, 'BBC', 'From 5 GeV/c:'))
            else:
                public.append(record(hf, i, dv, pairs, list(range(len(pairs))), 'MPC', ''))
    for hf, desc in sub.items():
        if hf in V2_TABLES:
            continue
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            q = quals(dv)
            sysname = q.get('Collision system', '')
            reaction, energy, name = SYSTEMS.get(sysname, ('AU AU --> PI0 X', 200.0, 'Au+Au'))
            staged.append({'observable_type': 'flow_vn' if hf.startswith(('Figure_6', 'Figure_7')) else
                           'nuclear_modification', 'inspire_id': '3187176', 'version': VERSION, 'hepdata_file': hf,
                           'dv_index': i, 'hepdata_table_label': f"{dv['header']['name']}, {q} ({hf})",
                           'collision_system': {'energy_gev': energy, 'reaction': reaction},
                           'particle_id': {'species': 'pi0'}, 'value_type': dv['header']['name'],
                           'notes': {'value': f'{CITE}. {desc}'},
                           'staging_reason': STAGED_SCALED if hf.startswith(('Figure_6', 'Figure_7')) else STAGED_RAB})
    out = 'data/flow_vn/phenix_flow_vn_pi0_v2_CuAu_UU_ins3187176.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if args.staged_out:
        path = os.path.join(args.staged_out, 'data', 'flow_vn', 'phenix_flow_vn_pi0_v2_scaled_RAB_CuAu_UU_ins3187176.yaml')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(staged)} staged records')


if __name__ == '__main__':
    main()
