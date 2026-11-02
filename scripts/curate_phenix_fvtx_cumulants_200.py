#!/usr/bin/env python3
"""Curate PHENIX multi-particle v2, v3 at forward rapidity in Au+Au at 200 GeV (PPG218), INSPIRE 1670164.

C. Aidala et al. (PHENIX), "Multiparticle azimuthal correlations for
extracting event-by-event elliptic and triangular flow in Au+Au collisions
at sqrt(sNN) = 200 GeV", PRC 99, 024903 (2019), arXiv:1804.10024.

From the paper (Secs. II-III):
- FVTX tracks, two arms: south -3 < eta < -1 and north 1 < eta < 3; no
  momentum measurement, efficiency flat for pT > 0.3 GeV/c (Fig. 1);
- centrality from the BBC charge (3.1 < |eta| < 3.9), 1% classes;
- single-acceptance cumulants combine the tracks of both arms; two-subevent
  forms take the arms as subevents a, b: v2{2}ab (|Delta eta| > 2; the
  v2{SP} of Fig. 7a), v2{4}aa|bb = (2<2>^2_a|b - <4>_aa|bb)^(1/4) and
  v2{4}ab|ab (short-range pairs allowed), v3{2, |Delta eta| > 2};
- sigma_v/<v> from cumulants and from unfolding (Figs. 8, 11, 12);
- errors: stat. and syst.

Records (public): v2{2}ab (Fig. 6a, magenta), v2{4}aa|bb (Fig. 6b, red)
and v3{2, |Delta eta| > 2} (Fig. 9a), two subevents (particles AB, AABB).
Staged (--staged OUT): the single-acceptance v2{2}, {4}, {6}, {8} (Figs.
6a red, 6b black, 7b-d: both arms in one acceptance, which a record
cannot state -- one arm would drop the cross-arm pairs), v2{4}ab|ab (form
not implemented: vn4_gap computes aa|bb), c3{4} (the cumulant, Fig. 10),
sigma_v2/v2 (Fig. 8), the unfolding results (Figs. 9b: sqrt(<v3^2>), 11,
12).  Not curated: Fig. 7a (repeats Fig. 6a magenta).  The old records
had one arm for the single acceptance, the ab|ab form as aa|bb, c3{4} as
v3{4}, and v3 without the gap.

Run from the repository root:
    python scripts/curate_phenix_fvtx_cumulants_200.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1670164/HEPData-ins1670164-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Aidala et al. (PHENIX), PRC 99, 024903 (2019), arXiv:1804.10024'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.1}, {'min': 3.1, 'max': 3.9}]}
PT = {'min': 0.3, 'max': 100.0}
ARMS = {'A': {'kinematic_range': {'eta': {'min': -3.0, 'max': -1.0}, 'pT': dict(PT)}},
        'B': {'kinematic_range': {'eta': {'min': 1.0, 'max': 3.0}, 'pT': dict(PT)}}}
# file -> (harmonic, method fields, public?, what, staging reason)
SINGLE = 'both FVTX arms (1 < |eta| < 3) in one acceptance: a record states one window (one arm drops the cross-arm pairs)'
TABLES = {
    'ppg218fig6a1.yaml': (2, {'cumulant_order': 2}, False, 'v2{2}, no gap', SINGLE),
    'ppg218fig6a2.yaml': (2, {'cumulant_order': 2, 'particles': 'AB', 'eta_gap': 2.0}, True, 'v2{2}ab, |Delta eta| > 2 (the arms)', None),
    'ppg218fig6b1.yaml': (2, {'cumulant_order': 4}, False, 'v2{4}, no gap', SINGLE),
    'ppg218fig6b2.yaml': (2, {'cumulant_order': 4, 'particles': 'AABB', 'eta_gap': 2.0}, False, 'v2{4}ab|ab',
                          'the ab|ab four-particle form is not implemented (vn4_gap computes aa|bb)'),
    'ppg218fig6b3.yaml': (2, {'cumulant_order': 4, 'particles': 'AABB', 'eta_gap': 2.0}, True, 'v2{4}aa|bb, |Delta eta| > 2 (the arms)', None),
    'ppg218fig7b.yaml': (2, {'cumulant_order': 4}, False, 'v2{4} (Fig. 7)', SINGLE),
    'ppg218fig7c.yaml': (2, {'cumulant_order': 6}, False, 'v2{6}', SINGLE),
    'ppg218fig7d.yaml': (2, {'cumulant_order': 8}, False, 'v2{8}', SINGLE),
    'ppg218fig9a.yaml': (3, {'cumulant_order': 2, 'particles': 'AB', 'eta_gap': 2.0}, True, 'v3{2, |Delta eta| > 2} (the arms)', None),
}
STAGED_OTHER = {'ppg218fig8.yaml': 'sigma_v2/<v2> from cumulants: a derived ratio',
                'ppg218fig9b.yaml': 'sqrt(<v3^2>) from the unfolding (Fig. 9 caption): not a cumulant',
                'ppg218fig10a1.yaml': 'c3{4}, the cumulant (not a flow coefficient)',
                'ppg218fig10a2.yaml': 'c3{4}, the cumulant (not a flow coefficient)',
                'ppg218fig10b.yaml': 'c3{4}, the cumulant (not a flow coefficient)',
                'ppg218fig11a.yaml': 'unfolding result', 'ppg218fig11b.yaml': 'unfolding result',
                'ppg218fig11c.yaml': 'unfolding result', 'ppg218fig12a.yaml': 'unfolding result',
                'ppg218fig12b.yaml': 'unfolding result', 'ppg218fig12c.yaml': 'unfolding result'}
OLD = 'data/flow_vn/phenix_flow_vn_multi-particle_azimuthal_correlations_200gev.yaml'


def classes_rows(t, dv):
    iv = t['independent_variables'][0]
    rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
    return [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows], rows, len(iv['values'])


def base(hf, i, dv, classes):
    return {'inspire_id': '1670164', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f"{dv['header']['name']} ({hf})",
            'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
            'centrality': {'classes': classes, **BBC}, 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': 1.0, 'max': 3.0}, 'pT': dict(PT)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf, (n, fields, pub, what, reason) in TABLES.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv = t['dependent_variables'][0]
        classes, rows, n_iv = classes_rows(t, dv)
        d = {'observable_type': 'flow_vn', **base(hf, 0, dv, classes), 'harmonic': n, 'method': 'cumulant', **fields,
             'value_type': dv['header']['name'], 'dimensionality': 'integrated',
             'notes': {'value': f'{CITE}. {what} of FVTX tracks (no momentum; pT > 0.3 GeV/c assumed), BBC '
                                'centrality. Errors: stat. and syst.'}}
        if fields.get('particles'):
            d['subevents'] = {k: {'kinematic_range': {kk: dict(vv) for kk, vv in v['kinematic_range'].items()}}
                              for k, v in ARMS.items()}
        if rows != list(range(n_iv)):
            d['table_rows'] = rows
        if pub:
            public.append(d)
        else:
            d['staging_reason'] = reason
            staged.append(d)
    for hf, reason in STAGED_OTHER.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            classes, rows, n_iv = classes_rows(t, dv)
            d = {'observable_type': 'flow_vn', **base(hf, i, dv, classes), 'harmonic': 3 if '3' in dv['header']['name'] else 2,
                 'method': 'cumulant', 'value_type': dv['header']['name'], 'dimensionality': 'integrated',
                 'notes': {'value': f"{CITE}. {dv['header']['name']}. Errors: stat. and syst."},
                 'staging_reason': reason}
            if rows != list(range(n_iv)):
                d['table_rows'] = rows
            staged.append(d)
    if os.path.exists(OLD):
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_fvtx_v2_v3_subevents_AuAu_200gev_ins1670164.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', [d['value_type'] for d in public])
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
