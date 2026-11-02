#!/usr/bin/env python3
"""Curate PHENIX charged-hadron v2 from the event plane and two-particle cumulants, Au+Au 200 GeV, INSPIRE 819672.

S. Afanasiev et al. (PHENIX), "Systematic studies of elliptic flow
measurements in Au+Au collisions at sqrt(sNN) = 200 GeV", PRC 80, 024909
(2009), arXiv:0905.1070 (PPG043).

From the paper:
- charged hadrons in the central arms, |eta| < 0.35; centrality from the
  BBC charge (with the ZDC);
- event planes from the BBC (3.1 < |eta| < 3.9) and the ZDC-SMD
  (spectator neutrons, |eta| > 6.5);
- two-particle cumulant (generating function) within the central arms:
  integral particles 0.25 < pT < 2.0 GeV/c, differential particles in
  small (pT, eta) bins, the integral ones excluded;
- errors: stat and sys (v2(eta): stat).

Records (public): v2{2}(pT) in 0-10 ... 50-60% and 20-60% (Figs. 18a,
23), v2{BBC}(pT) in 0-10 ... 50-60% (Fig. 18b), v2{BBC}(eta) and
v2{2}(eta) in 20-40% for three pT windows (Fig. 21); the BBC plane
computed with the scalar-product estimator.  Staged (--staged OUT):
v2{ZDC-SMD} (spectator plane: models have no spectators).  Not curated:
resolutions (Fig. 15), the subevent comparisons (Fig. 22), the modified
event-plane method (Fig. 24), the Glauber table.  The 3 old records read
the resolution and N_part as v2.

Run from the repository root:
    python scripts/curate_phenix_v2_methods_ppg043.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins819672/HEPData-ins819672-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Afanasiev et al. (PHENIX), PRC 80, 024909 (2009), arXiv:0905.1070'
BBC_CENT = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
            'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
            'notes': 'BBC charge (with the ZDC energy in the paper).'}
BBC_REF = {'detector': 'BBC', 'eta': [{'min': -3.9, 'max': -3.1}, {'min': 3.1, 'max': 3.9}],
           'pT': {'min': 0.0, 'max': 100.0}, 'notes': 'Second-order event plane of both BBCs.'}
ZDC_REF = {'detector': 'ZDC-SMD', 'plane': 'spectator', 'eta': [{'min': -100.0, 'max': -6.5}, {'min': 6.5, 'max': 100.0}],
           'notes': 'First-order spectator-neutron plane of the ZDC-SMDs.'}
REF_PT = {'min': 0.25, 'max': 2.0}
ETA = {'min': -0.35, 'max': 0.35}
OLD = 'data/flow_vn/phenix_flow_vn_systematic_studies_of_elliptic_flow_AuAu_200gev.yaml'


def method_fields(h):
    if '{2}' in h:
        return {'method': 'cumulant', 'cumulant_order': 2, 'flow_reference': {'pT': dict(REF_PT)}}, 'v2{2}', None
    if 'ZDC' in h:
        return ({'method': 'event_plane', 'flow_reference': dict(ZDC_REF)}, 'v2{ZDC-SMD}',
                'relative to the spectator-neutron plane (ZDC-SMD): models have no spectators')
    return {'method': 'event_plane', 'flow_reference': dict(BBC_REF)}, 'v2{BBC}', None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    tables = sorted(f for f in os.listdir(RAW) if re.fullmatch(r'ppg043fig18[abc]\d\.yaml', f)) + \
        ['ppg043fig23.yaml', 'ppg043fig21.yaml']
    for hf in tables:
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        eta_axis = 'eta' in iv['header']['name']
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            if not h.startswith('$v_2$'):
                continue                                        # the <pT> column
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            f, name, why = method_fields(h)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            if eta_axis:
                lo, hi = (float(x) for x in re.findall(r'[\d.]+', q['$p_T$ Range (GeV/$c$)']))
                cls, dim = '20-40%', 'differential_eta'
                kr = {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}, 'pT': {'min': lo, 'max': hi}}
                where = f'vs eta, 20-40%, {lo:g} < pT < {hi:g} GeV/c'
            else:
                cls, dim = q['Centrality (%)'], 'differential_pT'
                kr = {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
                where = f'vs pT, {cls}'
            d = {'observable_type': 'flow_vn', 'inspire_id': '819672', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{name} {where} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
                 'centrality': {'classes': [cls], **BBC_CENT}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': kr, 'harmonic': 2, 'value_type': h, 'dimensionality': dim, **f}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            how = {'v2{2}': 'two-particle cumulant (integral particles 0.25 < pT < 2 GeV/c)',
                   'v2{BBC}': 'BBC event plane, computed with the scalar-product estimator',
                   'v2{ZDC-SMD}': 'ZDC-SMD spectator plane'}[name]
            errs = 'stat and sys' if any(e.get('label', '').startswith('sys') for v in dv['values']
                                         for e in v.get('errors') or []) else 'stat'
            d['notes'] = {'value': f'{CITE}. {name} of charged hadrons {where}, |eta| < 0.35, {how}. Errors: {errs}.'}
            if why:
                d['staging_reason'] = why
                staged.append(d)
            else:
                public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '819672' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_v2_methods_AuAu_200gev_ins819672.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
