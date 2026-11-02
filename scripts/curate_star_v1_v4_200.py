#!/usr/bin/env python3
"""Curate STAR v2 (and the first and fourth harmonics) in Au+Au at 200 GeV, INSPIRE 631713.

J. Adams et al. (STAR), "Azimuthal anisotropy at RHIC: the first and
fourth harmonics", PRL 92, 062301 (2004), arXiv:nucl-ex/0310029.

From the paper:
- charged particles in the TPC, |eta| < 1.2, pT > 0.15 GeV/c (FTPCs 2.4 <
  |eta| < 4.2 for v1); centrality as in the STAR 200 GeV analyses (RefMult,
  |eta| < 0.5);
- v_n{EP_2}: relative to the second-harmonic event plane of the TPC; v4{3}
  from the three-particle correlation <cos(4 phi_a - 2 phi_b - 2 phi_c)>;
  v1{3} from the three-particle cumulant with the second harmonic;
- errors: statistical (non-flow systematic in the paper); values in percent.

Records (public): v2{EP_2}(pT) in 0-80% (Fig. 2) and v2{EP_2} vs
centrality (Fig. 3), TPC event plane computed with the scalar-product
estimator (value_scale 100).  Staged (--staged OUT): v4{EP_2}, v6{EP_2},
v4{3} and v1{3} (mixed-harmonic correlations relative to Psi_2: no
computation).  The 4 old records computed v4, v6 and v4{3} as v2-type
scalar products in 0-80%.

Run from the repository root:
    python scripts/curate_star_v1_v4_200.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins631713/HEPData-ins631713-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRL 92, 062301 (2004), arXiv:nucl-ex/0310029'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.5, 'max': 0.5}]}
ETA = {'min': -1.2, 'max': 1.2}
PT = {'min': 0.15, 'max': 100.0}
REF = {'detector': 'TPC', 'eta': [{'min': -1.2, 'max': 1.2}], 'pT': dict(PT),
       'notes': 'Second-harmonic TPC event plane, |eta| < 1.2, pT > 0.15 GeV/c (upper limit not stated).'}
OLD = 'data/flow_vn/star_flow_vn_azimuthal_anisotropy_AuAu_200gev_631713.yaml'
STAGE = 'mixed-harmonic correlation relative to the second-order plane (v_n{EP_2}, n > 2, or v_n{3}): no computation'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf in ('fig1.yaml', 'fig2_v2EP.yaml', 'fig2_v4EP.yaml', 'fig2_v6EP.yaml', 'fig2_v43.yaml', 'fig3.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            n = int(re.search(r'v_(\d)', h).group(1))
            q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            if hf == 'fig3.yaml':
                classes = [re.sub(r'\s*TO\s*', '-', str(iv['values'][j]['value'])).replace('.0', '') + '%' for j in rows]
                kr, dim, where = {'eta': dict(ETA), 'pT': dict(PT)}, 'integrated', 'vs centrality'
            else:
                pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
                lo, hi = (float(x) for x in re.findall(r'[\d.]+', q['CENTRALITY']))
                classes = [f'{lo:g}-{hi:g}%']
                var = 'eta' if hf == 'fig1.yaml' else 'pT'
                kr = ({'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}, 'pT': dict(PT)}
                      if var == 'eta' else
                      {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}})
                dim, where = f'differential_{var}', f'vs {var}, {classes[0]}'
            d = {'observable_type': 'flow_vn', 'inspire_id': '631713', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{h} {where} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
                 'centrality': {'classes': classes, **REFMULT}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': kr, 'harmonic': n, 'value_type': h, 'dimensionality': dim, 'value_scale': 100.0}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if n == 2:
                d.update({'method': 'event_plane', 'flow_reference': dict(REF)})
                how = 'TPC second-harmonic event plane, computed with the scalar-product estimator'
            else:
                d.update({'method': 'cumulant' if '{3}' in h.replace('\\', '') else 'event_plane',
                          'staging_reason': STAGE})
                if d['method'] == 'cumulant':
                    d['cumulant_order'] = 3
                how = h
            d['notes'] = {'value': f'{CITE}. {h} of charged particles {where}, {how}; tabulated in percent. '
                                   'Errors: statistical.'}
            (staged if n != 2 else public).append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '631713' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/star_flow_vn_v2_EP_AuAu_200gev_ins631713.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
