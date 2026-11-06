#!/usr/bin/env python3
"""Curate PHENIX phi spectra and v2 in Cu+Au at 200 GeV and U+U at 193 GeV, INSPIRE 2623245.

U. A. Acharya et al. (PHENIX), "Measurement of phi-meson production in
Cu+Au collisions at sqrt(sNN) = 200 GeV and U+U collisions at sqrt(sNN) =
193 GeV", PRC 107 (2023) 014907, arXiv:2207.10745.

From the paper:
- phi -> K+ K- in the central arms, |eta| < 0.35; invariant pT spectra
  from 1.1 to 7 GeV/c in five classes per system (Cu+Au 0-20, 20-40,
  40-60, 60-80, 0-93%; U+U 0-20, 20-40, 40-60, 60-80, 0-80%);
- centrality from the total charge in the north and south BBCs (3.1 <
  |eta| < 3.9);
- v2(pT) relative to the second-order event plane of the FVTX (1 < |eta| <
  3) in Cu+Au and of the MPC (-3.7 < eta < -3.1, 3.1 < eta < 3.9) in U+U,
  at mean pT per bin (no edges given);
- errors: stat and sys.

Records (public): the spectra (Fig. 3a, b; 10 records, -0.35 < y < 0.35)
and v2(pT) (Fig. 8; Cu+Au 0-20, 20-40, 40-60, 20-60%, U+U 0-50%) at the
tabulated mean pT in 0.1 GeV/c windows, event planes as flow_reference
(the template of PHENIX 3187176's pi0 v2).  Cu+Au in the nn frame with Cu
toward +eta (rapidity_shift 0; the windows are symmetric).  Staged
(--staged OUT): data-to-Levy-fit ratios (Fig. 3c, d), R_AB (Figs. 4, 5:
need the pp reference), v2 over eps N_part^(1/3) (model-dependent), and
Figs. 9-11 (the v2 of Fig. 8 again, and vs KE_T).

Run from the repository root:  python scripts/curate_phenix_phi_cuau_uu.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2623245/HEPData-ins2623245-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'U. A. Acharya et al. (PHENIX), PRC 107 (2023) 014907, arXiv:2207.10745'
OUT_S = 'data/particle_spectra/phenix_particle_spectra_phi_CuAu_UU_ins2623245.yaml'
OUT_V = 'data/flow_vn/phenix_flow_vn_phi_v2_CuAu_UU_ins2623245.yaml'
WINDOW = 0.1
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.1}, {'min': 3.1, 'max': 3.9}],
       'notes': 'total charge in the north and south BBCs, 3.1 < |eta| < 3.9.'}
REFS = {'Cu+Au': {'detector': 'FVTX', 'eta': [{'min': -3.0, 'max': -1.0}, {'min': 1.0, 'max': 3.0}],
                  'pT': {'min': 0.0, 'max': 100.0},
                  'notes': 'Second-order event plane of both FVTX arms (tracks: all pT).'},
        'U+U': {'detector': 'MPC', 'eta': [{'min': -3.7, 'max': -3.1}, {'min': 3.1, 'max': 3.9}],
                'pT': {'min': 0.0, 'max': 100.0},
                'notes': 'Event plane of both MPC arms (charged and neutral particles: all; the energy weighting is '
                         'not modelled).'}}


def system(name):
    if name == 'Cu+Au':
        return {'energy_gev': 200.0, 'reaction': 'CU AU --> PHI X', 'frame': 'nn_cms', 'rapidity_shift': 0.0,
                'forward_beam': 'Cu'}
    return {'energy_gev': 193.0, 'reaction': 'U U --> PHI X'}


def qual(dv):
    return list({k['name']: k['value'] for k in dv.get('qualifiers', [])}.values())[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    spectra, flows, staged = [], [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        xs = t['independent_variables'][0]['values']
        for i, dv in enumerate(t['dependent_variables']):
            q = qual(dv)
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
            if hf == 'fig3_1.yaml':
                name, lo, hi = re.fullmatch(r'\$(Cu\+Au|U\+U)\$ \$(\d+)\\%-(\d+)\\%\$', q).groups()
                cls = f'{lo}-{hi}%'
                bins = [[float(xs[r]['low']), float(xs[r]['high'])] for r in rows]
                assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat.', 'sys.')}
                d = {'observable_type': 'particle_spectra', 'inspire_id': '2623245', 'version': VERSION,
                     'hepdata_file': hf, 'dv_index': i,
                     'hepdata_table_label': f'Fig. 3: phi invariant yield, {name}, {cls} ({hf} column {i})',
                     'collision_system': system(name), 'centrality': {'classes': [cls], **BBC},
                     'particle_id': {'species': 'phi'},
                     'kinematic_range': {'y': {'min': -0.35, 'max': 0.35},
                                         'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                     'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                     'hepdata_corrections': {'pT': "HEPData's qualifier named p_T holds the system and class"},
                     'notes': {'value': f'{CITE}, Fig. 3. Invariant yield of phi (K+ K-), |eta| < 0.35 (here |y| < '
                                        f'0.35), {name} {system(name)["energy_gev"]:g} GeV {cls}. Errors: stat. and '
                                        'sys.'}}
                if rows != list(range(len(xs))):
                    d['table_rows'] = rows
                spectra.append(d)
                continue
            if hf == 'fig8.yaml' and q.startswith('$v_2$ '):
                lo, hi, name = re.fullmatch(r'\$v_2\$ \$(\d+)\\%-(\d+)\\% (Cu\+Au|U\+U)\$', q).groups()
                cls = f'{lo}-{hi}%'
                pts = [float(xs[r]['low']) for r in rows]
                assert all(xs[r]['low'] == xs[r]['high'] for r in rows) and pts == sorted(pts), (hf, i)
                assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat.', 'sys.')}
                d = {'observable_type': 'flow_vn', 'inspire_id': '2623245', 'version': VERSION, 'hepdata_file': hf,
                     'dv_index': i, 'hepdata_table_label': f'Fig. 8: phi v2(pT), {name}, {cls} ({hf} column {i})',
                     'collision_system': system(name), 'centrality': {'classes': [cls], **BBC},
                     'particle_id': {'species': 'phi'},
                     'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35},
                                         'pT': {'min': round(pts[0] - WINDOW / 2, 4),
                                                'max': round(pts[-1] + WINDOW / 2, 4),
                                                'points': pts, 'point_window': WINDOW}},
                     'harmonic': 2, 'method': 'event_plane', 'value_type': 'v_2', 'dimensionality': 'differential_pT',
                     'flow_reference': dict(REFS[name]), 'table_rows': rows,
                     'hepdata_corrections': {'pT': "HEPData's qualifier named p_T holds the quantity, class and "
                                                   "system; the pT axis gives mean pT points"},
                     'notes': {'value': f'{CITE}, Fig. 8. v2 of phi at the tabulated mean pT (0.1 GeV/c windows), '
                                        f'|eta| < 0.35, {name} {cls}, relative to the '
                                        f'{"FVTX" if name == "Cu+Au" else "MPC"} event plane. Errors: stat. and '
                                        'sys.'}}
                flows.append(d)
                continue
            staged.append({'observable_type': 'flow_vn' if hf.startswith(('fig8', 'fig9', 'fig10', 'fig11')) else
                           'nuclear_modification' if hf.startswith(('fig4', 'fig5')) else 'particle_spectra',
                           'inspire_id': '2623245', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f'{q} ({hf} column {i})', 'value_type': dv['header']['name'][:80],
                           'notes': {'value': f'{CITE}. {desc[:200]}'},
                           'staging_reason': ('data-to-Levy-fit ratio' if hf == 'fig3_2.yaml' else
                                              'R_AB: needs the pp reference' if hf.startswith(('fig4', 'fig5')) else
                                              'v2 over eps N_part^(1/3): model-dependent scaling' if hf == 'fig8.yaml'
                                              else 'the v2 of Fig. 8 again, or vs KE_T (/n_q)')})
    for out, docs in ((OUT_S, spectra), (OUT_V, flows)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
