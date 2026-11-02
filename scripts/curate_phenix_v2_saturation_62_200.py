#!/usr/bin/env python3
"""Curate PHENIX charged-hadron v2 in Au+Au at 62.4-200 GeV, INSPIRE 664944.

S. S. Adler et al. (PHENIX), "Saturation of azimuthal anisotropy in Au+Au
collisions at sqrt(sNN) = 62-200 GeV", PRL 94, 232302 (2005),
nucl-ex/0411040 (PPG047).

From the paper:
- charged tracks in the central arms, |eta| < 0.35; centrality from the
  BBC (62.4 GeV) or BBC vs ZDC (130, 200 GeV);
- three methods: the BBC event plane (|eta| ~ 3-3.9; 62.4 and 200 GeV),
  the two-particle cumulant v2{2} corrected for the PHENIX acceptance (~30%;
  62.4 and 200 GeV), and assorted-pT correlation functions (the reference
  pairs 0.65 < pT < 2.5 GeV/c excluding the pT range of interest; 62.4, 130,
  200 GeV); "a similarly good agreement between the different methods";
- Fig. 2 is v2(pT) at 62.4 GeV (0-20%, 20-40%) for the three methods;
  Fig. 3 v2 vs centrality at <pT> = 0.4, 0.75, 1.35 GeV/c (cumulant and
  correlation-function results) and v2(pT) in 13-26% at 62.4 and 200 GeV;
- errors: stat and sys (Fig. 3: sys in HEPData).

Records (public): v2{2}(pT) from the cumulant method at 62.4 GeV, 0-20% and
20-40% (Fig. 2, table f2a); HEPData describes Fig. 2 as 200 GeV
(hepdata_corrections.energy).  The reference particles are not stated
(here the record acceptance).  Staged (--staged OUT): the other two Fig. 2
tables (their Method qualifiers, reaction plane and correlation function,
contradict the HEPData descriptions), v2 vs centrality (Fig. 3a-c: points at
a mean pT without windows, two methods combined) and v2(pT) in 13-26% (Fig.
3d, 3e: method not stated).  Not curated: correlation functions (Fig. 1)
and the excitation function (Fig. 4: with STAR, CERES and E895 data).  The 2
old records read Fig. 3c (two pT points, one record) and Fig. 4 as event-plane
v2 at 200 GeV.

Run from the repository root:
    python scripts/curate_phenix_v2_saturation_62_200.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins664944/HEPData-ins664944-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. S. Adler et al. (PHENIX), PRL 94, 232302 (2005), nucl-ex/0411040'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge (62.4 GeV); with the ZDC energy at 130 and 200 GeV.'}
ETA = {'min': -0.35, 'max': 0.35}
OLD = 'data/flow_vn/phenix_flow_vn_saturation_of_azimuthal_anisotropy_AuAu_200gev.yaml'
OUT = 'data/flow_vn/phenix_flow_vn_v2_charged_AuAu_62gev_ins664944.yaml'
FIG2_ENERGY = "HEPData describes Fig. 2 as 200 GeV; the paper's Fig. 2 (caption and text) is 62.4 GeV"
BLANK = ('', '-', '--', 'none')


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def pt_bins(t, dv):
    iv = t['independent_variables'][0]
    rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
    pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
    return rows, pairs, len(iv['values'])


def base(hf, i, h, energy, classes, kr, dim, note, **extra):
    d = {'observable_type': 'flow_vn', 'inspire_id': '664944', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
         'hepdata_table_label': f'{note.split(". ")[0]} ({hf})',
         'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> X'}, 'particle_id': {'species': 'charged'},
         'value_type': h, 'method': 'cumulant', 'cumulant_order': 2, 'harmonic': 2,
         'centrality': {'classes': classes, **BBC}, 'dimensionality': dim, 'kinematic_range': kr,
         'notes': {'value': f'{CITE}. {note}'}}
    d.update(extra)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    # Fig. 2, 62.4 GeV: f2a cumulant (public); f2b, f2c (method labels contradict the descriptions)
    for hf in ('ppg047f2a.yaml', 'ppg047f2b.yaml', 'ppg047f2c.yaml'):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            cls = q['Centrality'].replace(' ', '')
            rows, pairs, n = pt_bins(t, dv)
            kr = {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
            if hf == 'ppg047f2a.yaml':
                d = base(hf, i, dv['header']['name'], 62.4, [cls], kr, 'differential_pT',
                         f'v2{{2}}(pT) of charged hadrons, {cls}, 62.4 GeV, cumulant method. |eta| < 0.35, BBC '
                         'centrality; two-particle cumulant corrected for the PHENIX acceptance; the reference '
                         'particles are not stated (here the record acceptance). Errors: stat and sys.',
                         hepdata_corrections={'energy': FIG2_ENERGY})
                public.append(d)
            else:
                desc = 'correlation function' if hf == 'ppg047f2b.yaml' else 'reaction plane'
                d = base(hf, i, dv['header']['name'], 62.4, [cls], kr, 'differential_pT',
                         f"v2(pT) of charged hadrons, {cls}, 62.4 GeV, Method qualifier '{q['Method']}'. The "
                         f'HEPData description says {desc}. Errors: stat and sys.',
                         hepdata_corrections={'energy': FIG2_ENERGY},
                         staging_reason=f"the method is ambiguous: Method qualifier '{q['Method']}', HEPData "
                                        f"description '{desc}' (the BBC event plane or assorted-pT pairs whose "
                                        'reference excludes each bin)')
                staged.append(d)
            if rows != list(range(n)):
                d['table_rows'] = rows
    # Fig. 3a-c: v2 vs centrality at a mean pT (no window)
    for hf in ('ppg047f3a.yaml', 'ppg047f3b.yaml', 'ppg047f3c.yaml'):
        t = load(hf)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            pt = float(re.search(r'[\d.]+', q['$p_T$ (GeV/$c$)']).group())
            e = float(re.search(r'[\d.]+', q['$\\sqrt{s_{NN}}$']).group())
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
            d = base(hf, i, dv['header']['name'], e, classes,
                     {'eta': dict(ETA), 'pT': {'min': round(pt - 0.05, 3), 'max': round(pt + 0.05, 3)}}, 'integrated',
                     f'v2 of charged hadrons vs centrality at <pT> = {pt:g} GeV/c, {e:g} GeV. Cumulant and '
                     'correlation-function results; the pT window is not stated (here +-0.05). Errors: sys.',
                     staging_reason='a mean pT without a window, and two methods combined')
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            staged.append(d)
    # Fig. 3d, 3e: v2(pT), 13-26%
    for hf, e in (('ppg047f3d.yaml', 62.4), ('ppg047f3e.yaml', 200.0)):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            rows, pairs, n = pt_bins(t, dv)
            d = base(hf, i, dv['header']['name'], e, ['13-26%'],
                     {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                     'differential_pT', f'v2(pT) of charged hadrons, 13-26%, {e:g} GeV. Method not stated. Errors: sys.',
                     staging_reason='the method is not stated (cumulant, correlation function or event plane)')
            if rows != list(range(n)):
                d['table_rows'] = rows
            staged.append(d)
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '664944' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
