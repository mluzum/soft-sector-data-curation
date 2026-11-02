#!/usr/bin/env python3
"""Curate ALICE anisotropic flow of charged particles in Pb-Pb at 5.02 (and 2.76) TeV, INSPIRE 1419244.

J. Adam et al. (ALICE), "Anisotropic flow of charged particles in Pb-Pb
collisions at sqrt(sNN) = 5.02 TeV", PRL 116, 132302 (2016),
arXiv:1602.01119.

From the paper:
- 2015 run (140k events), compared with 2.76 TeV analysed the same way;
  centrality from the V0 amplitudes (V0A 2.8 < eta < 5.1, V0C -3.7 < eta
  < -1.7); charged tracks |eta| < 0.8, integrated over 0.2 < pT < 5 GeV/c;
- v_n{2, |Delta eta| > 1}: two-particle cumulant with a gap (here the
  subevents -0.8 < eta < -0.5 and 0.5 < eta < 0.8); v2{4}, v2{6}, v2{8}:
  multi-particle cumulants, no gap;
- the fully pT-integrated v2{4} (20-30%): a weighted average of the
  spectrum-weighted v2{4}(pT) extrapolated to pT = 0 and v2{4} from SPD
  tracklets (pT >~ 50 MeV/c);
- errors: stat and sys.

Records (public): vs centrality, 0.2 < pT < 5 GeV/c, at 5.02 TeV (Tables
1, 2) and 2.76 TeV (Tables 3, 4): v2{2, |Delta eta| > 1}, v2{4}, v2{6},
v2{8}, v3 and v4{2, |Delta eta| > 1}; vs pT: v_n{2, |Delta eta| > 1} (n =
2-4) in 0-5% and 30-40% (Tables 7, 8) and v2{4} in 10-20, 20-30, 30-40%
(Table 9), the reference 0.2 < pT < 5 GeV/c (the integrated range; the
paper does not state it); v2{4} in 20-30% over all pT (Table 11, one record
per energy).  HEPData gives the 2.76 TeV energy as 2076 GeV
(hepdata_corrections.energy).  Staged (--staged OUT): the ratios 5.02 /
2.76 TeV (Tables 5, 6, 10; energy-ratio records are not computed, and the
2.76 TeV v2{4}(pT) of Table 10 is not tabulated).  The 2 old records had
the ratio of Table 10 as v2{4}(pT) at 2.76 TeV and Table 11 as 20-30% v2{4}
at 2.76 TeV.

Run from the repository root:
    python scripts/curate_alice_flow_pbpb_5020_first.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1419244/HEPData-ins1419244-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (ALICE), PRL 116, 132302 (2016), arXiv:1602.01119'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
ETA = {'min': -0.8, 'max': 0.8}
PT_INT = {'min': 0.2, 'max': 5.0}
OUT = 'data/flow_vn/alice_flow_vn_PbPb_2760_5020gev_ins1419244.yaml'
OLD = 'data/flow_vn/alice_flow_vn_anisotropic_flow_of_charged_particles_PbPb_multigev.yaml'
ENERGY_FIX = 'HEPData SQRT(S)/NUCLEON 2076.0 for the 2.76 TeV tables (the caption and the paper: 2.76 TeV)'
BLANK = ('', '-', '--', 'none')


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def subevents(pt):
    return {'particles': 'AB', 'eta_gap': 1.0,
            'subevents': {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.5}, 'pT': dict(pt)}},
                          'B': {'kinematic_range': {'eta': {'min': 0.5, 'max': 0.8}, 'pT': dict(pt)}}}}


def parse_header(h):
    """'V3{2, |DELTAETA|>1}' -> (3, 'gap'); 'V2{4} (10-20%)' -> (2, 4)."""
    m = re.match(r'V(\d)\{(\d)(, \|DELTAETA\|>1)?\}', h)
    return int(m.group(1)), ('gap' if m.group(3) else int(m.group(2)))


def record(hf, i, h, energy, kind, n, classes, kr, differential, fig):
    d = {'observable_type': 'flow_vn', 'inspire_id': '1419244', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i}
    if kind == 'gap':
        f = {'method': 'cumulant', 'cumulant_order': 2, **subevents(PT_INT)}
        desc = (f'v{n}{{2, |Delta eta| > 1}} (subevents -0.8 < eta < -0.5 and 0.5 < eta < 0.8'
                + ('; reference 0.2 < pT < 5 GeV/c in the other subevent)' if differential else ')'))
    else:
        f = {'method': 'cumulant', 'cumulant_order': kind}
        if differential:
            f['flow_reference'] = {'pT': dict(PT_INT)}
        desc = f'v{n}{{{kind}}}' + (' (reference particles 0.2 < pT < 5 GeV/c)' if differential else '')
    what = f'{desc} vs pT, {classes[0]}' if differential else f"{desc} vs centrality, {kr['pT']['min']:g} < pT < " \
                                                              f"{kr['pT']['max']:g} GeV/c"
    d.update({'hepdata_table_label': f'{what}, {energy:g} GeV ({hf})',
              'collision_system': {'energy_gev': energy, 'reaction': 'PB PB --> CHARGED X'},
              'particle_id': {'species': 'charged'}, 'value_type': h, 'centrality': {'classes': classes, **V0M},
              'kinematic_range': kr, 'harmonic': n, **f,
              'dimensionality': 'differential_pT' if differential else 'integrated',
              'notes': {'value': f'{CITE}, {fig}. {what}, charged particles |eta| < 0.8, {energy:g} GeV. Errors: stat '
                                 'and sys.'}})
    if energy == 2760.0:
        d['hepdata_corrections'] = {'energy': ENERGY_FIX}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    # vs centrality
    for hf, energy in (('Table1.yaml', 5020.0), ('Table2.yaml', 5020.0), ('Table3.yaml', 2760.0), ('Table4.yaml', 2760.0)):
        t = load(hf)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            n, kind = parse_header(h)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
            d = record(hf, i, h, energy, kind, n, classes, {'eta': dict(ETA), 'pT': dict(PT_INT)}, False, 'Fig. 1a')
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    # vs pT
    for hf in ('Table7.yaml', 'Table8.yaml', 'Table9.yaml'):
        t = load(hf)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            n, kind = parse_header(h)
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            kr = {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
            d = record(hf, i, h, 5020.0, kind, n, [f'{lo:g}-{hi:g}%'], kr, True, 'Fig. 2')
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    # fully pT-integrated v2{4}, 20-30%, one record per energy
    t = load('Table11.yaml')
    dv = t['dependent_variables'][0]
    for j, v in enumerate(t['independent_variables'][0]['values']):
        energy = 2760.0 if float(v['value']) < 3000 else 5020.0
        d = record('Table11.yaml', 0, dv['header']['name'], energy, 4, 2, ['20-30%'],
                   {'eta': dict(ETA), 'pT': {'min': 0.0, 'max': 20.0}}, False, 'Fig. 3')
        d['table_rows'] = [j]
        d['notes']['value'] = (f'{CITE}, Fig. 3. v2{{4}} of charged particles over all pT, 20-30%, |eta| < 0.8, '
                               f'{energy:g} GeV: the weighted average of the spectrum-weighted v2{{4}}(pT) extrapolated '
                               'to pT = 0 and v2{4} of SPD tracklets. Errors: stat and sys.')
        d.pop('hepdata_corrections', None)
        public.append(d)
    # ratios 5.02 / 2.76 TeV (staged)
    index = {(d['hepdata_file'], d['dv_index']): k for k, d in enumerate(public)}
    links = {'Table5.yaml': ('Table1.yaml', 'Table3.yaml'), 'Table6.yaml': ('Table2.yaml', 'Table4.yaml')}
    for hf in ('Table5.yaml', 'Table6.yaml', 'Table10.yaml'):
        t = load(hf)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            n, kind = parse_header(h)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            if hf == 'Table10.yaml':
                pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
                classes = ['20-30%']
                kr = {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
            else:
                classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
                kr = {'eta': dict(ETA), 'pT': dict(PT_INT)}
            d = {'observable_type': 'observable_ratios', 'inspire_id': '1419244', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{h} ({hf})',
                 'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': 'charged'}, 'kinematic_range': kr,
                 'observable': f'v{n}', 'value_type': h, 'ratio_type': 'system_ratio', 'numerator': 'Pb+Pb 5.02 TeV',
                 'denominator': 'Pb+Pb 2.76 TeV', 'numerator_system': 'Pb+Pb 5.02 TeV',
                 'denominator_system': 'Pb+Pb 2.76 TeV',
                 'notes': {'value': f'{CITE}. {h}, charged particles |eta| < 0.8. Errors: stat and sys.'},
                 'staging_reason': 'ratio between collision energies: energy-ratio records are not computed'
                                   + ('; the 2.76 TeV v2{4}(pT) is not tabulated' if hf == 'Table10.yaml' else '')}
            if hf in links:
                num, den = links[hf]
                d['numerator_record'] = f'{OUT}#{index[(num, i)]}'
                d['denominator_record'] = f'{OUT}#{index[(den, i)]}'
            else:
                d['numerator_record'] = f"{OUT}#{index[('Table9.yaml', 1)]}"
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            staged.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1419244' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records;', dict(collections.Counter(
        (d['collision_system']['energy_gev'], d['dimensionality'], d.get('cumulant_order'), d.get('particles'))
        for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
