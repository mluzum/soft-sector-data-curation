#!/usr/bin/env python3
"""Curate PHENIX dNch/deta(eta) and v2(eta) in p+Al, p+Au, d+Au, 3He+Au at 200 GeV (PPG221), INSPIRE 1684475.

C. Aidala et al. (PHENIX), "Pseudorapidity dependence of particle
production and elliptic flow in asymmetric nuclear collisions of p+Al,
p+Au, d+Au, and 3He+Au at sqrt(sNN) = 200 GeV", PRL 121, 222301 (2018),
arXiv:1807.11928.

From the paper:
- multiplicity classes from the total charge of the BBC in the Al- or
  Au-going direction (-3.9 < eta < -3.1); dNch/deta corrected for the
  trigger bias;
- dNch/deta from the central arms (|eta| < 0.35) and the FVTX (1 < |eta|
  < 3);
- v2(eta) in 0-5% from the event plane of the Al/Au-going BBC (-3.9 < eta
  < -3.1), corrected to all pT in each eta bin;
- Al and Au go toward eta < 0, the light projectile toward +eta (lab frame
  = nucleon-nucleon frame);
- errors: dNch/deta syst.; v2 stat. and syst.

Records (public): dNch/deta(eta) of each system and class (Fig. 2) and
v2(eta) of each system in 0-5% (Fig. 4; the first row, edges -2.0 to
-2.8 against 0.2-wide bins, left out).  Staged (--staged OUT): dNch/deta
per participating quark pair (Fig. 3: divided by the Glauber N_qp/2).
Not curated: Fig. 1 (the 0-5% series of Fig. 2).  The old records called
the four systems 3He+Au and filed Fig. 3 as flow.

Run from the repository root:
    python scripts/curate_phenix_small_systems_eta_200.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1684475/HEPData-ins1684475-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Aidala et al. (PHENIX), PRL 121, 222301 (2018), arXiv:1807.11928'
BBCS = [{'min': -3.9, 'max': -3.1}]
SYSTEMS = {'a': ('HE3 AU', '3He', '3He+Au'), 'b': ('D AU', 'd', 'd+Au'), 'c': ('P AU', 'p', 'p+Au'),
           'd': ('P AL', 'p', 'p+Al')}
FIG4 = {'$^3$He+Au': 'a', '$d$+Au': 'b', '$p$+Au': 'c', '$p$+Al': 'd'}
OLD_GLOB = 'data/*/phenix_*pseudorapidity_dependence_of_particle_production*'


def base(hf, i, key, classes):
    beams, proj, name = SYSTEMS[key]
    heavy = 'Al' if 'AL' in beams else 'Au'
    return {'inspire_id': '1684475', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'collision_system': {'energy_gev': 200.0, 'reaction': f'{beams} --> CHARGED X', 'frame': 'lab',
                                 'rapidity_shift': 0.0, 'forward_beam': proj},
            'centrality': {'classes': classes,
                           'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'},
                           'eta': [dict(w) for w in BBCS],
                           'notes': f'Total charge in the BBC on the {heavy}-going side.'},
            'particle_id': {'species': 'charged'}}


def eta_axis(iv, rows):
    pairs = [sorted([float(iv['values'][j]['low']), float(iv['values'][j]['high'])]) for j in rows]
    return {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for key in 'abcd':
        name = SYSTEMS[key][2]
        hf = f'ppg221fig2{key}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            cls = dv['qualifiers'][0]['value'].replace(' ', '')
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            d = {'observable_type': 'particle_yields', **base(hf, i, key, [cls]),
                 'hepdata_table_label': f'dNch/deta(eta), {name} {cls} ({hf})',
                 'kinematic_range': {'eta': eta_axis(iv, rows), 'pT': {'min': 0.0, 'max': 100.0}},
                 'observable': 'dNch_deta', 'value_type': 'dNch_deta',
                 'notes': {'value': f'{CITE}. dNch/deta(eta) of charged hadrons, {name} {cls} (BBC charge on the '
                                    'heavy-nucleus side), trigger-bias corrected; lab frame, the heavy nucleus toward '
                                    '-eta. Errors: syst.'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    hf = 'ppg221fig4.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        key = FIG4[dv['qualifiers'][0]['value']]
        name = SYSTEMS[key][2]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')
                and float(iv['values'][j]['low']) < float(iv['values'][j]['high'])]
        heavy = 'Al' if key == 'd' else 'Au'
        d = {'observable_type': 'flow_vn', **base(hf, i, key, ['0-5%']),
             'hepdata_table_label': f'v2(eta), {name} 0-5% ({hf})',
             'kinematic_range': {'eta': eta_axis(iv, rows), 'pT': {'min': 0.0, 'max': 100.0}},
             'harmonic': 2, 'method': 'event_plane', 'value_type': 'v_2', 'dimensionality': 'differential_eta',
             'flow_reference': {'detector': 'BBC', 'eta': [dict(w) for w in BBCS], 'pT': {'min': 0.0, 'max': 100.0},
                                'notes': f'Event plane of the {heavy}-going BBC. Reference pT: all particles.'},
             'table_rows': rows,
             'notes': {'value': f'{CITE}. v2(eta) of charged hadrons at all pT (corrected), {name} 0-5%, event plane '
                                f'of the {heavy}-going BBC (-3.9 < eta < -3.1); computed with the scalar-product '
                                'estimator; lab frame. The first row (eta edges -2.0, -2.8) left out. Errors: stat. '
                                'and syst.'}}
        public.append(d)
    for key in 'abcd':
        hf = f'ppg221fig3{key}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
        d = {'observable_type': 'particle_yields', **base(hf, 0, key, classes),
             'hepdata_table_label': f'dNch/deta / (N_qp/2), {SYSTEMS[key][2]} ({hf})',
             'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35}, 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': 'dNch_deta', 'value_type': '(dN_ch/deta)/(N_qp/2)',
             'notes': {'value': f'{CITE}. Midrapidity dNch/deta per participating quark pair.'},
             'staging_reason': 'divided by the Glauber number of participating quark pairs'}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        staged.append(d)
    for f in glob.glob(OLD_GLOB):
        docs = [x for x in yaml.safe_load_all(open(f)) if x]
        assert all(str(x['inspire_id']) == '1684475' for x in docs), f
        os.remove(f)
    out = {}
    for d in public:
        out.setdefault(d['observable_type'], []).append(d)
    for otype, docs in out.items():
        path = f'data/{otype}/phenix_{otype}_eta_pAl_pAu_dAu_He3Au_200gev_ins1684475.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records;', dict(collections.Counter(
            d['collision_system']['reaction'].split(' -->')[0] for d in docs)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
