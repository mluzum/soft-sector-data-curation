#!/usr/bin/env python3
"""Curate PHENIX identified-hadron v2 in central p+Au and 3He+Au at 200 GeV, INSPIRE 1632759.

A. Adare et al. (PHENIX), "Measurements of mass-dependent azimuthal
anisotropy in central p+Au, d+Au, and 3He+Au collisions at sqrt(sNN) = 200
GeV", PRC 97, 064904 (2018), arXiv:1710.09736 (PPG207).

From the paper (Sec. II):
- pi+ + pi-, K+ + K-, p + pbar at |eta| < 0.35; 0-5% centrality from the
  charged multiplicity in the BBC-S (Au-going side, -3.9 < eta < -3.0);
- the second-order event plane from the FVTX-S (Au-going side, -3.0 < eta <
  -1.0), resolution from three subevents (BBC-S, FVTX-S, central arms);
- the light projectile goes toward +eta (north), Au toward -eta;
- errors: stat and (asymmetric) sys.

Records (public): v2(pT) of pi, p (Fig. 1) and K (Table I) in 0-5% p+Au
and 3He+Au, FVTX-S event plane computed with the scalar-product
estimator; lab frame (equal beam energies per nucleon), the light nucleus
toward +eta.  The d+Au results are not in this HEPData record.  The 4 old
records filed p+Au as 3He+Au, the species as p or charged, and had no
reference.

Run from the repository root:  python scripts/curate_phenix_pid_v2_small_systems.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1632759/HEPData-ins1632759-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRC 97, 064904 (2018), arXiv:1710.09736'
BBCS = {'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'}, 'eta': [{'min': -3.9, 'max': -3.0}],
        'notes': 'BBC charge on the Au-going side.'}
FVTXS = {'detector': 'FVTX-S', 'eta': [{'min': -3.0, 'max': -1.0}], 'pT': {'min': 0.0, 'max': 100.0},
         'notes': 'Second-order event plane of the FVTX on the Au-going side (hit clusters: all pT).'}
SYSTEMS = {'a': ('P AU', 'p'), 'b': ('HE3 AU', '3He')}
SPECIES = {'pi^+': 'pi+ + pi-', 'p$+': 'p + pbar', 'K^+': 'K+ + K-'}
OLD = 'data/flow_vn/phenix_flow_vn_measurements_of_mass-dependent_azimuthal_anisotropy_AuAu_200gev.yaml'


def species_of(label):
    for k, v in SPECIES.items():
        if k in label:
            return v
    raise KeyError(label)


def main():
    public = []
    for hf in ('ppg207fig1a.yaml', 'ppg207fig1b.yaml', 'ppg207tab1a.yaml', 'ppg207tab1b.yaml'):
        beams, light = SYSTEMS[hf[-6]]
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            sp = species_of(str(dv['qualifiers'][0]['value']))
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = {'observable_type': 'flow_vn', 'inspire_id': '1632759', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'v2(pT) of {sp}, 0-5% {beams.title()} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': f'{beams} --> X', 'frame': 'lab',
                                      'rapidity_shift': 0.0, 'forward_beam': light},
                 'centrality': {'classes': ['0-5%'], **BBCS}, 'particle_id': {'species': sp},
                 'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35},
                                     'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'harmonic': 2, 'method': 'event_plane', 'flow_reference': dict(FVTXS), 'value_type': 'v_2',
                 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}. v2(pT) of {sp}, 0-5% {beams.title()}, |eta| < 0.35, relative to the '
                                    'FVTX-S event plane (-3 < eta < -1), computed with the scalar-product estimator; '
                                    f'lab frame, {light} toward +eta. Errors: stat and sys.'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1632759' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_pid_v2_pAu_HeAu_200gev_ins1632759.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', [(d['collision_system']['reaction'][:6], d['particle_id']['species']) for d in public])


if __name__ == '__main__':
    main()
