#!/usr/bin/env python3
"""Curate PHENIX v2 of identified hadrons in Au+Au at 200 GeV, INSPIRE 619061.

S. S. Adler et al. (PHENIX), "Elliptic flow of identified hadrons in Au+Au
collisions at sqrt(sNN) = 200 GeV", PRL 91, 182301 (2003),
arXiv:nucl-ex/0305013.

From the paper:
- particles in the central arms, |eta| < 0.35, PID with the TOF;
- reaction plane from the combined BBCs (|eta| = 3-4; every particle);
  resolution about 0.3 on average;
- minimum-bias results (no percentile range stated) and 0-20, 20-40,
  40-60% (Fig. 3);
- errors: statistical (systematic in the paper).

Records (public): v2(pT) of p and pbar in 0-20, 20-40, 40-60% (Fig. 3).
Staged (--staged OUT): the h-, h+, pi- + K-, pi+ + K+ columns of Fig. 3
(species the pipeline does not provide) and the minimum-bias v2 (Figs.
1c, 2: no percentile class).  Not curated: the BBC reaction-plane
correlation (Fig. 1a) and the resolution correction factor (Fig. 1b: the
one public record, filed as v2).  The <pT> bin-centre columns were removed
earlier.

Run from the repository root:
    python scripts/curate_phenix_pid_v2_auau200_2003.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins619061/HEPData-ins619061-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. S. Adler et al. (PHENIX), PRL 91, 182301 (2003), arXiv:nucl-ex/0305013'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
BBC_REF = {'detector': 'BBC', 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
           'pT': {'min': 0.0, 'max': 100.0},
           'notes': 'Combined reaction plane of the two BBCs. Reference pT: all particles.'}
SPECIES = {'\\bar{p}': 'pbar', 'p': 'p'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    public, staged = [], []
    for x in sub:
        hf = x['data_file']
        if not re.match(r'ppg022f(1c|2|3)', hf):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        i, dv = 1, t['dependent_variables'][1]
        assert dv['header']['name'] == '$v_2$', (hf, dv['header'])
        q = [str(qq.get('value')) for qq in dv['qualifiers'] if qq['name'] not in ('RE', 'SQRT(S)/NUCLEON')]
        label = q[0].strip('$')
        cls = q[1].replace(' ', '') if len(q) > 1 else None
        sp = SPECIES.get(label)
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        vals = [iv['values'][j] for j in rows]
        if 'low' in vals[0]:
            pairs = [[float(v['low']), float(v['high'])] for v in vals]
            pt = {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}
        else:
            pts = [float(v['value']) for v in vals]
            w = round(min(b - a for a, b in zip(pts[:-1], pts[1:])), 4)
            pt = {'min': round(pts[0] - w / 2, 4), 'max': round(pts[-1] + w / 2, 4), 'points': pts,
                  'point_window': w}
        d = {'observable_type': 'flow_vn', 'inspire_id': '619061', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': i, 'hepdata_table_label': ' '.join(x['description'].split())[:150],
             'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
             'centrality': {'classes': [cls] if cls else ['minimum bias'], **BBC},
             'particle_id': {'species': sp or label},
             'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35}, 'pT': pt},
             'harmonic': 2, 'method': 'event_plane', 'value_type': 'v_2', 'dimensionality': 'differential_pT',
             'flow_reference': dict(BBC_REF),
             'notes': {'value': f'{CITE}. v2(pT) of {sp or label}, {cls or "minimum bias"}, |eta| < 0.35, combined '
                                'BBC reaction plane; computed with the scalar-product estimator. Errors: '
                                'statistical.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        if sp and cls:
            public.append(d)
        else:
            d['staging_reason'] = ('minimum bias: no percentile class' if not cls else
                                   f'species {label!r} (one charge, or a pi + K sum) is not provided by the pipeline')
            staged.append(d)
    old = 'data/flow_vn/phenix_flow_vn_identified_hadrons_AuAu_200gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/flow_vn/phenix_flow_vn_p_pbar_v2_AuAu_200gev_ins619061.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['particle_id']['species'], d['centrality']['classes'][0]) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
