#!/usr/bin/env python3
"""Curate BRAHMS dNch/deta in d+Au at 200 GeV, INSPIRE 643085.

I. Arsene et al. (BRAHMS), "Centrality dependence of charged-particle
pseudorapidity distributions from d+Au collisions at sqrt(sNN) = 200
GeV", PRL 94, 032301 (2005), nucl-ex/0401025.

From the paper:
- dNch/deta from the Si (SiMA) and tile (TMA) multiplicity arrays,
  averaged (-2.2 <= eta <= 2.2 for a central vertex), and the Beam-Beam
  Counters (out to |eta| = 4.2);
- centrality from a geometry-weighted average of the SiMA and TMA
  multiplicities; classes 0-30, 30-60, 60-80% and minimum bias;
- the deuteron goes toward +eta (the deuteron-fragmentation region at
  positive rapidity), the Au nucleus toward -eta;
- errors: stat and sys (8% SiMA/TMA, 12% BBC).

Records (public): dNch/deta(eta) in 0-30, 30-60, 60-80% (lab frame, the
deuteron toward +eta), differential_eta (computed on 3D containers).  The
table gives the eta of each point, not bins: the SiMA/TMA points (spacing
0.293) take bins of that width, the BBC points (irregular, as close as
0.067) windows of +-0.03 around the point; for a density the window only
needs to be narrow.  Staged (--staged OUT): minimum bias (no centrality
class).  The old record held the three classes as one record with dN/dy,
bin edges from midpoints across the SiMA/BBC gaps, and no frame.

Run from the repository root:
    python scripts/curate_brahms_dndeta_dau_200.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins643085/HEPData-ins643085-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'I. Arsene et al. (BRAHMS), PRL 94, 032301 (2005), nucl-ex/0401025'
MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'Multiplicity array'},
      'eta': [{'min': -2.2, 'max': 2.2}],
      'notes': 'Geometry-weighted average of the SiMA and TMA multiplicities.'}
SYSTEM = {'energy_gev': 200.0, 'reaction': 'D AU --> CHARGED X', 'frame': 'lab', 'rapidity_shift': 0.0,
          'forward_beam': 'd'}
OLD = 'data/particle_yields/brahms_particle_yields_centrality_dependence_of_charged-particle_pseudorapidity_AuAu_200gev.yaml'
OUT = 'data/particle_yields/brahms_particle_yields_dNch_deta_dAu_200gev_ins643085.yaml'
MA_HALF, BBC_HALF = 0.1464, 0.03
BLANK = ('', '-', '--', 'none')


def window(x):
    h = MA_HALF if abs(x) < 2.2 else BBC_HALF
    return [round(x - h, 4), round(x + h, 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    t = yaml.safe_load(open(os.path.join(RAW, 'tab2.yaml')))
    x = [float(v['value']) for v in t['independent_variables'][0]['values']]
    public, staged = [], []
    for i, dv in enumerate(t['dependent_variables']):
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        mb = q['CENTRALITY'] == 'minbias'
        cls = 'MB' if mb else q['CENTRALITY'] + '%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        pairs = [window(x[j]) for j in rows]
        d = {'observable_type': 'particle_yields', 'inspire_id': '643085', 'version': VERSION, 'hepdata_file': 'tab2.yaml',
             'dv_index': i, 'hepdata_table_label': f'dNch/deta(eta), d+Au, {cls} (tab2.yaml)',
             'collision_system': dict(SYSTEM), 'centrality': {'classes': [cls], **MA},
             'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': min(p[0] for p in pairs), 'max': max(p[1] for p in pairs),
                                         'bin_ranges': pairs},
                                 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': 'dNch_deta', 'value_type': 'dNch_deta', 'dimensionality': 'differential_eta',
             'notes': {'value': f'{CITE}, Fig. 2. dNch/deta of charged particles vs eta in d+Au, {cls} (SiMA/TMA '
                                'multiplicity); SiMA/TMA averaged for |eta| < 2.2, BBC beyond; lab frame, the deuteron '
                                'toward +eta. Points at the tabulated eta: bins of 0.293 (SiMA/TMA) and +-0.03 (BBC). '
                                'Errors: stat and sys.'}}
        if rows != list(range(len(x))):
            d['table_rows'] = rows
        if mb:
            del d['centrality']
            d['staging_reason'] = 'minimum bias: no centrality class'
            staged.append(d)
        else:
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '643085' for r in yaml.safe_load_all(open(OLD)) if r), OLD
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
