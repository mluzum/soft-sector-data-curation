#!/usr/bin/env python3
"""Curate CMS elliptic-flow cumulants from unfolded p(v2) in Pb-Pb at 5.02 TeV, INSPIRE 1636197.

A. M. Sirunyan et al. (CMS), "Non-Gaussian elliptic-flow fluctuations in
PbPb collisions at sqrt(sNN) = 5.02 TeV", PLB 789 (2019) 643,
arXiv:1711.05594.

From the paper:
- charged tracks |eta| < 1.0, 0.3 < pT < 3.0 GeV/c; centrality from the HF
  transverse energy; 5%-wide classes 5-10 ... 55-60%;
- event-by-event v2 distributions unfolded for the multiplicity-related
  (statistical) resolution; v2{2}, v2{4}, v2{6}, v2{8} from the moments of
  the unfolded p(v2); their ratios and the standardized skewness estimate
  gamma1^exp; elliptic-power fits;
- errors: stat and sys.

Records (public): v2{2}, v2{4}, v2{6}, v2{8} vs centrality (Fig. 2),
computed as the cumulants of the record acceptance (the unfolding removes
the finite-multiplicity resolution, as the cumulants' self-correlation
removal does).  Staged (--staged OUT): v2{6}/v2{4}, v2{8}/v2{4},
v2{8}/v2{6} (Fig. 3) and gamma1^exp (Fig. 4): ratios of cumulant orders
are not computed; the unfolded p(v2) in 15-20, 30-35, 55-60% (Fig. 1: an
event-by-event distribution, not computed).  Not curated: elliptic-power
fit parameters (Fig. 5).  The old public record had v2{2} only; the 14 old
staged ones were typed as fluctuations at 0 GeV.

Run from the repository root:
    python scripts/curate_cms_v2_cumulants_unfolded_pbpb_5020.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1636197/HEPData-ins1636197-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. M. Sirunyan et al. (CMS), PLB 789 (2019) 643, arXiv:1711.05594'
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
KR = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.3, 'max': 3.0}}
OLD = 'data/flow_vn/cms_flow_vn_non_gaussian_elliptic_flow_fluctuations_PbPb_5020gev.yaml'
OUT = 'data/flow_vn/cms_flow_vn_v2_cumulants_unfolded_PbPb_5020gev_ins1636197.yaml'
# staged tables: description and the first cumulant order involved (the record's cumulant_order)
STAGED = {'fig3_v26_v24.yaml': ('v2{6}/v2{4}', 6), 'fig3_v28_v24.yaml': ('v2{8}/v2{4}', 8),
          'fig3_v28_v26.yaml': ('v2{8}/v2{6}', 8), 'fig4_g1e.yaml': ('gamma1^exp from v2{4}, v2{6}, v2{8}', 4)}
BLANK = ('', '-', '--', 'none')


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def base(hf, h, classes, order, note):
    return {'observable_type': 'flow_vn', 'inspire_id': '1636197', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
            'hepdata_table_label': f'{note.split(".")[0]} ({hf})',
            'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
            'particle_id': {'species': 'charged'}, 'value_type': h, 'method': 'cumulant', 'cumulant_order': order,
            'harmonic': 2, 'centrality': {'classes': classes, **HF}, 'dimensionality': 'integrated',
            'kinematic_range': {k: dict(v) for k, v in KR.items()},
            'notes': {'value': f'{CITE}. {note} Errors: stat and sys.'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf in ('fig2_v22.yaml', 'fig2_v24.yaml', 'fig2_v26.yaml', 'fig2_v28.yaml') + tuple(STAGED):
        t = load(hf)
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
        h = dv['header']['name']
        if hf in STAGED:
            what, order = STAGED[hf]
            d = base(hf, h, classes, order, f'{what} of charged particles vs centrality, |eta| < 1, 0.3 < pT < 3 '
                                            'GeV/c. From the moments of the unfolded p(v2).')
            d['staging_reason'] = 'ratio of v2 cumulant orders (or the skewness estimate built from them): not computed'
            staged.append(d)
        else:
            order = int(re.search(r'\{(\d)\}', h).group(1))
            d = base(hf, h, classes, order, f'v2{{{order}}} of charged particles vs centrality, |eta| < 1, 0.3 < pT < 3 '
                                            'GeV/c (HF centrality). From the moments of the unfolded event-by-event '
                                            'p(v2).')
            public.append(d)
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
    for hf in sorted(glob.glob(os.path.join(RAW, 'fig1_pv2_*.yaml'))):
        hf = os.path.basename(hf)
        t = load(hf)
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        cls = q['Centrality'] + '%'
        pairs = [[float(v['low']), float(v['high'])] for v in iv['values']]
        d = base(hf, dv['header']['name'], [cls], 2, f'Unfolded p(v2), {cls}, |eta| < 1, 0.3 < pT < 3 GeV/c, in v2 bins '
                                                    f"{pairs[0][0]:g}-{pairs[-1][1]:g}.")
        d.update({'observable': 'p(v2)', 'v2_bins': pairs,
                  'staging_reason': 'event-by-event v2 distribution (unfolded) in the v2 bins of v2_bins: not computed'})
        staged.append(d)
    for f in (OLD,):
        if os.path.exists(f):
            assert all(str(r['inspire_id']) == '1636197' for r in yaml.safe_load_all(open(f)) if r), f
            os.remove(f)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
