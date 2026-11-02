#!/usr/bin/env python3
"""Curate ATLAS charged-hadron spectra in Pb+Pb at 2.76 TeV, INSPIRE 1360290.

G. Aad et al. (ATLAS), "Measurement of charged-particle spectra in Pb+Pb
collisions at sqrt(sNN) = 2.76 TeV with the ATLAS detector at the LHC",
JHEP 09 (2015) 050, arXiv:1504.04337.

From the paper (Secs. 2-5, Table 2):
- charged hadrons, 0.5 < pT < 150 GeV, |eta| < 2 (and |eta| slices of
  0.25 for 0-5%); centrality from the FCal Sum E_T (3.2 < |eta| < 4.9);
- the Pb+Pb spectra are tabulated as 1/N_evt 1/(2 pi pT) 1/<T_AA>
  d2N/(deta dpT) [mb/GeV^2], i.e. per-event yields divided by the Glauber
  <T_AA> of the class (Table 2: 26.3, 20.6, 14.4, 8.73, 5.05, 2.70, 1.34,
  0.41 mb^-1 for 0-5 ... 60-80%);
- errors: statistical and systematic (the systematic includes the <T_AA>
  uncertainty, which cancels in the per-event yield: kept, conservative).

Records: the Pb+Pb spectra of Tables 2-9 (|eta| < 2, eight classes) and
11-25 (0-5%, |eta| slices; a symmetric system, computed at eta > 0) as
per-event invariant yields: value_scale = 1/<T_AA> multiplies the
tabulated values back by <T_AA> (an exact inverse of the published
normalization).  The old records had |eta| < 2.5 for every table and no
<T_AA>.  Not curated: the pp spectra (no centrality classes), R_AA and
R_CP (ratios to pp or to a peripheral class with Glauber factors).

Run from the repository root:  python scripts/curate_atlas_charged_spectra_pbpb_2760.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1360290/HEPData-ins1360290-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'G. Aad et al. (ATLAS), JHEP 09 (2015) 050, arXiv:1504.04337'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
TAA = {'0-5': 26.3, '5-10': 20.6, '10-20': 14.4, '20-30': 8.73, '30-40': 5.05, '40-50': 2.70,
       '50-60': 1.34, '60-80': 0.41}                                   # mb^-1, Table 2


def main():
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    docs = []
    for n in list(range(2, 10)) + list(range(11, 26, 2)):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        assert q['RE'] == 'PB PB --> CHARGED X' and dv['header']['units'] == 'MB/GeV2', (hf, q)
        cent = q['CENTRALITY']
        lo_e, hi_e = (float(x) for x in q['ABS(ETA)'].split('-'))
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert rows == list(range(len(iv['values']))), hf
        edges = [float(iv['values'][0]['low'])] + [float(v['high']) for v in iv['values']]
        eta = {'min': lo_e, 'max': hi_e} if lo_e > 0 or hi_e < 2.0 else {'min': -2.0, 'max': 2.0}
        where = (f'|eta| < 2' if eta['min'] < 0 else
                 f'{lo_e:g} < |eta| < {hi_e:g} (a symmetric system: computed at eta > 0)')
        docs.append({
            'observable_type': 'particle_spectra', 'inspire_id': '1360290', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': ' '.join(sub[hf]['description'].split()) + f' {cent}%, {where}',
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
            'centrality': {'classes': [f'{cent}%'], **FCAL}, 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': eta, 'pT': {'min': edges[0], 'max': edges[-1], 'bins': edges}},
            'observable': 'invariant_yield', 'value_type': 'invariant_yield',
            'value_scale': round(1.0 / TAA[cent], 8),
            'notes': {'value': f'{CITE}. 1/N_evt 1/(2 pi pT) d2N/(deta dpT) of charged hadrons, {cent}%, {where}; '
                               f'tabulated divided by <T_AA> = {TAA[cent]} mb^-1 (Table 2), multiplied back by '
                               'value_scale = 1/<T_AA>. Errors: statistical and systematic (incl. the <T_AA> '
                               'uncertainty, which cancels in the per-event yield).'}})
    old = 'data/particle_spectra/atlas_particle_spectra_charged-particle_spectra_PbPb_pp_2760gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/particle_spectra/atlas_particle_spectra_charged_PbPb_2760gev_ins1360290.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
