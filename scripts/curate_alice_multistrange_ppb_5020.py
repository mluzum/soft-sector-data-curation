#!/usr/bin/env python3
"""Curate ALICE Xi and Omega production in p-Pb at 5.02 TeV, INSPIRE 1411084.

J. Adam et al. (ALICE), "Multi-strange baryon production in p-Pb
collisions at sqrt(sNN) = 5.02 TeV", PLB 758 (2016) 389,
arXiv:1512.07227.

From the paper and tables:
- (Xi- + Xibar+)/2 and (Omega- + Omegabar+)/2 invariant yields
  1/(2 pi pT N_ev) d2N/(dy dpT) at -0.5 < y_cms < 0 in seven V0A
  multiplicity classes 0-5 ... 80-100% (V0A: 2.8 < eta_lab < 5.1, the
  Pb-going side); 2013 beams: the nucleon-nucleon cms moves by 0.465
  toward the proton (y_lab = y_cms - 0.465), Pb toward +eta;
- <pT> of Xi-, Xibar+, Omega-, Omegabar+ per class (Tables 15-18), errors
  stat, 'sys,total' and 'sys,uncorr' (a part of the total);
- yields and ratios vs <dNch/deta> (Tables 19-24).

Records (public): the Xi spectra (Tables 1-7), species Xi- + Xibar+ with
charge_average, frame nn_cms (the V0A window in nn_cms: eta_lab + 0.465).
Staged (--staged OUT): the Omega spectra (Tables 8-14) and the <pT>
(Tables 15-18): Omega and single-charge Xi are not provided by the
pipeline.  The old records called every species 'p' and the system Pb-Pb.
Tables 19-24 stay staged (vs <dNch/deta>; single-charge species).

Run from the repository root:
    python scripts/curate_alice_multistrange_ppb_5020.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1411084/HEPData-ins1411084-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (ALICE), PLB 758 (2016) 389, arXiv:1512.07227'
SHIFT = -0.465
SYSTEM = {'energy_gev': 5020.0, 'frame': 'nn_cms', 'rapidity_shift': SHIFT, 'forward_beam': 'Pb'}
V0A = {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
       'eta': [{'min': round(2.8 - SHIFT, 3), 'max': round(5.1 - SHIFT, 3)}],
       'notes': 'V0A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side), here in the nn_cms frame.'}
SPECIES = {'(XI- + XIBAR+)/2': 'Xi- + Xibar+', '(OMEGA- + OMEGABAR+)/2': 'Omega- + Omegabar+',
           'XI-': 'Xi-', 'XIBAR+': 'Xibar+', 'OMEGA-': 'Omega-', 'OMEGABAR+': 'Omegabar+'}


def classes_of(iv):
    return [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for n in range(1, 19):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
        assert q['YRAP(C=CM)'] == '-0.5-0.0', (hf, q)
        re_ = q['RE']
        sp = SPECIES[re_.split('-->')[1].rsplit(' X', 1)[0].strip()]
        base = {'observable_type': None, 'inspire_id': '1411084', 'version': VERSION, 'hepdata_file': hf,
                'dv_index': 0, 'hepdata_table_label': ' '.join(dv['header']['name'].split()) + f' ({hf})',
                'collision_system': {**SYSTEM, 'reaction': re_}}
        y = {'min': -0.5, 'max': 0.0}
        if n <= 14:
            lo, hi = (float(x) for x in q['Multiplicity class'].split('-'))
            cls = f'{lo:g}-{hi:g}%'
            pairs = [[float(v['low']), float(v['high'])] for v in iv['values']]
            assert all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs[:-1], pairs[1:])), hf
            d = dict(base, observable_type='particle_spectra',
                     centrality={'classes': [cls], **V0A},
                     particle_id={'species': sp, 'charge_average': True},
                     kinematic_range={'y': y, 'pT': {'min': pairs[0][0], 'max': pairs[-1][1],
                                                     'bins': [p[0] for p in pairs] + [pairs[-1][1]]}},
                     observable='invariant_yield', value_type='invariant_yield',
                     notes={'value': f'{CITE}. 1/(2 pi pT N_ev) d2N/(dy dpT) of ({sp})/2, -0.5 < y_cms < 0, V0A '
                                     f'class {cls}. Errors: stat and sys.'})
            if sp == 'Xi- + Xibar+':
                public.append(d)
            else:
                d['staging_reason'] = 'Omega is not provided by the pipeline (analysis.species)'
                staged.append(d)
        else:
            d = dict(base, observable_type='mean_pT',
                     centrality={'classes': classes_of(iv), **V0A},
                     particle_id={'species': sp},
                     kinematic_range={'y': y, 'pT': {'min': 0.0, 'max': 100.0}},
                     value_type='mean_pT', errors_excluded=['sys,uncorr'],
                     notes={'value': f"{CITE}. <pT> of {sp}, -0.5 < y_cms < 0, full pT (spectra extrapolated). "
                                     "Errors: stat and sys,total ('sys,uncorr', the part uncorrelated across "
                                     "classes, is contained in it)."},
                     staging_reason=f'single-charge {sp} is not provided by the pipeline (analysis.species)')
            staged.append(d)
    old = 'data/mean_pT/alice_mean_pT_multi-strange_baryon_production_PbPb_5020gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/particle_spectra/alice_particle_spectra_Xi_pPb_5020gev_ins1411084.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(d['centrality']['classes'][0] for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
