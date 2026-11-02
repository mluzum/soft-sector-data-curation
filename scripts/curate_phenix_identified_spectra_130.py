#!/usr/bin/env python3
"""Curate PHENIX identified-hadron spectra in Au+Au at 130 GeV, INSPIRE 623413.

K. Adcox et al. (PHENIX), "Single identified hadron spectra from sqrt(sNN)
= 130 GeV Au+Au collisions", PRC 69, 024904 (2004), nucl-ex/0307010
(PPG009).

From the paper:
- pi+-, K+-, p, pbar in the east-arm TOF (|eta| < 0.35), yields per unit
  rapidity at mid-rapidity;
- centrality from the BBC charge vs ZDC energy (the trigger takes 92 +- 4%
  of the cross section): 0-5, 5-15, 15-30, 30-60, 60-92%;
- p and pbar inclusive (not corrected for Lambda feed-down);
- errors: stat and sys; <pT> in MeV/c.

Records (public): invariant yields of pi+-, K+-, p, pbar in the five classes
(Fig. 10), dN/dy (Table II) and <pT> (Fig. 13) vs centrality of each
species, |y| < 0.35 used (the arm acceptance).  Staged (--staged OUT): the
minimum-bias spectra (Fig. 9: no class of the model events).  Not curated:
the same spectra rebinned for ratios (Fig. 11, with the kaon table
labelled pi), systematic tables, inverse slopes, blast-wave and N_part-fit
parameters (Tables I, IV, VI-VIII, Fig. 23).  The old record had the
proton <pT> at 200 GeV in |eta| < 0.8.

Run from the repository root:
    python scripts/curate_phenix_identified_spectra_130.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins623413/HEPData-ins623413-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'K. Adcox et al. (PHENIX), PRC 69, 024904 (2004), nucl-ex/0307010'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge with the ZDC energy.'}
Y = {'min': -0.35, 'max': 0.35}
SPECIES = {'pi^{+}': 'pi+', 'pi^{-}': 'pi-', 'pi^+': 'pi+', 'pi^-': 'pi-', 'K^{+}': 'K+', 'K^{-}': 'K-', 'K^+': 'K+',
           'K^-': 'K-', 'p': 'p', 'bar{p}': 'pbar', 'overline{p}': 'pbar'}
SYSTEM = {'energy_gev': 130.0, 'reaction': 'AU AU --> X'}
OLD = 'data/mean_pT/phenix_mean_pT_single_identified_hadron_spectra_from_AuAu_200gev.yaml'
OUT = {'particle_spectra': 'data/particle_spectra/phenix_particle_spectra_identified_AuAu_130gev_ins623413.yaml',
       'particle_yields': 'data/particle_yields/phenix_particle_yields_identified_AuAu_130gev_ins623413.yaml',
       'mean_pT': 'data/mean_pT/phenix_mean_pT_identified_AuAu_130gev_ins623413.yaml'}
FIG10 = ['ppg009fig10a-1.yaml', 'ppg009fig10a-2.yaml', 'ppg009fig10b-1.yaml', 'ppg009fig10b-2.yaml'] + \
        [f'ppg009fig10c-{k}.yaml' for k in range(1, 8)]
BLANK = ('', '-', '--', 'none')


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def species(text):
    return SPECIES[str(text).replace('$', '').replace('\\', '').strip()]


def feed(sp):
    return '; inclusive (not corrected for Lambda feed-down)' if sp in ('p', 'pbar') else ''


def spectrum(hf, i, sp, cls, pairs, fig):
    return {'observable_type': 'particle_spectra', 'inspire_id': '623413', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{sp} invariant yield, {cls or "minimum bias"} ({hf})',
            'collision_system': dict(SYSTEM), 'centrality': {'classes': [cls or 'MB'], **BBC},
            'particle_id': {'species': sp},
            'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
            'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
            'notes': {'value': f'{CITE}, Fig. {fig}. 1/(2 pi pT) d2N/(dpT dy) of {sp}, {cls or "minimum bias"}, '
                               f'mid-rapidity (|y| < 0.35 used){feed(sp)}. Errors: stat and sys.'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    out, staged = collections.defaultdict(list), []
    for hf in FIG10 + ['ppg009fig9a.yaml', 'ppg009fig9b.yaml', 'ppg009fig9c.yaml']:
        t = load(hf)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            sp = species(dv['header']['name'])
            q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers') or []}
            cls = (q['Centrality (%)'] + '%') if 'Centrality (%)' in q else None
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = spectrum(hf, i, sp, cls, pairs, '10' if cls else '9')
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if cls:
                out['particle_spectra'].append(d)
            else:
                d['staging_reason'] = 'minimum-bias spectra: no class of the model events'
                staged.append(d)
    t = load('ppg009tab2.yaml')
    iv = t['independent_variables'][0]
    classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
    for i, dv in enumerate(t['dependent_variables']):
        h = dv['header']['name']
        if 'N_{' in h:
            continue                                                  # N_part, N_coll
        sp = species(h)
        out['particle_yields'].append({
            'observable_type': 'particle_yields', 'inspire_id': '623413', 'version': VERSION,
            'hepdata_file': 'ppg009tab2.yaml', 'dv_index': i, 'hepdata_table_label': f'{sp} dN/dy vs centrality (ppg009tab2.yaml)',
            'collision_system': dict(SYSTEM), 'centrality': {'classes': list(classes), **BBC},
            'particle_id': {'species': sp}, 'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
            'observable': 'dN_dy', 'value_type': 'dN_dy',
            'notes': {'value': f'{CITE}, Table II. dN/dy of {sp} at mid-rapidity (|y| < 0.35 used) vs centrality, '
                               f'from fits extrapolated to all pT{feed(sp)}. Errors: stat and sys.'}})
    t = load('ppg009fig13.yaml')
    iv = t['independent_variables'][0]
    classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
    for i, dv in enumerate(t['dependent_variables']):
        sp = species({qq['name']: qq['value'] for qq in dv['qualifiers']}['Particle'])
        out['mean_pT'].append({
            'observable_type': 'mean_pT', 'inspire_id': '623413', 'version': VERSION, 'hepdata_file': 'ppg009fig13.yaml',
            'dv_index': i, 'hepdata_table_label': f'<pT> of {sp} vs centrality (ppg009fig13.yaml)',
            'collision_system': dict(SYSTEM), 'centrality': {'classes': list(classes), **BBC},
            'particle_id': {'species': sp}, 'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
            'observable': 'mean_pT', 'value_type': 'mean_pT',
            'notes': {'value': f'{CITE}, Fig. 13. <pT> of {sp} at mid-rapidity (|y| < 0.35 used) vs centrality, from '
                               f'fits extrapolated to all pT{feed(sp)}; MeV/c. Errors: stat and sys.'}})
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '623413' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    for otype, docs in out.items():
        with open(OUT[otype], 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{OUT[otype]}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
