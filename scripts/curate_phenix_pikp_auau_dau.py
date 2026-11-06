#!/usr/bin/env python3
"""Curate PHENIX pi, K, p spectra and ratios in Au+Au and d+Au at 200 GeV, INSPIRE 1227971.

A. Adare et al. (PHENIX), "Spectra and ratios of identified particles in
Au+Au and d+Au collisions at sqrt(sNN) = 200 GeV", PRC 88 (2013) 024906,
arXiv:1304.3410.

From the paper:
- pi+-, K+-, p, pbar in the central arms (TOF-West, |eta| < 0.35);
- centrality from the BBC charge (3.0 < |eta| < 3.9): Au+Au both BBCs,
  0-10, 10-20, 20-40, 40-60, 60-92%; d+Au the south (Au-going) BBC only,
  0-20, 20-40, 40-60, 60-88% and minimum bias (bias-corrected, 0-100%);
- invariant yields; p and pbar corrected for the hyperon feed-down
  (Lambda/p assumed 0.89 and Lambdabar/pbar 0.95 in Au+Au, 0.85 and 0.99
  in d+Au); errors stat and sys;
- ratios of invariant yields pi-/pi+, K-/K+, pbar/p, K/pi, p/pi vs pT
  (Figs. 5-9).

Records (public): the spectra (60; |y| < 0.35, from |eta| < 0.35) and the
ratios (70), particle_ratio records linked to the spectra.  d+Au in the nn
frame with d toward +y (rapidity_shift 0).  Staged (--staged OUT): R_CP,
R_AA, R_dA (Glauber-scaled or need the pp reference) and the three
'Ratio of Inv Yield' comparison tables.

Run from the repository root:  python scripts/curate_phenix_pikp_auau_dau.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1227971/HEPData-ins1227971-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRC 88 (2013) 024906, arXiv:1304.3410'
OUT_S = 'data/particle_spectra/phenix_particle_spectra_pi_K_p_AuAu_dAu_200gev_ins1227971.yaml'
OUT_R = 'data/observable_ratios/phenix_observable_ratios_pi_K_p_AuAu_dAu_200gev_ins1227971.yaml'
CENT = {'Au+Au': {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
                  'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
                  'notes': 'summed charge of the north and south BBCs, 3.0 < |eta| < 3.9.'},
        'd+Au': {'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'},
                 'eta': [{'min': -3.9, 'max': -3.0}],
                 'notes': 'charge in the south (Au-going) BBC, -3.9 < eta < -3.0.'}}
SYSTEM = {'Au+Au': {'energy_gev': 200.0, 'reaction': 'AU AU --> {} X'},
          'd+Au': {'energy_gev': 200.0, 'reaction': 'D AU --> {} X', 'frame': 'nn_cms', 'rapidity_shift': 0.0,
                   'forward_beam': 'd'}}
SPECIES = {'\\pi^-': ('pi-', 'PI-'), '\\pi^+': ('pi+', 'PI+'), 'K^-': ('K-', 'K-'), 'K^+': ('K+', 'K+'),
           '\\bar{p}': ('pbar', 'PBAR'), '\\bar{p}^-': ('pbar', 'PBAR'), 'p': ('p', 'P')}
FEED = {'p': ' Corrected for the hyperon feed-down.', 'pbar': ' Corrected for the hyperon feed-down.'}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def parse(q):
    m = re.fullmatch(r'(Au\+Au|d\+AU|d\+Au) \$(.+?)\$ \$(\d+)-(\d+)\\%\$ centrality', q.strip())
    assert m, q
    system = 'Au+Au' if m.group(1) == 'Au+Au' else 'd+Au'
    return system, m.group(2), f'{m.group(3)}-{m.group(4)}%'


def system_of(name, prod):
    s = dict(SYSTEM[name])
    s['reaction'] = s['reaction'].format(prod)
    return s


def bins_of(t, rows):
    ivs = t['independent_variables'][0]['values']
    b = [[float(ivs[r]['low']), float(ivs[r]['high'])] for r in rows]
    assert all(x[1] <= y[0] + 1e-9 for x, y in zip(b, b[1:])), b
    return b


def filled(dv):
    return [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, ratios, staged = [], [], []
    side = {}
    for part in ('pion', 'kaon', 'prot'):
        for sysname in ('AuAu', 'dAu'):
            hf = f'{part}_{sysname}.yaml'
            t = load(hf)
            for i, dv in enumerate(t['dependent_variables']):
                system, sp, cls = parse(dv['qualifiers'][0]['value'])
                species, prod = SPECIES[sp]
                rows = filled(dv)
                assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == \
                    {('Stat error', 'Sys error')}, (hf, i)
                bins = bins_of(t, rows)
                mb = ' (minimum bias, corrected for the BBC trigger bias: 0-100%)' if cls == '0-100%' else ''
                d = {'observable_type': 'particle_spectra', 'inspire_id': '1227971', 'version': VERSION,
                     'hepdata_file': hf, 'dv_index': i,
                     'hepdata_table_label': f'{species} invariant yield, {system} 200 GeV {cls} ({hf} column {i})',
                     'collision_system': system_of(system, prod), 'centrality': {'classes': [cls], **CENT[system]},
                     'particle_id': {'species': species},
                     'kinematic_range': {'y': {'min': -0.35, 'max': 0.35},
                                         'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                     'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                     'hepdata_corrections': {'pT': "HEPData's qualifier named p_T holds the system, species and "
                                                   'class; the pT bins are the table\'s'},
                     'notes': {'value': f'{CITE}, Figs. 1, 2. Invariant yield of {species}, |eta| < 0.35 (here |y| < '
                                        f'0.35), {system} 200 GeV {cls}{mb}.{FEED.get(species, "")} Errors: stat and '
                                        'sys.'}}
                if rows != list(range(len(t['independent_variables'][0]['values']))):
                    d['table_rows'] = rows
                side[(system, species, cls)] = len(spectra)
                spectra.append(d)
    for hf in sorted(f for f in os.listdir(RAW) if f.startswith('ratio')):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            system, pair, cls = parse(dv['qualifiers'][0]['value'])
            num_s, den_s = pair.split('/')
            num, den = SPECIES[num_s][0], SPECIES[den_s][0]
            rows = filled(dv)
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == \
                {('Stat error', 'Sys error')}, (hf, i)
            bins = bins_of(t, rows)
            d = {'observable_type': 'observable_ratios', 'inspire_id': '1227971', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f'{num}/{den} vs pT, {system} 200 GeV {cls} ({hf} column {i})',
                 'collision_system': system_of(system, 'X'), 'centrality': {'classes': [cls], **CENT[system]},
                 'kinematic_range': {'y': {'min': -0.35, 'max': 0.35},
                                     'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                 'observable': 'invariant_yield', 'value_type': f'{num}/{den}', 'ratio_type': 'particle_ratio',
                 'numerator': num, 'denominator': den, 'numerator_particle': num, 'denominator_particle': den,
                 'numerator_system': system, 'denominator_system': system,
                 'numerator_record': f'{OUT_S}#{side[(system, num, cls)]}',
                 'denominator_record': f'{OUT_S}#{side[(system, den, cls)]}',
                 'hepdata_corrections': {'pT': "HEPData's qualifier named p_T holds the system, ratio and class; the "
                                               "pT bins are the table's"},
                 'notes': {'value': f'{CITE}, Figs. 5-9. {num}/{den} vs pT, |eta| < 0.35 (here |y| < 0.35), {system} '
                                    f'200 GeV {cls}. Errors: stat and sys.'}}
            if rows != list(range(len(t['independent_variables'][0]['values']))):
                d['table_rows'] = rows
            ratios.append(d)
    for hf in sorted(f for f in os.listdir(RAW) if f.startswith(('rcp', 'raa', 'rda', 'aud_sr'))):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            q = ', '.join(str(k['value']) for k in dv.get('qualifiers', []))
            staged.append({'observable_type': 'nuclear_modification' if not hf.startswith('aud') else
                           'particle_spectra', 'inspire_id': '1227971', 'version': VERSION, 'hepdata_file': hf,
                           'dv_index': i, 'hepdata_table_label': f"{dv['header']['name']} {q} ({hf} column {i})",
                           'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}.'},
                           'staging_reason': ("a 'Ratio of Inv Yield' comparison table (its reference not stated)"
                                              if hf.startswith('aud') else
                                              'R_CP / R_AA / R_dA: N_coll scaled or needs the pp reference')})
    for out, docs in ((OUT_S, spectra), (OUT_R, ratios)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
