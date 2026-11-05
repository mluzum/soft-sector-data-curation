#!/usr/bin/env python3
"""Curate ALICE pi, K, K0S, p, Lambda vs multiplicity in p-Pb at 5.02 TeV, INSPIRE 1244523.

B. Abelev et al. (ALICE), "Multiplicity dependence of pion, kaon, proton
and lambda production in p-Pb collisions at sqrt(sNN) = 5.02 TeV", PLB 728
(2014) 25, arXiv:1307.6796.

From the paper (Secs. 2, 3, Table 1):
- 2013 run: the lead beam toward the ALICE A side, positive rapidity in the
  paper; the nucleon-nucleon centre of mass moves with y_NN = -0.465 (toward
  the proton); every result is at 0 < y_cms < 0.5 (the Pb-going half);
- seven event classes, percentiles of the V0A amplitude (2.8 < eta_lab <
  5.1, the Pb-going side) in the analysed (V0A and V0C) sample, not
  corrected for trigger efficiency (about 2% for NSD events): 0-5, 5-10,
  10-20, 20-40, 40-60, 60-80, 80-100%;
- primary particles, sums of particle and antiparticle: pi+ + pi-, K+ + K-,
  p + pbar, Lambda + Lambdabar, and K0S, invariant yields 1/N_ev 1/(2 pi pT)
  d2N/(dpT dy) over 0.1-3, 0.2-2.5, 0-8, 0.3-4 and 0.6-8 GeV/c;
- pT-integrated yields dN/dy and <pT> extrapolated to 0 and to 10 GeV/c
  with blast-wave fits (Sec. 3);
- K/pi = (K+ + K-)/(pi+ + pi-), p/pi = (p + pbar)/(pi+ + pi-) and
  Lambda/K0S vs pT, and K/pi, p/pi, 2 Lambda/pi pT-integrated; in the
  tables Lambda/K0S is (Lambda + Lambdabar) / (2 K0S) (the spectra's ratio
  is 2.007 x the table; K/pi and p/pi: 1.000), 2 Lambda/pi is
  (Lambda + Lambdabar) / (pi+ + pi-) (= the dN/dy ratio);
- errors: stat and the total systematic; the ratio, <pT> tables give a
  second 'sys', the part uncorrelated across multiplicity classes (shaded
  boxes, Figs. 2-4), contained in the total: errors_excluded 'sys#2'.

Records (public, 69): the 35 spectra (one per species and class); dN/dy
and <pT> of each species vs class (pT 0-100 GeV, the extrapolated yields);
the pT-differential K/pi, p/pi, Lambda/K0S per class and the integrated
K/pi, p/pi, 2 Lambda/pi vs class as particle ratios linking those records
(computed on one simulation set, compute_ratios; Lambda/K0S with scale
0.5).  Frame nn_cms, Pb toward +y; the V0A window in that frame (eta_lab +
0.465).

Run from the repository root:  python scripts/curate_alice_pid_ppb_multiplicity_5020.py
"""
import copy
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1244523/HEPData-ins1244523-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. Abelev et al. (ALICE), PLB 728 (2014) 25, arXiv:1307.6796'
SHIFT = -0.465
V0A = {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
       'eta': [{'min': round(2.8 - SHIFT, 3), 'max': round(5.1 - SHIFT, 3)}],
       'notes': 'V0A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side), here in the nn_cms frame; percentiles of the '
                'analysed sample (V0A and V0C coincidence), not corrected for trigger efficiency.'}
Y = {'min': 0.0, 'max': 0.5}
CLASSES = ['0-5%', '5-10%', '10-20%', '20-40%', '40-60%', '60-80%', '80-100%']
# HEPData reaction -> (species, reaction code, label)
SPECIES = {'P PB --> (PI+ + PI-) X': ('pi+ + pi-', 'PI+ PI-', 'pi+ + pi-'),
           'P PB --> (K+ + K-) X': ('K+ + K-', 'K+ K-', 'K+ + K-'),
           'P PB --> K0S X': ('K0_S', 'K0S', 'K0S'),
           'P PB --> (P + PBAR) X': ('p + pbar', 'P PBAR', 'p + pbar'),
           'P PB --> (LAMBDA + LAMBDABAR) X': ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR', 'Lambda + Lambdabar'),
           'P PB --> (LAMBDA+LAMDBABAR) X': ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR', 'Lambda + Lambdabar')}
SPECTRA = {'pi+ + pi-': ('data1.yaml', 'data2.yaml'), 'K+ + K-': ('data3.yaml', 'data4.yaml'),
           'K0_S': ('data5.yaml', 'data6.yaml'), 'p + pbar': ('data7.yaml', 'data8.yaml'),
           'Lambda + Lambdabar': ('data9.yaml', 'data10.yaml')}
MEAN_PT = {'Lambda + Lambdabar': 'data17.yaml', 'p + pbar': 'data18.yaml', 'K0_S': 'data19.yaml',
           'K+ + K-': 'data20.yaml', 'pi+ + pi-': 'data21.yaml'}
YIELDS = {'pi+ + pi-': 'data23.yaml', 'K+ + K-': 'data24.yaml', 'K0_S': 'data25.yaml', 'p + pbar': 'data26.yaml',
          'Lambda + Lambdabar': 'data27.yaml'}
# pT-differential ratios: (files, numerator, denominator, label, scale)
RATIOS_PT = [(('data11.yaml', 'data12.yaml'), 'K+ + K-', 'pi+ + pi-', 'K/pi = (K+ + K-)/(pi+ + pi-)', None),
             (('data13.yaml', 'data14.yaml'), 'p + pbar', 'pi+ + pi-', 'p/pi = (p + pbar)/(pi+ + pi-)', None),
             (('data15.yaml', 'data16.yaml'), 'Lambda + Lambdabar', 'K0_S', 'Lambda/K0S = (Lambda + Lambdabar)/(2 K0S)',
              0.5)]
RATIOS_INT = [(0, 'K+ + K-', 'pi+ + pi-', 'K/pi = (K+ + K-)/(pi+ + pi-), pT-integrated'),
              (1, 'p + pbar', 'pi+ + pi-', 'p/pi = (p + pbar)/(pi+ + pi-), pT-integrated'),
              (2, 'Lambda + Lambdabar', 'pi+ + pi-', '2 Lambda/pi = (Lambda + Lambdabar)/(pi+ + pi-), pT-integrated')]
OUT = {'particle_spectra': 'data/particle_spectra/alice_particle_spectra_pi_K_K0S_p_Lambda_pPb_5020gev_ins1244523.yaml',
       'particle_yields': 'data/particle_yields/alice_particle_yields_pi_K_K0S_p_Lambda_pPb_5020gev_ins1244523.yaml',
       'mean_pT': 'data/mean_pT/alice_mean_pT_pi_K_K0S_p_Lambda_pPb_5020gev_ins1244523.yaml',
       'observable_ratios': 'data/observable_ratios/alice_observable_ratios_pid_pPb_5020gev_ins1244523.yaml'}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def quals(dv):
    return {q['name']: str(q['value']) for q in dv.get('qualifiers', [])}


def cls(text):
    lo, hi = (float(x) for x in text.split('TO'))
    return f'{lo:g}-{hi:g}%'


def system(species):
    return {'energy_gev': 5020.0, 'reaction': f'P PB --> {species} X', 'frame': 'nn_cms', 'rapidity_shift': SHIFT,
            'forward_beam': 'Pb'}


def note(text, uncorrelated=False):
    return {'value': f'{CITE}. {text}; 0 < y_cms < 0.5 (Pb-going), V0A classes. Errors: stat and the total '
                     'systematic' + (" (the second 'sys', its multiplicity-uncorrelated part, is contained in it: "
                                     "errors_excluded)" if uncorrelated else '') + '.'}


def main():
    out = {k: [] for k in OUT}
    refs = {}

    def add(otype, d, key):
        refs[key] = f'{OUT[otype]}#{len(out[otype])}'
        out[otype].append(d)

    for sp, files in SPECTRA.items():
        for hf in files:
            t = load(hf)
            iv = t['independent_variables'][0]['values']
            assert all(abs(a['high'] - b['low']) < 1e-9 for a, b in zip(iv, iv[1:])), hf
            edges = [float(iv[0]['low'])] + [float(v['high']) for v in iv]
            for i, dv in enumerate(t['dependent_variables']):
                q = quals(dv)
                assert SPECIES[q['RE']][0] == sp and q['YRAP(C=CM)'] == '0.0 TO 0.5', (hf, q)
                assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), hf
                c = cls(q['class'])
                add('particle_spectra', {
                    'observable_type': 'particle_spectra', 'inspire_id': '1244523', 'version': VERSION,
                    'hepdata_file': hf, 'dv_index': i,
                    'hepdata_table_label': f'{sp} invariant yield, V0A {c} ({hf})',
                    'collision_system': system(SPECIES[q['RE']][1]),
                    'centrality': {'classes': [c], **copy.deepcopy(V0A)}, 'particle_id': {'species': sp},
                    'kinematic_range': {'y': dict(Y), 'pT': {'min': edges[0], 'max': edges[-1], 'bins': edges}},
                    'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                    'notes': note(f'1/N_ev 1/(2 pi pT) d2N/(dpT dy) of {sp} (sum of particle and antiparticle), {c}')},
                    ('spectra', sp, c))
    for otype, tables, obs in (('particle_yields', YIELDS, 'dN_dy'), ('mean_pT', MEAN_PT, 'mean_pT')):
        for sp, hf in tables.items():
            t = load(hf)
            ivs = t['independent_variables']
            civ = next(iv for iv in ivs if iv['header']['name'] == 'EVENT CLASS')
            classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in civ['values']]
            assert classes == CLASSES, (hf, classes)
            dv = t['dependent_variables'][0]
            labels = [e['label'] for e in dv['values'][0]['errors']]
            what = 'dN/dy' if obs == 'dN_dy' else '<pT>'
            d = {'observable_type': otype, 'inspire_id': '1244523', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': 0, 'hepdata_table_label': f'{what} of {sp} vs V0A class ({hf})',
                 'collision_system': system(SPECIES[quals(dv)['RE']][1]),
                 'centrality': {'classes': list(CLASSES), **copy.deepcopy(V0A)}, 'particle_id': {'species': sp},
                 'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
                 'observable': obs, 'value_type': obs}
            if labels.count('sys') == 2:
                d['errors_excluded'] = ['sys#2']
            d['notes'] = note(f'{what} of {sp} (sum of particle and antiparticle), extrapolated to 0 and 10 GeV/c '
                              'with blast-wave fits', labels.count('sys') == 2)
            add(otype, d, (obs, sp))
    ratios = out['observable_ratios']
    for files, num, den, label, scale in RATIOS_PT:
        for hf in files:
            t = load(hf)
            iv = t['independent_variables'][0]['values']
            edges = [float(iv[0]['low'])] + [float(v['high']) for v in iv]
            assert all(abs(a['high'] - b['low']) < 1e-9 for a, b in zip(iv, iv[1:])), hf
            for i, dv in enumerate(t['dependent_variables']):
                c = cls(quals(dv)['class'])
                r = {'observable_type': 'observable_ratios', 'inspire_id': '1244523', 'version': VERSION,
                     'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{label}, V0A {c} ({hf})',
                     'collision_system': system('X'), 'centrality': {'classes': [c], **copy.deepcopy(V0A)},
                     'kinematic_range': {'y': dict(Y), 'pT': {'min': edges[0], 'max': edges[-1], 'bins': edges}},
                     'observable': 'invariant_yield', 'value_type': f'{label}, p+Pb 5.02 TeV',
                     'ratio_type': 'particle_ratio', 'numerator': num, 'denominator': den,
                     'numerator_particle': num, 'denominator_particle': den,
                     'numerator_system': 'p+Pb', 'denominator_system': 'p+Pb',
                     'numerator_record': refs[('spectra', num, c)], 'denominator_record': refs[('spectra', den, c)],
                     'errors_excluded': ['sys#2']}
                if scale:
                    r['scale'] = scale
                r['notes'] = note(f'{label} vs pT, {c}', True)
                ratios.append(r)
    t = load('data22.yaml')
    for i, num, den, label in RATIOS_INT:
        dv = t['dependent_variables'][i]
        ratios.append({'observable_type': 'observable_ratios', 'inspire_id': '1244523', 'version': VERSION,
                       'hepdata_file': 'data22.yaml', 'dv_index': i,
                       'hepdata_table_label': f'{label} vs V0A class (data22.yaml)',
                       'collision_system': system('X'), 'centrality': {'classes': list(CLASSES), **copy.deepcopy(V0A)},
                       'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
                       'observable': 'dN_dy', 'value_type': f'{label}, p+Pb 5.02 TeV',
                       'ratio_type': 'particle_ratio', 'numerator': num, 'denominator': den,
                       'numerator_particle': num, 'denominator_particle': den,
                       'numerator_system': 'p+Pb', 'denominator_system': 'p+Pb',
                       'numerator_record': refs[('dN_dy', num)], 'denominator_record': refs[('dN_dy', den)],
                       'errors_excluded': ['sys#2'],
                       'notes': note(f'{label} ({dv["header"]["name"]}) vs class, the ratio of the dN/dy', True)})
    for otype, path in OUT.items():
        with open(path, 'w') as fh:
            yaml.safe_dump_all(out[otype], fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(out[otype])} records')


if __name__ == '__main__':
    main()
