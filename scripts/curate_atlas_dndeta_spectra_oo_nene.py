#!/usr/bin/env python3
"""Curate ATLAS dn/deta, <pT> and pT spectra in O+O and Ne+Ne at 5.36 TeV, INSPIRE 3170344.

ATLAS Collaboration, "Measurements of charged-particle pseudorapidity and
transverse momentum distributions in O+O and Ne+Ne collisions at sqrt(sNN)
= 5.36 TeV with the ATLAS detector", arXiv:2606.20257.

From the paper (Secs. 1, 3, 4):
- centrality from the FCal Sigma E_T, 3.2 < |eta| < 4.9, Glauber fit;
  1% classes to 5%, 5% classes to 80% (and wider ones in the tables);
- primary charged particles (mean lifetime > 0.3e-10 s), per event, in the
  fiducial range |eta| < 2.5, 0.27 < pT < 5 GeV, and extrapolated to
  0 < pT < 5 GeV with fits to the spectra;
- y_pi: rapidity with the charged-pion mass for every particle;
- spectra (1/pT) d2n/(dpT deta) (no 1/(2 pi): value_scale 2 pi gives the
  invariant yield) at |eta| < 0.5 and 1.5 < |eta| < 2.25, and in y_pi;
- errors: statistical and systematic.

Public:
- dn/deta(eta) and dn/dy_pi(y_pi), 0.25-wide bins, every centrality
  column, pT > 270 MeV (0.27-5 GeV) and extrapolated (0-5 GeV);
- <pT>(eta) and <pT>(y_pi) per class (dimensionality differential_eta,
  differential_y), both thresholds;
- dn/deta, dn/dy_pi and <pT> at |eta| < 0.5 or |y_pi| < 0.5 vs centrality,
  both thresholds;
- pT spectra at |eta| < 0.5, 1.5 < |eta| < 2.25, |y_pi| < 0.5 and
  1.5 < |y_pi| < 2.25 (kinematic_range.y.mass 0.13957); the two-sided
  forward windows are computed on the positive side (symmetric systems, as
  1093733's mirrored v_n(|eta|));
- the Ne+Ne / O+O ratios of all of these (system ratios linking the two
  systems' records);
- the spectra in each class over the 70-80% spectrum (ratio_type
  centrality_ratio, denominator_class 70-80%) and at |eta| < 0.5 over
  1.5 < |eta| < 2.25 (|y_pi| likewise; rapidity_ratio), each spectrum
  first divided by its pT-integrated yield (Sec. 6.2): side_normalization
  pT 0.27-5 GeV, the fiducial range, although the spectra start at 0.21
  GeV.  Checked against the tables: the spectra with that normalization
  reproduce the tabulated ratios to 70-80% within 0.1-0.4%; ATLAS's
  normalization is its measured fiducial dn/deta (dn/dy_pi) in the window,
  which the spectra's integrals match within 0.5% (mid, forward eta) and
  3% (forward y_pi, within the 2% + 2% systematic errors), so the mid /
  forward ratios differ from the spectra's own by up to 1% (eta).  The
  model computes both sides' yields itself.
The HEPData spectra at 1.5 < |y_pi| < 2.25 are 1.5 x ATLAS's (staged until
November 2026): ATLAS's mid / forward y_pi ratio, the mid spectra and the
dn/dy_pi tables give HEPData / ATLAS = 1.500 in every class, bin and system
(0.9998-1.0002 x 1.5; the forward eta spectra 1.000), and their ratio to the
1.5 < |eta| < 2.25 spectra tends to 1.500 at high pT, where y_pi -> eta (a
rapidity-width normalization: 1.5 is the two-sided window's width).
value_scale 3 pi = 2 pi x 1.5 (hepdata_corrections); the Ne+Ne / O+O
ratio, in which the factor cancels, is the tabulated one.  Nothing is staged.

Run from the repository root:  python scripts/curate_atlas_dndeta_spectra_oo_nene.py [--staged OUT]
"""
import argparse
import copy
import glob
import math
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins3170344/HEPData-ins3170344-v*-yaml')[0]
VERSION = int(re.search(r'-v(\d+)-yaml', RAW).group(1))
CITE = 'ATLAS Collaboration, arXiv:2606.20257'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
SYS = {'OO': ('O+O', 'O O --> CHARGED X'), 'NeNe': ('Ne+Ne', 'NE NE --> CHARGED X')}
PT = {'270': {'min': 0.27, 'max': 5.0}, '0': {'min': 0.0, 'max': 5.0}}
PT_TXT = {'270': '0.27 < pT < 5 GeV (fiducial)', '0': '0 < pT < 5 GeV (extrapolated with fits to the spectra)'}
PION = 0.13957
WINDOWS = {'mid_eta': ('eta', 0.0, 0.5, '|eta| < 0.5'), 'forward_eta': ('eta', 1.5, 2.25, '1.5 < |eta| < 2.25'),
           'mid_ypi': ('y', 0.0, 0.5, '|y_pi| < 0.5'), 'forward_ypi': ('y', 1.5, 2.25, '1.5 < |y_pi| < 2.25')}
FWD_YPI = ('HEPData tabulates the spectra at 1.5 < |y_pi| < 2.25 as 1.5 x ATLAS\'s: ATLAS\'s mid / forward y_pi '
           'ratio table with the mid spectra and the dn/dy_pi tables gives 1.500 (0.9998-1.0002 x 1.5) in every class, '
           'bin and system (forward eta: 1.000), and their ratio to the 1.5 < |eta| < 2.25 spectra tends to 1.500 at '
           'high pT where y_pi -> eta; value_scale 3 pi = 2 pi x 1.5')
NORM = {'pT': {'min': 0.27, 'max': 5.0}}         # Sec. 6.2: spectra over their pT-integrated (fiducial) yield
OUT = {'particle_yields': 'data/particle_yields/atlas_particle_yields_dNch_deta_OO_NeNe_5360gev_ins3170344.yaml',
       'mean_pT': 'data/mean_pT/atlas_mean_pT_OO_NeNe_5360gev_ins3170344.yaml',
       'particle_spectra': 'data/particle_spectra/atlas_particle_spectra_charged_OO_NeNe_5360gev_ins3170344.yaml',
       'observable_ratios': 'data/observable_ratios/atlas_observable_ratios_NeNe_over_OO_5360gev_ins3170344.yaml'}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def quals(dv):
    return {q['name']: str(q['value']) for q in dv.get('qualifiers', [])}


def cent(c):
    return {'classes': [c] if isinstance(c, str) else c, **copy.deepcopy(FCAL)}


def window(kind):
    var, lo, hi, txt = WINDOWS[kind]
    lo_s = 0.0 if lo == 0.0 else lo
    w = {'min': -hi if lo == 0.0 else lo_s, 'max': hi}
    if var == 'y':
        w['mass'] = PION
    return var, w, txt + ('' if lo == 0.0 else ', computed at the positive side (symmetric system)')


def base(otype, hf, i, system, label):
    return {'observable_type': otype, 'inspire_id': '3170344', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf})',
            'collision_system': {'energy_gev': 5360.0, 'reaction': SYS[system][1]},
            'particle_id': {'species': 'charged'}}


def note(fig, text):
    return {'value': f'{CITE}, {fig}. {text}. Centrality from the FCal Sigma E_T (3.2 < |eta| < 4.9). '
                     'Errors: stat and sys.'}


def ratio_of(num, den, num_ref, den_ref, hf, i, label, extra):
    r = {'observable_type': 'observable_ratios', 'inspire_id': '3170344', 'version': VERSION, 'hepdata_file': hf,
         'dv_index': i, 'hepdata_table_label': f'{label} ({hf})',
         'collision_system': {'energy_gev': 5360.0, 'reaction': 'Ne+Ne and O+O'},
         'centrality': copy.deepcopy(num['centrality']), 'particle_id': {'species': 'charged'},
         'kinematic_range': copy.deepcopy(num['kinematic_range']),
         'observable': num.get('observable'), 'value_type': label, 'ratio_type': 'system_ratio',
         'numerator': 'Ne+Ne', 'denominator': 'O+O', 'numerator_system': 'Ne+Ne', 'denominator_system': 'O+O',
         'numerator_record': num_ref, 'denominator_record': den_ref}
    r.update(extra)
    return r


def within(num_ref, den_ref, out, hf, i, system, c, label, ratio_type, numerator, denominator, extra, notes):
    """A ratio of spectrum shapes within one system: both sides divided by
    their pT-integrated yields (side_normalization), the pT bins of the sides."""
    num = out['particle_spectra'][int(num_ref.split('#')[1])]
    name = SYS[system][0]
    return {'observable_type': 'observable_ratios', 'inspire_id': '3170344', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{label} ({hf})',
            'collision_system': {'energy_gev': 5360.0, 'reaction': SYS[system][1]},
            'centrality': cent(c), 'particle_id': {'species': 'charged'},
            'kinematic_range': {'pT': copy.deepcopy(num['kinematic_range']['pT'])},
            'observable': 'invariant_yield', 'value_type': label, 'ratio_type': ratio_type,
            'numerator': numerator, 'denominator': denominator, 'numerator_system': name, 'denominator_system': name,
            'numerator_record': num_ref, 'denominator_record': den_ref, **extra,
            'side_normalization': copy.deepcopy(NORM), 'notes': notes}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    out = {k: [] for k in OUT}
    staged = []
    refs = {}                                     # (kind, system, thr/window, column) -> 'path#index'

    def add(otype, d, key):
        refs[key] = f'{OUT[otype]}#{len(out[otype])}'
        out[otype].append(d)

    # dn/deta(eta) and dn/dy_pi(y_pi) per class
    for var in ('deta', 'dypi'):
        for system in SYS:
            for thr in PT:
                hf = f'dN_{var}_{system}_pT{thr}MeV.yaml'
                t = load(hf)
                iv = t['independent_variables'][0]['values']
                bins = [[float(v['low']), float(v['high'])] for v in iv]
                for i, dv in enumerate(t['dependent_variables']):
                    q = quals(dv)
                    c = q['Centrality']
                    what = 'dn/deta' if var == 'deta' else 'dn/dy_pi'
                    d = base('particle_yields', hf, i, system, f'{what}, {c}, {SYS[system][0]}, pT > {thr} MeV')
                    axis = 'eta' if var == 'deta' else 'y'
                    kr = {axis: {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}, 'pT': dict(PT[thr])}
                    if axis == 'y':
                        kr['y']['mass'] = PION
                    d.update({'centrality': cent(c), 'kinematic_range': kr,
                              'observable': 'dNch_deta' if var == 'deta' else 'dNch_dy',
                              'value_type': 'dNch_deta' if var == 'deta' else 'dNch_dy',
                              'dimensionality': 'differential_eta' if var == 'deta' else 'differential_y',
                              'notes': note('Figs. 13, 17', f'Per-event {what} of primary charged particles vs '
                                            f'{"eta" if var == "deta" else "y_pi (pion mass)"}, {c}, '
                                            f'{SYS[system][0]} 5.36 TeV, {PT_TXT[thr]}')})
                    add('particle_yields', d, (var, system, thr, c))
    # dn/deta, dn/dy_pi and <pT> at midrapidity vs centrality
    for what in ('dN_deta', 'dN_dypi', 'MeanpT_vs_eta', 'MeanpT_vs_ypi'):
        for system in SYS:
            for thr in PT:
                hf = f'{what}_{system}_centdep_midrapidity_pT{thr}MeV.yaml'
                t = load(hf)
                iv = t['independent_variables'][0]['values']
                classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv]
                dv = t['dependent_variables'][0]
                mean = what.startswith('Mean')
                otype = 'mean_pT' if mean else 'particle_yields'
                ypi = what.endswith('ypi')
                label = (('<pT>' if mean else ('dn/dy_pi' if ypi else 'dn/deta')) +
                         f' at |{"y_pi" if ypi else "eta"}| < 0.5 vs centrality, {SYS[system][0]}, pT > {thr} MeV')
                d = base(otype, hf, 0, system, label)
                kr = ({'y': {'min': -0.5, 'max': 0.5, 'mass': PION}} if ypi else {'eta': {'min': -0.5, 'max': 0.5}})
                kr['pT'] = dict(PT[thr])
                obs = 'mean_pT' if mean else ('dNch_dy' if ypi else 'dNch_deta')
                d.update({'centrality': cent(classes), 'kinematic_range': kr, 'observable': obs, 'value_type': obs,
                          'notes': note('Figs. 18, 19', f'{label}; {PT_TXT[thr]}')})
                add(otype, d, ('centdep', what, system, thr))
    # <pT>(eta), <pT>(y_pi) per class
    for what in ('MeanpT_vs_eta', 'MeanpT_vs_ypi'):
        for system in SYS:
            for thr in PT:
                hf = f'{what}_{system}_pT{thr}MeV.yaml'
                t = load(hf)
                iv = t['independent_variables'][0]['values']
                bins = [[float(v['low']), float(v['high'])] for v in iv]
                axis = 'eta' if what.endswith('eta') else 'y'
                for i, dv in enumerate(t['dependent_variables']):
                    c = quals(dv)['Centrality']
                    d = base('mean_pT', hf, i, system, f'<pT> vs {axis}, {c}, {SYS[system][0]}, pT > {thr} MeV')
                    kr = {axis: {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}, 'pT': dict(PT[thr])}
                    if axis == 'y':
                        kr['y']['mass'] = PION
                    d.update({'centrality': cent(c), 'kinematic_range': kr, 'observable': 'mean_pT',
                              'value_type': 'mean_pT', 'dimensionality': f'differential_{axis}',
                              'notes': note('Figs. 14, 17', f'<pT> vs {axis}, {c}, {SYS[system][0]}, {PT_TXT[thr]}')})
                    add('mean_pT', d, (what, system, thr, c))
    # spectra
    for kind in WINDOWS:
        for system in SYS:
            hf = f'pT_spectra_{kind}_{system}.yaml'
            t = load(hf)
            iv = t['independent_variables'][0]['values']
            bins = [[float(v['low']), float(v['high'])] for v in iv]
            var, w, wtxt = window(kind)
            for i, dv in enumerate(t['dependent_variables']):
                c = quals(dv)['Centrality']
                d = base('particle_spectra', hf, i, system, f'(1/pT) d2n/dpT d{var}, {WINDOWS[kind][3]}, {c}, '
                                                            f'{SYS[system][0]}')
                d.update({'centrality': cent(c),
                          'kinematic_range': {var: dict(w), 'pT': {'min': bins[0][0], 'max': bins[-1][1],
                                                                  'bin_ranges': bins}},
                          'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                          'value_scale': 2 * math.pi,
                          'notes': note('Figs. 10, 11', f'Per-event (1/pT) d2n/(dpT d{"eta" if var == "eta" else "y_pi"})'
                                        f' of primary charged particles, {wtxt}, {c}, {SYS[system][0]} 5.36 TeV; '
                                        'value_scale 2 pi: the table has no 1/(2 pi)')})
                if kind == 'forward_ypi':
                    d['value_scale'] = 3 * math.pi
                    d['hepdata_corrections'] = {'values': FWD_YPI}
                    d['notes']['value'] = d['notes']['value'].replace(
                        'value_scale 2 pi: the table has no 1/(2 pi)',
                        'value_scale 3 pi: the table has no 1/(2 pi) and is 1.5 x the spectrum (hepdata_corrections)')
                add('particle_spectra', d, ('spectra', kind, system, c))
    # Ne+Ne / O+O ratios
    ratios = out['observable_ratios']
    names = {'dN_deta': 'dn/deta', 'dN_dypi': 'dn/dy_pi', 'MeanpT_vs_eta': '<pT>', 'MeanpT_vs_ypi': '<pT>'}
    for thr in PT:
        # per class: dn/deta(eta), dn/dy_pi(y_pi), <pT>(eta), <pT>(y_pi)
        for what, otype, key, dim in (('dN_deta', 'particle_yields', 'deta', 'differential_eta'),
                                      ('dN_dypi', 'particle_yields', 'dypi', 'differential_y'),
                                      ('MeanpT_vs_eta', 'mean_pT', 'MeanpT_vs_eta', 'differential_eta'),
                                      ('MeanpT_vs_ypi', 'mean_pT', 'MeanpT_vs_ypi', 'differential_y')):
            hf = f'{what}_NeNe_to_OO_ratio_pT{thr}MeV.yaml'
            axis = 'eta' if dim.endswith('eta') else 'y_pi'
            for i, dv in enumerate(load(hf)['dependent_variables']):
                c = quals(dv)['Centrality']
                num_ref = refs[(key, 'NeNe', thr, c)]
                num = out[otype][int(num_ref.split('#')[1])]
                lab = f'{names[what]}({axis}) Ne+Ne/O+O, {c}, pT > {thr} MeV'
                ratios.append(ratio_of(num, None, num_ref, refs[(key, 'OO', thr, c)], hf, i, lab,
                                       {'dimensionality': dim,
                                        'notes': note('Figs. 15, 17', f'{names[what]}({axis}) in Ne+Ne over O+O, {c}, '
                                                      f'{PT_TXT[thr]}')}))
        # at mid-rapidity vs centrality
        for what in ('dN_deta', 'dN_dypi', 'MeanpT_vs_eta', 'MeanpT_vs_ypi'):
            otype = 'mean_pT' if what.startswith('Mean') else 'particle_yields'
            hf = f'{what}_NeNe_to_OO_ratio_centdep_midrapidity_pT{thr}MeV.yaml'
            num_ref = refs[('centdep', what, 'NeNe', thr)]
            num = out[otype][int(num_ref.split('#')[1])]
            lab = f'{names[what]} at |{"y_pi" if what.endswith("ypi") else "eta"}| < 0.5, Ne+Ne/O+O, pT > {thr} MeV'
            ratios.append(ratio_of(num, None, num_ref, refs[('centdep', what, 'OO', thr)], hf, 0, lab,
                                   {'notes': note('Figs. 18, 19', f'{lab} vs centrality; {PT_TXT[thr]}')}))
    for kind in WINDOWS:
        hf = f'pT_spectra_{kind}_NeNe_to_OO_ratio.yaml'
        for i, dv in enumerate(load(hf)['dependent_variables']):
            c = quals(dv)['Centrality']
            num_ref = refs[('spectra', kind, 'NeNe', c)]
            num = out['particle_spectra'][int(num_ref.split('#')[1])]
            lab = f'pT spectrum Ne+Ne/O+O, {WINDOWS[kind][3]}, {c}'
            r = ratio_of(num, None, num_ref, refs[('spectra', kind, 'OO', c)], hf, i, lab,
                         {'notes': note('Fig. 12', f'{lab} ({window(kind)[2]})')})
            ratios.append(r)
    # within one system: spectrum shapes over 70-80%, and mid over forward rapidity
    for kind in WINDOWS:
        for system in SYS:
            hf = f'pT_spectra_{kind}_{system}_cent_to_70_80_ratio.yaml'
            for i, dv in enumerate(load(hf)['dependent_variables']):
                c = quals(dv)['Centrality']
                lab = f'pT spectrum shape {c} / 70-80%, {WINDOWS[kind][3]}, {SYS[system][0]}'
                ratios.append(within(refs[('spectra', kind, system, c)], refs[('spectra', kind, system, '70-80%')],
                                     out, hf, i, system, c, lab, 'centrality_ratio', c, '70-80%',
                                     {'denominator_class': '70-80%'},
                                     note('Figs. 13, 33-36', f'{lab}: each spectrum over its fiducial yield '
                                          f'(0.27 < pT < 5 GeV) in {window(kind)[2]}')))
    for var in ('eta', 'ypi'):
        for system in SYS:
            hf = f'pT_spectra_mid_{var}_to_forward_{var}_{system}_ratio.yaml'
            mid, fwd = WINDOWS[f'mid_{var}'][3], WINDOWS[f'forward_{var}'][3]
            for i, dv in enumerate(load(hf)['dependent_variables']):
                c = quals(dv)['Centrality']
                lab = f'pT spectrum shape {mid} / {fwd}, {c}, {SYS[system][0]}'
                ratios.append(within(refs[('spectra', f'mid_{var}', system, c)],
                                     refs[('spectra', f'forward_{var}', system, c)],
                                     out, hf, i, system, c, lab, 'rapidity_ratio', mid, fwd, {},
                                     note('Figs. 14, 37-40', f'{lab}: each spectrum over its fiducial yield '
                                          f'(0.27 < pT < 5 GeV) in its window (the forward one computed at the '
                                          f'positive side)')))
    for otype, path in OUT.items():
        with open(path, 'w') as fh:
            yaml.safe_dump_all(out[otype], fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(out[otype])} records')
    if staged_out and staged:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} staged records')


if __name__ == '__main__':
    main()
