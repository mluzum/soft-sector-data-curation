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
- dn/deta(eta), 0.25-wide eta bins, every centrality column, pT > 270 MeV
  (0.27-5 GeV) and extrapolated (0-5 GeV);
- dn/deta at |eta| < 0.5 vs centrality, and <pT> at |eta| < 0.5 vs
  centrality, both thresholds;
- pT spectra at |eta| < 0.5, 1.5 < |eta| < 2.25, |y_pi| < 0.5 and 1.5 <
  |y_pi| < 2.25 (kinematic_range.y.mass 0.13957); the two-sided forward
  windows are computed on the positive side (symmetric systems, as
  1093733's mirrored v_n(|eta|));
- the Ne+Ne / O+O ratios of all of these (system ratios linking the two
  systems' records).
Staged (--staged OUT): dn/dy_pi (a rapidity with an assumed mass is
implemented for spectra only), <pT>(eta) and <pT>(y_pi) per class (the
<pT> core is integrated only), their Ne+Ne / O+O ratios, the spectra
ratios to 70-80% and mid / forward (ratios within one system: no
computation).

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

    # dn/deta(eta) per class, and dn/dy_pi(y_pi) (staged)
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
                              'dimensionality': 'differential_eta' if var == 'deta' else None,
                              'notes': note('Figs. 13, 17', f'Per-event {what} of primary charged particles vs '
                                            f'{"eta" if var == "deta" else "y_pi (pion mass)"}, {c}, '
                                            f'{SYS[system][0]} 5.36 TeV, {PT_TXT[thr]}')})
                    if var == 'deta':
                        add('particle_yields', d, ('dndeta', system, thr, c))
                    else:
                        d.pop('dimensionality')
                        d['staging_reason'] = ('dn/dy_pi: a rapidity window with an assumed mass is implemented for '
                                               'spectra only')
                        staged.append(d)
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
                if ypi:
                    d['staging_reason'] = ('a rapidity window with an assumed mass (y_pi) is implemented for '
                                           'spectra only')
                    staged.append(d)
                else:
                    add(otype, d, ('centdep', what, system, thr))
    # <pT>(eta), <pT>(y_pi) per class: staged
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
                              'notes': note('Figs. 14, 17', f'<pT> vs {axis}, {c}, {SYS[system][0]}, {PT_TXT[thr]}'),
                              'staging_reason': '<pT> vs eta or y_pi: the <pT> core is integrated only'})
                    staged.append(d)
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
                add('particle_spectra', d, ('spectra', kind, system, c))
    # Ne+Ne / O+O ratios
    ratios = out['observable_ratios']
    for thr in PT:
        hf = f'dN_deta_NeNe_to_OO_ratio_pT{thr}MeV.yaml'
        for i, dv in enumerate(load(hf)['dependent_variables']):
            c = quals(dv)['Centrality']
            num = out['particle_yields'][int(refs[('dndeta', 'NeNe', thr, c)].split('#')[1])]
            ratios.append(ratio_of(num, None, refs[('dndeta', 'NeNe', thr, c)], refs[('dndeta', 'OO', thr, c)], hf, i,
                                   f'dn/deta Ne+Ne/O+O, {c}, pT > {thr} MeV',
                                   {'dimensionality': 'differential_eta',
                                    'notes': note('Fig. 15', f'dn/deta(eta) in Ne+Ne over O+O, {c}, {PT_TXT[thr]}')}))
        for what, otype in (('dN_deta', 'particle_yields'), ('MeanpT_vs_eta', 'mean_pT')):
            hf = f'{what}_NeNe_to_OO_ratio_centdep_midrapidity_pT{thr}MeV.yaml'
            num_ref = refs[('centdep', what, 'NeNe', thr)]
            num = out[otype][int(num_ref.split('#')[1])]
            lab = ('<pT>' if otype == 'mean_pT' else 'dn/deta') + f' at |eta| < 0.5, Ne+Ne/O+O, pT > {thr} MeV'
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
    # ratios staged: dn/dy_pi, <pT>(eta, y_pi), spectra to 70-80%, mid / forward
    for hf in sorted(os.listdir(RAW)):
        if not hf.endswith('.yaml') or hf == 'submission.yaml':
            continue
        within = 'cent_to_70_80' in hf or 'mid_eta_to_forward' in hf or 'mid_ypi_to_forward' in hf
        other = ('NeNe_to_OO' in hf and not hf.startswith('pT_spectra')
                 and ('ypi' in hf or ('MeanpT' in hf and 'centdep' not in hf)))
        if not (within or other):
            continue
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            q = quals(dv)
            staged.append({'observable_type': 'observable_ratios', 'inspire_id': '3170344', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{dv['header']['name']}, {q.get('Centrality', '')} ({hf})",
                           'collision_system': {'energy_gev': 5360.0,
                                                'reaction': 'Ne+Ne and O+O' if 'NeNe_to_OO' in hf else
                                                SYS['NeNe' if '_NeNe' in hf else 'OO'][1]},
                           'observable': dv['header']['name'], 'value_type': dv['header']['name'],
                           'ratio_type': 'quantity_ratio' if within else 'system_ratio',
                           'numerator': ('Ne+Ne' if not within else
                                         ('the centrality class' if 'cent_to_70_80' in hf else 'mid-rapidity')),
                           'denominator': ('O+O' if not within else
                                           ('70-80%' if 'cent_to_70_80' in hf else 'forward rapidity')),
                           'notes': note('Figs. 12, 16', f"{dv['header']['name']}, {q}"),
                           'staging_reason': ('a ratio within one system (to 70-80%, or mid- to forward '
                                              'rapidity): no computation' if within else
                                              'its sides are staged (y_pi yields, <pT> vs eta or y_pi)')})
    for otype, path in OUT.items():
        with open(path, 'w') as fh:
            yaml.safe_dump_all(out[otype], fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(out[otype])} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} staged records')


if __name__ == '__main__':
    main()
