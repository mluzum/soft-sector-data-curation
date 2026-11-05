#!/usr/bin/env python3
"""Curate ATLAS charged-hadron spectra in p+Pb, Pb+Pb and Xe+Xe, INSPIRE 2601282.

G. Aad et al. (ATLAS), "Charged-hadron production in pp, p+Pb, Pb+Pb, and
Xe+Xe collisions at sqrt(sNN) = 5 TeV with the ATLAS detector at the LHC",
JHEP 07 (2023) 074, arXiv:2211.15257.

From the paper (Secs. 1, 4-6, Table 2):
- primary charged hadrons; Pb+Pb at 5.02 TeV (2015) and Xe+Xe at 5.44 TeV
  (2017) in |eta| < 2.5, p+Pb at 5.02 TeV (2013) in -2.5 < y* < 2.0;
  pT > 0.5 GeV in Pb+Pb, > 0.3 GeV otherwise (first bin 0.31-0.407);
- HEPData also gives each class in |eta| slices (0.2 wide, 2.2-2.5 last)
  and y* slices (0.3 wide);
- centrality from the FCal E_T, 3.2 < |eta| < 4.9: both sides for Pb+Pb
  and Xe+Xe, the Pb-going side only for p+Pb; classes 0-5, 5-10, 10-20,
  20-30, 30-40, 40-50, 50-60, 60-80% (A+A) and 0-5, 5-10, 10-20, 20-30,
  30-40, 40-60, 60-90% (p+Pb);
- p+Pb: 4 TeV protons on 1.58 TeV/nucleon Pb, the nucleon-nucleon frame
  boosted by 0.465 in rapidity; results with the p-going direction toward
  positive rapidity; y* computed with the pion mass and corrected to the
  true rapidity with the generated masses (Sec. 6);
- the A+A and p+Pb spectra are tabulated as 1/N_evt 1/(2 pi pT) 1/<T>
  d2N/(dpT deta) (d y* for p+Pb) [mb/GeV^2]: per-event yields divided by
  the Glauber <T_AA> (<T_pPb>) of the class, Table 2;
- errors: stat and ten systematic components, one of them 'syst, TAA',
  the uncertainty of <T> (Table 3).

Records (public): per-event invariant yields.  value_scale = 1/<T> [mb]
multiplies the tabulated values back by <T> (an exact inverse of the
published normalization), and errors_excluded drops 'syst, TAA': the
per-event yield does not carry the uncertainty of <T>.  |eta| slices of
the symmetric systems are computed at eta > 0.  p+Pb: frame nn_cms, the
proton toward +y, the FCal window in that frame, y the true rapidity (as
INSPIRE 1463284: on 3D containers a charged rapidity window has no single
mass and gives NaN; 2D containers compute it).  Every pT bin is kept
(0.31-300 GeV); the soft sector is the first ~15.

Staged (--staged-out <collection root>): the pp cross sections (no
centrality classes), the p+Pb 0-90% spectra (Table 2 gives no <T_pPb> for
0-90%; the 0-90% spectra of the same 2013 data are public from INSPIRE
1463284), and R_pPb, R_AA vs pT, y* and |eta| (nuclear modification: a
ratio to the pp cross section with <T>, a high-pT observable, not
computed).  The 378 earlier staged records called the tables invariant
yields of every system and had no <T>.

Run from the repository root:
    python scripts/curate_atlas_charged_spectra_ppb_pbpb_xexe_5tev.py [--staged-out <collection root>]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2601282/HEPData-ins2601282-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'G. Aad et al. (ATLAS), JHEP 07 (2023) 074, arXiv:2211.15257'
SHIFT = 0.465
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
FCAL_PB = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
           'eta': [{'min': round(-4.9 - SHIFT, 3), 'max': round(-3.2 - SHIFT, 3)}],
           'notes': 'FCal transverse energy, -4.9 < eta_lab < -3.2 (the Pb-going side), here in the nn_cms frame.'}
# Table 2: <T_AA> [mb^-1] (Pb+Pb, Xe+Xe) and <T_pPb> [mub^-1 / 1000 = mb^-1]
T = {'PB PB': {'0-5': 26.0, '5-10': 20.4, '10-20': 14.4, '20-30': 8.77, '30-40': 5.09, '40-50': 2.75,
               '50-60': 1.35, '60-80': 0.42},
     'XE XE': {'0-5': 13.9, '5-10': 10.9, '10-20': 7.54, '20-30': 4.48, '30-40': 2.55, '40-50': 1.37,
               '50-60': 0.69, '60-80': 0.23},
     'P PB': {'0-5': 0.222, '5-10': 0.194, '10-20': 0.172, '20-30': 0.148, '30-40': 0.126, '40-60': 0.092,
              '60-90': 0.043}}
SYSTEMS = {'PB PB': ('Pb+Pb', 5020.0), 'XE XE': ('Xe+Xe', 5440.0), 'P PB': ('p+Pb', 5020.0),
           'P P': ('pp', None)}
STAGED_PP = 'p+p cross section: no centrality classes (the pipeline needs them)'
STAGED_PPB_MB = ('p+Pb 0-90%: Table 2 gives no <T_pPb> for 0-90%, so the per-event yield cannot be recovered; '
                 'the 0-90% spectra of the same 2013 data are public from INSPIRE 1463284')
STAGED_R = ('nuclear modification factor (the spectrum over <T> times the pp cross section): a high-pT '
            'observable, not computed')


def interval(text):
    m = re.fullmatch(r'\s*(-?[\d.]+)\s*-\s*(-?[\d.]+)\s*', str(text))
    return float(m.group(1)), float(m.group(2))


def cls_label(text):
    lo, hi = interval(text)
    return f'{lo:g}-{hi:g}'


def pt_axis(iv):
    vals = iv['values']
    assert all(abs(a['high'] - b['low']) < 1e-9 for a, b in zip(vals, vals[1:]))
    edges = [float(vals[0]['low'])] + [float(v['high']) for v in vals]
    return {'min': edges[0], 'max': edges[-1], 'bins': edges}


def window(q, beams):
    """(kinematic_range entry, text) of the table's eta or y* window."""
    if 'Y*' in q:
        lo, hi = interval(q['Y*'])
        return {'y': {'min': lo, 'max': hi}}, f'{lo:g} < y* < {hi:g}'
    lo, hi = interval(q['ABS(ETA)'])
    if lo == 0.0 and hi == 2.5:
        return {'eta': {'min': -2.5, 'max': 2.5}}, '|eta| < 2.5'
    return {'eta': {'min': lo, 'max': hi}}, f'{lo:g} < |eta| < {hi:g} (a symmetric system: computed at eta > 0)'


def collision_system(beams, energy):
    cs = {'energy_gev': energy, 'reaction': f'{beams} --> CHARGED X'}
    if beams == 'P PB':
        cs.update(frame='nn_cms', rapidity_shift=SHIFT, forward_beam='p')
    return cs


def tables():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    for x in sub:
        t = yaml.safe_load(open(os.path.join(RAW, x['data_file'])))
        if not t.get('dependent_variables') or t['dependent_variables'][0]['header']['name'] == 'Page':
            continue                                  # the overview page
        dv = t['dependent_variables'][0]
        assert len(t['dependent_variables']) == 1, x['data_file']
        q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
        assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), x['data_file']
        yield x['data_file'], ' '.join(str(x.get('description', '')).split()), t, dv, q


def spectrum(hf, t, q, beams, cls):
    name, energy = SYSTEMS[beams]
    kr, where = window(q, beams)
    taa = T[beams][cls]
    tname = '<T_pPb>' if beams == 'P PB' else '<T_AA>'
    return {
        'observable_type': 'particle_spectra', 'inspire_id': '2601282', 'version': VERSION, 'hepdata_file': hf,
        'dv_index': 0, 'hepdata_table_label': f'charged-hadron invariant yield, {cls}% {name} at {energy / 1000:g} '
                                              f'TeV, {where.split(" (")[0]} ({hf})',
        'collision_system': collision_system(beams, energy),
        'centrality': {'classes': [f'{cls}%'], **(FCAL_PB if beams == 'P PB' else FCAL)},
        'particle_id': {'species': 'charged'},
        'kinematic_range': {**kr, 'pT': pt_axis(t['independent_variables'][0])},
        'observable': 'invariant_yield', 'value_type': 'invariant_yield',
        'value_scale': round(1.0 / taa, 8),
        'errors_excluded': ['syst, TAA'],
        'notes': {'value': (
            f'{CITE}. 1/N_evt 1/(2 pi pT) d2N/(dpT d{"y*" if beams == "P PB" else "eta"}) of charged hadrons, '
            f'{cls}% {name}, {where}; tabulated divided by {tname} = {taa:g} mb^-1 (Table 2), multiplied back by '
            f'value_scale = 1/{tname}. '
            + ('nn_cms frame, the proton toward +y; y* the true rapidity (computed with the pion mass, corrected '
               'with the generated masses). ' if beams == 'P PB' else '')
            + "Errors: stat and the systematic components, without 'syst, TAA' (the uncertainty of "
              f'{tname}, which the per-event yield does not carry).')},
    }


def staged_record(hf, desc, t, dv, q, beams, cls, reason):
    name, energy = SYSTEMS[beams]
    energy = energy or float(q['SQRT(S)'])
    iv = t['independent_variables'][0]
    ivn = iv['header']['name']
    kr = {}
    if ivn == 'PT':
        kr.update(window(q, beams)[0])
        kr['pT'] = pt_axis(iv)
    else:                                             # R vs y* or |eta| at one pT bin
        lo, hi = interval(q['PT'])
        kr['pT'] = {'min': lo, 'max': hi}
        pairs = [[float(v['low']), float(v['high'])] for v in iv['values']]
        kr['y' if ivn == 'Y*' else 'eta'] = {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}
    is_r = dv['header']['name'] in ('RAA', 'RPPB')
    d = {'observable_type': 'nuclear_modification' if is_r else 'particle_spectra', 'inspire_id': '2601282',
         'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
         'hepdata_table_label': f'{desc} ({hf})',
         'collision_system': collision_system(beams, energy),
         'centrality': {'classes': [f'{cls}%'] if cls else ['MB'],
                        **(FCAL_PB if beams == 'P PB' else FCAL if beams != 'P P' else {})},
         'particle_id': {'species': 'charged'},
         'kinematic_range': kr,
         'value_type': dv['header']['name'],
         'notes': {'value': f'{CITE}. {desc}'},
         'staging_reason': reason}
    if not d['centrality'].get('estimator'):
        d['centrality'].pop('estimator', None)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged-out', help='collection root for the staged records (data/<type>/ below it)')
    args = ap.parse_args()
    public, staged = [], {'particle_spectra': [], 'nuclear_modification': []}
    for hf, desc, t, dv, q in tables():
        beams = q['RE'].split(' -->')[0].strip()
        cls = cls_label(q['CENTRALITY']) if 'CENTRALITY' in q else None
        header = dv['header']['name']
        if header in ('RAA', 'RPPB'):
            staged['nuclear_modification'].append(staged_record(hf, desc, t, dv, q, beams, cls, STAGED_R))
        elif beams == 'P P':
            staged['particle_spectra'].append(staged_record(hf, desc, t, dv, q, beams, None, STAGED_PP))
        elif beams == 'P PB' and cls == '0-90':
            staged['particle_spectra'].append(staged_record(hf, desc, t, dv, q, beams, cls, STAGED_PPB_MB))
        else:
            assert 'D2(N)' in header and dv['header']['units'] == 'MB/GeV2', (hf, header)
            assert float(q['SQRT(S)/NUCLEON']) == SYSTEMS[beams][1], (hf, q)
            labels = {e['label'] for v in dv['values'] for e in v['errors']}
            assert 'syst, TAA' in labels and 'stat' in labels, (hf, labels)
            public.append(spectrum(hf, t, q, beams, cls))
    out = 'data/particle_spectra/atlas_particle_spectra_charged_pPb_PbPb_XeXe_5tev_ins2601282.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if not args.staged_out:
        return
    for otype, docs in staged.items():
        path = os.path.join(args.staged_out, 'data', otype, f'atlas_{otype}_charged_pp_pPb_PbPb_XeXe_5tev_ins2601282.yaml')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} staged records')


if __name__ == '__main__':
    main()
