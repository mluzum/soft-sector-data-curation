#!/usr/bin/env python3
"""Curate PHENIX pi0 spectra in Cu+Au at 200 GeV and U+U at 192 GeV.

- INSPIRE 1672859: A. Adare et al. (PHENIX), "Production of pi0 and eta
  mesons in Cu+Au collisions at sqrt(sNN) = 200 GeV", PRC 98, 054903
  (2018), arXiv:1805.04389.  Tables Pi0<class>Spectra (Fig. 3a; 60-90% is
  supplemental): 0-10, 10-20, 0-20, 20-40, 40-60, 60-90%.
- INSPIRE 1798526: U. A. Acharya et al. (PHENIX), "Production of pi0 and
  eta mesons in U+U collisions at sqrt(sNN) = 192 GeV", PRC 102, 064905
  (2020), arXiv:2005.14686.  Tables fig2a_spect_UU_pi0_<class> (Fig. 2a):
  0-20, 20-40, 40-60, 60-80%.

From the papers: pi0 -> gamma gamma in the EMCal, |eta| < 0.35; invariant
yields 1/N_evt d2N/(2 pi pT dpT dy); centrality from the BBC charge,
3.0 < |eta| < 3.9, with |z_BBC| < 20 cm.  Errors: type A (statistical and
uncorrelated systematic), B (correlated in pT) and C (normalization).
The records use the rapidity window |y| < 0.35 (y ~ eta for pT > 1 GeV/c)
and pT bins from the tables (1-20 GeV/c; the soft sector covers the first
few bins).  pi0 is computed by isospin, (pi+ + pi-)/2 (analysis.species).

Staged (--staged-out): the minimum-bias pi0 spectra (Cu+Au: the MB trigger
sees 93% of the inelastic cross section; a trigger-bias correction is not
stated) and the eta-meson spectra (eta is not provided by the simulations).
Not curated: eta/pi0 ratios, R_AB / R_AA, PbSc/PbGl and data/fit ratios,
N_part tables.

Run from the repository root:
  python scripts/curate_phenix_pi0_cuau_uu.py [--staged-out <collection root>]
"""
import argparse
import glob
import os

import yaml

BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
CUAU = ('1672859', 'A. Adare et al. (PHENIX), PRC 98, 054903 (2018), arXiv:1805.04389',
        200.0, 'CU AU', 'Cu+Au')
UU = ('1798526', 'U. A. Acharya et al. (PHENIX), PRC 102, 064905 (2020), arXiv:2005.14686',
      192.0, 'U U', 'U+U')


def table(iid, name):
    d = glob.glob(f'hepdata_raw/records/ins{iid}/HEPData-ins{iid}-v*-yaml')[0]
    return yaml.safe_load(open(os.path.join(d, name)))


def pt_axis(t):
    pairs = [[float(x['low']), float(x['high'])] for x in t['independent_variables'][0]['values']]
    ax = {'min': pairs[0][0], 'max': pairs[-1][1]}
    if all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs, pairs[1:])):
        ax['bins'] = [pairs[0][0]] + [p[1] for p in pairs]
    else:
        ax['bin_ranges'] = pairs
    return ax


def record(paper, hf, species, meson, cls, extra_note=''):
    iid, cite, energy, beams, system = paper
    t = table(iid, hf)
    label = 'minimum bias' if cls is None else cls
    d = {'observable_type': 'particle_spectra', 'inspire_id': iid, 'version': 1, 'hepdata_file': hf,
         'dv_index': 0, 'hepdata_table_label': f'{meson} invariant yield, {label} {system} at {energy:g} GeV',
         'collision_system': {'energy_gev': energy, 'reaction': f'{beams} --> {meson.upper()} X'},
         'centrality': {'classes': [cls] if cls else [], **BBC},
         'kinematic_range': {'y': {'min': -0.35, 'max': 0.35}, 'pT': pt_axis(t)},
         'particle_id': {'species': species},
         'observable': 'invariant_yield', 'value_type': 'invariant_yield',
         'notes': {'value': (f'{cite}. {meson} -> gamma gamma in the EMCal (|eta| < 0.35), invariant yield per unit '
                             'rapidity (|y| < 0.35 used; y ~ eta above 1 GeV/c). Centrality from the BBC charge '
                             '(3.0 < |eta| < 3.9). Errors: type A, B and C.' + extra_note)}}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staged-out', help='collection root for the staged records (data/<type>/ below it)')
    args = ap.parse_args()
    pi0_note = ' pi0 computed by isospin, (pi+ + pi-)/2.'
    cuau = [record(CUAU, f'Pi0{c}Spectra.yaml', 'pi0', 'pi0', f'{c}%',
                   pi0_note + (' 60-90% is supplemental to Fig. 3a.' if c == '60-90' else ''))
            for c in ('0-10', '10-20', '0-20', '20-40', '40-60', '60-90')]
    uu = [record(UU, f'fig2a_spect_UU_pi0_{c}.yaml', 'pi0', 'pi0', f'{c[:-2]}-{c[-2:]}%' if c != '020' else '0-20%',
                 pi0_note) for c in ('020', '2040', '4060', '6080')]
    public = {'data/particle_spectra/phenix_particle_spectra_pi0_CuAu_200gev_ins1672859.yaml': cuau,
              'data/particle_spectra/phenix_particle_spectra_pi0_UU_192gev_ins1798526.yaml': uu}
    for out, docs in public.items():
        for d in docs:
            t = table(d['inspire_id'], d['hepdata_file'])
            q = t['dependent_variables'][0]
            assert len(q['values']) == len(t['independent_variables'][0]['values'])
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if not args.staged_out:
        return
    eta_reason = 'species eta (meson) is not provided by the simulations'
    mb_reason = 'minimum bias: no percentile centrality class (trigger-bias correction not stated)'
    staged = {
        'phenix_particle_spectra_pi0_eta_CuAu_200gev_ins1672859.yaml':
            [dict(record(CUAU, 'Pi0MinBiasSpectra.yaml', 'pi0', 'pi0', None,
                         pi0_note + ' The MB trigger sees 93% of the inelastic cross section.'),
                  staging_reason=mb_reason)]
            + [dict(record(CUAU, f'Eta{c}Spectra.yaml', 'eta', 'eta', None if c == 'MinBias' else f'{c}%'),
                    staging_reason=eta_reason) for c in ('MinBias', '0-20', '20-40', '40-60', '60-90')],
        'phenix_particle_spectra_pi0_eta_UU_192gev_ins1798526.yaml':
            [dict(record(UU, 'fig2a_spect_UU_pi0_MinBias.yaml', 'pi0', 'pi0', None, pi0_note), staging_reason=mb_reason)]
            + [dict(record(UU, f'fig2b_spect_UU_eta_{c}.yaml', 'eta', 'eta',
                           None if c == 'MinBias' else ('0-20%' if c == '020' else f'{c[:-2]}-{c[-2:]}%')),
                    staging_reason=eta_reason) for c in ('MinBias', '020', '2040', '4060', '6080')],
    }
    os.makedirs(os.path.join(args.staged_out, 'data/particle_spectra'), exist_ok=True)
    for name, docs in staged.items():
        out = os.path.join(args.staged_out, 'data/particle_spectra', name)
        for d in docs:
            if not d['centrality']['classes']:
                d['centrality']['classes'] = ['MB']
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} staged records')


if __name__ == '__main__':
    main()
