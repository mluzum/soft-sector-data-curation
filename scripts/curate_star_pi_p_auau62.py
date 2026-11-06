#!/usr/bin/env python3
"""Curate STAR pi+-, p and pbar spectra in Au+Au at 62.4 GeV, INSPIRE 747299.

B. I. Abelev et al. (STAR), "Energy dependence of pi+-, p and pbar
transverse momentum spectra for Au+Au collisions at sqrt(sNN) = 62.4 and
200 GeV", PLB 655 (2007) 104, arXiv:nucl-ex/0703040.

From the paper:
- |y| < 0.5, 0.2 < pT < 12 GeV/c (TPC dE/dx at low pT, TOF, and the
  relativistic rise for pT >= 2.5 GeV/c);
- centrality at 62.4 GeV from the uncorrected charged multiplicity in
  |eta| < 0.5: 0-10, 10-20, 20-40, 40-80%;
- pions corrected for weak-decay feed-down (K0_S, Lambda); p and pbar
  inclusive (no hyperon feed-down correction, up to ~40%); errors stat and
  syst;
- the 200 GeV spectra (0-12, 20-40, 40-80%) are STAR's earlier results
  (Ref. [21]), shown for comparison, with overlapping TOF and dE/dx bins.

Records (public): the 62.4 GeV spectra (16; each column's bins, which
HEPData interleaves with the 200 GeV binning), pi-/pi+ and pbar/p (0-10%)
and p/pi+, pbar/pi- (0-10, 40-80%) at 62.4 GeV, particle ratios linked to
the spectra.  Staged (--staged OUT): the 200 GeV comparison spectra and
ratios, R_CP and R_AA (Fig. 2), and the digitized minimum-bias spectra.

Run from the repository root:  python scripts/curate_star_pi_p_auau62.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins747299/HEPData-ins747299-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. I. Abelev et al. (STAR), PLB 655 (2007) 104, arXiv:nucl-ex/0703040'
OUT_S = 'data/particle_spectra/star_particle_spectra_pi_p_AuAu_62.4gev_ins747299.yaml'
OUT_R = 'data/observable_ratios/star_observable_ratios_pi_p_AuAu_62.4gev_ins747299.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': 'uncorrected charged particle multiplicity in |eta| < 0.5.'}
SPECIES = {'$\\pi^-$': ('pi-', 'PI-'), '$\\pi^+$': ('pi+', 'PI+'), 'anti-proton': ('pbar', 'PBAR'),
           'proton': ('p', 'P')}
FEED = {'pi-': ' Corrected for weak-decay feed-down.', 'pi+': ' Corrected for weak-decay feed-down.',
        'p': ' Inclusive (no hyperon feed-down correction).', 'pbar': ' Inclusive (no hyperon feed-down correction).'}
RATIO_NAMES = {'$\\pi^-/\\pi^+$': ('pi-', 'pi+'), '$\\bar{p}/p$': ('pbar', 'p'), 'proton/piplus': ('p', 'pi+'),
               'anti-proton/piminus': ('pbar', 'pi-')}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def filled(dv):
    return [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]


def stage(hf, i, label, reason, otype='particle_spectra'):
    return {'observable_type': otype, 'inspire_id': '747299', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{label} ({hf} column {i})', 'value_type': label,
            'notes': {'value': f'{CITE}.'}, 'staging_reason': reason}


def bins_of(t, rows):
    ivs = t['independent_variables'][0]['values']
    b = [[float(ivs[r]['low']), float(ivs[r]['high'])] for r in rows]
    assert all(x[1] <= y[0] + 1e-9 for x, y in zip(b, b[1:])), b
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, ratios, staged = [], [], []
    side = {}
    hf = 'fig1_new.yaml'
    t = load(hf)
    for i, dv in enumerate(t['dependent_variables']):
        q = dv['qualifiers'][0]['value']
        m = re.fullmatch(r'\$\\sqrt\{s_\{NN\}\}\$ = ([\d.]+) GeV, (.+) (\d+-\d+%)', q)
        energy, sp, cls = float(m.group(1)), m.group(2), m.group(3)
        species, prod = SPECIES[sp]
        if energy != 62.4:
            staged.append(stage(hf, i, f'{species} invariant yield, 200 GeV {cls}',
                                "the 200 GeV comparison spectra (STAR's earlier results; overlapping TOF and dE/dx "
                                'bins)'))
            continue
        rows = filled(dv)
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'syst')}
        bins = bins_of(t, rows)
        d = {'observable_type': 'particle_spectra', 'inspire_id': '747299', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': i, 'hepdata_table_label': f'Fig. 1: {species} invariant yield, 62.4 GeV {cls} ({hf} '
                                                  f'column {i})',
             'collision_system': {'energy_gev': 62.4, 'reaction': f'AU AU --> {prod} X'},
             'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': species},
             'kinematic_range': {'y': {'min': -0.5, 'max': 0.5},
                                 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
             'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'table_rows': rows,
             'notes': {'value': f'{CITE}, Fig. 1. 1/(2 pi pT) d2N/(dpT dy) of {species}, |y| < 0.5, Au+Au 62.4 GeV '
                                f'{cls}.{FEED[species]} Errors: stat and syst.'}}
        side[(species, cls)] = len(spectra)
        spectra.append(d)
    for hf in ('subfig1.yaml', 'fig4.yaml'):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            q = dv['qualifiers'][0]['value']
            m = re.fullmatch(r'\$\\sqrt\{s_\{NN\}\}\$ = ([\d.]+) GeV, (\d+-\d+%)', q)
            energy, cls = float(m.group(1)), m.group(2)
            num, den = RATIO_NAMES[dv['header']['name']]
            if energy != 62.4:
                staged.append(stage(hf, i, f'{num}/{den}, 200 GeV {cls}', 'a ratio of the 200 GeV comparison '
                                    'spectra', 'observable_ratios'))
                continue
            rows = filled(dv)
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'syst')}
            bins = bins_of(t, rows)
            d = {'observable_type': 'observable_ratios', 'inspire_id': '747299', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f'{num}/{den} vs pT, 62.4 GeV {cls} ({hf} column {i})',
                 'collision_system': {'energy_gev': 62.4, 'reaction': 'AU AU --> X X'},
                 'centrality': {'classes': [cls], **REFMULT},
                 'kinematic_range': {'y': {'min': -0.5, 'max': 0.5},
                                     'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                 'observable': 'invariant_yield', 'value_type': f'{num}/{den}', 'ratio_type': 'particle_ratio',
                 'numerator': num, 'denominator': den, 'numerator_particle': num, 'denominator_particle': den,
                 'numerator_system': 'Au+Au', 'denominator_system': 'Au+Au',
                 'numerator_record': f'{OUT_S}#{side[(num, cls)]}', 'denominator_record': f'{OUT_S}#{side[(den, cls)]}',
                 'table_rows': rows,
                 'notes': {'value': f"{CITE}, Fig. {1 if hf == 'subfig1.yaml' else 4}. {num}/{den} vs pT, |y| < 0.5, "
                                    f'Au+Au 62.4 GeV {cls}. Errors: stat and syst.'}}
            ratios.append(d)
    for hf, reason, otype in (('fig2.yaml', 'R_CP / R_AA: N_bin scaled, or needs the pp reference',
                               'nuclear_modification'),
                              ('fig1_mb.yaml', 'minimum-bias spectra digitized from the figure (a lone MB class)',
                               'particle_spectra')):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            staged.append(stage(hf, i, f"{dv['header']['name'][:40]}, {dv['qualifiers'][0]['value']}", reason, otype))
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
