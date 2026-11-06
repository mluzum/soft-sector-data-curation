#!/usr/bin/env python3
"""Curate STAR pi, K, p spectra, yields and <pT> in Cu+Cu at 62.4 and 200 GeV, INSPIRE 865572.

M. M. Aggarwal et al. (STAR), "Scaling properties at freeze-out in
relativistic heavy ion collisions", PRC 83 (2011) 034910,
arXiv:1008.3133.

From the paper:
- Cu+Cu at 200 and 62.4 GeV, 0-60% in six 10% classes (STAR's charged
  reference multiplicity, |eta| < 0.5);
- pi+-, K+-, p and pbar from TPC dE/dx at |y| < 0.1: 1/(2 pi pT)
  d2N/(dy dpT) (Fig. 3, 4); pions corrected for weak-decay feed-down,
  muons and background; (anti)protons inclusive (not corrected for weak
  decays, ~30% feed-down); stat and syst (<= 7%);
- dN/dy and <pT> (extrapolated with Blast-wave and Bose-Einstein fits;
  Figs. 6, 7), yield ratios (Fig. 8), and dNch/deta at mid-rapidity
  (Appendix I: the identified yields, Jacobian-corrected and corrected for
  the neutral strange feed-down), which the summary tables use as their
  row label.

Records (public): spectra (72; 0.05 GeV/c bins around the points), dN/dy
and <pT> (24; rows of each energy via table_rows), dNch/deta (2), and the
ratios pbar/pi-, K-/pi-, p/pi+, K+/pi+, pi-/pi+, K-/K+, pbar/p (14),
particle_ratio records linked to the dN/dy records.  Staged (--staged
OUT): (p + pbar)/pi and K/pi summed over charges (Fig. 8c), enhancement
factors (relative to pp per N_part), Blast-wave and statistical-model
parameters.

Run from the repository root:  python scripts/curate_star_pikp_cucu.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins865572/HEPData-ins865572-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'M. M. Aggarwal et al. (STAR), PRC 83 (2011) 034910, arXiv:1008.3133'
OUT_S = 'data/particle_spectra/star_particle_spectra_pi_K_p_CuCu_62.4_200gev_ins865572.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_pi_K_p_CuCu_62.4_200gev_ins865572.yaml'
OUT_M = 'data/mean_pT/star_mean_pT_pi_K_p_CuCu_62.4_200gev_ins865572.yaml'
OUT_R = 'data/observable_ratios/star_observable_ratios_pi_K_p_CuCu_62.4_200gev_ins865572.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': "10% intervals of the geometric cross section; STAR's charged reference multiplicity, |eta| < 0.5."}
CLASSES = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%']
SPECIES = {'pi^-': ('pi-', 'PI-'), 'k^-': ('K-', 'K-'), 'p^-': ('pbar', 'PBAR'),
           'pi^+': ('pi+', 'PI+'), 'k^+': ('K+', 'K+'), 'p^+': ('p', 'P')}
FEED = {'pi-': ' Corrected for weak-decay feed-down, muons and background pions.',
        'pi+': ' Corrected for weak-decay feed-down, muons and background pions.',
        'p': ' Inclusive (not corrected for weak decays, ~30% feed-down).',
        'pbar': ' Inclusive (not corrected for weak decays, ~30% feed-down).'}
ENERGY_ROWS = {200.0: list(range(0, 6)), 62.4: list(range(6, 12))}
DNCH = {200.0: [170, 119, 82.9, 56.9, 38.7, 25.3], 62.4: [117, 81.2, 56.4, 38.7, 26.3, 17.4]}
AXIS = {'centrality': "HEPData's row axis (named Centrality, or dNch/deta) holds the class's dNch/deta (Appendix "
                      'I), 200 GeV rows then 62.4 GeV rows; the classes are the six 10% classes'}
RATIOS = {'pbar/pi^-': ('pbar', 'pi-'), 'k^-/pi^-': ('K-', 'pi-'), 'p/pi^+': ('p', 'pi+'), 'k^+/pi^+': ('K+', 'pi+'),
          'pi^-/pi^+': ('pi-', 'pi+'), 'k^-/k^+': ('K-', 'K+'), 'pbar/p': ('pbar', 'p')}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def quals(dv):
    return {k['name']: k['value'] for k in dv.get('qualifiers', [])}


def filled(dv):
    return [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]


def system(energy, prod):
    return {'energy_gev': energy, 'reaction': f'CU CU --> {prod} X'}


def head(otype, hf, i, label, energy, prod, classes, species):
    return {'observable_type': otype, 'inspire_id': '865572', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf} column {i})', 'collision_system': system(energy, prod),
            'centrality': {'classes': classes, **REFMULT}, 'particle_id': {'species': species}}


def errs(dv, rows):
    labels = {tuple(e.get('label') for e in dv['values'][r]['errors']) for r in rows}
    assert labels <= {('stat', 'syst'), ('stat', 'stat')}, labels
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, means, ratios, staged = [], [], [], [], []
    for hf in sorted(f for f in os.listdir(RAW) if f[:6] in ('data_3', 'data_4') and len(f) == 12):
        t = load(hf)
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        assert all(abs(b - a - 0.05) < 1e-6 for a, b in zip(pts, pts[1:])), (hf, pts)
        bins = [[round(p - 0.025, 4), round(p + 0.025, 4)] for p in pts]
        for i, dv in enumerate(t['dependent_variables']):
            q = quals(dv)
            species, prod = SPECIES[q['Channel']]
            energy = float(q['SQRT(SNN)'])
            cls = dv['header']['name']
            assert cls in CLASSES and q['System'] == 'CU CU' and filled(dv) == list(range(len(pts))), (hf, i)
            errs(dv, range(len(pts)))
            d = head('particle_spectra', hf, i, f'{species} invariant yield, Cu+Cu {energy:g} GeV, {cls}', energy,
                     prod, [cls], species)
            d['kinematic_range'] = {'y': {'min': -0.1, 'max': 0.1},
                                    'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
            d.update(observable='invariant_yield', value_type='invariant_yield')
            d['notes'] = {'value': f'{CITE}, Figs. 3, 4. 1/(2 pi pT) d2N/(dy dpT) of {species}, |y| < 0.1, Cu+Cu '
                                   f'{energy:g} GeV {cls}.{FEED.get(species, "")} pT bins of 0.05 GeV/c around the '
                                   'tabulated points. Errors: stat and syst.'}
            spectra.append(d)
    side = {}
    for hf, otype in (('data_61.yaml', 'mean_pT'), ('data_62.yaml', 'mean_pT'),
                      ('data_71.yaml', 'particle_yields'), ('data_72.yaml', 'particle_yields')):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            q = quals(dv)
            species, prod = SPECIES[q['Channel']]
            energy = float(q['SQRT(SNN)'])
            rows = filled(dv)
            assert rows == ENERGY_ROWS[energy] and [float(t['independent_variables'][0]['values'][r]['value'])
                                                    for r in rows] == DNCH[energy], (hf, i)
            errs(dv, rows)
            mean = otype == 'mean_pT'
            d = head(otype, hf, i, f"{'<pT>' if mean else 'dN/dy'} of {species} vs class, Cu+Cu {energy:g} GeV",
                     energy, prod, list(CLASSES), species)
            d['kinematic_range'] = {'y': {'min': -0.1, 'max': 0.1}, 'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='mean_pT' if mean else 'dN_dy', value_type='mean_pT' if mean else 'dN_dy',
                     table_rows=rows, hepdata_corrections=dict(AXIS))
            d['notes'] = {'value': f"{CITE}, Fig. {6 if mean else 7}. {'<pT>' if mean else 'dN/dy'} of {species} at "
                                   f'|y| < 0.1, extrapolated to all pT (Blast-wave and Bose-Einstein fits), Cu+Cu '
                                   f'{energy:g} GeV vs class.{FEED.get(species, "")} Errors: stat and syst.'}
            if not mean:
                side[(species, energy)] = (OUT_Y, len(yields))
            (means if mean else yields).append(d)
    t = load('data_apxI.yaml')
    assert [v['value'] for v in t['independent_variables'][0]['values']] == [c[:-1] for c in CLASSES]
    for i, dv in enumerate(t['dependent_variables']):
        energy = float(quals(dv)['SQRT(SNN)'])
        assert [v['value'] for v in dv['values']] == DNCH[energy]
        d = head('particle_yields', 'data_apxI.yaml', i, f'dNch/deta vs class, Cu+Cu {energy:g} GeV', energy,
                 'CHARGED', list(CLASSES), 'charged')
        d['kinematic_range'] = {'eta': {'min': -0.1, 'max': 0.1}, 'pT': {'min': 0.0, 'max': 100.0}}
        d.update(observable='dNch_deta', value_type='dNch_deta')
        d['notes'] = {'value': f'{CITE}, Appendix I. dNch/deta at mid-rapidity (the pi, K, p yields at |y| < 0.1, '
                               'extrapolated, Jacobian-corrected and corrected for the weak decays of neutral strange '
                               f'particles: primordial charged hadrons), Cu+Cu {energy:g} GeV vs class. Errors: syst.'}
        yields.append(d)
    for hf in ('data_81.yaml', 'data_82.yaml', 'data_83.yaml', 'data_84.yaml'):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            q = quals(dv)
            energy = float(q['SQRT(SNN)'])
            rows = filled(dv)
            if name not in RATIOS:
                staged.append({'observable_type': 'observable_ratios', 'inspire_id': '865572', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f'{name}, Cu+Cu {energy:g} GeV ({hf} column {i})',
                               'value_type': name, 'notes': {'value': f'{CITE}. Fig. 8c.'},
                               'staging_reason': 'a ratio of charge-summed yields: the separate ratios are public'})
                continue
            assert rows == ENERGY_ROWS[energy], (hf, i)
            num, den = RATIOS[name]
            n_ref, d_ref = side[(num, energy)], side[(den, energy)]
            d = head('observable_ratios', hf, i, f'{num}/{den} vs class, Cu+Cu {energy:g} GeV', energy, 'X',
                     list(CLASSES), None)
            d.pop('particle_id')
            d['kinematic_range'] = {'y': {'min': -0.1, 'max': 0.1}, 'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='dN_dy', value_type=name, ratio_type='particle_ratio', numerator=num,
                     denominator=den, numerator_particle=num, denominator_particle=den,
                     numerator_system='Cu+Cu', denominator_system='Cu+Cu',
                     numerator_record=f'{n_ref[0]}#{n_ref[1]}', denominator_record=f'{d_ref[0]}#{d_ref[1]}',
                     table_rows=rows, hepdata_corrections=dict(AXIS))
            d['notes'] = {'value': f'{CITE}, Fig. 8. {num}/{den}, the ratio of the extrapolated dN/dy at |y| < 0.1, '
                                   f'Cu+Cu {energy:g} GeV vs class. Errors: one component (unlabelled).'}
            ratios.append(d)
    for hf in ('data_9.yaml', 'data_101.yaml', 'data_102.yaml', 'data_13.yaml', 'data_141.yaml'):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '865572', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                           'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}.'},
                           'staging_reason': 'enhancement factor relative to pp per N_part' if hf == 'data_9.yaml'
                           else 'Blast-wave or statistical-model fit parameter'})
    for out, docs in ((OUT_S, spectra), (OUT_Y, yields), (OUT_M, means), (OUT_R, ratios)):
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
