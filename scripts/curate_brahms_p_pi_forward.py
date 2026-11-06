#!/usr/bin/env python3
"""Curate BRAHMS p/pi+ and pbar/pi- at forward pseudorapidity in Au+Au at 200 and 62.4 GeV, INSPIRE 834310.

I. G. Arsene et al. (BRAHMS), "Rapidity dependence of the proton-to-pion
ratio in Au+Au and p+p collisions at sqrt(sNN) = 62.4 and 200 GeV", PLB 684
(2010) 22, arXiv:0910.3328.

From the paper:
- p/pi+ and pbar/pi- vs pT measured with the movable spectrometers in
  pseudorapidity windows (the analysis is done in eta vs pT);
- centrality from the charged-particle multiplicity in -2.2 < eta < 2.2,
  classes 0-10, 10-20, 20-40, 40-80%;
- no weak-decay correction: vertex cuts limit the feed-down, leaving (from
  AMPT) a Lambda/K0_S contamination that raises p/pi+ by 6-20% (pbar/pi- by
  5-14%), most at low pT and mid-rapidity;
- errors stat and sys at 200 GeV (Tables 1, 2), stat at 62.4 GeV (Table 4).

Records (public): the ratios at 200 GeV (p/pi+ and pbar/pi-, six eta
windows from 2.17-2.38 to 3.70-3.90, four classes: 48) and at 62.4 GeV
(p/pi+ at eta = 2.67, 3.2, 3.5, taken in windows of 0.2 around them: 6),
particle_ratio records linked to side records that carry no values
(experimental_values: false; BRAHMS publishes only the ratios), one per
energy, eta window and species.  HEPData's 62.4 GeV bins overlap by up to
1 MeV at a few edges (rounded edges): the upper edge of a bin is kept and
the next bin starts there.  Staged (--staged OUT): p+p (no centrality),
Fig. 3 (a repeat of Table 1a at 0-10%, and 62.4 GeV at eta = 0 without its
class), Fig. 5 (0-10% at an eta not stated).

Run from the repository root:  python scripts/curate_brahms_p_pi_forward.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins834310/HEPData-ins834310-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'I. G. Arsene et al. (BRAHMS), PLB 684 (2010) 22, arXiv:0910.3328'
OUT_R = 'data/observable_ratios/brahms_observable_ratios_p_pi_forward_AuAu_62.4_200gev_ins834310.yaml'
OUT_S = 'data/particle_spectra/brahms_particle_spectra_p_pi_sides_AuAu_62.4_200gev_ins834310.yaml'
MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'SiMA+TMA'},
      'eta': [{'min': -2.2, 'max': 2.2}], 'notes': 'charged-particle multiplicity in -2.2 < eta < 2.2.'}
CLASSES = ['0-10%', '10-20%', '20-40%', '40-80%']
PAIRS = {'P/PI+': ('p', 'pi+', 'P', 'PI+'), 'PBAR/PI-': ('pbar', 'pi-', 'PBAR', 'PI-')}
FEED = ('Not corrected for weak decays: vertex cuts leave (AMPT) a Lambda/K0_S contamination raising p/pi+ by '
        '6-20% and pbar/pi- by 5-14%, most at low pT.')


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def cls_of(s):
    lo, hi = (int(x) for x in re.fullmatch(r'(\d+) TO (\d+)', s).groups())
    return f'{lo}-{hi}%'


def bins_of(t):
    """The table's pT bins; a bin starting up to 1 MeV below the previous upper edge starts there."""
    b = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
    for x, y in zip(b, b[1:]):
        assert y[0] > x[1] - 0.0012, (x, y)
        y[0] = max(y[0], x[1])
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sides, ratios, staged = [], [], []
    side_index = {}

    def side(energy, species, prod, eta, bins, hf):
        key = (energy, species, eta, tuple(map(tuple, bins)))
        if key not in side_index:
            side_index[key] = len(sides)
            sides.append({
                'observable_type': 'particle_spectra', 'inspire_id': '834310', 'version': VERSION,
                'experimental_values': False,
                'hepdata_table_label': f'{species} invariant yield, {eta[0]:g} < eta < {eta[1]:g}, Au+Au {energy:g} '
                                       f'GeV (the computation of the {hf} ratio; no values published)',
                'collision_system': {'energy_gev': energy, 'reaction': f'AU AU --> {prod} X'},
                'centrality': {'classes': list(CLASSES), **MA}, 'particle_id': {'species': species},
                'kinematic_range': {'eta': {'min': eta[0], 'max': eta[1]},
                                    'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                'notes': {'value': f'{CITE}. {species} spectrum at {eta[0]:g} < eta < {eta[1]:g}, Au+Au {energy:g} GeV: '
                                   'the side of the published p/pi ratios (BRAHMS gives only the ratios; the '
                                   f'ratio records hold the values). {FEED}'}})
        return side_index[key]

    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = load(hf)
        pp = '\\mathrm{p}\\mathrm{p}' in desc
        energy = 62.4 if '62.4' in desc else 200.0
        if pp or hf.startswith(('tab3', 'tab5')):
            for i, dv in enumerate(t['dependent_variables']):
                staged.append({'observable_type': 'observable_ratios', 'inspire_id': '834310', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc.strip()}'},
                               'staging_reason': ('p+p: no centrality classes' if pp else
                                                  'Fig. 3: a repeat of Table 1a (0-10%), or 62.4 GeV at eta = 0 '
                                                  'without its class' if hf.startswith('tab3') else
                                                  'Fig. 5: 0-10%, the eta of the table not stated')})
            continue
        bins = bins_of(t)
        for i, dv in enumerate(t['dependent_variables']):
            q = {k['name']: k['value'] for k in dv['qualifiers']}
            name = dv['header']['name']
            num, den, pn, pd = PAIRS[name]
            cls = cls_of(q['CENTRALITY'])
            e = q['ETARAP']
            if isinstance(e, str):
                lo, hi = (float(x) for x in e.split('-'))
                ewin, etxt = (lo, hi), f'{lo:g} < eta < {hi:g}'
            else:
                ewin, etxt = (round(e - 0.1, 4), round(e + 0.1, 4)), f'eta = {e:g} (here {e - 0.1:g} < eta < ' \
                                                                     f'{e + 0.1:g})'
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
            labels = {tuple(x['label'] for x in dv['values'][r]['errors']) for r in rows}
            assert labels == ({('stat', 'sys')} if hf.startswith(('tab1', 'tab2')) else {('stat',)}), (hf, labels)
            rb = [bins[r] for r in rows]
            n_ref = side(energy, num, pn, ewin, bins, hf)
            d_ref = side(energy, den, pd, ewin, bins, hf)
            d = {'observable_type': 'observable_ratios', 'inspire_id': '834310', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f'{num}/{den} vs pT, {etxt}, Au+Au {energy:g} GeV {cls} ({hf} column {i})',
                 'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> X X'},
                 'centrality': {'classes': [cls], **MA},
                 'kinematic_range': {'eta': {'min': ewin[0], 'max': ewin[1]},
                                     'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}},
                 'observable': 'invariant_yield', 'value_type': f'{num}/{den}', 'ratio_type': 'particle_ratio',
                 'numerator': num, 'denominator': den, 'numerator_particle': num, 'denominator_particle': den,
                 'numerator_system': 'Au+Au', 'denominator_system': 'Au+Au',
                 'numerator_record': f'{OUT_S}#{n_ref}', 'denominator_record': f'{OUT_S}#{d_ref}',
                 'notes': {'value': f'{CITE}, Table {hf[3]}. {num}/{den} vs pT, {etxt}, Au+Au {energy:g} GeV {cls}. '
                                    f'{FEED} Errors: '
                                    + ('stat and sys.' if hf.startswith(('tab1', 'tab2')) else 'stat.')}}
            if rows != list(range(len(bins))):
                d['table_rows'] = rows
            ratios.append(d)
    for out, docs in ((OUT_S, sides), (OUT_R, ratios)):
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
