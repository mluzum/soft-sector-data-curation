#!/usr/bin/env python3
"""Curate the STAR isobar multiplicity measurements (ins1914564, arXiv:2109.00131).

- Fig. 2 (left): P(N_trk^offline) in Ru+Ru and Zr+Zr, minimum bias
  -> multiplicity_distribution (detector level);
- Fig. 2 (upper right): their Ru+Ru / Zr+Zr ratio -> observable_ratios;
- Fig. 3 (lower): the <N_trk^offline> ratio vs centrality -> observable_ratios,
  linked to the per-system <N_trk^offline> records; STAR applies no
  efficiency correction "because the effects of inefficiency cancel out in
  these ratios" (Sec. II), hence efficiency_cancels: true.

The Glauber-model curves of Fig. 2 (Case-2, Case-3) are model output and
are not curated.  N_trk^offline: TPC tracks with |eta|<0.5, pT>0.2 GeV/c,
DCA<3 cm and >=10 fit points, corrected for luminosity and vertex position,
not for the 85-90% tracking efficiency (Secs. II-III, Fig. 2 caption).

Run from the repository root:  python scripts/curate_isobar_1914564_multiplicity.py
"""

import glob
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = glob.glob(os.path.join(ROOT, 'hepdata_raw/records/ins1914564/HEPData-*'))[0]
NTRK = ('N_trk^offline: reconstructed TPC tracks with |eta|<0.5, pT>0.2 GeV/c, DCA<3 cm and '
        '>=10 fit points; corrected for luminosity and vertex position, not for the 85-90% '
        'tracking efficiency (arXiv:2109.00131, Secs. II-III and Fig. 2 caption)')
KR = {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.2, 'max': 100.0}}
CENT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
        'eta': [{'min': -0.5, 'max': 0.5}]}
SYSTEM = {'ruru': ('Ru+Ru', 'RuRu'), 'zrzr': ('Zr+Zr', 'ZrZr')}


def label(hf):
    for d in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))):
        if d and d.get('data_file') == hf:
            return ' '.join(str(d.get('description', '')).split())
    raise KeyError(hf)


def common(otype, hf, reaction):
    return {'observable_type': otype, 'inspire_id': '1914564', 'version': 1, 'hepdata_file': hf,
            'hepdata_table_label': label(hf),
            'collision_system': {'energy_gev': 200.0, 'reaction': reaction}}


def write(path, doc):
    with open(os.path.join(ROOT, path), 'w') as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False, allow_unicode=True, width=120)
    print('wrote', path)


def main():
    dist = {}
    for key, part in (('ruru', 'left_top'), ('zrzr', 'left_low')):
        reaction, tag = SYSTEM[key]
        hf = f'fig2_{part}_isobarpaper_star_grey_data_{key}_nonzeros.yaml'
        doc = common('multiplicity_distribution', hf, reaction)
        doc.update(particle_id={'species': 'charged'}, kinematic_range=KR,
                   observable='multiplicity_distribution', value_type='P(N_trk^offline)',
                   efficiency_corrected=False, measured_quantity=NTRK,
                   event_selection='minimum bias (0-80% trigger-efficiency corrected via Glauber), as published',
                   dimensionality='differential_multiplicity',
                   notes={'value': 'Fig. 2 (gray circles). Glauber Case-2/Case-3 fits not curated.'})
        dist[key] = f'data/multiplicity_distribution/star_multiplicity_distribution_isobar_Ntrk_{tag}_200gev.yaml'
        os.makedirs(os.path.join(ROOT, 'data/multiplicity_distribution'), exist_ok=True)
        write(dist[key], doc)

    hf = 'fig2_right_isobarpaper_star_grey_data_nonzero.yaml'
    doc = common('observable_ratios', hf, 'Ru+Ru and Zr+Zr')
    doc.update(kinematic_range=KR, observable='multiplicity_distribution',
               value_type='P(N_trk^offline)_Ru_over_Zr', numerator='Ru+Ru', denominator='Zr+Zr',
               ratio_type='system_ratio', numerator_system='Ru+Ru', denominator_system='Zr+Zr',
               numerator_record=dist['ruru'], denominator_record=dist['zrzr'], efficiency_cancels=False,
               notes='Ratio of the minimum-bias N_trk^offline distributions (Fig. 2 upper right), in bins of '
                     'N_trk^offline; detector level -- the efficiency does not cancel at fixed track count.')
    write('data/observable_ratios/star_observable_ratios_isobar_multiplicity_distribution_Ru_over_Zr_200gev.yaml', doc)

    hf = 'fig3_olow_isobarpaper_star_blue_mean_multiplicity_ratio.yaml'
    doc = common('observable_ratios', hf, 'Ru+Ru and Zr+Zr')
    doc.update(centrality=dict(classes=['70-80%', '60-70%', '50-60%', '40-50%', '30-40%', '20-30%',
                                        '10-20%', '5-10%', '0-5%'], **CENT),
               particle_id={'species': 'charged'}, kinematic_range=KR, observable='Nch',
               value_type='Nch_Ru_over_Zr', numerator='Ru+Ru', denominator='Zr+Zr',
               ratio_type='system_ratio', numerator_system='Ru+Ru', denominator_system='Zr+Zr',
               numerator_record='data/particle_yields/star_particle_yields_isobar_mean_Nch_RuRu_200gev.yaml',
               denominator_record='data/particle_yields/star_particle_yields_isobar_mean_Nch_ZrZr_200gev.yaml',
               efficiency_cancels=True,
               notes='Ratio of <N_trk^offline> vs centrality (Fig. 3 lower panel); table listed '
                     'peripheral-first. ' + NTRK)
    write('data/observable_ratios/star_observable_ratios_isobar_mean_Nch_Ru_over_Zr_200gev.yaml', doc)


if __name__ == '__main__':
    main()
