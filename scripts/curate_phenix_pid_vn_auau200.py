#!/usr/bin/env python3
"""Curate PHENIX v_n of identified hadrons in Au+Au at 200 GeV, INSPIRE 1332239.

A. Adare et al. (PHENIX), "Measurement of the higher-order anisotropic flow
coefficients for identified hadrons in Au+Au collisions at sqrt(sNN) = 200
GeV", PRC 93, 051902(R) (2016), arXiv:1412.1038.

From the paper (data taking, experimental technique):
- pi+-, K+-, p + pbar (charge signs combined), |eta| < 0.35 (drift and pad
  chambers, TOF identification); centrality from the BBC (3.0 < |eta| <
  3.9);
- event plane (EP): Psi_m from the south and north RXN (1.0 < |eta| < 2.8,
  scintillators: every particle), v_n{Psi_n}, resolution from sub-events;
- long-range two-particle correlations (2PC): v_n = <v_n^HAD v_n^RXN> /
  sqrt(<v_n^RXN-N v_n^RXN-S>), the same reference;
- "the results for wider centrality ranges are obtained by averaging
  across several smaller ranges, weighted by the multiplicity of the
  selected particle": 0-50% and 30-50% (here from the 10% classes);
- errors: statistical, pT-uncorrelated and pT-correlated systematic.

Records: v2, v3, v4 of each species, EP (Figs. 1-2, appendix: 0-50%,
0-10%, 30-50%, 10% classes to 60%) and 2PC (Fig. 1, 0-50%), pT points
(0.2 GeV/c windows; the two highest EP points sit in wider bins).  EP is
computed with the scalar-product estimator, 2PC as the scalar product with
the RXN reference.  The 0-50% and 30-50% records (multiplicity-weighted
averages of 10% classes: centrality.combined_from analysis_classes) are
written to --staged OUT, not to data/.  Not curated: v4{Psi_2}
(mixed-harmonic, no computation) and the blast-wave fit parameters (Fig. 4).

Run from the repository root:
    python scripts/curate_phenix_pid_vn_auau200.py [--staged OUT.yaml]
"""
import argparse
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1332239/HEPData-ins1332239-v1-yaml'
CITE = 'A. Adare et al. (PHENIX), PRC 93, 051902(R) (2016), arXiv:1412.1038'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
RXN = {'detector': 'RXN', 'eta': [{'min': -2.8, 'max': -1.0}, {'min': 1.0, 'max': 2.8}],
       'pT': {'min': 0.0, 'max': 100.0},
       'notes': 'South and north RXN scintillators, 1.0 < |eta| < 2.8 (every particle: no pT selection; the '
                'scintillator weighting is not modelled).'}
SPECIES = {r'\pi^{\pm}': 'pi+ + pi-', 'K^{\\pm}': 'K+ + K-', r'p\bar{p}': 'p + pbar'}
WINDOW = 0.2
COMBINED = {'0-50%': ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%'], '30-50%': ['30-40%', '40-50%']}
STAGING = ('PHENIX averages v_n of narrower classes weighted by the multiplicity of the selected particle '
           '(centrality.combined_from analysis_classes); the pipeline computes over the whole class.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the combined-class records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    public, staged = [], []
    for x in sub:
        hf = x['data_file']
        if hf.startswith('Fig4'):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        desc = ' '.join(x['description'].split())
        two_pc = 'two-particle correlation' in desc
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            m = re.fullmatch(r'\$(.+?)\$ \$v_(\d)(\\\{\\Psi_2\\\})?\$ in centrality (\d+)-(\d+)%', h.strip())
            assert m, (hf, h)
            if m.group(3):                                   # v4{Psi_2}
                continue
            sp, n, cls = SPECIES[m.group(1)], int(m.group(2)), f'{m.group(4)}-{m.group(5)}%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            assert rows == list(range(len(iv['values']))), (hf, i)
            pts = [float(v['value']) for v in iv['values']]
            how = ('long-range two-particle correlations with the RXN, <v^HAD v^RXN>/sqrt(<v^RXN-N v^RXN-S>), '
                   'computed as the scalar product' if two_pc else
                   'event plane from the RXN, computed with the scalar-product estimator')
            d = {'observable_type': 'flow_vn', 'inspire_id': '1332239', 'version': 1, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{desc[:150]} | {h}',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                 'centrality': {'classes': [cls], **BBC},
                 'particle_id': {'species': sp},
                 'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35},
                                     'pT': {'min': round(pts[0] - WINDOW / 2, 4),
                                            'max': round(pts[-1] + WINDOW / 2, 4),
                                            'points': pts, 'point_window': WINDOW}},
                 'harmonic': n, 'method': 'scalar_product' if two_pc else 'event_plane',
                 'value_type': h, 'dimensionality': 'differential_pT', 'flow_reference': dict(RXN),
                 'notes': {'value': f'{CITE}. v{n}(pT) of {sp}, {cls}, |eta| < 0.35; {how}; pT points '
                                    f'({WINDOW} GeV/c windows). Errors: statistical, pT-uncorrelated and '
                                    'pT-correlated systematic.'}}
            if cls in COMBINED:
                d['centrality'].update({'combined_from': 'analysis_classes', 'analysis_classes': COMBINED[cls]})
                d['staging_reason'] = STAGING
                staged.append(d)
            else:
                public.append(d)
    out = 'data/flow_vn/phenix_flow_vn_higher-order_anisotropic_flow_coefficients_200gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['particle_id']['species'], d['centrality']['classes'][0]) for d in public)))
    if staged_out:
        os.makedirs(os.path.dirname(staged_out) or '.', exist_ok=True)
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records;', dict(collections.Counter(
            (d['method'], d['centrality']['classes'][0]) for d in staged)))


if __name__ == '__main__':
    main()
