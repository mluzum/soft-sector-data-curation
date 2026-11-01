#!/usr/bin/env python3
"""Curate ALICE dNch/deta in p-Pb at 8.16 TeV, INSPIRE 1706753.

S. Acharya et al. (ALICE), "Charged-particle pseudorapidity density at
mid-rapidity in p-Pb collisions at sqrt(sNN) = 8.16 TeV", EPJC 79, 307
(2019), arXiv:1812.01312.

From the paper (Secs. 2-4):
- primary charged particles from SPD tracklets, |eta_lab| < 1.8, corrected
  to pT = 0 (tracklets select pT > 50 MeV/c);
- protons toward -z, Pb toward +z: the nucleon-nucleon centre of mass
  moves by Delta y = 0.465 toward the proton (rapidity_shift -0.465, Pb
  toward +eta_lab);
- non-single-diffractive (NSD) events; centrality classes are percentiles
  of the visible cross section from CL1 (clusters in the outer SPD layer,
  |eta_lab| < 1.4), V0A (2.8 < eta_lab < 5.1, Pb-going) or ZNA (neutron
  energy at zero degrees, Pb-going, the hybrid method);
- errors: statistical and systematic.

Records (public): dNch/deta_lab of the CL1 and V0A classes (Tables 3-16).
Staged (--staged OUT): the NSD distribution and the NSD average (Tables
1-2: an event selection, not percentile classes of the model's events),
the ZNA classes (Tables 17-23: spectator neutrons), and <dNch/deta> *
2/<N_part> (Tables 24-29: normalized by Glauber <N_part>).  The records
were all at 8.16 GeV with the V0M estimator.

Run from the repository root:
    python scripts/curate_alice_dndeta_ppb_8160.py [--staged OUT.yaml]
"""
import argparse
import collections
import os

import yaml

RAW = 'hepdata_raw/records/ins1706753/HEPData-ins1706753-v1-yaml'
CITE = 'S. Acharya et al. (ALICE), EPJC 79, 307 (2019), arXiv:1812.01312'
SYSTEM = {'energy_gev': 8160.0, 'reaction': 'P PB --> CHARGED X', 'frame': 'lab', 'rapidity_shift': -0.465,
          'forward_beam': 'Pb'}
ESTIMATORS = {
    'CL1': {'estimator': {'name': 'CL1', 'observable': 'Nch', 'detector': 'SPD'},
            'eta': [{'min': -1.4, 'max': 1.4}],
            'notes': 'Clusters in the outer SPD layer, |eta_lab| < 1.4.'},
    'V0A': {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
            'eta': [{'min': 2.8, 'max': 5.1}], 'notes': 'V0A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side).'},
    'ZNA': {'estimator': {'name': 'ZNA', 'observable': 'spectator_neutrons', 'detector': 'ZNA'},
            'eta': [{'min': 8.7, 'max': 20.0}],
            'notes': 'Neutron energy in the Pb-going zero-degree calorimeter (eta_lab > 8.7), hybrid method.'},
}
FULL_PT = {'min': 0.0, 'max': 100.0}
STAGING = {
    'nsd': 'NSD event selection (no percentile classes of the model events).',
    'zna': 'Centrality from spectator neutrons (ZNA); a model without spectators cannot form it.',
    'npart': '<dNch/deta> * 2/<N_part>: normalized by the Glauber <N_part> of each class.',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    public, staged = [], []
    for n in range(1, 30):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
        assert q['RE'] == 'P Pb --> CHARGED X' and q['SQRT(S)'] == '8.16', (hf, q)
        desc = ' '.join(sub[hf]['description'].split())
        d = {'observable_type': 'particle_yields', 'inspire_id': '1706753', 'version': 1, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': desc, 'collision_system': dict(SYSTEM)}
        if n <= 23 and n != 2:
            edges = [float(iv['values'][0]['low'])] + [float(v['high']) for v in iv['values']]
            kr = {'eta': {'min': edges[0], 'max': edges[-1], 'bins': edges}, 'pT': dict(FULL_PT)}
            dim = 'differential_eta'
        if n <= 2:                                                   # NSD
            est, cls = ESTIMATORS['V0A'], '0-100%'
            if n == 2:
                kr, dim = {'eta': {'min': -0.5, 'max': 0.5}, 'pT': dict(FULL_PT)}, 'integrated'
            what, reason = 'NSD events', STAGING['nsd']
        elif n <= 23:
            est = ESTIMATORS[q['ESTIMATOR']]
            cls = q['CENTRALITY']
            what, reason = f"{cls} ({q['ESTIMATOR']} classes)", (STAGING['zna'] if q['ESTIMATOR'] == 'ZNA' else None)
        else:                                                        # vs <N_part>
            est = ESTIMATORS[{'ZNAmult': 'ZNA', 'ZNAPb': 'ZNA'}.get(q['ESTIMATOR'], q['ESTIMATOR'])]
            cls = None
            kr, dim = {'eta': {'min': -0.5, 'max': 0.5}, 'pT': dict(FULL_PT)}, 'integrated'
            what, reason = f"<dNch/deta>*2/<N_part> vs <N_part> ({q['ESTIMATOR']})", STAGING['npart']
        cent = {'classes': [cls] if cls else [], 'estimator': dict(est['estimator']), 'eta': est['eta'],
                'notes': est['notes']}
        d.update({'centrality': cent, 'particle_id': {'species': 'charged'}, 'kinematic_range': kr,
                  'observable': 'dNch_deta', 'value_type': dv['header']['name'], 'dimensionality': dim,
                  'notes': {'value': f'{CITE}. dNch/deta_lab of primary charged particles, {what}, lab frame '
                                     '(y_lab = y_cms - 0.465, Pb toward +eta_lab), tracklets corrected to pT = 0. '
                                     'Errors: statistical and systematic.'}})
        if reason:
            if not cls:
                d['centrality']['classes'] = [f'N_part = {v["value"]}' for v in iv['values']]
            d['staging_reason'] = reason
            staged.append(d)
        else:
            public.append(d)
    old = 'data/particle_yields/alice_particle_yields_charged-particle_pseudorapidity_density_PbPb_8.16gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/particle_yields/alice_particle_yields_dNch_deta_pPb_8160gev_ins1706753.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        d['centrality']['estimator']['name'] for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records;', dict(collections.Counter(d['staging_reason'][:30]
                                                                                for d in staged)))


if __name__ == '__main__':
    main()
