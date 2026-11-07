#!/usr/bin/env python3
"""Curate ALICE v_n{2}(eta) and r_{n|n}(eta) in Pb-Pb at 5.02 TeV and Xe-Xe at 5.44 TeV, INSPIRE 2679248.

S. Acharya et al. (ALICE), "Pseudorapidity dependence of anisotropic flow
and its decorrelations using long-range multiparticle correlations in
Pb-Pb and Xe-Xe collisions", PLB 850 (2024) 138477 (corrigendum PLB 853
(2024) 138659), arXiv:2307.11116.

From the paper (Secs. 2, 3; Figs. 1, 2):
- regions A (FMD, -3.5 < eta < -1.8), B and C (TPC, eta < 0 and eta > 0,
  tracks |eta| < 1.2, 0.2 < pT < 5 GeV/c), D (FMD, 1.8 < eta < 5); the FMD
  counts all pT; midrapidity v_n are extrapolated to pT = 0;
- v'_n{2}(eta) = <<cos n(phi' - phi^Y)>> / sqrt(<<cos n(phi^X -
  phi^Y)>>) (Eqs. 1, 2): with the reference at midrapidity, Y the TPC
  side opposite the POI restricted to 0.8 < |eta| < 1.0 (Xe-Xe: 0.4 <
  |eta| < 1.0) and X its mirror; with the forward reference, Y the FMD
  opposite the POI (A for POI at eta > 0, D for eta < 0) and X the other
  FMD; the tables list the POI bins symmetrized in eta;
- r_{n|n}(eta) (Eq. 5): opposite- over same-side correlation of POI at
  -+eta in the TPC with the FMD reference at 2.0 < |eta_ref| < 2.4 or 2.8 <
  |eta_ref| < 3.2;
- V0 centrality (V0M); errors stat and sys (r_{n|n}: stat).

Records (public): v_n{2}(eta) (flow_vn, cumulant_order 2, the reference
per side in flow_reference.segments with its partner region; key
vn2_eta_partner) and r_{n|n}(eta) (flow_harmonic_correlations,
longitudinal_decorrelation, quantity r_nn; POI 0.2 < pT < 5 GeV/c, the
ratio's pT extrapolation cancelling), analysis/eta_regions.py.  The
paper's 'mid |Delta eta| > 2.0' with the FMD reference corresponds to the
FMD edge at |eta| = 1.8 used here.  Staged (--staged OUT): v2{4}(eta)
(Table 7): the subevents of forward POI differ between Fig. 2b and the
text.

Run from the repository root:
    python scripts/curate_alice_flow_eta_decorrelation.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins2679248/HEPData-ins2679248-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = ('S. Acharya et al. (ALICE), PLB 850 (2024) 138477 (corrigendum PLB 853 (2024) 138659), '
        'arXiv:2307.11116')
OUT_VN = 'data/flow_vn/alice_flow_vn_eta_long_range_PbPb_XeXe_ins2679248.yaml'
OUT_R = 'data/flow_harmonic_correlations/alice_flow_harmonic_correlations_eta_decorrelation_PbPb_5020gev_ins2679248.yaml'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
SYSTEMS = {'Pb': (5020.0, 'PB PB --> CHARGED X', 'Pb-Pb 5.02 TeV'), 'Xe': (5440.0, 'XE XE --> CHARGED X',
                                                                            'Xe-Xe 5.44 TeV')}
TPC_PT = {'min': 0.2, 'max': 5.0}
ALL_PT = {'min': 0.0, 'max': 100.0}
FMD_A, FMD_D = {'min': -3.5, 'max': -1.8}, {'min': 1.8, 'max': 5.0}


def segments(gap):
    """flow_reference segments for the tables' gap configuration."""
    if gap in ('(0.8, 2.6)', '(0.4, 2.2)'):
        lo = float(gap.strip('()').split(',')[0])
        pos, neg = {'min': lo, 'max': 1.0}, {'min': -1.0, 'max': -lo}
        return [{'applies_to_eta': {'min': 0.0, 'max': 5.0}, 'eta': [neg], 'pT': dict(TPC_PT), 'partner': [pos],
                 'notes': f'POI at eta > 0: reference TPC {-1.0} < eta < {-lo}, partner {lo} < eta < 1.0.'},
                {'applies_to_eta': {'min': -3.5, 'max': 0.0}, 'eta': [pos], 'pT': dict(TPC_PT), 'partner': [neg],
                 'notes': f'POI at eta < 0: reference TPC {lo} < eta < 1.0, partner {-1.0} < eta < {-lo}.'}], \
            f'reference particles from the TPC on the other side, {lo:g} < |eta| < 1.0'
    assert gap == '(2.0, 3.8)', gap
    return [{'applies_to_eta': {'min': 0.0, 'max': 5.0}, 'eta': [dict(FMD_A)], 'pT': dict(ALL_PT),
             'partner': [dict(FMD_D)], 'notes': 'POI at eta > 0: reference FMD A, partner FMD D.'},
            {'applies_to_eta': {'min': -3.5, 'max': 0.0}, 'eta': [dict(FMD_D)], 'pT': dict(ALL_PT),
             'partner': [dict(FMD_A)], 'notes': 'POI at eta < 0: reference FMD D, partner FMD A.'}], \
        'reference particles from the FMD on the other side (A: -3.5 < eta < -1.8, D: 1.8 < eta < 5)'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    subs = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    vn, rnn, staged = [], [], []
    for x in subs:
        hf = x['data_file']
        desc = ' '.join(x['description'].split())
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        bins = [[round(float(v['low']), 4) + 0.0, round(float(v['high']), 4) + 0.0]
                for v in t['independent_variables'][0]['values']]
        sysname = 'Xe' if 'Xe}' in desc else 'Pb'
        energy, reaction, sys_label = SYSTEMS[sysname]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            lo, hi = q['CENTRALITY'].split(' TO ')
            cls = f'{int(float(lo))}-{int(float(hi))}%'
            n = int(q['HARMONIC'])
            name = dv['header']['name']
            head = {'inspire_id': '2679248', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                    'hepdata_table_label': f'{desc} ({name}, {cls})',
                    'collision_system': {'energy_gev': energy, 'reaction': reaction},
                    'centrality': {'classes': [cls], **V0M}, 'particle_id': {'species': 'charged'}}
            if q.get('CORRELATOR') == '4':
                staged.append({'observable_type': 'flow_vn', **head, 'value_type': name,
                               'notes': {'value': f'{CITE}. {desc}.'},
                               'staging_reason': 'v2{4}(eta): the subevents of forward POI differ between Fig. 2b '
                                                 '(phi_D + phi_D - phi_C - phi_C) and the text (references B or C '
                                                 'only)'})
                continue
            labels = {tuple(e['label'] for e in v['errors']) for v in dv['values']}
            if 'ETARAP REF' in q:
                assert labels == {('stat',)}, (hf, labels)
                r_lo, r_hi = (round(float(v), 4) for v in q['ETARAP REF'].split(' TO '))
                d = {'observable_type': 'flow_harmonic_correlations', **head,
                     'kinematic_range': {'eta': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins},
                                         'pT': dict(TPC_PT)},
                     'observable': f'r_{n}|{n}', 'correlation_type': 'longitudinal_decorrelation',
                     'harmonics': [n], 'quantity': 'r_nn', 'dimensionality': 'differential_eta',
                     'decorrelation_reference': {'eta': {'min': r_lo, 'max': r_hi}, 'pT': dict(ALL_PT),
                                                 'detector': 'FMD',
                                                 'notes': f'FMD reference at {r_lo:g} < |eta| < {r_hi:g}, both '
                                                          'sides.'},
                     'value_type': name,
                     'notes': {'value': f'{CITE}. r_{{{n}|{n}}}(eta) = <<cos {n}(phi(-eta) - phi_ref)>> / <<cos '
                                        f'{n}(phi(eta) - phi_ref)>> (Eq. 5), POI charged particles in the TPC at '
                                        f'+-|eta| (0.2 < pT < 5 GeV/c), reference FMD {r_lo:g} < |eta_ref| < '
                                        f'{r_hi:g} (both orientations pooled), {sys_label} {cls}. Errors: stat.'}}
                rnn.append(d)
                continue
            assert labels == {('stat', 'sys')}, (hf, labels)
            segs, how = segments(q['ETARAP GAP'])
            d = {'observable_type': 'flow_vn', **head,
                 'kinematic_range': {'eta': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins},
                                     'pT': dict(ALL_PT)},
                 'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2, 'dimensionality': 'differential_eta',
                 'flow_reference': {'segments': segs, 'notes': how}, 'value_type': name,
                 'notes': {'value': f'{CITE}. v_{n}{{2}}(eta) of charged particles (midrapidity extrapolated to pT = '
                                    f'0, FMD all pT), {how}, normalized by the correlation of the reference with its '
                                    f'mirror region (Eqs. 1, 2; HEPData gap {q["ETARAP GAP"]}); bins symmetrized in '
                                    f'eta by the collaboration; {sys_label} {cls}. Errors: stat and sys.'}}
            vn.append(d)
    for out, docs in ((OUT_VN, vn), (OUT_R, rnn)):
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
