#!/usr/bin/env python3
"""Curate STAR relative dynamical pT correlations in Au+Au at 3.0-7.7 GeV, INSPIRE 3141716.

STAR Collaboration, "Non-monotonicity of transverse momentum correlations
in Au+Au collisions at RHIC", arXiv:2604.06434.

From the paper:
- C_pT = sqrt(<Delta pT_i Delta pT_j>) / <<pT>> (Eq. 3) of charged
  particles with |eta_cm| < 0.5 and 0.2 < pT < 2.0 GeV/c, in percent;
- fixed-target (FXT) Au+Au at sqrt(sNN) = 3.0, 3.2, 3.5, 3.9, 4.5, 5.2 and
  7.7 GeV, and collider (COL) mode at 7.7 GeV; in FXT mode mid-rapidity
  moves to negative eta_lab (eta_cm = eta_lab + y_cm, y_cm half the beam
  rapidity);
- centrality: FXT from all TPC tracks, -2.0 < eta_lab < 0 (to -2.24 with
  the iTPC, from 3.2 GeV on); COL from tracks with |eta| < 0.5 and pT >
  0.15 GeV/c; Fig. 3 in 10% classes 0-10 ... 60-70% (HEPData gives their
  <N_part>), Fig. 4 the 0-5% and 30-40% classes vs energy;
- errors: stat and sys.

Records (public): C_pT vs class per energy and mode (Fig. 3, 8 records)
and the 0-5% class of Fig. 4 (8 one-point records; its 30-40% points are
Fig. 3's class).  Windows in the nucleon-nucleon frame: the FXT estimator
window -2.0 (-2.24) < eta_lab < 0 maps to eta_lab + y_cm (a symmetric
system: no frame fields, which would ask the container for an
orientation).  value_scale 100 (percent).  Not curated: the event-by-event
<pT> distributions (Fig. 2), the BES-I points of Fig. 4 (INSPIRE 1712047,
public) and the model curves.

Run from the repository root:  python scripts/curate_star_pt_correlations_fxt.py
"""
import glob
import math
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins3141716/HEPData-ins3141716-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'STAR Collaboration, arXiv:2604.06434'
M_N = 0.938272
CLASSES = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%']
FIG3 = {(3.0, 'FXT'): 'fig3_3p0.yaml', (3.2, 'FXT'): 'fig3_3p2.yaml', (3.5, 'FXT'): 'fig3_3p5.yaml',
        (3.9, 'FXT'): 'fig3_3p9.yaml', (4.5, 'FXT'): 'fig3_4p5.yaml', (5.2, 'FXT'): 'fig3_5p2.yaml',
        (7.7, 'FXT'): 'fig3_7p7_fxt.yaml', (7.7, 'COL'): 'fig3_7p7_col.yaml'}
OUT = 'data/mean_pT_fluctuations/star_mean_pT_fluctuations_pT_correlations_AuAu_3_7.7gev_ins3141716.yaml'


def y_cm(sqrt_s):
    """Half the beam rapidity of a fixed-target collision at sqrt(sNN)."""
    e_lab = (sqrt_s ** 2 - 2 * M_N ** 2) / (2 * M_N)
    return 0.5 * math.acosh(e_lab / M_N)


def estimator(energy, mode):
    if mode == 'COL':
        return {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
                'eta': [{'min': -0.5, 'max': 0.5}], 'pT': {'min': 0.15, 'max': 100.0},
                'notes': 'charged tracks with |eta| < 0.5 and pT > 0.15 GeV/c (collider mode).'}
    lo = -2.0 if energy == 3.0 else -2.24
    yc = y_cm(energy)
    return {'estimator': {'name': 'FXTMult', 'observable': 'Nch', 'detector': 'TPC'},
            'eta': [{'min': round(lo + yc, 3), 'max': round(yc, 3)}],
            'notes': f'all TPC tracks, {lo:g} < eta_lab < 0 ({"TPC" if energy == 3.0 else "with the iTPC"}), here in '
                     f'the nucleon-nucleon frame: eta_lab + y_cm, y_cm = {yc:.3f}.'}


def record(hf, energy, mode, classes, dv_index=0, rows=None, label=''):
    d = {'observable_type': 'mean_pT_fluctuations', 'inspire_id': '3141716', 'version': VERSION, 'hepdata_file': hf,
         'dv_index': dv_index, 'hepdata_table_label': f'C_pT, Au+Au {energy:g} GeV {mode}{label} ({hf})',
         'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
         'centrality': {'classes': list(classes), **estimator(energy, mode)},
         'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.2, 'max': 2.0}},
         'observable': 'sqrt_dpt_dpt_over_mean_pT', 'value_type': 'sqrt_<dpt1_dpt2>_over_<pT>', 'method': 'two_particle',
         'value_scale': 100.0}
    if rows is not None:
        d['table_rows'] = rows
    d['notes'] = {'value': f'{CITE}. C_pT of charged particles, |eta_cm| < 0.5, 0.2 < pT < 2.0 GeV/c, Au+Au {energy:g} '
                           f'GeV ({"fixed target" if mode == "FXT" else "collider"}), {", ".join(classes)}, in percent '
                           '(value_scale 100)' + ('; the rows of Fig. 3 are the 10% classes 0-10 ... 60-70%, given in '
                                                  'HEPData by their <N_part>' if rows is None else '')
                           + '. Errors: stat and sys.'}
    return d


def main():
    docs = []
    for (energy, mode), hf in FIG3.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        npart = [float(v['value']) for v in t['independent_variables'][0]['values']]
        dv = t['dependent_variables'][0]
        q = {x['name']: x['value'] for x in dv['qualifiers']}
        assert len(npart) == len(CLASSES) and npart == sorted(npart, reverse=True), hf
        assert float(q['SQRT(S)/NUCLEON']) == energy and q.get('MODE', 'FXT') == mode, (hf, q)
        assert q['ETARAP'] == 'ABS(ETARAP) < 0.5' and q['PT'] == '0.2 - 2.0' and dv['header']['units'] == 'PCT', hf
        docs.append(record(hf, energy, mode, CLASSES))
    for hf, mode in (('fig4_besii_fxt.yaml', 'FXT'), ('fig4_besii_col.yaml', 'COL')):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        energies = [float(v['value']) for v in t['independent_variables'][0]['values']]
        k = next(i for i, dv in enumerate(t['dependent_variables'])
                 if {x['name']: x['value'] for x in dv['qualifiers']}['CENTRALITY'] == '0-5%')
        for j, energy in enumerate(energies):
            docs.append(record(hf, energy, mode, ['0-5%'], dv_index=k, rows=[j], label=', 0-5%'))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()
