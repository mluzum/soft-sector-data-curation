#!/usr/bin/env python3
"""Curate STAR bulk properties in Au+Au at 14.5 GeV, INSPIRE 1748776.

J. Adam et al. (STAR), "Bulk properties of the system formed in Au+Au
collisions at sqrt(sNN) = 14.5 GeV", PRC 101, 024905 (2020),
arXiv:1908.03585.

From the paper (Secs. II-III, Table I):
- pi+-, K+-, p, pbar spectra at |y| < 0.1 in nine RefMult classes (0-5% ...
  70-80%); <pT> and dN/dy from the spectra extrapolated with fits;
- charged v1, v2 with |eta| < 1: TPC eta-subevent plane (gap 0.1, the
  opposite subevent), Q-cumulants v2{2}, v2{4}, and the BBC first-order
  plane (3.8 < |eta| < 5.2);
- errors: statistical and systematic.

Records (public): the spectra (54), <pT> vs centrality (classes from the
<N_part> of Fig. 17), v2{eta-sub}(pT) of six classes (Fig. 16 top), v2{2}(pT)
and v2{4}(pT) of 10-20, 20-30, 30-40% (Fig. 15), and v2{eta-sub}(eta) of six
classes and 10-40% (Figs. 16 bottom, 19).  The pT window of the integrated
v2(eta) and of the event-plane tracks is not stated: 0.2 < pT < 2 GeV/c,
as in STAR's BES v2 analyses (PRC 86, 054908), is assumed.  Staged
(--staged OUT): dN/dy / (0.5 <N_part>) (Glauber-normalized), and v1 and
v2{BBC}: the BBC first-order plane lies beyond beam rapidity (2.7) at this
energy, a beam-fragment (spectator-like) reference no model provides.  Not
curated: the 7.7-39 GeV v1 of other STAR papers shown for comparison, the
blast-wave parameters, the event-plane resolution, v2/epsilon and the
Glauber table.  The old records had energy 0, 'MB' classes, pi and K as
'charged', and only the 70-80% spectra.

Run from the repository root:
    python scripts/curate_star_bulk_auau14p5.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1748776/HEPData-ins1748776-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (STAR), PRC 101, 024905 (2020), arXiv:1908.03585'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
CLASSES = ['0-5%', '5-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%']
SPECIES = {'proton': 'p', 'antiproton': 'pbar', 'piplus': 'pi+', 'piminus': 'pi-', 'kaplus': 'K+', 'kaminus': 'K-'}
FIG8 = dict(zip('abcdef', ['pi+', 'pi-', 'K+', 'K-', 'p', 'pbar']))
EP_REF = {'detector': 'TPC', 'mirror': True, 'eta': [{'min': 0.05, 'max': 1.0}], 'pT': {'min': 0.2, 'max': 2.0},
          'notes': 'eta-subevent plane from the opposite TPC subevent (gap 0.1); event-plane pT 0.2-2 GeV/c '
                   'assumed (not stated; STAR BES v2, PRC 86, 054908).'}
V2_PT = {'min': 0.2, 'max': 2.0}
FULL_PT = {'min': 0.0, 'max': 100.0}


def base(otype, hf, i, label, classes, sp):
    return {'observable_type': otype, 'inspire_id': '1748776', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': 14.5, 'reaction': 'AU AU --> X' if sp != 'charged'
                                 else 'AU AU --> CHARGED X'},
            'centrality': {'classes': classes, **REFMULT}, 'particle_id': {'species': sp}}


def points(iv):
    pts = [float(v['value']) for v in iv['values']]
    w = round(min(b - a for a, b in zip(pts[:-1], pts[1:])), 4)
    return {'min': round(pts[0] - w / 2, 4), 'max': round(pts[-1] + w / 2, 4), 'points': pts, 'point_window': w}


def note(what):
    return {'value': f'{CITE}. {what}. Errors: statistical and systematic.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    load = lambda hf: yaml.safe_load(open(os.path.join(RAW, hf)))  # noqa: E731
    npart = [float(v) for v in [d['value'] for d in load('figure17.yaml')['dependent_variables'][0]['values']]]
    cls_of_npart = dict(zip(npart, CLASSES))
    docs = {'spectra': [], 'mean_pT': [], 'flow': [], 'staged': []}
    for stem, sp in SPECIES.items():                                  # spectra
        hf = f'{stem}.yaml'
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            cls = dv['qualifiers'][0]['value']
            d = base('particle_spectra', hf, i, f"{dv['header']['name']} {cls}", [cls], sp)
            d.update({'kinematic_range': {'y': {'min': -0.1, 'max': 0.1}, 'pT': points(t['independent_variables'][0])},
                      'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                      'notes': note(f'1/(2 pi pT) d2N/(dpT dy) of {sp}, |y| < 0.1, {cls}; pT points')})
            docs['spectra'].append(d)
    for k, sp in FIG8.items():                                        # <pT>, dN/dy
        for fig, otype in (('figure8', 'mean_pT'), ('figure9', 'particle_yields')):
            hf = f'{fig}{k}.yaml'
            t = load(hf)
            classes = [cls_of_npart[float(v['value'])] for v in t['independent_variables'][0]['values']]
            d = base(otype, hf, 0, ' '.join(sub[hf]['description'].split())[:150], classes, sp)
            d['kinematic_range'] = {'y': {'min': -0.1, 'max': 0.1}, 'pT': dict(FULL_PT)}
            if otype == 'mean_pT':
                d.update({'observable': 'mean_pT', 'value_type': 'mean_pT',
                          'notes': note(f'<pT> of {sp}, |y| < 0.1, extrapolated to all pT with fits; classes from '
                                        'the <N_part> of Fig. 17')})
                docs['mean_pT'].append(d)
            else:
                d.update({'observable': 'dN_dy', 'value_type': 'dN/dy / (0.5 N_part)',
                          'notes': note(f'dN/dy of {sp} divided by 0.5 <N_part> (Glauber)'),
                          'staging_reason': 'dN/dy divided by the Glauber 0.5 <N_part> of each class.'})
                docs['staged'].append(d)
    t = load('figure16.1.yaml')                                       # v2{eta-sub}(pT)
    for i, dv in enumerate(t['dependent_variables']):
        cls = list(dv['qualifiers'][0].values())[1] if 'value' not in dv['qualifiers'][0] else dv['qualifiers'][0]['value']
        d = base('flow_vn', 'figure16.1.yaml', i, f'v2(EtaSub) {cls}', [cls], 'charged')
        d.update({'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': points(t['independent_variables'][0])},
                  'harmonic': 2, 'method': 'event_plane', 'value_type': 'v2(EtaSub)', 'dimensionality': 'differential_pT',
                  'flow_reference': dict(EP_REF),
                  'notes': note(f'v2(pT) of charged hadrons, |eta| < 1, {cls}, eta-subevent plane; computed with '
                                'the scalar-product estimator; pT points')})
        docs['flow'].append(d)
    for k in 'abc':                                                   # v2{2}, v2{4}, v2{BBC} (pT)
        hf = f'figure15{k}.yaml'
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            h, cls = dv['header']['name'], dv['qualifiers'][0]['value']
            if h == 'v2(EtaSub)':
                continue                                              # = Fig. 16 top
            d = base('flow_vn', hf, i, f'{h} {cls}', [cls], 'charged')
            d.update({'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': points(t['independent_variables'][0])},
                      'harmonic': 2, 'value_type': h, 'dimensionality': 'differential_pT'})
            if h == 'v2(BBC)':
                d.update({'method': 'event_plane',
                          'flow_reference': {'detector': 'BBC', 'eta': [{'min': -5.2, 'max': -3.8},
                                                                        {'min': 3.8, 'max': 5.2}],
                                             'plane': 'spectator', 'notes': 'BBC first-order plane (Psi_1), beyond '
                                             'beam rapidity (2.7) at 14.5 GeV: beam fragments.'},
                          'notes': note(f'v2 relative to the BBC first-order plane, {cls}'),
                          'staging_reason': 'v2 relative to the BBC first-order plane beyond beam rapidity '
                                            '(beam fragments, spectator-like; mixed harmonic).'})
                docs['staged'].append(d)
                continue
            order = 2 if h == 'v2(QC2)' else 4
            d.update({'method': 'cumulant', 'cumulant_order': order,
                      'notes': note(f'v2{{{order}}}(pT) of charged hadrons, |eta| < 1, {cls}, Q-cumulants without '
                                    'gap; reference the record window')})
            docs['flow'].append(d)
    for hf in ('figure16.2.yaml', 'figure19.yaml'):                  # v2{eta-sub}(eta)
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            cls = [v for q, v in ((q['name'], q.get('value')) for q in dv['qualifiers']) if 'entrality' in q][0]
            d = base('flow_vn', hf, i, f'v2(eta) {cls}', [cls], 'charged')
            d.update({'kinematic_range': {'eta': points(t['independent_variables'][0]), 'pT': dict(V2_PT)},
                      'harmonic': 2, 'method': 'event_plane', 'value_type': 'v2', 'dimensionality': 'differential_eta',
                      'flow_reference': dict(EP_REF),
                      'notes': note(f'pT-integrated v2(eta) of charged hadrons, {cls}, eta-subevent plane; pT window '
                                    'not stated, 0.2-2 GeV/c assumed; computed with the scalar-product estimator')})
            docs['flow'].append(d)
    for fig, dim in (('figure20c.yaml', 'differential_pT'), ('figure21c.yaml', 'differential_eta')):   # v1, 14.5
        t = load(fig)
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: qq.get('value') for qq in dv['qualifiers']}
            assert float(q['Energy']) == 14.5, (fig, q)
            d = base('flow_vn', fig, i, f"v1 {q['Centrality']}", [q['Centrality']], 'charged')
            ax = points(t['independent_variables'][0])
            d.update({'kinematic_range': ({'eta': {'min': -1.0, 'max': 1.0}, 'pT': ax} if dim == 'differential_pT'
                                          else {'eta': ax, 'pT': dict(V2_PT)}),
                      'harmonic': 1, 'method': 'event_plane', 'value_type': 'v1', 'dimensionality': dim,
                      'flow_reference': {'detector': 'BBC', 'odd': True, 'plane': 'spectator',
                                         'eta': [{'min': -5.2, 'max': -3.8}, {'min': 3.8, 'max': 5.2}],
                                         'notes': 'BBC first-order plane beyond beam rapidity: beam fragments.'},
                      'notes': note(f"rapidity-odd v1 of charged hadrons, {q['Centrality']}, BBC first-order plane"),
                      'staging_reason': 'v1 relative to the BBC first-order plane beyond beam rapidity (beam '
                                        'fragments, spectator-like).'})
            docs['staged'].append(d)
    outs = {'spectra': 'data/particle_spectra/star_particle_spectra_bulk_properties_AuAu_14p5gev_ins1748776.yaml',
            'mean_pT': 'data/mean_pT/star_mean_pT_bulk_properties_AuAu_14p5gev_ins1748776.yaml',
            'flow': 'data/flow_vn/star_flow_vn_bulk_properties_AuAu_14p5gev_ins1748776.yaml'}
    for old in glob.glob('data/*/star_*_bulk_properties_of_the_system_0gev.yaml'):
        os.remove(old)
    old_y = 'data/particle_yields/star_particle_yields_bulk_properties_of_the_system_0gev.yaml'
    if os.path.exists(old_y):
        os.remove(old_y)
    for k, out in outs.items():
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs[k], fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs[k])} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(docs['staged'], fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(docs["staged"])} records;',
              dict(collections.Counter(d['staging_reason'][:25] for d in docs['staged'])))


if __name__ == '__main__':
    main()
