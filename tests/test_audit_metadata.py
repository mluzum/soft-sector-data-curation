"""scripts/audit_metadata.py: its parsers, and every public record free of errors.

The audit compares each record with its raw HEPData table (qualifiers,
independent variables, header).  Every paper of data/ was vetted against
the paper itself (November 2026), and the departures from HEPData's own
labels that the curation made on purpose are stated in the records
(`hepdata_corrections`), so a public record that the audit reports as an
error is a regression: a new record, or a changed one, that disagrees
with its table.  Warnings stay informational.
Checks:
  - parsers: signed ranges ('-0.8 - 0.8' was read as (0.8, 0.8)),
    |eta| < x, collision systems in LaTeX or run together ('Pb$-$Pb',
    'OO', 'Ne+Ne and O+O', two reactions joined by '+'), qualifier names
    and centrality labels in LaTeX;
  - the checks still fire: a wrong reaction or energy, a flow table in
    percent; a ratio of cumulant orders (~1) is not read as percent;
  - the whole public database: 0 errors.
Run: python3 tests/test_audit_metadata.py
"""
import copy
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

import audit_metadata as A  # noqa: E402

results = []


def check(name, cond, detail=''):
    results.append((name, bool(cond)))
    print(('  PASS ' if cond else '  FAIL ') + name + (f'  {detail}' if detail else ''))


def test_parsers():
    cases = {'-0.8 - 0.8': (-0.8, 0.8), '0.2-3.0': (0.2, 3.0), '0.2 < pT < 5': (0.2, 5.0),
             '-1.0 < eta < -0.5': (-1.0, -0.5), 'pT > 0.2': (0.2, None), '|eta| < 0.8': (-0.8, 0.8),
             '$|\\eta| < 1$': (-1.0, 1.0), '0.2 TO 5.0': (0.2, 5.0),
             'pT > 0.2 GeV/c and |eta| < 0.8': (0.2, None)}
    got = {k: A._num_range(k) for k in cases}
    check('ranges: signed, |x| < a, a < x < b, a TO b', got == cases,
          '' if got == cases else str({k: v for k, v in got.items() if v != cases[k]}))
    sy = {'Pb$-$Pb': {('Pb', 'Pb')}, 'PB PB --> X': {('Pb', 'Pb')}, 'OO': {('O', 'O')},
          'Ne+Ne and O+O': {('Ne', 'Ne'), ('O', 'O')}, 'Ne$-$Ne, OO': {('Ne', 'Ne'), ('O', 'O')},
          'Ne Ne --> CHARGED X  +  O O --> CHARGED X': {('Ne', 'Ne'), ('O', 'O')},
          'DEUT AU --> CHARGED X': {('Au', 'd')}, 'd+Au': {('Au', 'd')}, 'Au+Cu': {('Au', 'Cu')},
          'Au+Au 200 GeV': {('Au', 'Au')}, 'unknown': None}
    got = {k: A._systems(k) for k in sy}
    check('collision systems: LaTeX, run together, several, order-free', got == sy,
          '' if got == sy else str({k: v for k, v in got.items() if v != sy[k]}))
    check('qualifier names without LaTeX',
          (A._qname('$p_{\\rm T}$'), A._qname('$Centrality (\\%)$'), A._qname('$\\sqrt{s_{NN}}$'))
          == ('p_t', 'centrality (%)', 'sqrts_nn'))
    check('centrality labels without LaTeX',
          (A._norm_cent('$0-10\\%$'), A._norm_cent('10$-$20%'), A._norm_cent('$5\\% - 10\\%$'))
          == ('0-10%', '10-20%', '5-10%'))


def _doc(path, pred):
    for d in yaml.safe_load_all(open(path)):
        if d and pred(d):
            return d
    raise LookupError(path)


def _errors(path, doc):
    return [f['message'] for f in A.audit_doc(path, doc)['flags'] if f['severity'] == 'error']


def test_checks_fire():
    f = 'data/flow_vn/alice_flow_vn_OO_NeNe_5360gev_ins2967353.yaml'
    d = _doc(f, lambda d: d['hepdata_file'] == 'Table_v22.yaml'
             and d['collision_system']['reaction'].startswith('NE NE'))
    check('ALICE Ne-Ne v2{2} (qualifier Ne$-$Ne): no error', not _errors(f, d), ' '.join(_errors(f, d)))
    bad = copy.deepcopy(d)
    bad['collision_system']['reaction'] = 'PB PB --> CHARGED X'
    check('a wrong reaction is an error', any('reaction qualifier' in m for m in _errors(f, bad)))
    bad = copy.deepcopy(d)
    bad['collision_system']['energy_gev'] = 5020.0
    check('a wrong energy is an error', any('energy qualifier' in m for m in _errors(f, bad)))
    y = 'data/particle_spectra/atlas_particle_spectra_charged_OO_NeNe_5360gev_ins3170344.yaml'
    flags = A.audit_doc(y, _doc(y, lambda d: 'y' in d['kinematic_range']))['flags']
    check('y_pi (pion-mass rapidity) names no species', not any('species' in x['message'] for x in flags),
          ' '.join(x['message'] for x in flags))
    r = _doc(f, lambda d: (d.get('cumulant_combination') or {}).get('kind') == 'ratio')
    check('a ratio of cumulant orders is not read as percent', not _errors(f, r), ' '.join(_errors(f, r)))
    bad = copy.deepcopy(r)
    del bad['cumulant_combination']
    check('the same values as a flow coefficient: percent without a unit',
          any('percent without a unit' in m for m in _errors(f, bad)))


def test_public_database():
    n, errors = 0, []
    for path in sorted(A.glob.glob(os.path.join(A.DATA, '**', '*.yaml'), recursive=True)):
        for doc in yaml.load_all(open(path), Loader=A._LOADER):
            if not isinstance(doc, dict):
                continue
            e = A.audit_doc(path, doc)
            if e is None:
                continue
            n += 1
            errors += [f"{e['file']} :: {e['hepdata_file']}: {f['message']}"
                       for f in e['flags'] if f['severity'] == 'error']
    check(f'public records audited ({n}): no errors', n > 5000 and not errors,
          '\n    ' + '\n    '.join(errors[:20]) if errors else '')


if __name__ == '__main__':
    for t in (test_parsers, test_checks_fire, test_public_database):
        print(t.__name__)
        t()
    failed = [n for n, ok in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} passed')
    if failed:
        raise SystemExit('FAILED: ' + ', '.join(failed))
