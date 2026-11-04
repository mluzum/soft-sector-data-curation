"""Records that share their prepared data in run_pipeline.

Records of one compute key that differ only in their centrality classes
(one record per class of a table) prepare the same data: run_pipeline
prepares it once (prepare_key) and masks it once per jackknife replicate
(masked_entries).  Checks:
  - prepare_key: equal for the per-class records of a table; different when
    the acceptance, species, value type, compute key or container differ;
    None for a param it cannot compare;
  - masked_entries: entries of one key get one masked copy, made at the
    first and released after the last of them (entries in between keep
    theirs); entries without a key get their own;
  - run_pipeline end to end (toy 2D package, Au+Au 200 GeV): two sets of
    per-class records from data/ (dN/deta(eta) and spectra, whose cores fit
    the spectrum per class), computed together (shared) and each record
    alone: identical values and errors.
Run: python3 tests/test_shared_prepares.py
"""
import collections
import gc
import importlib.util
import os
import subprocess
import sys
import tempfile
import weakref

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
os.chdir(ROOT)

from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from scripts.search_records import generate_manifest, load_records  # noqa: E402

_spec = importlib.util.spec_from_file_location('run_pipeline', os.path.join(ROOT, 'scripts', 'run_pipeline.py'))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

results = []


def check(name, cond, detail=''):
    results.append((name, bool(cond)))
    print(('  PASS ' if cond else '  FAIL ') + name + ('' if cond else f'  {detail}'))


def sibling_sets(otype, n=3):
    """Au+Au 200 GeV records of one type, grouped by prepare_key: the first
    group of at least n records with different centrality classes."""
    groups = collections.defaultdict(list)
    for fpath, idx, doc in load_records(f'data/{otype}'):
        cs = doc.get('collision_system') or {}
        if cs.get('reaction') != 'AU AU --> CHARGED X' or abs(float(cs.get('energy_gev') or 0) - 200) > 1:
            continue
        try:
            key = resolve(doc)['compute']
        except ValueError:
            continue
        groups[rp.prepare_key(key, extract_measurement_params(doc), '2d')].append((fpath, idx, doc))
    for pk, v in groups.items():
        classes = {str((d.get('centrality') or {}).get('classes')) for _, _, d in v}
        if pk is not None and len(v) >= n and len(classes) == len(v):
            return v[:n]
    raise RuntimeError(f'no {otype} records sharing a prepare key')


def test_prepare_key():
    sets = sibling_sets('particle_spectra')
    p = [extract_measurement_params(d) for _, _, d in sets]
    key = resolve(sets[0][2])['compute']
    k0 = rp.prepare_key(key, p[0], '2d')
    check('per-class records of a table: one key',
          all(rp.prepare_key(key, q, '2d') == k0 for q in p) and p[0]['cent_classes'] != p[1]['cent_classes'])
    q = dict(p[0])
    q['single_range'] = dict(q['single_range'], pT=(0.3, 2.0))
    check('another pT window: another key', rp.prepare_key(key, q, '2d') != k0)
    q = dict(p[0], particle_id={'species': 'p' if (p[0]['particle_id'] or {}).get('species') != 'p' else 'pi+'})
    check('another species: another key', rp.prepare_key(key, q, '2d') != k0)
    q = dict(p[0], value_type='dN_dpT_dy' if p[0]['value_type'] != 'dN_dpT_dy' else 'invariant_yield')
    check('another value type: another key', rp.prepare_key(key, q, '2d') != k0)
    check('another compute key or container: another key',
          rp.prepare_key('mean_pt', p[0], '2d') != k0 and rp.prepare_key(key, p[0], '3d') != k0)
    q = dict(p[0], differential_bins=[(0.2, 0.4), (0.4, 0.6)])
    q2 = dict(p[0], differential_bins=[[0.2, 0.4], [0.4, 0.6]])
    check('a tuple and a list are told apart', rp.prepare_key(key, q, '2d') != rp.prepare_key(key, q2, '2d'))
    q = dict(p[0], cent_valid=np.ones(3, bool), cent_partial=np.zeros(3, bool), hepdata_file='x.yaml')
    check('centrality and label fields do not enter', rp.prepare_key(key, q, '2d') == k0)
    q = dict(p[0], sp_ref_eta=np.array([0.0, 1.0]))
    check('a param that cannot be compared: no key', rp.prepare_key(key, q, '2d') is None)


def test_masked_entries():
    n = 6
    mask = np.array([1, 0, 1, 1, 0, 1], bool)
    shared = {'Q0': np.arange(n, dtype=float), 'nested': {'s': np.arange(2 * n).reshape(n, 2)}, 'edges': (0, 1)}
    other = {'Q0': np.arange(n, dtype=float) * 10}
    entries = [{'prep_key': 'a', 'data_dict': shared}, {'prep_key': None, 'data_dict': other},
               {'prep_key': 'a', 'data_dict': shared}, {'prep_key': 'b', 'data_dict': other},
               {'data_dict': shared}]
    gen = rp.masked_entries(entries, mask)
    got = []
    _, d0 = next(gen)
    ref = weakref.ref(d0['Q0'])
    got.append(d0)
    for _ in range(4):
        got.append(next(gen)[1])
    check('masked as before (nested arrays, other values kept)',
          np.array_equal(got[0]['Q0'], shared['Q0'][mask]) and
          np.array_equal(got[0]['nested']['s'], shared['nested']['s'][mask]) and got[0]['edges'] == (0, 1)
          and np.array_equal(got[1]['Q0'], other['Q0'][mask]))
    check('one key: one masked copy for its entries', got[2] is got[0])
    check('no key: a copy of its own', got[4] is not got[0] and got[3] is not got[1])
    del got, d0
    gc.collect()
    check('released after the last entry of its key', ref() is None)


def _by_record(path):
    import h5py
    out = {}
    with h5py.File(path, 'r') as f:
        for g in f.values():
            if 'labels' not in g:
                continue
            for i, k in enumerate(g['label_record'][:]):
                eg = g[f'exp_{k}']
                key = (os.path.basename(str(eg.attrs['record_file'])), int(eg.attrs['doc_index']))
                out.setdefault(key, []).append((g['labels'][i], g['values'][i], g['errors'][i]))
    return out


def test_pipeline():
    import test_sim_package as tsp
    a, b = sibling_sets('particle_yields'), sibling_sets('particle_spectra')
    print(f'  {a[0][0]}: docs {[i for _, i, _ in a]};  {b[0][0]}: docs {[i for _, i, _ in b]}')
    with tempfile.TemporaryDirectory() as tmp:
        pkg, *_ = tsp._package(tmp, 'toy_adapter_shared')
        joint, alone = os.path.join(tmp, 'joint.yaml'), os.path.join(tmp, 'alone.yaml')
        generate_manifest(a + b, joint)
        generate_manifest([a[1], b[2]], alone)        # records that reuse data in the joint run
        base = [sys.executable, 'scripts/run_pipeline.py', '--sim-package', str(pkg), '--variant', 'plain',
                '--design-point', '0', '--type', 'particle_yields', 'particle_spectra', '--n-jk-blocks', '10']
        runs = {}
        for name, mf in (('joint', joint), ('alone', alone)):
            out = os.path.join(tmp, f'{name}.h5')
            r = subprocess.run(base + ['--manifest', mf, '--output', out], capture_output=True, text=True)
            runs[name] = (r, out)
        ok = all(r.returncode == 0 for r, _ in runs.values())
        check('both runs succeed', ok)
        if not ok:
            for r, _ in runs.values():
                print(r.stdout[-1500:], r.stderr[-1500:])
            return
        check('joint run: 4 records reuse an earlier one\'s prepared data',
              runs['joint'][0].stdout.count('Prepared data shared') == 4)
        check('alone: none shared', 'Prepared data shared' not in runs['alone'][0].stdout)
        j, s = _by_record(runs['joint'][1]), _by_record(runs['alone'][1])
        check('all six records computed', len(j) == 6 and len(s) == 2, f'{sorted(j)} {sorted(s)}')
        same = bool(s) and all(
            k in j and len(j[k]) == len(v) and all(x[0] == y[0] for x, y in zip(j[k], v))
            and np.allclose([x[1] for x in j[k]], [y[1] for y in v], rtol=1e-12, atol=0, equal_nan=True)
            and np.allclose([x[2] for x in j[k]], [y[2] for y in v], rtol=1e-12, atol=0, equal_nan=True)
            for k, v in s.items())
        check('shared data: the values and errors of each record alone', same)
        check('each record has finite values', all(np.isfinite([x[1] for x in v]).any() for v in s.values()))


if __name__ == '__main__':
    for t in (test_prepare_key, test_masked_entries, test_pipeline):
        print(t.__name__)
        t()
    failed = [n for n, ok in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} passed')
    if failed:
        raise SystemExit('FAILED: ' + ', '.join(failed))
