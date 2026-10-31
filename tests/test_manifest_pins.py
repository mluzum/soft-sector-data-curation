"""Manifest pins (fpath + doc_index) must still hold the record they name.

Record files change: documents are added, files split, records moved to
another collection.  A stale pin would silently compute another record --
often a sibling of the same table, which shares the obs_ id and differs only
in its centrality classes.  run_pipeline.py checks each pin against the
manifest entry and skips a stale one with a warning, without falling back to
the id lookup (which cannot tell those siblings apart).

Run: python3 tests/test_manifest_pins.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from analysis.record_id import record_short_id  # noqa: E402
from scripts.search_records import generate_manifest, load_records  # noqa: E402

_spec = importlib.util.spec_from_file_location('run_pipeline', os.path.join(ROOT, 'scripts', 'run_pipeline.py'))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

results = []


def check(name, cond, detail=''):
    results.append((name, bool(cond)))
    print(('  PASS ' if cond else '  FAIL ') + name + ('' if cond else f'  {detail}'))


def siblings():
    """(type, file, two documents of the file that share the obs_ id but not the
    centrality classes, a record of the same type with another id)."""
    for otype in sorted(os.listdir('data')):
        if not os.path.isdir(f'data/{otype}'):
            continue
        records = list(load_records(f'data/{otype}'))
        ids = {}
        for fpath, idx, doc in records:
            ids.setdefault((fpath, record_short_id(doc)), []).append((idx, doc))
        for (fpath, rid), v in sorted(ids.items()):
            if len(v) >= 2 and (v[0][1].get('centrality') or {}).get('classes') \
                    != (v[1][1].get('centrality') or {}).get('classes'):
                other = next(d for _, _, d in records if record_short_id(d) != rid)
                return otype, fpath, v[0], v[1], other
    raise RuntimeError('no file in data/ with per-centrality siblings')


def main():
    otype, fpath, (i, a), (j, b), c = siblings()
    print(f'  {fpath}: documents {i} and {j} share {record_short_id(a)}')
    with tempfile.TemporaryDirectory() as tmp:
        mf = os.path.join(tmp, 'm.yaml')
        generate_manifest([(fpath, i, a)], mf)
        entry = yaml.safe_load(open(mf))['records'][0]
    check('generated entries carry the centrality classes',
          entry.get('centrality_classes') == (a.get('centrality') or {}).get('classes'))
    check('pin to the named record passes', rp._pin_mismatch(entry, a) is None, rp._pin_mismatch(entry, a))
    check('pin to a sibling of the same table (same id) is stale',
          record_short_id(b) == entry['id'] and rp._pin_mismatch(entry, b) is not None)
    check('pin to another record of the file is stale', rp._pin_mismatch(entry, c) is not None)
    old = {kk: v for kk, v in entry.items() if kk != 'centrality_classes'}
    check('older manifests without classes still check the id',
          rp._pin_mismatch(old, a) is None and rp._pin_mismatch(old, c) is not None)

    # run_pipeline: a shifted pin is skipped, not computed and not looked up by id.
    with tempfile.TemporaryDirectory() as tmp:
        good = dict(entry)
        stale = dict(entry, doc_index=j)
        out = {}
        for name, e in (('good', good), ('stale', stale)):
            mf = os.path.join(tmp, f'{name}.yaml')
            yaml.safe_dump({'manifest_version': 1, 'records': [e]}, open(mf, 'w'))
            r = subprocess.run([sys.executable, 'scripts/run_pipeline.py', '--manifest', mf,
                                '--type', otype, '--print-fingerprint'],
                               capture_output=True, text=True)
            out[name] = r
        fp = lambda r: yaml.safe_load(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}  # noqa: E731
        check('the valid pin selects one record', fp(out['good']).get('n_records') == 1, out['good'].stdout[-500:])
        check('the stale pin is reported and selects nothing',
              'is stale' in out['stale'].stdout and fp(out['stale']).get('n_records') == 0,
              out['stale'].stdout[-500:])

    failed = [n for n, ok in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
