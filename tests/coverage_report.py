"""Measure which compute keys the test suite actually exercises.

Static analysis of test sources is unreliable here: the toy-MC harness goes
through ``analysis.toy_mc.evaluate``, which resolves the dispatch key
internally, so a well-covered key never appears by name in a test file.
Some tests also call a core directly rather than through the registry.

So this measures rather than guesses.  Every ``_core_*`` / ``compute_*``
function in ``analysis.observables_2d`` is wrapped, and each dispatch key is
mapped to the cores its wrapper ultimately calls.  Running every test
module then yields, empirically, which keys are exercised.

Usage: python3 tests/coverage_report.py [--json OUT.json] [--quiet]
"""
import argparse
import collections
import glob
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import traceback
from contextlib import redirect_stdout

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import analysis.observables_2d as obs2d          # noqa: E402
import analysis.pipeline_dispatch as pd         # noqa: E402

WRAPPED = []          # (attr_name, original) for restore
HITS = collections.defaultdict(set)   # observable-calls -> {test tag}
HITS_FILE = os.environ.get('_COV_TRACE') or '/tmp/_cov_hits.json'
_TMPDIR = os.environ.get('_COV_TMP') or '/tmp'
TAG = ['?']


def _wrap_observable_cores():
    """Wrap every public compute core in observables_2d."""
    for name in dir(obs2d):
        if not re.fullmatch(r'(_core|compute)_[A-Za-z0-9_]+', name):
            continue
        real = getattr(obs2d, name)
        if not callable(real):
            continue
        WRAPPED.append((name, real))

        def make(name=name, real=real):
            def wrapper(*a, **kw):
                HITS[name].add(TAG[0])
                return real(*a, **kw)
            wrapper.__name__ = getattr(real, '__name__', name)
            return wrapper
        setattr(obs2d, name, make())


def _restore():
    for name, real in WRAPPED:
        setattr(obs2d, name, real)
    WRAPPED.clear()


def _calls_of(fn):
    """Names of observables_2d compute cores called (or imported) by fn."""
    names = set()
    if not callable(fn):
        return names
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return names
    for m in re.findall(r'\b(_core_[A-Za-z0-9_]+|compute_[A-Za-z0-9_]+)\s*\(', src):
        names.add(m)
    for m in re.findall(r'import\s+(_core_[A-Za-z0-9_]+|compute_[A-Za-z0-9_]+)', src):
        names.add(m)
    return {n for n in names if hasattr(obs2d, n)}


def _key_to_cores():
    """Map each dispatch key to every observable core it reaches.

    Resolution has to be transitive: a registry wrapper typically calls a
    registered ``compute_*`` dispatcher, which in turn calls a ``_core_*``
    function, and tests often call that ``_core_*`` directly.  Stopping at
    the first hop would miss every such test.
    """
    graph = {}
    for name in dir(obs2d):
        fn = getattr(obs2d, name)
        if callable(fn):
            graph[name] = _calls_of(fn)
    out = {}
    for key, entry in pd.PIPELINE_DISPATCH.items():
        seen, stack = set(), [entry.get('core_fn')]
        while stack:
            fn = stack.pop()
            if not callable(fn):
                continue
            nm = getattr(fn, '__name__', '')
            if nm in seen:
                continue
            seen.add(nm)
            for callee in graph.get(nm, ()):
                stack.append(getattr(obs2d, callee))
        out[key] = seen
    return out


def _record_counts():
    import yaml
    from analysis.dispatcher import resolve
    n = collections.Counter()
    for f in glob.glob(os.path.join(_ROOT, 'data', '*', '*.yaml')):
        try:
            docs = list(yaml.safe_load_all(open(f)))
        except Exception:
            continue
        for d in docs:
            if not d:
                continue
            try:
                n[resolve(d)['compute']] += 1
            except Exception:
                pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    # Build the call graph BEFORE wrapping: inspect.getsource on a wrapper
    # closure returns the wrapper's own source, not the real function's.
    key2cores = _key_to_cores()
    _wrap_observable_cores()

    # Each test module runs in its OWN subprocess: they mutate module-level
    # state and call sys.exit(), so running them in one interpreter both
    # leaks state and hides which module did what.
    tracer = os.path.join(_ROOT, 'tests', '_cov_child.py')
    for stale in glob.glob(os.path.join(_TMPDIR, 'hits_*.json')):
        os.remove(stale)
    with open(tracer, 'w') as fh:
        fh.write(TRACER_SRC)
    statuses = {}
    for path in sorted(glob.glob(os.path.join(_ROOT, 'tests', 'test_*.py'))):
        tag = os.path.basename(path)[len('test_'):-3]
        env = dict(os.environ, _COV_TAG=tag,
                   _COV_TRACE=os.path.join(_TMPDIR, f'hits_{tag}.json'))
        r = subprocess.run([sys.executable, tracer, path],
                           capture_output=True, text=True, env=env)
        statuses[tag] = 'ran' if r.returncode == 0 else (
            'FAILED' if 'FAIL' in r.stdout else
            'ERROR: ' + (r.stderr.strip().splitlines() or ['?'])[-1])

    # Merge every child's hit file (each child writes its own; a shared
    # path would leave only the last module's hits).
    for hf in sorted(glob.glob(os.path.join(_TMPDIR, 'hits_*.json'))):
        try:
            with open(hf) as fh:
                for nm, tags in json.load(fh).items():
                    HITS[nm].update(tags)
        except Exception:
            pass

    # Attribute hits to keys (a key is covered if any of its cores was hit).
    covered = collections.defaultdict(set)
    for key, names in key2cores.items():
        for nm in names:
            for t in HITS.get(nm, ()):
                covered[key].add(t)

    counts = _record_counts()
    allkeys = sorted(set(pd.PIPELINE_DISPATCH) | set(counts))
    rows = []
    for k in allkeys:
        rows.append({'compute_key': k, 'n_records': counts.get(k, 0),
                     'covered_by': sorted(covered.get(k, set()))})
    rows.sort(key=lambda r: -r['n_records'])

    if not args.quiet:
        print(f"{'compute key':28s} {'#recs':>6s}  exercised by")
        for r in rows:
            cov = ', '.join(r['covered_by']) or '*** NOT EXERCISED ***'
            print(f"{r['compute_key']:28s} {r['n_records']:6d}  {cov}")
    gaps = [r for r in rows if not r['covered_by']]
    print(f"\n{len(rows)-len(gaps)}/{len(rows)} keys exercised; "
          f"{len(gaps)} not exercised, carrying "
          f"{sum(r['n_records'] for r in gaps)} records")
    if gaps:
        print(f"  {sorted(r['compute_key'] for r in gaps)}")
    bad = {t: s for t, s in statuses.items() if s != 'ran'}
    if bad:
        print('\nmodules that did not run cleanly:')
        for t, s in bad.items():
            print(f'  {t}: {s}')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'rows': rows, 'module_status': statuses}, fh, indent=1)
        print(f'\nwrote {args.json}')


TRACER_SRC = '''
"""Child process: run one test module with the cores traced, dump hits."""
import collections, importlib.util, json, os, runpy, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import analysis.observables_2d as obs2d
import re
hits = collections.defaultdict(set)
tag = os.environ.get('_COV_TAG', '?')
for _n in dir(obs2d):
    if not re.fullmatch(r'(_core|compute)_[A-Za-z0-9_]+', _n):
        continue
    _real = getattr(obs2d, _n)
    if not callable(_real):
        continue
    def _mk(nm=_n, real=_real):
        def w(*a, **k):
            hits[nm].add(tag)
            return real(*a, **k)
        w.__name__ = getattr(real, '__name__', nm)
        return w
    setattr(obs2d, _n, _mk())
_path = sys.argv[1]
_spec = importlib.util.spec_from_file_location('_cov_mod', _path)
_mod = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_mod)
    if hasattr(_mod, 'main'):
        _mod.main()
    else:
        runpy.run_path(_path, run_name='__main__')
except SystemExit:
    pass
json.dump({k: sorted(v) for k, v in hits.items()},
          open(os.environ['_COV_TRACE'], 'w'))
'''


class _NullOut:
    def write(self, *a, **k):
        return len(a[0]) if a else 0

    def flush(self):
        pass


if __name__ == '__main__':
    main()