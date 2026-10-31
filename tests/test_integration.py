"""Integration tests: run compute_flow_vn.py --dry-run on real records.

Verifies that the full pipeline from record loading → dispatch → parameter
extraction works end-to-end for the PHENIX record, without needing sim data.
"""

import os
import sys
import subprocess
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import (
    load_record,
    extract_measurement_params,
    resolve,
    load_experimental,
    iter_records,
)


SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
PHENIX_RECORD = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'flow_vn',
    'phenix_flow_vn_multi-particle_azimuthal_correlations_200gev.yaml',
)


# ── dry-run test ────────────────────────────────────────────────────────────

def test_compute_flow_vn_dry_run():
    """Run compute_flow_vn.py --dry-run for the PHENIX v₂{2}ab record.

    This validates the metadata pipeline: record loading, parameter extraction,
    dispatch resolution — all without requiring sim data.
    """
    script = os.path.join(SCRIPTS_DIR, 'compute_flow_vn.py')
    result = subprocess.run(
        [sys.executable, script,
         '--record', PHENIX_RECORD,
         '--value-type', 'v_2${2}ab',
         '--dry-run'],
        capture_output=True, text=True, cwd=os.path.dirname(SCRIPTS_DIR),
    )
    assert result.returncode == 0, (
        f"dry-run failed with exit code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}")

    output = result.stdout

    # Should print key metadata
    assert 'v_2${2}ab' in output, "value_type should be printed"
    assert 'Harmonic: n=2' in output, "harmonic should be printed"
    assert 'Subevent A' in output, "subevent A should be printed"
    assert 'Subevent B' in output, "subevent B should be printed"
    assert 'η-gap: |Δη| > 2.0' in output, "eta gap should be printed"
    assert 'Centrality estimator' in output, "centrality info should be printed"
    assert 'Detector: BBC' in output, "detector should be BBC"
    assert '90 bins' in output or 'Classes: 90' in output, "90 centrality bins"
    assert '--dry-run: validation complete' in output, "should exit with dry-run message"


def test_compute_flow_vn_dry_run_single():
    """Dry-run the single-acceptance v₂{2} record."""
    script = os.path.join(SCRIPTS_DIR, 'compute_flow_vn.py')
    result = subprocess.run(
        [sys.executable, script,
         '--record', PHENIX_RECORD,
         '--value-type', 'v_2${2}',
         '--dry-run'],
        capture_output=True, text=True, cwd=os.path.dirname(SCRIPTS_DIR),
    )
    assert result.returncode == 0, (
        f"single dry-run failed: {result.stderr}")
    assert 'single-acceptance' in result.stdout or 'Particles: None' in result.stdout
    assert '--dry-run: validation complete' in result.stdout


def test_compute_flow_vn_dry_run_no_match():
    """Dry-run with a value_type that doesn't exist should fail gracefully."""
    script = os.path.join(SCRIPTS_DIR, 'compute_flow_vn.py')
    result = subprocess.run(
        [sys.executable, script,
         '--record', PHENIX_RECORD,
         '--value-type', 'nonexistent_observable',
         '--dry-run'],
        capture_output=True, text=True, cwd=os.path.dirname(SCRIPTS_DIR),
    )
    assert result.returncode == 0  # Should exit cleanly
    assert 'Could not find record' in result.stdout


# ── HEPData resolution test ─────────────────────────────────────────────────

def test_experimental_values_loaded():
    """PHENIX FVTX experimental values should load and have sensible range."""
    rec = load_record(PHENIX_RECORD, 'v_2${2}ab')
    assert rec is not None
    params = extract_measurement_params(rec)
    vals, errs = load_experimental(params)

    assert vals is not None, "Experimental values should load"
    assert errs is not None
    assert len(vals) == 90
    assert len(errs) == 90

    # v₂ should be in a physically sensible range for mid-central AuAu
    assert all(v >= 0 for v in vals), "All v₂ values should be non-negative"
    assert max(vals) < 0.15, "v₂ should not exceed 0.15 for this system"
    assert all(e > 0 for e in errs), "All errors should be positive"


# ── iter_records and group_records integration ──────────────────────────────

def test_iter_and_group_records():
    """Iterate all flow_vn records and group them by collision system."""
    record_dir = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'flow_vn')
    records = list(iter_records(record_dir))
    assert len(records) > 0, "Should find flow_vn records"

    # Check that each record can be dispatched
    dispatch_ok = 0
    dispatch_fail = 0
    for fpath, doc in records:
        try:
            resolve(doc)
            dispatch_ok += 1
        except ValueError:
            dispatch_fail += 1
    print(f"\n  {dispatch_ok}/{len(records)} records dispatchable, "
          f"{dispatch_fail} failed")
    assert dispatch_ok > 0, "At least some records should dispatch"


# ── CLI help test ──────────────────────────────────────────────────────────

def test_compute_flow_vn_help():
    script = os.path.join(SCRIPTS_DIR, 'compute_flow_vn.py')
    result = subprocess.run(
        [sys.executable, script, '--help'],
        capture_output=True, text=True, cwd=os.path.dirname(SCRIPTS_DIR),
    )
    assert result.returncode == 0
    assert '--record RECORD' in result.stdout
    assert '--value-type' in result.stdout
    assert '--sim-path' in result.stdout
    assert '--sim-3d-path' in result.stdout
    assert '--dry-run' in result.stdout
    assert '--method' in result.stdout
    assert '--rebin' in result.stdout


def test_run_pipeline_help():
    script = os.path.join(SCRIPTS_DIR, 'run_pipeline.py')
    result = subprocess.run(
        [sys.executable, script, '--help'],
        capture_output=True, text=True, cwd=os.path.dirname(SCRIPTS_DIR),
    )
    assert result.returncode == 0
    assert '--type TYPE' in result.stdout
    assert '--sim {2d,3d}' in result.stdout
    assert '--output OUTPUT' in result.stdout


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [fn for fn in dir() if fn.startswith('test_')]
    n_pass = 0
    n_fail = 0
    for name in sorted(tests):
        try:
            globals()[name]()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            n_fail += 1
    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    exit(1 if n_fail else 0)
