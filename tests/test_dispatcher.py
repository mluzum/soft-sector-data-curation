"""Unit tests for analysis.dispatcher module."""

import os
import sys
import tempfile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import (
    DISPATCH_TABLE,
    COMPUTE_REGISTRY,
    register,
    resolve,
    _field_exists,
    load_record,
    extract_measurement_params,
    load_experimental,
    iter_records,
)


# ── _field_exists ────────────────────────────────────────────────────────────

def test_field_exists_top_level():
    assert _field_exists({'a': 1}, 'a')
    assert not _field_exists({'a': 1}, 'b')


def test_field_exists_nested():
    assert _field_exists({'a': {'b': 2}}, 'a.b')
    assert not _field_exists({'a': {'b': 2}}, 'a.c')
    assert not _field_exists({'a': {'b': 2}}, 'x.y')


def test_field_exists_none_value():
    assert not _field_exists({'a': None}, 'a')


# ── Dispatch table structure ────────────────────────────────────────────────

def test_dispatch_table_has_known_entries():
    assert DISPATCH_TABLE
    for key, entry in DISPATCH_TABLE.items():
        assert 'compute' in entry
        assert 'requires' in entry
        assert 'description' in entry
        assert len(key) == 4  # (method, order, particles, dim)


def test_dispatch_table_requires_are_strings():
    for entry in DISPATCH_TABLE.values():
        for req in entry['requires']:
            assert isinstance(req, str), f"requires entry not a string: {req}"


# ── resolve ──────────────────────────────────────────────────────────────────

def test_resolve_single_acceptance():
    record = {
        'method': 'cumulant',
        'cumulant_order': 2,
        'particles': None,
        'dimensionality': 'integrated',
        'kinematic_range': {'eta': {'min': 0, 'max': 1},
                            'pT': {'min': 0.2, 'max': 3}},
    }
    entry = resolve(record)
    assert entry['compute'] == 'vn2_single'
    assert 'v_n{2} single-acceptance' in entry['description']


def test_resolve_subevent():
    record = {
        'method': 'cumulant',
        'cumulant_order': 2,
        'particles': 'AB',
        'dimensionality': 'integrated',
        'subevents': {
            'A': {'kinematic_range': {'eta': {'min': 1, 'max': 3},
                                       'pT': {'min': 0.2, 'max': 20}}},
            'B': {'kinematic_range': {'eta': {'min': -3, 'max': -1},
                                       'pT': {'min': 0.2, 'max': 20}}},
        },
    }
    entry = resolve(record)
    assert entry['compute'] == 'vn2_gap'
    assert 'two-subevent' in entry['description']


def test_resolve_raises_on_missing_field():
    record = {
        'method': 'cumulant',
        'cumulant_order': 2,
        'particles': None,
        'dimensionality': 'integrated',
        # Missing kinematic_range.eta
    }
    try:
        resolve(record)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'kinematic_range.eta' in str(e)


def _sp_record(**ref):
    return {
        'method': 'event_plane', 'dimensionality': 'integrated', 'value_type': 'v2{ZDC}',
        'kinematic_range': {'eta': {'min': -1, 'max': 1}, 'pT': {'min': 0.2, 'max': 2}},
        'flow_reference': dict(detector='ZDC-SMD', eta=[{'min': 6.3, 'max': 99}], **ref),
    }


def test_resolve_refuses_spectator_plane():
    """Flow relative to the spectator plane cannot come from a model without spectators."""
    try:
        resolve(_sp_record(plane='spectator'))
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'spectator plane' in str(e)
    # the same reference read as a participant plane still dispatches (SP estimator)
    assert resolve(_sp_record(plane='participant'))['compute'] == 'vn2_sp'
    assert resolve(_sp_record())['compute'] == 'vn2_sp'


def test_resolve_refuses_detector_level():
    """Quantities not corrected for efficiency are not model observables."""
    record = {'observable_type': 'particle_yields', 'value_type': 'Nch',
              'efficiency_corrected': False,
              'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5},
                                  'pT': {'min': 0.2, 'max': 100}}}
    try:
        resolve(record)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'detector-level' in str(e)
    record['efficiency_corrected'] = True
    resolve(record)


def test_resolve_raises_on_unknown_key():
    record = {
        'method': 'nonexistent',
        'cumulant_order': 99,
        'particles': None,
        'dimensionality': 'integrated',
    }
    try:
        resolve(record)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert 'No dispatch entry' in str(e)


# ── register decorator ──────────────────────────────────────────────────────

def test_register_and_lookup():
    @register('test_fn')
    def dummy():
        pass
    assert 'test_fn' in COMPUTE_REGISTRY
    assert COMPUTE_REGISTRY['test_fn'] is dummy
    # Clean up
    del COMPUTE_REGISTRY['test_fn']


# ── load_record ──────────────────────────────────────────────────────────────

def test_load_record_found():
    record_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'data', 'flow_vn',
        'phenix_flow_vn_multi-particle_azimuthal_correlations_200gev.yaml',
    )
    value_type = 'v_2${2}ab'
    rec = load_record(record_path, value_type)
    assert rec is not None, f"load_record returned None for '{value_type}' in {record_path}"
    vt = rec.get('value_type', '')
    assert vt == value_type, f"Expected '{value_type}', got '{vt}'"
    iid = rec.get('inspire_id', 0)
    assert int(iid) == 1670164, f"Expected inspire_id 1670164, got {iid!r}"


def test_load_record_not_found():
    record_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'data', 'flow_vn',
        'phenix_flow_vn_multi-particle_azimuthal_correlations_200gev.yaml',
    )
    rec = load_record(record_path, 'nonexistent_value_type')
    assert rec is None


def test_load_record_returns_first_match():
    record_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'data', 'flow_vn',
        'phenix_flow_vn_multi-particle_azimuthal_correlations_200gev.yaml',
    )
    rec = load_record(record_path, 'v_2')
    assert rec is not None
    # Should find v_2${2} first (appears before v_2${2}ab in the file)
    vt = rec.get('value_type', '')
    assert 'v_2' in vt


# ── extract_measurement_params ──────────────────────────────────────────────

def test_extract_params_subevent():
    record = {
        'harmonic': 2,
        'particles': 'AB',
        'kinematic_range': {'eta': {'min': -1, 'max': 1},
                             'pT': {'min': 0.1, 'max': 5}},
        'subevents': {
            'A': {'kinematic_range': {'eta': {'min': 1, 'max': 3},
                                       'pT': {'min': 0.2, 'max': 20}}},
            'B': {'kinematic_range': {'eta': {'min': -3, 'max': -1},
                                       'pT': {'min': 0.2, 'max': 20}}},
        },
        'eta_gap': 2.0,
        'centrality': {
            'classes': ['0-5%', '5-10%'],
            'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
            'estimator': {'detector': 'BBC', 'observable': 'amplitude'},
        },
        'hepdata_file': 'test.yaml',
        'inspire_id': 12345,
        'version': 1,
    }
    params = extract_measurement_params(record)

    assert params['harmonic'] == 2
    assert params['particles'] == 'AB'
    assert params['eta_gap'] == 2.0
    assert params['has_gap'] is True

    # Subevent ranges
    assert 'A' in params['subevent_ranges']
    assert 'B' in params['subevent_ranges']
    assert params['subevent_ranges']['A']['eta'] == (1, 3)
    assert params['subevent_ranges']['B']['eta'] == (-3, -1)

    # Single range should default to subevent A
    assert params['single_range']['eta'] == (1, 3)

    # Centrality
    assert len(params['cent_edges']) == 2
    assert params['cent_edges'][0] == (0, 5)
    assert params['cent_classes'] == ['0-5%', '5-10%']
    assert params['cent_eta_ranges'] == [(-3.9, -3.0), (3.0, 3.9)]


def test_extract_params_single_acceptance():
    record = {
        'harmonic': 3,
        'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5},
                             'pT': {'min': 0.15, 'max': 3}},
        'centrality': {
            'classes': ['0-10%', '10-20%'],
            'eta': [{'min': -1, 'max': 1}],
        },
    }
    params = extract_measurement_params(record)

    assert params['harmonic'] == 3
    assert params['particles'] is None
    assert params['single_range']['eta'] == (-0.5, 0.5)
    assert params['single_range']['pT'] == (0.15, 3)
    assert params['subevent_ranges'] == {}


def test_extract_params_missing_top_kr_fallback():
    """When subevents have kinematic_range but top-level doesn't."""
    record = {
        'harmonic': 2,
        'particles': 'AB',
        'subevents': {
            'A': {'kinematic_range': {'eta': {'min': 1, 'max': 3},
                                       'pT': {'min': 0.2, 'max': 5}}},
            'B': {'kinematic_range': {'eta': {'min': -3, 'max': -1},
                                       'pT': {'min': 0.2, 'max': 5}}},
        },
    }
    params = extract_measurement_params(record)
    assert params['single_range']['eta'] == (1, 3)


# ── load_experimental ────────────────────────────────────────────────────────

def test_load_experimental_phenix():
    """PHENIX FVTX record's raw data file should exist and parse."""
    params = {
        'inspire_id': 1670164,
        'version': 1,
        'hepdata_file': 'ppg218fig6a2.yaml',
    }
    vals, errs = load_experimental(params)
    assert vals is not None, "Experimental values should load"
    assert len(vals) == 90
    assert errs is not None
    assert len(errs) == 90
    # First bin should have reasonable v₂ value
    assert 0.01 < vals[0] < 0.1


def test_load_experimental_missing_file():
    vals, errs = load_experimental({
        'inspire_id': 999999,
        'version': 99,
        'hepdata_file': 'nonexistent.yaml',
    })
    assert vals is None
    assert errs is None


# ── iter_records ─────────────────────────────────────────────────────────────

def test_iter_records_flow_vn():
    record_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'flow_vn')
    records = list(iter_records(record_dir))
    assert len(records) > 0
    for fpath, doc in records:
        assert fpath.endswith('.yaml')
        assert 'value_type' in doc


def test_iter_records_filtered():
    record_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'flow_vn')
    records = list(iter_records(record_dir, value_type_filter='PHENIX'))
    # All returned records should have 'PHENIX' in value_type
    for _, doc in records:
        assert 'PHENIX' in doc.get('value_type', '')


# ── PHENIX record end-to-end extraction ──────────────────────────────────────

def test_phenix_record_full_extraction():
    """End-to-end: load the PHENIX record and verify all extracted parameters."""
    record_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'data', 'flow_vn',
        'phenix_flow_vn_multi-particle_azimuthal_correlations_200gev.yaml',
    )
    rec = load_record(record_path, 'v_2${2}ab')
    assert rec is not None

    params = extract_measurement_params(rec)

    # Harmonic
    assert params['harmonic'] == 2

    # Particles / subevent
    assert params['particles'] == 'AB'
    assert params['has_gap'] is True
    assert params['eta_gap'] == 2.0

    # Subevent kinematic ranges
    assert params['subevent_ranges']['A']['eta'] == (1.0, 3.0)
    assert params['subevent_ranges']['A']['pT'] == (0.2, 20)
    assert params['subevent_ranges']['B']['eta'] == (-3.0, -1.0)
    assert params['subevent_ranges']['B']['pT'] == (0.2, 20)

    # Centrality
    assert len(params['cent_classes']) == 90
    assert params['cent_classes'][0] == '0-1%'
    assert params['cent_classes'][-1] == '89-90%'
    assert params['cent_eta_ranges'] == [(-3.9, -3.0), (3.0, 3.9)]
    assert params['cent_estimator']['detector'] == 'BBC'

    # Link to raw data
    assert params['inspire_id'] == '1670164'
    assert params['version'] == 1
    assert params['hepdata_file'] == 'ppg218fig6a2.yaml'


# ── Run tests when called directly ──────────────────────────────────────────

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
