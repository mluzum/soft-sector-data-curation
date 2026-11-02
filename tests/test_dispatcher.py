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
    """Each requirement is a field path, or a list of alternative paths (one of)."""
    for entry in DISPATCH_TABLE.values():
        for req in entry['requires']:
            ok = isinstance(req, str) or (isinstance(req, list) and req and all(isinstance(r, str) for r in req))
            assert ok, f"requires entry not a path or list of paths: {req}"


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
        'phenix_flow_vn_fvtx_v2_v3_subevents_AuAu_200gev_ins1670164.yaml',
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
        'phenix_flow_vn_fvtx_v2_v3_subevents_AuAu_200gev_ins1670164.yaml',
    )
    rec = load_record(record_path, 'nonexistent_value_type')
    assert rec is None


def test_load_record_returns_first_match():
    record_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'data', 'flow_vn',
        'phenix_flow_vn_fvtx_v2_v3_subevents_AuAu_200gev_ins1670164.yaml',
    )
    rec = load_record(record_path, 'v_2')
    assert rec is not None
    # the first record matching 'v_2' (v_2${2}ab: the single-acceptance v_2${2} is staged)
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
        'phenix_flow_vn_fvtx_v2_v3_subevents_AuAu_200gev_ins1670164.yaml',
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
    assert params['subevent_ranges']['A']['eta'] == (-3.0, -1.0)
    assert params['subevent_ranges']['A']['pT'] == (0.3, 100.0)
    assert params['subevent_ranges']['B']['eta'] == (1.0, 3.0)
    assert params['subevent_ranges']['B']['pT'] == (0.3, 100.0)

    # Centrality
    assert len(params['cent_classes']) == 90
    assert params['cent_classes'][0] == '0-1%'
    assert params['cent_classes'][-1] == '89-90%'
    assert params['cent_eta_ranges'] == [(-3.9, -3.1), (3.1, 3.9)]
    assert params['cent_estimator']['detector'] == 'BBC'

    # Link to raw data
    assert params['inspire_id'] == '1670164'
    assert params['version'] == 1
    assert params['hepdata_file'] == 'ppg218fig6a2.yaml'


# ── Run tests when called directly ──────────────────────────────────────────

def test_axis_bins_points_and_windows():
    """A measurement published at mean positions: windows of point_window
    around each point (possibly disjoint); edges keep their meaning."""
    from analysis.dispatcher import axis_bins, extract_differential_bins
    assert axis_bins({'bins': [0.0, 0.5, 1.0]}) == [(0.0, 0.5), (0.5, 1.0)]
    assert axis_bins({'bin_ranges': [[-3.0, -2.5], [2.5, 3.0]]}) == [(-3.0, -2.5), (2.5, 3.0)]
    w = axis_bins({'points': [-4.0, 0.3], 'point_window': 0.2})
    assert [tuple(round(x, 9) for x in b) for b in w] == [(-4.1, -3.9), (0.2, 0.4)], w
    try:
        axis_bins({'points': [0.3]})
        raise AssertionError('points without point_window accepted')
    except ValueError:
        pass
    rec = {'dimensionality': 'differential_eta',
           'kinematic_range': {'eta': {'min': -5.4, 'max': 5.4, 'points': [-3.0, 1.0], 'point_window': 0.2},
                               'pT': {'min': 0.0, 'max': 5.0e9}}}
    var, bins, fixed = extract_differential_bins(rec)
    assert var == 'eta' and len(bins) == 2 and abs(bins[1][0] - 0.9) < 1e-9 and fixed == (0.0, 5.0e9)


def test_validator_points_rules():
    """points need a positive point_window and exclude bins; bin_ranges are lo < hi pairs."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from validate import validate_doc
    base = {'observable_type': 'flow_vn', 'inspire_id': '1', 'version': 1, 'hepdata_file': 'x.yaml',
            'value_type': 'v2', 'harmonic': 2, 'method': 'event_plane', 'dimensionality': 'differential_eta',
            'particle_id': {'species': 'charged'},
            'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
            'centrality': {'classes': ['0-50%'], 'estimator': {'name': 'Paddle', 'observable': 'energy'},
                           'eta': [{'min': 3.0, 'max': 4.5}]}}
    def errs(eta):
        rec = dict(base, kinematic_range={'eta': eta, 'pT': {'min': 0.0, 'max': 5.0}})
        return [e for e in validate_doc(rec, 'r') if 'kinematic_range' in e]
    assert not errs({'min': -5, 'max': 5, 'points': [0.3], 'point_window': 0.2})
    assert errs({'min': -5, 'max': 5, 'points': [0.3]})
    assert errs({'min': -5, 'max': 5, 'points': [0.3], 'point_window': 0.2, 'bins': [0.0, 1.0]})
    assert not errs({'min': -5, 'max': 5, 'bin_ranges': [[-3.0, -2.5], [2.5, 3.0]]})
    assert errs({'min': -5, 'max': 5, 'bin_ranges': [[-2.5, -3.0]]})
    assert errs({'min': -5, 'max': 5, 'bin_ranges': [[2.5, 3.0]], 'bins': [0.0, 1.0]})

    def cent_errs(**c):
        cent = dict(base['centrality'], **c)
        rec = dict(base, centrality=cent, kinematic_range={'eta': {'min': -1, 'max': 1}, 'pT': {'min': 0, 'max': 5}})
        return [e for e in validate_doc(rec, 'r') if 'centrality' in e]
    assert not cent_errs(classes=['58-60%', '1-3%'], points=[59.0, 2.0], point_window=2.0)
    assert not cent_errs(classes=['0-1.5%'], points=[0.5], point_window=2.0)          # clipped at 0
    assert cent_errs(classes=['55-62%'], points=[59.0], point_window=2.0)              # not the window
    assert cent_errs(classes=['58-60%'], points=[59.0])                                # no window
    assert cent_errs(classes=['58-60%', '1-3%'], points=[59.0], point_window=2.0)      # lengths


def test_load_experimental_total_error():
    """The experimental error is every HEPData component in quadrature: an
    asymmetric one as the mean of its sides, 'x%' relative to the value.
    (Until Nov 2026 a third component was ignored and an asymmetric second
    one dropped.)"""
    import numpy as np
    from analysis.dispatcher import load_experimental
    table = {'independent_variables': [{'header': {'name': 'X'}, 'values': [{'value': i} for i in range(4)]}],
             'dependent_variables': [{'header': {'name': 'V2'}, 'values': [
                 {'value': 1.0, 'errors': [{'symerror': 0.3, 'label': 'stat'},
                                           {'asymerror': {'plus': 0.5, 'minus': -0.3}, 'label': 'sys'}]},
                 {'value': 2.0, 'errors': [{'symerror': 0.1}, {'symerror': 0.2}, {'symerror': 0.2}]},
                 {'value': 4.0, 'errors': [{'symerror': '5%'}, {'symerror': 0.3}]},
                 {'value': 1.0, 'errors': [{'asymerror': {'plus': 0.2}}]}]}]}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 't.yaml')
        yaml.safe_dump(table, open(path, 'w'))
        v, e = load_experimental({'inspire_id': '0', 'version': 1, 'hepdata_file': 't.yaml', 'value_table': path})
        # errors_excluded: a component contained in another one (ALICE
        # 'syst. uncorr.' inside 'syst.') is not added again
        _, e2 = load_experimental({'inspire_id': '0', 'version': 1, 'hepdata_file': 't.yaml', 'value_table': path,
                                   'errors_excluded': ['sys']})
    assert list(v) == [1.0, 2.0, 4.0, 1.0], v
    want = [np.hypot(0.3, 0.4), 0.3, np.hypot(0.2, 0.3), 0.2]
    assert np.allclose(e, want), (e, want)
    assert np.allclose(e2, [0.3, 0.3, np.hypot(0.2, 0.3), 0.2]), e2


def test_fractional_centrality_edges():
    """Fractional class edges stay fractional (they were truncated to int)."""
    record = {'harmonic': 2, 'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}},
              'centrality': {'classes': ['0.15-0.2%', '5-11.25%', '0-5%'],
                             'estimator': {'name': 'V0M', 'observable': 'amplitude'},
                             'eta': [{'min': -3.7, 'max': 5.1}]}}
    edges = extract_measurement_params(record)['cent_edges']
    assert edges == [(0.15, 0.2), (5.0, 11.25), (0.0, 5.0)], edges
    from analysis.weights import cent_weights
    import numpy as np
    mult = np.arange(20000, 0, -1).astype(float)          # event i at (i+0.5)/200 %
    w = cent_weights('simple', mult, edges)
    assert w[:, 0].sum() == 10 and w[:, 1].sum() == 1250 and w[:, 2].sum() == 1000, w.sum(axis=0)


def test_table_rows_selects_rows():
    """table_rows: a record holds some rows of a table (overlapping classes,
    e.g. 0-1% and 0-5% in one ATLAS table, go to separate records)."""
    import tempfile
    tab = {'independent_variables': [{'header': {'name': 'CENTRALITY'},
                                      'values': [{'low': 0, 'high': 1}, {'low': 0, 'high': 5}, {'low': 5, 'high': 10}]}],
           'dependent_variables': [{'header': {'name': 'V2'},
                                    'values': [{'value': 0.02, 'errors': [{'symerror': 0.001}]},
                                               {'value': 0.03, 'errors': [{'symerror': 0.002}]},
                                               {'value': 0.05, 'errors': [{'symerror': 0.003}]}]}]}
    d = tempfile.mkdtemp()
    path = os.path.join(d, 't.yaml')
    yaml.safe_dump(tab, open(path, 'w'))
    rec = {'value_table': path, 'inspire_id': '0', 'version': 1, 'hepdata_file': 't.yaml', 'table_rows': [1, 2],
           'centrality': {'classes': ['0-5%', '5-10%'], 'estimator': {'name': 'x', 'observable': 'ET'},
                          'eta': [{'min': 3, 'max': 4}]},
           'kinematic_range': {'eta': {'min': -1, 'max': 1}}}
    p = extract_measurement_params(rec)
    v, e = load_experimental(dict(p, inspire_id='0', version=1, hepdata_file='t.yaml', dv_index=0))
    assert list(v) == [0.03, 0.05] and list(e) == [0.002, 0.003], (v, e)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from validate import validate_doc
    base = {'observable_type': 'flow_vn', 'inspire_id': '1', 'version': 1, 'hepdata_file': 'x.yaml',
            'value_type': 'v2', 'harmonic': 2, 'method': 'cumulant', 'cumulant_order': 2,
            'dimensionality': 'integrated', 'particle_id': {'species': 'charged'},
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
            'centrality': rec['centrality'], 'kinematic_range': {'eta': {'min': -1, 'max': 1}, 'pT': {'min': 0, 'max': 5}}}
    def errs(rows):
        return [x for x in validate_doc(dict(base, table_rows=rows), 'r') if 'table_rows' in x]
    assert not errs([1, 2]) and errs([2, 1]) and errs([]) and errs([-1])



def test_asymmetric_frame_and_spectator_estimator():
    """Frame fields reach the params; a spectator-neutron centrality is refused."""
    rec = {'observable_type': 'particle_yields', 'inspire_id': '0', 'version': 1, 'hepdata_file': 't.yaml',
           'value_type': 'dNch/deta',
           'collision_system': {'energy_gev': 8160.0, 'reaction': 'P PB --> CHARGED X', 'frame': 'lab',
                                'rapidity_shift': -0.465, 'forward_beam': 'Pb'},
           'centrality': {'classes': ['0-5%'], 'estimator': {'name': 'V0A', 'observable': 'amplitude'},
                          'eta': [{'min': 2.8, 'max': 5.1}]},
           'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}}}
    resolve(rec)
    p = extract_measurement_params(rec)
    assert (p['frame'], p['rapidity_shift'], p['forward_beam']) == ('lab', -0.465, 'Pb'), p
    zna = dict(rec, centrality=dict(rec['centrality'], estimator={'name': 'ZNA', 'observable': 'spectator_neutrons'}))
    try:
        resolve(zna)
        raise AssertionError('spectator-neutron centrality not refused')
    except ValueError as e:
        assert 'spectator' in str(e), e


def test_eta_sub_event_plane_two_subevents():
    """STAR eta-sub event plane declared as subevents (method event_plane,
    particles AB): the two-subevent correlation (vn2_gap_poi for an
    identified POI, vn2_gap for charged), not the single-set amplitude."""
    sub = {'A': {'kinematic_range': {'eta': {'min': -1.0, 'max': -0.05}, 'pT': {'min': 0.15, 'max': 5.0}}},
           'B': {'kinematic_range': {'eta': {'min': 0.05, 'max': 1.0}, 'pT': {'min': 0.15, 'max': 5.0}}}}
    rec = {'observable_type': 'flow_vn', 'method': 'event_plane', 'harmonic': 2, 'particles': 'AB',
           'subevents': sub, 'particle_id': {'species': 'p'}, 'dimensionality': 'differential_pT',
           'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 2.0, 'bins': [0.2, 1.0, 2.0]}},
           'centrality': {'classes': ['0-10%'], 'eta': [{'min': -0.5, 'max': 0.5}]}}
    assert resolve(rec)['compute'] == 'vn2_gap_poi', resolve(rec)['compute']
    ch = dict(rec, particle_id={'species': 'charged'}, dimensionality='integrated',
              kinematic_range={'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.15, 'max': 2.0}})
    assert resolve(ch)['compute'] in ('vn2_gap', 'vn2_gap_poi'), resolve(ch)['compute']
    sp = dict(rec, method='scalar_product')        # ALICE p-Pb v2{SP}: POI in a, reference in b
    assert resolve(sp)['compute'] == 'vn2_gap_poi', resolve(sp)['compute']


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
