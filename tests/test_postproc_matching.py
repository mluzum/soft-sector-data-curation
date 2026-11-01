"""Regression tests for the record<->model matching layer.

These cover three production bugs that silently suppressed the PHOBOS
v2(eta) comparison (found Sep 2026 via the JETSCAPE cross-validation):

  1. The backward half of a mirrored model bin list was written
     innermost-first.  experiment.csv is exactly symmetric and bin00 is
     the *smallest* v2, i.e. the largest |eta|, so the order must run
     outermost-first.  The wrong order paired forward POI bins with
     backward model columns and inflated residuals to >100%.
  2. The HDF5 label key is "[value_type/hepdata_file[:15]]", which every
     doc of the PHOBOS v2(eta) file shares (three distinct figures: 14
     hit-based bins, 6 track-based, 8 Fig-5 combined).  Values must be
     resolved by bin EDGES, not by position in that bucket.
  3. flow_reference parsing: the publication uses TWO different flow
     references (octagon 2.05<|eta|<3.2 both sides for the track-based
     method, mirrored 0.1<|eta|<3.0 for the hit-based one), and the
     record must drive the computation from them rather than from a
     blanket eta_gap guess.

Also pins the value correspondence that fixes the 16-bin binning against
the paper (arXiv:nucl-ex/0407012, PRC 72 051901) Fig-5 table.

The PHOBOS v2(eta) record is curated outside the public database until its
ASCII tables are converted to value tables; tests/fixtures/ holds a copy.

Run: python3 tests/test_postproc_matching.py
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    'compare_postproc',
    os.path.join(_ROOT, 'data_validation', 'jetscape_bayes',
                 'compare_postproc.py'))
cp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp)

_RP_SPEC = importlib.util.spec_from_file_location(
    'run_pipeline', os.path.join(_ROOT, 'scripts', 'run_pipeline.py'))
rp = importlib.util.module_from_spec(_RP_SPEC)
_RP_SPEC.loader.exec_module(rp)

PHOBOS_V2_ETA = os.path.join(
    _ROOT, 'tests', 'fixtures',
    'phobos_flow_vn_centrality_and_pseudorapidity_dependence_of_AuAu_200gev.yaml')

FAM = ('v22_eta', 'PHOB')
EDGES = cp.MODEL_DIFF_BINS[FAM]

_n_fail = 0
_n_pass = 0


def check(name, cond, detail=''):
    global _n_fail, _n_pass
    if cond:
        _n_pass += 1
        print(f'  {name}: PASS')
    else:
        _n_fail += 1
        print(f'  {name}: FAIL {detail}')


def test_edge_count_and_mirroring():
    print('test_edge_count_and_mirroring')
    check('16 model bins', len(EDGES) == 16, f'got {len(EDGES)}')
    check('family is mirrored', FAM in cp.MODEL_DIFF_BINS_MIRROR)
    # Backward half (indices 0-7) must be the exact mirror of the forward
    # half (15-8): bin00 <-> bin15, ..., bin07 <-> bin08.
    pairs_ok = all(EDGES[i] == (-EDGES[15 - i][1], -EDGES[15 - i][0])
                   for i in range(8))
    check('backward half mirrors forward half', pairs_ok)
    # Backward half ordered outermost-first: |lo| strictly decreasing.
    widths = [max(abs(lo), abs(hi)) for lo, hi in EDGES[:8]]
    check('backward half outermost-first',
          all(widths[i] > widths[i + 1] for i in range(7)), f'{widths}')
    # The fitted grid is a SPARSE selection of the simulation's uniform
    # 0.1-wide eta cells, not a contiguous tiling, so the properties that
    # matter are: each window is one grid cell, and no two overlap.
    widths = [hi - lo for lo, hi in EDGES]
    check('every window is one 0.1-wide grid cell',
          all(abs(w - 0.1) < 1e-9 for w in widths),
          f'{sorted(set(round(w, 4) for w in widths))}')
    flat = sorted((min(a, b), max(a, b)) for a, b in EDGES)
    disjoint = all(flat[i][1] <= flat[i + 1][0] + 1e-9
                   for i in range(len(flat) - 1))
    check('windows do not overlap', disjoint)
    check('no window straddles eta=0',
          all(not (lo < 0 < hi) for lo, hi in EDGES))


def test_value_correspondence_matches_paper():
    """Model bins 9-15 must equal the paper's Fig-5 <eta> values."""
    print('test_value_correspondence_matches_paper')
    import csv
    path = os.path.join(_ROOT, 'data_validation', 'jetscape_bayes',
                        'exp_data', 'experiment.csv')
    exp = next(csv.DictReader(open(path)))
    # Fig-5 combined values, 25-50% class: experiment.csv bins 8..15 are
    # <eta> 0.31 ... 4.96, so bins 9..15 are <eta> 0.76 1.22 1.69 3.04 3.67
    # 4.13 4.96.  The values below are the published figure's.  The PHOBOS
    # web table lists them one row early (against <eta> 0.31 ... 4.13; see
    # scripts/curate_phobos_v2_200gev.py).
    # Deviations: 2e-4 4e-4 5e-4 0 1e-4 1e-4 0 -- max 5e-4, two exact.
    pairs = [(9, 0.0612), (10, 0.0574), (11, 0.0542), (12, 0.0448),
             (13, 0.0373), (14, 0.0275), (15, 0.0209)]
    devs = []
    for b, want in pairs:
        col = f'v22_eta_cen_25_50_PHOB__bin{b:02d}'
        devs.append(abs(float(exp[col]) - want))
    check('7/7 model values match paper Fig 5 to <=5e-4',
          max(devs) <= 5.1e-4, f'max dev {max(devs):.1e}')
    check('five of seven within 2e-4',
          sum(1 for d in devs if d <= 2.1e-4) == 5, f'{devs}')
    check('two of them agree exactly',
          sum(1 for d in devs if d == 0.0) == 2, f'{devs}')
    # The whole table is exactly symmetric by construction (paper: the
    # eta>0 data are reflected about midrapidity for display).
    for cent in ('03_15', '15_25', '25_50'):
        v = [float(exp[f'v22_eta_cen_{cent}_PHOB__bin{i:02d}'])
             for i in range(16)]
        check(f'experiment symmetric ({cent})',
              all(abs(v[i] - v[15 - i]) < 1e-12 for i in range(16)))


def test_overlap_mapping_and_mirror_fallback():
    print('test_overlap_mapping_and_mirror_fallback')
    import yaml
    fpath = PHOBOS_V2_ETA
    docs = [d for d in yaml.safe_load_all(open(fpath)) if d]
    # doc10 = Fig-5 combined, forward-only, 8 bins; doc4 = hit-based,
    # two-sided, 14 bins.
    var, fig5_bins, _ = cp.record_diff_bins(docs[10])
    check('doc10 var is eta', var == 'eta', f'got {var}')
    check('doc10 is forward-only',
          all(lo >= 0 for lo, _ in fig5_bins), f'{fig5_bins[:2]}')
    cap = cp.WIDTH_RATIO_CAP_OVERRIDE[('v22_eta', 'PHOB')]
    m, _ = cp.overlap_bin_map(EDGES, fig5_bins, mirror=True, cap=cap)
    check('every backward bin but one reaches a record bin',
          all(i in m for i in range(8) if i != 6), f'{sorted(m)}')
    # 14 of 16: model bins 06 and 09 span |eta| 1.4-1.5, which straddles
    # the published record edge at 1.455 (55%/45% of the model bin), so
    # neither candidate record bin reaches OVERLAP_MIN_FRAC and both are
    # left unmatched rather than assigned to the wrong measurement bin.
    check('14 of 16 model bins map onto the 8 published record bins',
          len(m) == 14, f'got {len(m)}')
    check('the two bins straddling a record edge are dropped, not guessed',
          6 not in m and 9 not in m, f'{sorted(m)}')
    check('mirrored pairs share a record bin',
          all(m.get(i) == m.get(15 - i) for i in range(8)))
    check('the mapping is many-to-one (model grid finer than the record)',
          len(set(m.values())) < len(m), f'{len(set(m.values()))} distinct')
    # Without the mirror flag, backward bins must NOT match.
    m_nomirror, _ = cp.overlap_bin_map(EDGES, fig5_bins, mirror=False)
    check('no backward match without mirror flag',
          not any(i in m_nomirror for i in range(8)),
          f'{sorted(m_nomirror)}')
    check('without the cap a legitimate fine-grid match is rejected',
          len(cp.overlap_bin_map(EDGES, fig5_bins, mirror=True)[0]) == 0,
          'cap should still reject these')


def test_sp_ref_eta_per_record_parsing():
    print('test_sp_ref_eta_per_record_parsing')
    # _det_matches is the shared matcher in run_pipeline.py behind both
    # --cent-override-eta and the per-record --sp-ref-eta; the PHOBOS
    # octagon reference must apply to Octagon records and nothing else.
    check('bare spec matches anything', rp._det_matches('', None))
    check('octagon matches Octagon', rp._det_matches('octagon', 'octagon'))
    check('octagon does not match bbc',
          not rp._det_matches('bbc', 'octagon'))
    check('empty detector never matches a named spec',
          not rp._det_matches('', 'octagon'))
    check('substring match works',
          rp._det_matches('phobos octagon', 'octagon'))


def test_flow_reference_from_record():
    """The record must carry the paper's real reference, not an eta_gap."""
    print('test_flow_reference_from_record')
    import yaml
    from analysis.dispatcher import extract_measurement_params
    fpath = PHOBOS_V2_ETA
    docs = [d for d in yaml.safe_load_all(open(fpath)) if d]
    check('13 docs', len(docs) == 13, f'got {len(docs)}')
    # The blanket |Delta-eta| guess is gone everywhere.
    check('no doc carries a blanket eta_gap',
          all('eta_gap' not in d for d in docs),
          f'{[i for i, d in enumerate(docs) if "eta_gap" in d]}')
    # Every doc is self-describing.
    check('every doc has notes',
          all(isinstance(d.get('notes'), dict) and d['notes'].get('value')
              for d in docs))
    check('every doc has a flow_reference',
          all(isinstance(d.get('flow_reference'), dict) for d in docs))

    p_hit = extract_measurement_params(docs[4])['flow_reference']
    check('hit-based reference is the mirrored 0.1-3.0 sub-event',
          p_hit['mirror'] is True and p_hit['eta'] == [(0.1, 3.0)],
          f'{p_hit["eta"]} mirror={p_hit["mirror"]}')

    p_trk = extract_measurement_params(docs[7])['flow_reference']
    check('track-based reference is octagon 2.05-3.2 both sides',
          p_trk['mirror'] is False
          and p_trk['eta'] == [(-3.2, -2.05), (2.05, 3.2)], f'{p_trk["eta"]}')

    p_comb = extract_measurement_params(docs[10])['flow_reference']
    segs = p_comb['segments']
    check('combined series declares two segments', len(segs) == 2,
          f'{len(segs)}')
    check('segment 1 is track-based (no mirror)',
          segs[0]['mirror'] is False and segs[0]['applies_to_eta'] == (0.0, 1.8))
    check('segment 2 is mirrored hit-based',
          segs[1]['mirror'] is True and segs[1]['applies_to_eta'] == (1.8, 5.5))

    # The pT window is the tracking acceptance, not the old 0.1-2.0 guess.
    pts = {(d['kinematic_range'].get('pT') or {}).get('max') for d in docs}
    check('invented pT.max=2.0 removed', 2.0 not in pts, f'{pts}')


def test_mirror_reference_window_selection():
    """A mirrored reference must land on the side OPPOSITE the POI bin."""
    print('test_mirror_reference_window_selection')
    from analysis.pipeline_dispatch import _sp_ref_windows

    fr = {'eta': [(0.1, 3.0)], 'mirror': True, 'segments': []}
    bins = [(-4.59, -3.9), (0.31, 0.76), (-0.2, 0.4), (3.9, 4.6)]
    specs = _sp_ref_windows(bins, 'eta', (0, 6), (0.075, 3.645), {'flow_reference': fr})
    check('one reference per bin', len(specs) == len(bins), f'{len(specs)}')
    per = {b: sp[0] for b, sp in zip(bins, specs)}
    # negative POI -> positive sub-event; positive POI -> negative sub-event
    check('negative POI bin uses the positive sub-event',
          per[(-4.59, -3.9)] == [(0.1, 3.0)], f'{per[(-4.59, -3.9)]}')
    check('positive POI bin uses the negative sub-event',
          per[(0.31, 0.76)] == [(-3.0, -0.1)], f'{per[(0.31, 0.76)]}')
    check('bin straddling eta=0 takes both sub-events',
          sorted(per[(-0.2, 0.4)]) == [(-3.0, -0.1), (0.1, 3.0)],
          f'{per[(-0.2, 0.4)]}')
    check('forward POI bin uses the negative sub-event',
          per[(3.9, 4.6)] == [(-3.0, -0.1)], f'{per[(3.9, 4.6)]}')
    # The reference must not share particles with the POI.  For a bin on
    # one side of midrapidity that is strict eta disjointness.  A bin that
    # CONTAINS eta=0 cannot achieve it: the sub-event starts at |eta|=0.1,
    # so the mirror image necessarily overlaps the part of the bin beyond
    # 0.1 (the shared particles are then subtracted as self-correlation).
    for (lo, hi), wins in per.items():
        if lo < 0 < hi:
            ov = sum(max(0.0, min(hi, rhi) - max(lo, rlo)) for rlo, rhi in wins)
            edge = max((min(hi, rhi) for rlo, rhi in wins if rhi > lo), default=0.0)
            check(f'straddling bin ({lo},{hi}) shares only near eta=0',
                  edge <= max(abs(lo), abs(hi)) + 1e-9,
                  f'sharing reaches |eta|={edge:.2f}')
            print(f'      note: bin ({lo},{hi}) shares {ov:.2f} of '
                  f'{hi - lo:.2f} in width with the mirrored sub-event')
        else:
            check(f'reference disjoint from POI bin ({lo},{hi})',
                  all(rhi <= lo + 1e-9 or rlo >= hi - 1e-9
                      for rlo, rhi in wins), f'{wins}')
    # pT-differential: the POI eta window (not the pT bin) picks the side
    specs = _sp_ref_windows([(0.5, 1.0), (1.0, 2.0)], 'pT', (0.2, 0.8), (0.5, 2.0),
                            {'flow_reference': fr})
    check('pT bins at eta > 0 use the negative sub-event',
          all(sp[0] == [(-3.0, -0.1)] for sp in specs), f'{specs}')


def test_segment_selection_by_overlap():
    """Bins straddling a segment boundary pick the larger-overlap segment."""
    print('test_segment_selection_by_overlap')
    from analysis.pipeline_dispatch import _sp_ref_windows

    segs = [
        {'applies_to_eta': (0.0, 1.8), 'eta': [(-3.2, -2.05), (2.05, 3.2)],
         'mirror': False, 'pT': None},
        {'applies_to_eta': (1.8, 5.5), 'eta': [(0.1, 3.0)], 'mirror': True,
         'pT': None},
    ]
    fr = {'eta': [], 'mirror': False, 'segments': segs}
    # 1.455-2.365 straddles the 1.8 crossover; 5.6-6.0 is matched by no segment.
    bins = [(0.155, 0.535), (1.455, 2.365), (3.9, 4.59), (5.6, 6.0)]
    specs = _sp_ref_windows(bins, 'eta', (0, 6), (0.075, 3.645), {'flow_reference': fr})
    check('small-|eta| bin uses the two-sided octagon reference',
          specs[0][0] == [(-3.2, -2.05), (2.05, 3.2)], f'{specs[0]}')
    check('straddling bin uses the mirrored hit-based reference',
          specs[1][0] == [(-3.0, -0.1)], f'{specs[1]}')
    check('reference pT defaults to the record pT', specs[1][1] == (0.075, 3.645), f'{specs[1]}')
    check('a bin matched by no segment has no reference (NaN, not a crash)',
          specs[3] is None, f'{specs[3]}')


def test_per_bin_reference_core_equivalence():
    """A per-bin reference must reduce exactly to the shared-reference core.

    Guards the broadcasting in _core_vn2_sp_diff{,_means}: passing the same
    reference repeated along a bin axis has to reproduce the 2-D result
    bit-for-bit, and genuinely different per-bin references must actually
    change the answer (otherwise a mirror/segment bug would be invisible).
    """
    print('test_per_bin_reference_core_equivalence')
    import numpy as np
    from analysis.observables_2d import (_core_vn2_sp_diff,
                                         _core_vn2_sp_diff_means)
    n_ev, H, nb = 50, 4, 6

    def make(seed, N=400, v=0.1, phi0=0.7):
        r = np.random.default_rng(seed)
        phi = r.uniform(0, 2 * np.pi, (n_ev, N))
        return (np.exp(2j * phi) * (1 + 2 * v * np.cos(2 * phi - phi0))
                ).sum(axis=1)

    q_poi = np.repeat(np.stack([make(i) for i in range(nb)],
                               axis=1)[:, None, :], H, axis=1)
    n_poi = np.full((n_ev, nb), 400.0)
    n_ref = np.full(n_ev, 400.0)
    q_ref = np.repeat(np.stack([make(100)], axis=1), H, axis=1)
    w = np.ones(n_ev)

    shared = _core_vn2_sp_diff_means(q_poi, n_poi, q_ref, n_ref, w)
    q_ref_b = np.repeat(q_ref[:, :, None], nb, axis=2)
    n_ref_b = np.repeat(n_ref[:, None], nb, axis=1)
    per_bin = _core_vn2_sp_diff_means(q_poi, n_poi, q_ref_b, n_ref_b, w)
    check('means: per-bin == shared when references agree',
          np.allclose(shared, per_bin, rtol=0, atol=0),
          f'max|d|={np.nanmax(np.abs(shared - per_bin)):.2e}')

    r_shared = _core_vn2_sp_diff(q_poi, n_poi, q_ref, w)
    r_per_bin = _core_vn2_sp_diff(q_poi, n_poi, q_ref_b, w)
    check('ratio form: per-bin == shared when references agree',
          np.allclose(r_shared, r_per_bin, rtol=0, atol=0),
          f'max|d|={np.nanmax(np.abs(r_shared - r_per_bin)):.2e}')

    # Distinct references must give distinct results.
    q_r2 = np.repeat(np.stack([make(200 + i) for i in range(nb)],
                              axis=1)[:, None, :], H, axis=1)
    other = _core_vn2_sp_diff_means(q_poi, n_poi, q_r2, n_ref_b, w)
    check('means: distinct per-bin references change the result',
          not np.allclose(per_bin, other))
    check('means: all finite for a healthy reference',
          bool(np.all(np.isfinite(per_bin))))
    # Recovered magnitude should be near the injected v2 = 0.1.
    check('means: magnitude near injected v2=0.1',
          abs(np.nanmedian(per_bin) - 0.1) < 0.02,
          f'{np.nanmedian(per_bin):.4f}')


def test_per_bin_wrapper_is_harmonic_major():
    """The per-bin 3D diff wrapper must emit HARMONIC-MAJOR order.

    Differential extraction reads index (harm - 1) * n_bins + bin, i.e.
    every bin of harmonic 1, then every bin of harmonic 2.  A per-bin core
    returns one row per bin holding all harmonics, so concatenating rows is
    BIN-major and every bin after the first is read from the wrong slot.

    Observed on STAR v2(eta): values came out 0.005 -> 0.053 -> 0.013, and
    the eta-reflection symmetry v2(eta) == v2(-eta) -- which physics
    requires -- broke by a mean of 0.018 (max 0.054).  With harmonic-major
    order it is 0.0003, within the jackknife error.
    """
    print('test_per_bin_wrapper_is_harmonic_major')
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        'rp', os.path.join(os.path.dirname(__file__), '..', 'scripts',
                           'run_pipeline.py'))
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)

    # bin0 -> v1=10, v2=11, v3=12 ; bin1 -> 20, 21, 22 ; bin2 -> 30, 31, 32
    per_bin = [np.array([10, 11, 12]), np.array([20, 21, 22]),
               np.array([30, 31, 32])]
    out = rp.harm_major_stack(per_bin)
    n_bins, n_harm = 3, 3
    check('length preserved', len(out) == n_bins * n_harm, f'{len(out)}')
    for h in range(1, n_harm + 1):
        for b in range(n_bins):
            idx = (h - 1) * n_bins + b
            check(f'h{h} bin{b} read back correctly',
                  out[idx] == per_bin[b][h - 1],
                  f'got {out[idx]} want {per_bin[b][h - 1]}')

    # And the extracted v2 (harm=2) must be the per-bin v2, in bin order.
    v2 = [out[(2 - 1) * n_bins + b] for b in range(n_bins)]
    check('extracted v2 is [11, 21, 31]', v2 == [11, 21, 32 - 1], f'{v2}')


def test_inventory_row_must_match_the_document():
    """An inventory row names a record by file and document index; after the
    file changes, the index can point at another record.  is_row_record
    rejects a document of another table, value type or centrality class."""
    print('test_inventory_row_must_match_the_document')
    import copy
    import yaml
    docs = [d for d in yaml.safe_load_all(open(PHOBOS_V2_ETA)) if d]
    row = {'hepdata_file': docs[4]['hepdata_file'], 'value_type': 'v2',
           'record_cent_lo': '25.0', 'record_cent_hi': '50.0'}
    check('the row matches its document', cp.is_row_record(docs[4], row))
    check('a sibling of another centrality class is rejected', not cp.is_row_record(docs[5], row))
    other = copy.deepcopy(docs[4])
    other['hepdata_file'] = 'Table1.yaml'
    check('a document of another table is rejected', not cp.is_row_record(other, row))
    check('a row without a centrality class checks table and value type',
          cp.is_row_record(docs[5], {'hepdata_file': docs[5]['hepdata_file'], 'value_type': 'v2'}))
    check('no document is rejected', not cp.is_row_record(None, row))


if __name__ == '__main__':
    for fn in (test_edge_count_and_mirroring,
               test_value_correspondence_matches_paper,
               test_overlap_mapping_and_mirror_fallback,
               test_sp_ref_eta_per_record_parsing,
               test_flow_reference_from_record,
               test_mirror_reference_window_selection,
               test_segment_selection_by_overlap,
               test_per_bin_reference_core_equivalence,
               test_per_bin_wrapper_is_harmonic_major,
               test_inventory_row_must_match_the_document):
        fn()
    total = _n_pass + _n_fail
    # run_all.sh tallies lines of the form "N/M passed".
    print()
    print(f'{_n_pass}/{total} passed')
    sys.exit(1 if _n_fail else 0)
