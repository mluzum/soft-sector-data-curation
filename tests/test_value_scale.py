"""Regression tests for the experimental VALUE SCALE of curated records.

A publication may tabulate a quantity pre-multiplied for legibility --
"v_3(TPC) (%)", "v_3*100 (TPC)", "v_3^2*10^4".  HEPData returns those
verbatim with ``units: none``, so a record built from such a table carries
values 100x (or 10^4x) the physical quantity and NOTHING downstream can
tell.  Compared against a simulation fraction it is wrong by that factor
while still looking like a plausible number.

Every expected value below was checked against the raw HEPData tables in
hepdata_raw/, not inferred from the column header.

Run: python3 tests/test_value_scale.py
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    'fix_value_scale', os.path.join(os.path.dirname(__file__), '..',
                                    'scripts', 'fix_value_scale.py'))
_fvs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fvs)
classify_header = _fvs.classify_header

_spec2 = importlib.util.spec_from_file_location(
    'validate_metadata', os.path.join(os.path.dirname(__file__), '..',
                                      'scripts', 'validate_metadata.py'))
_vm = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_vm)

n_pass = n_fail = 0


def check(name, cond, detail=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f'  PASS  {name}' + (f'  {detail}' if detail else ''))
    else:
        n_fail += 1
        print(f'  FAIL  {name}  {detail}')


def test_percent_and_multiplier_headers():
    """Headers that really do pre-multiply the value."""
    print('test_percent_and_multiplier_headers')
    for hdr, want in (
            ('v_3(TPC) (%)', 1e-2),
            ('$v_{2}(\\%)$', 1e-2),
            ('$v_1$ (%)', 1e-2),
            ('dv_{1}/dy(\\%)', 1e-2),
            ('v_3*100 (TPC)', 1e-2),
            ('A*100 (cent 0-5%)', 1e-2),
            ('v_3^2*10^4 (cent 0-5%)', 1e-4),
            ('v_3^4(4)*10^6 (STAR)', 1e-6),
            ('A*1000 (LS)', 1e-3),
            ('slope*1000', 1e-3)):
        factor, kind, _ = classify_header(hdr)
        check(f'{hdr!r} -> x1/factor', kind == 'scaled' and factor == want,
              f'got kind={kind} factor={factor} want={want}')


def test_centrality_labels_are_not_scales():
    """A '%' attached to a centrality/acceptance word is a LABEL.

    This is the trap that would have introduced a 100x error: PHENIX
    ins585347 column "[F] 00 < Centrality (%) < 15" has v2 values of
    0.04-0.13, i.e. already fractions.  Scaling it by 0.01 is wrong.
    """
    print('test_centrality_labels_are_not_scales')
    for hdr in ('[F] 00 < Centrality (%) < 15',
                '[F] 30 < Centrality (%) < 60',
                '[A] 30 < Centrality (%) < 60',
                'v2 in centrality 0-5%',
                'V2{4} (10-20%)',
                '0-5%',
                'v_2$ in 10-40% central Au+Au collisions',
                'Elliptic Flow (TXT, $p_{T}>0.07$ GeV) 3-15%'):
        factor, kind, _ = classify_header(hdr)
        check(f'{hdr!r} is not scaled', factor is None,
              f'got factor={factor} kind={kind}')


def test_exponentiation_is_not_a_scale():
    """`V2**2` is exponentiation, not a multiplier."""
    print('test_exponentiation_is_not_a_scale')
    for hdr in ('SQRT(MEAN(V2**2)) / MEAN(V2)',
                '( V2{EP}* - V2{4}*',
                'SQRT(MEAN(V3* )) / MEAN(V3)'):
        factor, kind, _ = classify_header(hdr)
        check(f'{hdr!r} is not scaled', factor is None,
              f'got factor={factor} kind={kind}')


def test_unstated_scale_is_never_guessed():
    """A column merely labelled "Scaled" states no factor: flag, do not guess."""
    print('test_unstated_scale_is_never_guessed')
    for hdr in ('v_{2}$(EP), Pb+Pb, Scaled', 'scaled_V2', 'scaled_V3'):
        factor, kind, _ = classify_header(hdr)
        check(f'{hdr!r} flagged, factor None',
              factor is None and kind == 'scale_unknown',
              f'got kind={kind} factor={factor}')


def test_validator_agrees_with_fixer():
    """validate_metadata must not disagree with the fixer on any header.

    A disagreement is how a scaled record slips through unchecked (or a
    clean record gets a spurious value_scale).
    """
    print('test_validator_agrees_with_fixer')
    headers = [
        'v_3(TPC) (%)', '$v_{2}(\\%)$', 'v_3*100 (TPC)', 'A*100 (cent 0-5%)',
        'v_3^2*10^4 (cent 0-5%)', 'v_3^4(4)*10^6 (STAR)', 'A*1000 (LS)',
        'slope*1000', '[F] 00 < Centrality (%) < 15', 'v2 in centrality 0-5%',
        'SQRT(MEAN(V2**2)) / MEAN(V2)', 'v_{2}$(EP), Pb+Pb, Scaled',
        'v_2', 'invariant_yield', 'dN_dy', 'gamma_ss',
    ]
    for hdr in headers:
        factor, kind, _ = classify_header(hdr)
        scaled = kind == 'scaled' and factor is not None
        doc = {'value_type': hdr, 'source_column': hdr}
        flagged = _vm.check_value_scale(doc)
        check(f'fixer/validator agree on {hdr!r}',
              flagged == (scaled and factor != 1.0),
              f'scaled={scaled} factor={factor} validator_flags={flagged}')
        # A record that already carries value_scale is never re-flagged.
        doc2 = {'value_type': hdr, 'source_column': hdr, 'value_scale': 0.01}
        check(f'{hdr!r}: value_scale suppresses the warning',
              not _vm.check_value_scale(doc2))


def test_curated_records_carry_the_scale():
    """Every record whose source header pre-scales must carry value_scale."""
    print('test_curated_records_carry_the_scale')
    bad = []
    n_scaled = 0
    for d in ('flow_vn',):
        ddir = os.path.join(os.path.dirname(__file__), '..', 'data', d)
        for fn in sorted(os.listdir(ddir)):
            if not fn.endswith('.yaml'):
                continue
            for i, doc in enumerate(yaml.safe_load_all(open(os.path.join(ddir, fn)))):
                if not doc:
                    continue
                if doc.get('value_scale') is not None:
                    n_scaled += 1
                if _vm.check_value_scale(doc):
                    bad.append(f'{fn} doc{i}: {doc.get("source_column") or doc.get("value_type")!r}')
    # 35 public flow_vn records declare one (46 before the records that do not
    # compute end to end moved out of data/); guards against an empty scan.
    check(f'{n_scaled} flow_vn records carry an explicit value_scale',
          n_scaled >= 35, f'got {n_scaled}')
    check('no flow_vn record has an undeclared scale', not bad,
          '; '.join(bad[:5]))


def test_centrality_classes_check():
    """centrality.classes must be disjoint percentile bins.

    The field is systematically polluted in the curated data: 2055 records
    carry jet xT/pT bin edges there (atlas_jets '0.31623-0.35481%',
    cms_jets '0.3-0.4%'), 57 carry OVERLAPPING selections that need the
    record split, 64 do not parse, 12 run past 100%.  A gapped but integer
    list is LEGAL and must NOT be flagged -- cent_weights slices those per
    class.
    """
    print('test_centrality_classes_check')
    def flagged(classes):
        return _vm.check_centrality_classes({'centrality': {'classes': classes}})

    # legal: contiguous, gapped, single, MB-style, and FRACTIONAL percentile
    # bins -- STAR/ALICE publish equal-multiplicity classes such as
    # '5-11.25%' and '42.5-60%', and flagging those would block ~76
    # legitimate flow records.
    for cs in (['0-5%', '5-10%', '10-20%'],
               ['0-5%', '30-40%'],                 # gapped: legal
               ['0-5%'],
               ['0.0-80.0%'],
               ['10-20%', '30-40%', '50-60%'],     # the STAR eta-sub case
               ['0-5%', '5-11.25%', '11.25-20%'],  # fractional percentiles
               ['5-11.25%', '11.25-20%', '20-30%']):
        check(f'legal {cs}', not flagged(cs))

    # overlap -> needs the record SPLIT
    for cs in (['2-6%', '3-5%'],
               ['40-50%', '10-20%', '40-60%', '20-30%'],
               ['0-20%', '75-90%', '3-4%']):
        check(f'overlapping {cs}', flagged(cs))

    # kinematic bins misfiled as centrality (the '%' suffix is the tell)
    for cs in (['0.31623-0.35481%', '0.35481-0.39811%'],
               ['0.3-0.4%', '0.4-0.5%'],
               ['0.725-1.075%', '1.075-1.4%'],
               ['24.75-29.65%', '29.65-34.6%']):
        check(f'kinematic bins in centrality {cs}', flagged(cs))

    # beyond 100% and unparseable
    for cs in (['0-60%', '0-120%'], ['MB', '0-5%'], ['0-5', 'central']):
        check(f'invalid {cs}', flagged(cs))

    # A non-integer list whose span does not look like a centrality axis.
    # Deliberately CONSERVATIVE and deliberately biased towards flagging: a
    # false positive costs a visible skip with a reason, while a
    # false negative costs a silent wrong number. So a peripheral-only
    # fractional list ('42.5-60%') is flagged even though a publication
    # could in principle publish it.
    check('peripheral-only fractional list is flagged (conservative)',
          flagged(['42.5-60%', '60-80%']))


if __name__ == '__main__':
    for fn in sorted(dir()):
        if fn.startswith('test_'):
            globals()[fn]()
            print()
    # tests/run_all.sh parses this exact form for the suite total.
    print(f'{n_pass}/{n_pass + n_fail} passed')
    sys.exit(1 if n_fail else 0)