"""Toy-MC test: centrality classes combined from narrower analysis classes.

A record with ``centrality.combined_from: analysis_classes`` (ALICE 1%
bins, PHENIX 10% classes) reports the yield-weighted average of the values
computed in its sub-classes (analysis.cent_combine).  The toy sample mixes
two populations, central (M ~ 400, v2 = 0.06) and peripheral (M ~ 200,
v2 = 0.12), so a 0-100% class differs between
  - one estimate over the class (pair-weighted v2{2}):
    sqrt((400^2 0.06^2 + 200^2 0.12^2) / (400^2 + 200^2)) = 0.0759, and
  - the yield-weighted average of the 0-50% and 50-100% values:
    (400 x 0.06 + 200 x 0.12) / 600 = 0.0800.
Checks, on 2D and 3D containers, through the production path
(toy_mc.evaluate = extract_measurement_params -> resolve -> prepare ->
weights -> core): the combined value equals the yield-weighted average of
the sub-class values evaluated separately (to rounding), and the truth; the
'events' weight; listed analysis_classes == analysis_class_width; and the
refusal of sub-classes that do not tile a class.
"""


def _run():
    import copy
    import sys
    import numpy as np
    sys.path.insert(0, '.')
    from analysis.toy_mc import (ToyGenConfig, generate_events, make_qvector_data,
                                 make_qgrid_data, record_flow_vn2, record_yield, evaluate)
    from analysis.dispatcher import resolve

    results = []

    def check(name, cond, detail=''):
        results.append((name, bool(cond)))
        print(("  PASS " if cond else "  FAIL ") + name + (f"  ({detail})" if detail else ''))

    cfg_c = ToyGenConfig(n_events=1000, mult_mean=400, mult_sigma=30, seed=11,
                         flow={2: dict(kind='const', value=0.06)})
    cfg_p = ToyGenConfig(n_events=1000, mult_mean=200, mult_sigma=30, seed=12,
                         flow={2: dict(kind='const', value=0.12)})
    ev_c, truth = generate_events(cfg_c)
    ev_p, _ = generate_events(cfg_p)
    events = ev_c + ev_p
    containers = {'2D': dict(qv=make_qvector_data(events, truth)),
                  '3D': dict(grid=make_qgrid_data(events, truth, n_pt=40))}

    def combined(rec, **cent):
        r = copy.deepcopy(rec)
        r['centrality'].update({'combined_from': 'analysis_classes', **cent})
        return r

    # centrality from the multiplicity in the toy's acceptance (|y| < 1)
    CE = ((-1.0, 1.0),)
    whole = record_flow_vn2(cent_classes=('0-100%',), cent_eta=CE)
    halves = record_flow_vn2(cent_classes=('0-50%', '50-100%'), cent_eta=CE)
    yields = record_yield(eta=(-1, 1), pt=(0.2, 2.0))
    yields['centrality'] = copy.deepcopy(halves['centrality'])
    truth_comb = (400 * 0.06 + 200 * 0.12) / 600
    truth_whole = np.sqrt((400**2 * 0.06**2 + 200**2 * 0.12**2) / (400**2 + 200**2))

    for name, cont in containers.items():
        kw = dict(cent_method='simple', with_jackknife=False, **cont)
        v_whole, _ = evaluate(whole, **kw)
        v_comb, _ = evaluate(combined(whole, analysis_class_width=50), **kw)
        v_sub, _ = evaluate(halves, **kw)
        y_sub, _ = evaluate(yields, **kw)
        ref = float(np.sum(y_sub * v_sub) / np.sum(y_sub))
        check(f'{name}: combined = yield-weighted sub-class values', abs(v_comb[0] - ref) < 1e-10,
              f'{v_comb[0]:.6f} vs {ref:.6f}')
        # the same events: their common fluctuation cancels in the difference
        check(f'{name}: combined - whole class ~ 0.0800 - 0.0759',
              abs((v_comb[0] - v_whole[0]) - (truth_comb - truth_whole)) < 0.001,
              f'{v_comb[0]:.5f} - {v_whole[0]:.5f}')
        check(f'{name}: combined ~ truth 0.0800', abs(v_comb[0] - truth_comb) < 0.004, f'{v_comb[0]:.5f}')
        v_ev, _ = evaluate(combined(whole, analysis_class_width=50, combination_weight='events'), **kw)
        check(f'{name}: events weight = plain mean', abs(v_ev[0] - v_sub.mean()) < 1e-10,
              f'{v_ev[0]:.6f} vs {v_sub.mean():.6f}')
        v_list, _ = evaluate(combined(whole, analysis_classes=['0-50%', '50-100%']), **kw)
        check(f'{name}: listed classes == width', abs(v_list[0] - v_comb[0]) < 1e-12)
        # several classes, each from its own sub-classes; jackknife path
        rec4 = combined(copy.deepcopy(halves), analysis_class_width=25)
        v4, e4 = evaluate(rec4, cent_method='simple', with_jackknife=True, n_blocks=10, **cont)
        check(f'{name}: two combined classes ~ 0.06, 0.12 with errors',
              abs(v4[0] - 0.06) < 0.004 and abs(v4[1] - 0.12) < 0.004 and np.all(e4 > 0),
              f'{v4} +- {e4}')

    bad = combined(whole, analysis_classes=['0-40%', '50-100%'])
    try:
        resolve(bad)
        check('non-tiling sub-classes refused', False)
    except ValueError as e:
        check('non-tiling sub-classes refused', 'tile' in str(e), str(e)[:60])
    unit = copy.deepcopy(whole)
    unit['centrality']['combined_from'] = 'unit_multiplicity'
    try:
        resolve(unit)
        check('unit_multiplicity still refused', False)
    except ValueError:
        check('unit_multiplicity still refused', True)

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))


if __name__ == "__main__":
    _run()
