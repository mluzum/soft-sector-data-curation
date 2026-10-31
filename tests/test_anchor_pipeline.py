"""Tests for run_pipeline.py centrality-anchor config helpers (chunk 3 glue)."""


def _run():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'rp', 'scripts/run_pipeline.py')
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)

    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        print(("  PASS " if cond else "  FAIL ") + name)

    cfg = {
        'cent_anchors': {
            'Au Au 200': 0.8,
            'Pb Pb 2760': {'b_max': 12.0, 'sigma_tot': 530.0},  # fm^2 (geo cross section)
            'Xe Xe 5440': {'sigma_eff': 35.0, 'sigma_tot': 42.0},
        }
    }

    # Present: bare fraction
    check('frac lookup', rp._get_cent_anchor(cfg, 'Au Au', '200') == 0.8)
    # Present: bmax/sigtot dict
    check('dict lookup',
          rp._get_cent_anchor(cfg, 'Pb Pb', '2760') ==
          {'b_max': 12.0, 'sigma_tot': 530.0})
    # Normalized reaction key matches
    check('normalized key',
          rp._get_cent_anchor(cfg, 'AU AU', '200') == 0.8)
    # Missing system returns None
    check('missing -> None', rp._get_cent_anchor(cfg, 'Cu Cu', '200') is None)
    # Empty cfg returns None
    check('empty cfg -> None', rp._get_cent_anchor({}, 'Au Au', '200') is None)
    # resolve_anchor on the bmax spec gives f = pi*b_max^2/sigma_tot
    import math
    a = rp.resolve_anchor(rp._get_cent_anchor(cfg, 'Pb Pb', '2760'), 1000)
    f_expected = math.pi * 12.0 ** 2 / 530.0
    check('bmax f value', abs(a['f'] - f_expected) < 1e-9)
    # full set default
    a0 = rp.resolve_anchor(1.0, 1000)
    check('default f=1', abs(a0['f'] - 1.0) < 1e-12 and a0['n_ref'] == 1000)

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))


if __name__ == "__main__":
    _run()
