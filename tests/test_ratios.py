"""Ratios between two simulation sets (analysis/ratios.py, scripts/compute_ratios.py).

Toy simulation packages (from test_sim_package) serve Au+Au 200 GeV; ratio
records are written to a temporary directory and link existing Au+Au
records.  Checks: the joint jackknife on paired replicates; a self-ratio
(same sample on both sides) is exactly 1 with zero error; for two
independent samples the ratio equals numerator/denominator within errors
and its error matches the propagation of the two independent errors; a
detector-level side is computed only when the ratio declares
efficiency_cancels; a spectator-plane side is refused; a side is computed
at the ratio's centrality classes (and, for a ratio vs pT, in its bins) when
its own table is binned otherwise, and the ratio's points align per bin.

Run: python3 tests/test_ratios.py
"""
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from analysis.dispatcher import iter_records, resolve  # noqa: E402
from analysis.ratios import at_ratio_bins, at_ratio_classes, joint_jackknife, label_key, resolve_link, side_record  # noqa: E402

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("  PASS " if cond else "  FAIL ") + name)


def has_classes(d):
    return any("%" in str(c) and "-" in str(c) for c in (d.get("centrality") or {}).get("classes") or [])


def find(record_dir, pred):
    """(path, document index) of the first dispatchable document with centrality classes matching pred."""
    for fn in sorted(os.listdir(ROOT / record_dir)):
        p = ROOT / record_dir / fn
        for i, d in enumerate(yaml.safe_load_all(open(p))):
            if d and has_classes(d) and pred(d):
                try:
                    resolve(d)
                    return str(p), i, d
                except ValueError:
                    pass
    raise RuntimeError(f"no record in {record_dir}")


def ratio_record(name, num, den, **extra):
    r = {"observable_type": "observable_ratios", "inspire_id": "0", "version": 1,
         "hepdata_file": "none.yaml", "collision_system": {"energy_gev": 200.0, "reaction": "AU AU"},
         "observable": name, "value_type": name, "numerator": "A", "denominator": "B",
         "ratio_type": "system_ratio", "numerator_record": f"{num[0]}#{num[1]}",
         "denominator_record": f"{den[0]}#{den[1]}"}
    r.update(extra)
    return r


def test_unit():
    rng = np.random.default_rng(0)
    nf, df = np.array([2.0, 3.0]), np.array([4.0, 6.0])
    rep_n = nf + 0.01 * rng.normal(size=(10, 2))
    corr, err, cov = joint_jackknife(nf, rep_n, nf, rep_n)
    check("self-ratio of paired replicates: exactly 1, zero error",
          np.allclose(corr, 1) and np.allclose(err, 0))
    corr, err, _ = joint_jackknife(nf, np.tile(nf, (10, 1)), df, np.tile(df, (10, 1)))
    check("constant replicates: plain ratio", np.allclose(corr, 0.5) and np.allclose(err, 0))
    check("label key drops the record tag",
          label_key("vn2_single 20-30% [v2{2}/fig4_left]#2 ~NO-REF") == "vn2_single 20-30%")
    try:
        resolve_link("data/observable_ratios")
        ok = False
    except ValueError:
        ok = True
    check("bad link raises", ok)

    side = {"centrality": {"classes": ["0-1%", "1-2%", "2-3%", "3-4%"], "combined_from": "analysis_classes",
                           "analysis_class_width": 1.0}, "table_rows": [0, 1, 2, 3]}
    got = at_ratio_classes(copy.deepcopy(side), {"centrality": {"classes": ["0-2%", "2-4%"]}})
    check("side at the ratio's classes, table_rows dropped",
          got["centrality"]["classes"] == ["0-2%", "2-4%"] and "table_rows" not in got)
    check("same classes: side unchanged",
          at_ratio_classes(copy.deepcopy(side), {"centrality": {"classes": side["centrality"]["classes"]}}) == side)
    act = {"centrality": {"classes": ["0-5%", "5-10%"], "actual_classes": ["0-5.01%", "5.01-9.94%"]}}
    got = at_ratio_classes(copy.deepcopy(act), {"centrality": {"classes": ["5-10%"]}})
    check("actual_classes follow a subset of classes", got["centrality"]["actual_classes"] == ["5.01-9.94%"])
    for name, s_, r_ in (("actual_classes outside the side's classes", act, ["0-10%"]),
                         ("sub-classes that do not tile", side, ["0-2.5%"])):
        try:
            at_ratio_classes(copy.deepcopy(s_), {"centrality": {"classes": r_}})
            ok = False
        except ValueError:
            ok = True
        check(f"refused: {name}", ok)

    diff = {"dimensionality": "differential_pT", "table_rows": [1, 2],
            "kinematic_range": {"pT": {"min": 0.2, "max": 3.0, "bins": [0.2, 0.5, 1.0, 2.0, 3.0]}}}
    got = at_ratio_bins(copy.deepcopy(diff), {"kinematic_range": {"pT": {"min": 0.2, "max": 2.0,
                                                                         "bin_ranges": [[0.2, 1.0], [1.0, 2.0]]}}})
    check("differential side in the ratio's bins, its min/max kept",
          got["kinematic_range"]["pT"] == {"min": 0.2, "max": 3.0, "bin_ranges": [[0.2, 1.0], [1.0, 2.0]]}
          and "table_rows" not in got)
    check("ratio without bins: side unchanged",
          at_ratio_bins(copy.deepcopy(diff), {"kinematic_range": {"pT": {"min": 0.2, "max": 3.0}}}) == diff)
    try:
        at_ratio_bins(copy.deepcopy(diff), {"kinematic_range": {"pT": {"bins": [2.0, 4.0]}}})
        ok = False
    except ValueError:
        ok = True
    check("refused: ratio bins outside the side's acceptance", ok)


def test_pipeline():
    import h5py
    import test_sim_package as tsp
    with tempfile.TemporaryDirectory() as tmp:
        pkg, data, *_ = tsp._package(tmp, "toy_adapter_ratio", points=(0, 1))
        for dp in (0, 1):            # enough events for well-measured v2 in STAR's fine classes
            qn, ns = tsp._jetscape_qns(n_ev=3000, seed=11 + dp)
            np.save(Path(data) / f"Qns_{dp}.npy", qn)
            np.save(Path(data) / f"Nsamples_{dp}.npy", ns)
        v2 = find("data/flow_vn", lambda d: str((d.get("collision_system") or {}).get("reaction")) == "AU AU --> CHARGED X"
                  and d.get("value_type") == "v2{2}" and d.get("dimensionality") == "integrated"
                  and abs(float((d.get("collision_system") or {}).get("energy_gev") or 0) - 200) < 1)
        # an Au+Au record (the toy package simulates Au+Au only; until November 2026 "AU" anywhere in
        # the reaction also matched Cu+Au, picked once a re-curation removed the earlier Au+Au file)
        nch = find("data/particle_yields", lambda d: str((d.get("collision_system") or {}).get("reaction")).upper()
                   .replace(" ", "").startswith("AUAU")
                   and abs(float((d.get("collision_system") or {}).get("energy_gev") or 0) - 200) < 1
                   and str(d.get("observable")).lower() in ("dnch_deta", "nch", "dn_dy")
                   and str((d.get("particle_id") or {}).get("species")) == "charged"
                   and len((d.get("centrality") or {}).get("classes") or []) >= 4)
        det = copy.deepcopy(nch[2])
        det["efficiency_corrected"] = False
        det_path = os.path.join(tmp, "detector_level.yaml")
        yaml.safe_dump(det, open(det_path, "w"))
        spec = copy.deepcopy(v2[2])
        spec["flow_reference"] = {"detector": "ZDC-SMD", "plane": "spectator",
                                  "eta": [{"min": 6.3, "max": 99.0}]}
        spec_path = os.path.join(tmp, "spectator.yaml")
        yaml.safe_dump(spec, open(spec_path, "w"))
        def edges(c):
            return tuple(float(x) for x in c.rstrip("%").split("-"))

        def adjacent(d):            # three classes sharing edges (classes with edges, not points)
            cl = sorted(edges(c) for c in d["centrality"]["classes"])
            return "points" not in d["centrality"] and len(cl) >= 3 and cl[0][1] == cl[1][0] and cl[1][1] == cl[2][0]
        v2c = find("data/flow_vn", lambda d: str((d.get("collision_system") or {}).get("reaction")) == "AU AU --> CHARGED X"
                   and d.get("observable_type") == "flow_vn" and d.get("dimensionality") == "integrated"
                   and d.get("method") == "cumulant" and d.get("harmonic") == 2
                   and abs(float((d.get("collision_system") or {}).get("energy_gev") or 0) - 200) < 1 and adjacent(d))
        cl = sorted(edges(c) for c in v2c[2]["centrality"]["classes"])
        coarse = [f"{cl[0][0]:g}-{cl[2][1]:g}%"]          # three of the side's classes merged
        def pt_bins(d):
            from analysis.dispatcher import axis_bins
            return axis_bins((d.get("kinematic_range") or {}).get("pT"))
        v2pt = find("data/flow_vn", lambda d: str((d.get("collision_system") or {}).get("reaction")) == "AU AU --> CHARGED X"
                    and d.get("dimensionality") == "differential_pT" and d.get("harmonic") == 2
                    and str((d.get("particle_id") or {}).get("species")) == "charged"
                    and len(d["centrality"]["classes"]) == 1 and "points" not in d["centrality"]
                    and abs(float((d.get("collision_system") or {}).get("energy_gev") or 0) - 200) < 1
                    and len(pt_bins(d)) >= 3 and pt_bins(d)[0][1] == pt_bins(d)[1][0])
        b = pt_bins(v2pt[2])
        coarse_pt = [[b[0][0], b[1][1]], list(b[2])]              # bins 0+1 merged, bin 2
        own = {k: v2pt[2][k] for k in ("inspire_id", "version", "hepdata_file", "value_type", "dv_index",
                                       "dv_average", "table_rows", "value_scale", "kinematic_range", "centrality")
               if k in v2pt[2]}
        own["_record_file"] = v2pt[0]
        recs = [ratio_record("v2_ratio", v2, v2),
                ratio_record("v2pt_ratio_coarse", v2pt, v2pt, centrality=v2pt[2]["centrality"],
                             kinematic_range={"eta": v2pt[2]["kinematic_range"].get("eta"),
                                              "pT": {"min": b[0][0], "max": b[2][1], "bin_ranges": coarse_pt}}),
                ratio_record("v2pt_own_table", v2pt, v2pt, **own),
                ratio_record("v2_ratio_coarse", v2c, v2c, centrality={"classes": coarse}),
                ratio_record("v2_ratio_points", v2, v2, centrality={"classes": coarse}),
                ratio_record("nch_ratio_cancels", (det_path, 0), (det_path, 0), efficiency_cancels=True),
                ratio_record("nch_ratio_detector", (det_path, 0), (det_path, 0)),
                ratio_record("spectator_ratio", (spec_path, 0), (spec_path, 0))]
        rfile = os.path.join(tmp, "ratios.yaml")
        yaml.safe_dump_all(recs, open(rfile, "w"))

        def run(num_dp, den_dp, out):
            cmd = [sys.executable, "scripts/compute_ratios.py", "--ratio-records", rfile,
                   "--num-package", str(pkg), "--num-design-point", str(num_dp), "--num-variant", "plain",
                   "--den-package", str(pkg), "--den-design-point", str(den_dp), "--den-variant", "plain",
                   "--n-jk-blocks", "10", "--output", out]
            return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

        self_out = os.path.join(tmp, "self.h5")
        r = run(0, 0, self_out)
        check("self-ratio run succeeds", r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return
        check("detector-level side without efficiency_cancels skipped",
              "skip nch_ratio_detector: detector-level side" in r.stdout)
        check("spectator-plane side skipped", "skip spectator_ratio" in r.stdout and "spectator plane" in r.stdout)
        with h5py.File(self_out, "r") as f:
            g = f["ratios"]
            labels = [x.decode() for x in g["labels"][:]]
            v, e = g["values"][:], g["errors"][:]
            ev = g["exp_values"][:]
            side_ok = np.isfinite(g["numerator_values"][:])   # a bin without particles is NaN on the sides
        check("ratios computed for v2 and the efficiency-cancelling N_ch",
              any(l.startswith("v2_ratio") for l in labels) and any(l.startswith("nch_ratio_cancels") for l in labels))
        cls = v2pt[2]["centrality"]["classes"][0]
        want = [f"pT {lo:.2f}-{hi:.2f} {cls}" for lo, hi in coarse_pt]
        check(f"pT-differential ratio computed in the ratio's bins {want}",
              [l.split(" | ", 1)[1].split(" ", 1)[1] for l in labels if l.startswith("v2pt_ratio_coarse")] == want)
        from analysis.dispatcher import extract_measurement_params, load_experimental
        params = extract_measurement_params(v2pt[2])
        params["record_file"] = v2pt[0]
        table, _ = load_experimental(params)
        got = ev[[i for i, l in enumerate(labels) if l.split(" | ")[0] == own["value_type"]]]
        check(f"experimental points aligned per pT bin ({len(got)} of {len(table)} bins, {os.path.basename(v2pt[0])})",
              len(got) == len(table) and np.allclose(got, np.asarray(table, float), equal_nan=True))
        check("point-valued side at other classes skipped",
              "skip v2_ratio_points: side with point-valued centrality" in r.stdout)
        check(f"coarse-class ratio computed at its own class {coarse} ({os.path.basename(v2c[0])})",
              sorted(l.split()[-1] for l in labels if l.startswith("v2_ratio_coarse")) == coarse)
        check(f"self-ratio (same sample both sides) is exactly 1 with zero error ({int(side_ok.sum())} "
              f"observables; {int((~side_ok).sum())} empty toy bins NaN on both sides)",
              side_ok.sum() > 0 and np.allclose(v[side_ok], 1, atol=1e-12) and np.allclose(e[side_ok], 0, atol=1e-12)
              and np.all(np.isnan(v[~side_ok])))

        ind_out = os.path.join(tmp, "ind.h5")
        r = run(0, 1, ind_out)
        check("independent-sample run succeeds", r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return
        with h5py.File(ind_out, "r") as f:
            g = f["ratios"]
            v, e = g["values"][:], g["errors"][:]
            nv, dv = g["numerator_values"][:], g["denominator_values"][:]
        ok = np.isfinite(v) & (e > 0)
        # error propagation from each side's own run (same jackknife, independent samples)
        n_err, d_err = [], []
        with tempfile.TemporaryDirectory() as w:
            cmd = [sys.executable, "scripts/compute_ratios.py", "--ratio-records", rfile,
                   "--num-package", str(pkg), "--num-design-point", "0", "--num-variant", "plain",
                   "--den-package", str(pkg), "--den-design-point", "1", "--den-variant", "plain",
                   "--n-jk-blocks", "10", "--output", os.path.join(w, "x.h5"), "--work-dir", w]
            subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            for side, store in (("num", n_err), ("den", d_err)):
                with h5py.File(os.path.join(w, f"{side}.h5"), "r") as f:
                    for g in f.values():
                        if "labels" in g:
                            store.extend(zip([label_key(x.decode()) for x in g["labels"][:]],
                                             g["values"][:], g["errors"][:]))
        nd = {k: (x, s) for k, x, s in n_err}
        dd = {k: (x, s) for k, x, s in d_err}
        with h5py.File(ind_out, "r") as f:
            labels = [x.decode().split(" | ", 1)[1] for x in f["ratios"]["labels"][:]]
        prop = np.array([abs(nd[k][0] / dd[k][0]) * np.hypot(nd[k][1] / nd[k][0], dd[k][1] / dd[k][0])
                         for k in labels])
        # Linear error propagation holds only where both sides are well measured;
        # where a toy v2 is consistent with zero the ratio is non-linear.
        good = ok & np.array([nd[k][1] < 0.05 * abs(nd[k][0]) and dd[k][1] < 0.05 * abs(dd[k][0])
                              for k in labels])
        rel = e[good] / prop[good]
        check(f"independent: ratio = numerator/denominator within errors ({int(good.sum())} observables)",
              good.sum() >= 3 and np.all(np.abs(v[good] - nv[good] / dv[good]) < 3 * e[good]))
        check(f"independent: jackknife error matches propagation where both sides are "
              f"measured to <5% ({int(good.sum())} observables, median ratio {np.median(rel):.2f})",
              good.sum() >= 3 and 0.7 < np.median(rel) < 1.4)


if __name__ == "__main__":
    test_unit()
    test_pipeline()
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
