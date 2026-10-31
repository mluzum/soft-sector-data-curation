"""Ratios between two simulation sets (analysis/ratios.py, scripts/compute_ratios.py).

Toy simulation packages (from test_sim_package) serve Au+Au 200 GeV; ratio
records are written to a temporary directory and link existing Au+Au
records.  Checks: the joint jackknife on paired replicates; a self-ratio
(same sample on both sides) is exactly 1 with zero error; for two
independent samples the ratio equals numerator/denominator within errors
and its error matches the propagation of the two independent errors; a
detector-level side is computed only when the ratio declares
efficiency_cancels; a spectator-plane side is refused.

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
from analysis.ratios import joint_jackknife, label_key, resolve_link, side_record  # noqa: E402

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
        nch = find("data/particle_yields", lambda d: "AU" in str((d.get("collision_system") or {}).get("reaction")).upper()
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
        recs = [ratio_record("v2_ratio", v2, v2),
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
        check("ratios computed for v2 and the efficiency-cancelling N_ch",
              any(l.startswith("v2_ratio") for l in labels) and any(l.startswith("nch_ratio_cancels") for l in labels))
        check(f"self-ratio (same sample both sides) is exactly 1 with zero error ({len(v)} observables)",
              len(v) > 0 and np.allclose(v, 1, atol=1e-12) and np.allclose(e, 0, atol=1e-12))

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
