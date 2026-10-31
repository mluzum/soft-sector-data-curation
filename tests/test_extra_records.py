"""Records outside data/: --extra-records, value_table, status: preliminary.

A record in an extra directory (laid out like data/<type>/) with its own
HEPData-format value table is computed by run_pipeline.py only with
--include-preliminary when it is marked preliminary, and its experimental
values come from its value_table.  Also: record_roots(), the validator's
status rules, and load_experimental with a relative value_table.

Run: python3 tests/test_extra_records.py
"""
import copy
import glob
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

from analysis.dispatcher import extract_measurement_params, load_experimental, record_roots  # noqa: E402

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("  PASS " if cond else "  FAIL ") + name)


def make_extra(tmp):
    """An extra record root with a preliminary copy of an Au+Au 200 v2{2} record
    whose value table is the original table with every value doubled."""
    import test_ratios as tr
    path, idx, doc = tr.find("data/flow_vn", lambda d: str((d.get("collision_system") or {}).get("reaction")) == "AU AU --> CHARGED X"
                             and d.get("value_type") == "v2{2}" and d.get("dimensionality") == "integrated"
                             and abs(float((d.get("collision_system") or {}).get("energy_gev") or 0) - 200) < 1)
    iid = doc["inspire_id"]
    raw = glob.glob(str(ROOT / f"hepdata_raw/records/ins{iid}/HEPData-ins{iid}-v{doc['version']}-yaml/{doc['hepdata_file']}"))[0]
    table = yaml.safe_load(open(raw))
    for dv in table["dependent_variables"]:
        for v in dv["values"]:
            if isinstance(v.get("value"), (int, float)):
                v["value"] = 2 * v["value"]
    root = Path(tmp) / "extra" / "data"          # records root; value tables live beside it
    (root / "flow_vn").mkdir(parents=True)
    (root.parent / "tables").mkdir()
    yaml.safe_dump(table, open(root.parent / "tables" / "doubled.yaml", "w"))
    rec = copy.deepcopy(doc)
    rec.update(status="preliminary", value_table="../../tables/doubled.yaml",
               source={"reference": "test", "method": "doubled copy of a HEPData table"},
               value_type="v2{2}", hepdata_file="doubled.yaml")
    yaml.safe_dump(rec, open(root / "flow_vn" / "prelim.yaml", "w"))
    return root, doc, raw


def test_units():
    with tempfile.TemporaryDirectory() as tmp:
        root, orig, raw = make_extra(tmp)
        os.environ["SOFT_SECTOR_EXTRA_RECORDS"] = str(root)
        try:
            check("record_roots: data/ first, then the environment's roots",
                  record_roots() == ["data", str(root)] and record_roots([str(root)]) == ["data", str(root)])
        finally:
            del os.environ["SOFT_SECTOR_EXTRA_RECORDS"]
        rec = yaml.safe_load(open(root / "flow_vn" / "prelim.yaml"))
        rec["_record_file"] = str(root / "flow_vn" / "prelim.yaml")
        v, _ = load_experimental(extract_measurement_params(rec))
        v0, _ = load_experimental(extract_measurement_params(orig))
        check("value_table relative to the record file", v is not None and np.allclose(v, 2 * np.asarray(v0)))
        r = subprocess.run([sys.executable, "scripts/validate.py", str(root)], cwd=ROOT, capture_output=True, text=True)
        check("validator accepts the preliminary record", r.returncode == 0)
        nosrc = dict(rec); nosrc.pop("source"); nosrc.pop("_record_file")
        yaml.safe_dump(nosrc, open(root / "flow_vn" / "prelim.yaml", "w"))
        r = subprocess.run([sys.executable, "scripts/validate.py", str(root)], cwd=ROOT, capture_output=True, text=True)
        check("validator requires source for preliminary records", r.returncode != 0 and "source" in r.stdout)


def test_pipeline():
    import h5py
    import test_sim_package as tsp
    with tempfile.TemporaryDirectory() as tmp:
        pkg, data, *_ = tsp._package(tmp, "toy_adapter_extra")
        root, orig, raw = make_extra(tmp)
        base = [sys.executable, "scripts/run_pipeline.py", "--sim-package", str(pkg), "--variant", "plain",
                "--design-point", "0", "--type", "flow_vn", "--reaction", "AU AU --> CHARGED X", "--energy", "200",
                "--value-type-regex", r"^v2\{2\}$", "--n-jk-blocks", "10", "--extra-records", str(root)]
        out1, out2 = os.path.join(tmp, "a.h5"), os.path.join(tmp, "b.h5")
        r1 = subprocess.run(base + ["--output", out1], cwd=ROOT, capture_output=True, text=True)
        r2 = subprocess.run(base + ["--include-preliminary", "--output", out2], cwd=ROOT, capture_output=True, text=True)
        check("both runs succeed", r1.returncode == 0 and r2.returncode == 0)
        if r1.returncode or r2.returncode:
            print(r1.stdout[-1500:], r2.stdout[-1500:])
            return
        check("preliminary record excluded by default", "Excluded 1 preliminary records" in r1.stdout)

        def by_file(path):
            out = {}
            with h5py.File(path, "r") as f:
                for g in f.values():
                    if "labels" not in g:
                        continue
                    for i, k in enumerate(g["label_record"][:]):
                        eg = g[f"exp_{k}"]
                        out.setdefault((str(eg.attrs["record_file"]), str(eg.attrs.get("status"))), []).append(
                            (g["labels"][i].decode().split(" [")[0], g["values"][i], g["exp_values"][i]))
            return out
        a, b = by_file(out1), by_file(out2)
        prelim = [k for k in b if k[0].endswith("prelim.yaml")]
        check("included with --include-preliminary, marked in the output",
              not any(k[0].endswith("prelim.yaml") for k in a) and prelim and prelim[0][1] == "preliminary")
        orig_rows = next(v for k, v in b.items() if os.path.samefile(k[0], ROOT / "data" / "flow_vn" / os.path.basename(k[0]))
                         and len(v) == len(b[prelim[0]])) if prelim else []
        pre_rows = b[prelim[0]] if prelim else []
        check("same simulation values as the original record",
              bool(pre_rows) and all(np.isclose(x[1], y[1]) for x, y in zip(pre_rows, orig_rows)))
        check("experimental values from the value table (doubled)",
              bool(pre_rows) and all(np.isclose(x[2], 2 * y[2]) for x, y in zip(pre_rows, orig_rows)))


if __name__ == "__main__":
    test_units()
    test_pipeline()
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
