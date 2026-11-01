#!/usr/bin/env python3
"""Compute observable ratios (observable_ratios records) from two simulation sets.

    python scripts/compute_ratios.py \
        --num-package <dir> --num-design-point 12 [--num-variant CE] \
        --den-package <dir> --den-design-point 40 [--den-variant CE] \
        [--reaction 'Ru\\+Ru and Zr\\+Zr'] [--value-type-regex REGEX] [--ratio-records FILE ...] \
        --output ratios.h5 [-- <extra run_pipeline.py options>]

For every selected ratio record whose numerator and denominator records are
linked (numerator_record / denominator_record), the linked records are
computed by run_pipeline.py: the numerators on the numerator simulation set,
the denominators on the denominator set.  Both sets may be design points of
one package (e.g. Ru-like and Zr-like nuclear parameters of an A=96
campaign) or two different packages (e.g. Xe+Xe over Pb+Pb).  Each ratio is
formed per centrality class and bin, with a joint block jackknife over both
samples (analysis/ratios.py).  A detector-level side is computed as the
corrected quantity when the ratio declares efficiency_cancels: true.
Records whose sides cannot be computed (composite double ratios, spectator
planes, detector-level distributions) are listed with the reason.

Output (HDF5, group "ratios"): values, errors, covariance, labels
("<ratio value_type> | <quantity>"), exp_values, exp_errors (the ratio
record's points aligned by centrality class), ratio_record (index into the
records list), numerator_values / denominator_values (each side's own
full-sample values), and the provenance of both runs.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.dispatcher import (extract_measurement_params, iter_records,  # noqa: E402
                                 load_experimental, record_roots)
from analysis.ratios import collection_base, joint_jackknife, label_key, resolve_link, side_record  # noqa: E402


def select_ratio_records(a):
    """[(file, doc)] of the ratio records to compute."""
    out = []
    if a.ratio_records:
        for f in a.ratio_records:
            out += [(f, d) for d in yaml.safe_load_all(open(f)) if d]
    else:
        for root in record_roots(a.extra_records):
            d = os.path.join(ROOT, root, "observable_ratios") if root == "data" else os.path.join(root, "observable_ratios")
            if os.path.isdir(d):
                out += list(iter_records(d))
    sel = []
    for f, d in out:
        cs = d.get("collision_system") or {}
        if a.reaction and not re.search(a.reaction, str(cs.get("reaction", "")), re.I):
            continue
        if a.energy is not None and abs(float(cs.get("energy_gev", 0) or 0) - a.energy) > 0.1:
            continue
        if a.value_type_regex and not re.search(a.value_type_regex, str(d.get("value_type", ""))):
            continue
        sel.append((f, d))
    return sel


def run_side(side, a, entries, work, extra):
    """Run run_pipeline.py for one side; return the output path."""
    manifest = work / f"{side}_manifest.yaml"
    with open(manifest, "w") as fh:
        yaml.safe_dump({"records": [{"fpath": p, "doc_index": i} for p, i in entries]}, fh)
    out = work / f"{side}.h5"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--manifest", str(manifest),
           "--sim-package", getattr(a, f"{side}_package"),
           "--design-point", str(getattr(a, f"{side}_design_point")),
           "--n-jk-blocks", str(a.n_jk_blocks), "--save-jackknife-replicates", "--output", str(out)]
    if getattr(a, f"{side}_variant"):
        cmd += ["--variant", getattr(a, f"{side}_variant")]
    if getattr(a, f"{side}_data_root"):
        cmd += ["--data-root", getattr(a, f"{side}_data_root")]
    if a.adapter_path:
        cmd += ["--adapter-path", a.adapter_path]
    for d in a.extra_records or []:
        cmd += ["--extra-records", d]
    cmd += extra
    with open(work / f"{side}.log", "w") as log:
        r = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"{side} run failed (exit {r.returncode}); see {work / (side + '.log')}")
    return out


def read_side(path):
    """{(record_file, doc_index): [(label key, full, replicates column, value)]} and provenance."""
    by_rec, prov = {}, {}
    with h5py.File(path, "r") as f:
        prov = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()
                if k.startswith("sim_package_") or k in ("design_point", "variant", "code_fingerprint")}
        for g in f.values():
            if "jk_replicates" not in g:
                continue
            rule = g.attrs.get("jk_block_rule", "contiguous")
            prov["jk_block_rule"] = rule.decode() if isinstance(rule, bytes) else rule
            labels = [x.decode() if isinstance(x, bytes) else str(x) for x in g["labels"][:]]
            recs = g["label_record"][:]
            full, rep, vals = g["jk_full"][:], g["jk_replicates"][:], g["values"][:]
            for i, (lab, k) in enumerate(zip(labels, recs)):
                eg = g.get(f"exp_{k}")
                if eg is None:
                    continue
                key = (str(eg.attrs.get("record_file", "")), int(eg.attrs.get("doc_index", -1)))
                by_rec.setdefault(key, []).append((label_key(lab), full[i], rep[:, i], vals[i]))
    return by_rec, prov


def experimental(ratio):
    """{centrality class: (value, error)} of the ratio record's own table."""
    try:
        params = extract_measurement_params(ratio)
        params["record_file"] = ratio.get("_record_file")
        vals, errs = load_experimental(params)
    except Exception:          # noqa: BLE001 -- a ratio without a readable table has no points
        return {}
    classes = (ratio.get("centrality") or {}).get("classes") or []
    if vals is None or len(vals) != len(classes):
        return {}
    return {c: (float(v), float(e)) for c, v, e in zip(classes, vals, errs)}


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for side in ("num", "den"):
        ap.add_argument(f"--{side}-package", required=True)
        ap.add_argument(f"--{side}-design-point", type=int, required=True)
        ap.add_argument(f"--{side}-variant", default=None)
        ap.add_argument(f"--{side}-data-root", default=None)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--extra-records", action="append", default=None, metavar="DIR",
                    help="additional record directory (like data/), also passed to the side runs")
    ap.add_argument("--ratio-records", nargs="+", default=None,
                    help="record files to use instead of data/observable_ratios")
    ap.add_argument("--reaction", default=None, help="regex on the ratio record's reaction")
    ap.add_argument("--energy", type=float, default=None)
    ap.add_argument("--value-type-regex", default=None)
    ap.add_argument("--n-jk-blocks", type=int, default=20)
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", default=None, help="keep the per-side runs here")
    a = ap.parse_args(argv)

    ratios, skipped = [], []
    entries = {"num": {}, "den": {}}
    work = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="ratios_"))
    work.mkdir(parents=True, exist_ok=True)
    for f, r in select_ratio_records(a):
        r["_record_file"] = os.path.abspath(f)
        if not (r.get("numerator_record") and r.get("denominator_record")):
            skipped.append((r.get("value_type"), "no numerator/denominator records linked"))
            continue
        try:
            sides = {}
            for side, link in (("num", r["numerator_record"]), ("den", r["denominator_record"])):
                path, idx, doc = resolve_link(link, collection_base(r.get("_record_file")))
                rec = side_record(doc, r)
                if rec != doc:      # computed as the corrected quantity: pin a modified copy
                    path = str(work / f"{side}_{len(entries[side])}.yaml")
                    with open(path, "w") as fh:
                        yaml.safe_dump(rec, fh, sort_keys=False)
                    idx = 0
                entries[side].setdefault((path, idx), None)
                sides[side] = (path, idx)
        except ValueError as e:
            skipped.append((r.get("value_type"), str(e)))
            continue
        ratios.append((f, r, sides))
    for vt, why in skipped:
        print(f"skip {vt}: {why}")
    if not ratios:
        raise SystemExit("no computable ratio records selected")
    print(f"{len(ratios)} ratio records; numerator records {len(entries['num'])}, "
          f"denominator records {len(entries['den'])}")

    num, num_prov = read_side(run_side("num", a, list(entries["num"]), work, extra))
    den, den_prov = read_side(run_side("den", a, list(entries["den"]), work, extra))
    # replicate k must delete the same events on both sides for paired
    # (common-random-number) samples: event-id blocks (run_pipeline --jk-blocks)
    if num_prov.get("jk_block_rule") != den_prov.get("jk_block_rule"):
        print(f"WARNING: jackknife blocks differ (numerator {num_prov.get('jk_block_rule')}, "
              f"denominator {den_prov.get('jk_block_rule')}): the joint jackknife pairs block k with "
              f"block k, which is right only for independent samples")

    labels, rec_idx, nf, nr, df, dr, nv, dv, ev, ee = [], [], [], [], [], [], [], [], [], []
    for i, (f, r, sides) in enumerate(ratios):
        n_side = {k: (fu, rep, v) for k, fu, rep, v in num.get(sides["num"], [])}
        d_side = {k: (fu, rep, v) for k, fu, rep, v in den.get(sides["den"], [])}
        exp = experimental(r)
        common = [k for k in n_side if k in d_side]
        if not common:
            print(f"skip {r.get('value_type')}: no quantity computed on both sides")
            continue
        for k in common:
            labels.append(f"{r.get('value_type')} | {k}")
            rec_idx.append(i)
            nf.append(n_side[k][0]); nr.append(n_side[k][1]); nv.append(n_side[k][2])
            df.append(d_side[k][0]); dr.append(d_side[k][1]); dv.append(d_side[k][2])
            cent = k.split()[-1]
            ev.append(exp.get(cent, (np.nan, np.nan))[0]); ee.append(exp.get(cent, (np.nan, np.nan))[1])
    corr, err, cov = joint_jackknife(np.array(nf), np.array(nr).T, np.array(df), np.array(dr).T)

    with h5py.File(a.output, "w") as h:
        g = h.create_group("ratios")
        g.create_dataset("values", data=corr)
        g.create_dataset("errors", data=err)
        g.create_dataset("covariance", data=cov)
        g.create_dataset("labels", data=labels, dtype=h5py.string_dtype())
        g.create_dataset("exp_values", data=np.array(ev, float))
        g.create_dataset("exp_errors", data=np.array(ee, float))
        g.create_dataset("ratio_record", data=np.array(rec_idx, int))
        g.create_dataset("numerator_values", data=np.array(nv, float))
        g.create_dataset("denominator_values", data=np.array(dv, float))
        g.create_dataset("records", data=[f"{f}|{r.get('value_type')}" for f, r, _ in ratios],
                         dtype=h5py.string_dtype())
        g.attrs["n_jk_blocks"] = a.n_jk_blocks
        g.attrs["skipped"] = json.dumps(skipped)
        for side, prov in (("numerator", num_prov), ("denominator", den_prov)):
            for k, v in prov.items():
                g.attrs[f"{side}_{k}"] = v
    print(f"{len(labels)} ratio observables from {len(set(rec_idx))} records -> {a.output}")
    if not a.work_dir:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
