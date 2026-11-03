#!/usr/bin/env python3
"""Stack run_campaign.py outputs into one table (runs x observables).

    python scripts/collect_campaign.py <out-dir> --output table.h5 [--csv long.csv]

Reads every ``*.h5`` written by run_campaign.py (or run_pipeline.py) in
``<out-dir>`` and writes:

- ``values``, ``errors``: (n_runs, n_obs), NaN where a run lacks an observable
  (full-sample estimates); ``jk_bias``: the jackknife bias estimates, a
  diagnostic (NaN for runs written before they were stored);
- with ``--replicates``: ``replicates`` (n_runs, n_blocks, n_obs), each run's
  delete-one-block jackknife replicates (runs made with
  ``--save-jackknife-replicates``), attrs ``jk_block_rule`` and
  ``n_jk_blocks``.  With event-id blocks (common random numbers) the noise
  covariance between runs i and j of observable k is
  (B-1)/B sum_b (r_ibk - rbar_ik)(r_jbk - rbar_jk);
- ``runs/design_point``, ``runs/variant``, ``runs/file``;
- ``observables/key`` (``<system group>|<label>``), ``observables/system``,
  ``observables/label``, ``observables/exp_values``, ``observables/exp_errors``
  (the experimental points aligned by run_pipeline; NaN where it could not
  align them);
- ``design/parameter_names``, ``design/values`` (n_runs, n_par): the
  package's design table, NaN for runs without a row;
- the model precision of each observable across the runs
  (analysis/observable_selection.py): ``observables/undefined_fraction``,
  ``observables/median_rel_error`` (|error/value|),
  ``observables/median_error_ratio`` (model error / experimental error), and
  a selection for emulator training, whole observables only:
  ``observables/selected`` and ``observables/rejected_by`` (undefined |
  no_data | noise | rel_noise), the criteria in the attrs of
  ``observables/selected``.  ``--max-undefined`` (default 0: an observable
  undefined in any run is rejected), ``--keep-without-data``,
  ``--max-error-ratio``, ``--max-rel-error``.  Values are not changed.

Per-run covariances stay in the run files.  Runs made with different
pipeline options (other than the design point and variant) are refused
unless ``--allow-mixed``.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _strip_point(argv: list[str]) -> list[str]:
    """Pipeline argv without the per-run options."""
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in ("--design-point", "--variant", "--output"):
            skip = True
            continue
        out.append(tok)
    return out


def read_run(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        argv = json.loads(f.attrs.get("pipeline_argv", "[]"))
        run = {"file": path.name, "design_point": int(f.attrs["design_point"]),
               "variant": str(f.attrs.get("variant", "")),
               "argv": _strip_point(argv),
               "manifest": str(f.attrs.get("sim_package_manifest", "")),
               "groups": {}}
        for g in f:
            G = f[g]
            if "labels" not in G:
                continue
            labels = [x.decode() if isinstance(x, bytes) else str(x) for x in G["labels"][:]]
            n = len(labels)
            rule = G.attrs.get("jk_block_rule", "contiguous")
            run["groups"][g] = {
                "labels": labels,
                "values": G["values"][:],
                "errors": G["errors"][:],
                "jk_bias": G["jk_bias"][:] if "jk_bias" in G else np.full(n, np.nan),
                "replicates": G["jk_replicates"][:] if "jk_replicates" in G else None,
                "jk_block_rule": rule.decode() if isinstance(rule, bytes) else str(rule),
                "exp_values": G["exp_values"][:] if "exp_values" in G else np.full(n, np.nan),
                "exp_errors": G["exp_errors"][:] if "exp_errors" in G else np.full(n, np.nan),
            }
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--csv", default=None, help="also write a long-format CSV")
    ap.add_argument("--sim-package", default=None,
                    help="package for the design table (default: the one recorded in the runs)")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="accept runs made with different pipeline options")
    ap.add_argument("--replicates", action="store_true",
                    help="also stack the jackknife replicates (runs need --save-jackknife-replicates "
                         "and one block count and rule)")
    sel = ap.add_argument_group("observable selection (observables/selected; values are not changed)")
    sel.add_argument("--max-undefined", type=float, default=0.0,
                     help="reject an observable undefined (NaN) in more than this fraction of runs (default 0)")
    sel.add_argument("--keep-without-data", action="store_true",
                     help="keep observables without an experimental point")
    sel.add_argument("--max-error-ratio", type=float, default=None,
                     help="reject an observable whose median model error / experimental error exceeds this")
    sel.add_argument("--max-rel-error", type=float, default=None,
                     help="reject an observable whose median |model error / value| exceeds this")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.in_dir).glob("*.h5"))
    if not files:
        raise SystemExit(f"no .h5 files in {a.in_dir}")
    runs = [read_run(p) for p in files]
    runs.sort(key=lambda r: (r["design_point"], r["variant"]))

    argvs = {json.dumps(r["argv"]) for r in runs}
    if len(argvs) > 1 and not a.allow_mixed:
        raise SystemExit("runs were made with different pipeline options:\n  "
                         + "\n  ".join(sorted(argvs)) + "\n(--allow-mixed to accept)")

    keys, index, system, label = [], {}, [], []
    exp_v, exp_e = [], []
    exp_conflicts = 0
    for r in runs:
        for g, d in r["groups"].items():
            for lab, ev, ee in zip(d["labels"], d["exp_values"], d["exp_errors"]):
                k = f"{g}|{lab}"
                if k not in index:
                    index[k] = len(keys)
                    keys.append(k)
                    system.append(g)
                    label.append(lab)
                    exp_v.append(ev)
                    exp_e.append(ee)
                elif not (np.isnan(ev) and np.isnan(exp_v[index[k]])) and ev != exp_v[index[k]]:
                    exp_conflicts += 1
    if exp_conflicts:
        raise SystemExit(f"{exp_conflicts} observables carry different experimental values "
                         f"in different runs; the runs do not share one record selection")

    n_runs, n_obs = len(runs), len(keys)
    vals = np.full((n_runs, n_obs), np.nan)
    errs = np.full((n_runs, n_obs), np.nan)
    bias = np.full((n_runs, n_obs), np.nan)
    reps, rules = None, set()
    if a.replicates:
        shapes = {d["replicates"].shape[0] if d["replicates"] is not None else None
                  for r in runs for d in r["groups"].values()}
        rules = {d["jk_block_rule"] for r in runs for d in r["groups"].values()}
        if None in shapes or len(shapes) != 1 or len(rules) != 1:
            raise SystemExit(f"--replicates: runs need jk_replicates with one block count and rule "
                             f"(block counts {sorted(map(str, shapes))}, rules {sorted(rules)})")
        reps = np.full((n_runs, shapes.pop(), n_obs), np.nan)
    for i, r in enumerate(runs):
        for g, d in r["groups"].items():
            cols = [index[f"{g}|{lab}"] for lab in d["labels"]]
            vals[i, cols] = d["values"]
            errs[i, cols] = d["errors"]
            bias[i, cols] = d["jk_bias"]
            if reps is not None:
                reps[i][:, cols] = d["replicates"]

    par_names, par_vals = [], np.zeros((n_runs, 0))
    manifest = a.sim_package or next((r["manifest"] for r in runs if r["manifest"]), None)
    if manifest:
        from analysis.sim_package import MANIFEST_NAME, read_design_table
        import yaml
        mpath = Path(manifest)
        mpath = mpath if mpath.is_file() else mpath / MANIFEST_NAME
        try:
            with open(mpath) as fh:
                par_names, table = read_design_table(yaml.safe_load(fh), mpath.parent)
            par_vals = np.array([table.get(r["design_point"], [np.nan] * len(par_names))
                                 for r in runs], dtype=float).reshape(n_runs, len(par_names))
        except (OSError, KeyError, ValueError) as e:
            print(f"warning: no design table ({e})", file=sys.stderr)

    from analysis.observable_selection import CRITERIA, quality, select
    q = quality(vals, errs, exp_v, exp_e)
    selected, rejected_by = select(q, max_undefined=a.max_undefined, require_data=not a.keep_without_data,
                                   max_error_ratio=a.max_error_ratio, max_rel_error=a.max_rel_error)

    str_dt = h5py.string_dtype()
    with h5py.File(a.output, "w") as f:
        f.attrs["source_dir"] = str(Path(a.in_dir).resolve())
        f.attrs["pipeline_argv"] = sorted(argvs)[0]
        if manifest:
            f.attrs["sim_package_manifest"] = str(manifest)
        f.create_dataset("values", data=vals)
        f.create_dataset("errors", data=errs)
        f.create_dataset("jk_bias", data=bias)
        if reps is not None:
            f.create_dataset("replicates", data=reps)
            f.attrs["jk_block_rule"] = rules.pop()
            f.attrs["n_jk_blocks"] = reps.shape[1]
        f.create_dataset("runs/design_point", data=[r["design_point"] for r in runs])
        f.create_dataset("runs/variant", data=[r["variant"] for r in runs], dtype=str_dt)
        f.create_dataset("runs/file", data=[r["file"] for r in runs], dtype=str_dt)
        f.create_dataset("observables/key", data=keys, dtype=str_dt)
        f.create_dataset("observables/system", data=system, dtype=str_dt)
        f.create_dataset("observables/label", data=label, dtype=str_dt)
        f.create_dataset("observables/exp_values", data=np.asarray(exp_v, float))
        f.create_dataset("observables/exp_errors", data=np.asarray(exp_e, float))
        f.create_dataset("design/parameter_names", data=par_names, dtype=str_dt)
        f.create_dataset("design/values", data=par_vals)
        for k in ("undefined_fraction", "median_rel_error", "median_error_ratio"):
            f.create_dataset(f"observables/{k}", data=q[k])
        ds = f.create_dataset("observables/selected", data=selected)
        f.create_dataset("observables/rejected_by", data=list(rejected_by), dtype=str_dt)
        ds.attrs["max_undefined"] = a.max_undefined
        ds.attrs["require_data"] = not a.keep_without_data
        ds.attrs["max_error_ratio"] = np.nan if a.max_error_ratio is None else a.max_error_ratio
        ds.attrs["max_rel_error"] = np.nan if a.max_rel_error is None else a.max_rel_error

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["design_point", "variant", "system", "label", "value", "error",
                        "exp_value", "exp_error"])
            for i, r in enumerate(runs):
                for j in np.flatnonzero(np.isfinite(vals[i])):
                    w.writerow([r["design_point"], r["variant"], system[j], label[j],
                                repr(vals[i, j]), repr(errs[i, j]),
                                repr(exp_v[j]), repr(exp_e[j])])

    missing = np.isnan(vals).sum(axis=1)
    variants = sorted({r["variant"] for r in runs})
    print(f"{n_runs} runs ({len({r['design_point'] for r in runs})} design points; "
          f"variants {variants or ['-']}) x {n_obs} observables -> {a.output}")
    print(f"  experimental points aligned for {int(np.isfinite(exp_v).sum())}/{n_obs} observables")
    if missing.any():
        print(f"  {int((missing > 0).sum())} runs lack some observables "
              f"(max {int(missing.max())} missing)")
    if par_names:
        print(f"  design: {len(par_names)} parameters")
    counts = {c: int((rejected_by == c).sum()) for c in CRITERIA if (rejected_by == c).any()}
    print(f"  selected {int(selected.sum())}/{n_obs} observables for emulation"
          + (f" (rejected: {', '.join(f'{c} {n}' for c, n in counts.items())})" if counts else ""))


if __name__ == "__main__":
    main()
