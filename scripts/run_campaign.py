#!/usr/bin/env python3
"""Run the pipeline over a simulation package's design points and variants.

    python scripts/run_campaign.py --sim-package <dir> --out-dir <dir> \
        [--design-points 1-100,205 | all] [--variants CE,PTB | all] [--jobs 8] \
        [-- <extra run_pipeline.py options>]

One ``run_pipeline.py`` process per (design point, variant), writing
``<out-dir>/dp<NNNN>[_<variant>].h5`` and a ``.log``.  The record selection
defaults to the manifest's ``selection`` (``--no-default-selection`` drops it);
options after ``--`` are appended.  A run is skipped while its output is
up to date: same pipeline arguments, the same source files (size and
mtime, as reported by the adapter), and the same selected records and
pipeline code (``run_pipeline.py --print-fingerprint``; ``--ignore-code-changes``
keeps outputs made with older code).  Outputs are written under a temporary
name and renamed on success, and discarded if a source file changed while
the run read it, so a live campaign is recomputed where it grew.  Then
``collect_campaign.py`` stacks the outputs into one table.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.sim_package import SimPackage, SimPackageError  # noqa: E402


def parse_points(spec: str, available: list[int]) -> list[int]:
    """'all' or a comma list of N and N-M ranges, restricted to points with data."""
    if spec == "all":
        return list(available)
    want = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = (int(x) for x in tok.split("-", 1))
            want.update(range(lo, hi + 1))
        else:
            want.add(int(tok))
    return sorted(want & set(available))


def selection_args(sel: dict, package_dir: Path) -> list[str]:
    """run_pipeline.py options from the manifest's selection block."""
    out = []
    for key, val in sel.items():
        flag = "--" + key.replace("_", "-")
        if key == "manifest":
            val = str((package_dir / val).resolve())
        if isinstance(val, (list, tuple)):
            out += [flag, *[str(v) for v in val]]
        elif isinstance(val, bool):
            if val:
                out.append(flag)
        else:
            out += [flag, str(val)]
    return out


def source_signature(pkg: SimPackage, dp: int) -> list:
    return [[str(p), p.stat().st_size, p.stat().st_mtime_ns]
            for p in pkg.source_files(dp) if p.exists()]


def output_name(dp: int, variant) -> str:
    return f"dp{dp:04d}" + (f"_{variant}" if variant else "") + ".h5"


def up_to_date(out: Path, argv: list[str], signature: list, fingerprint: dict,
               ignore_code=False) -> bool:
    if not out.exists():
        return False
    import h5py
    try:
        with h5py.File(out, "r") as f:
            if (json.loads(f.attrs.get("campaign_argv", "null")) != argv
                    or json.loads(f.attrs.get("campaign_sources", "null")) != signature):
                return False
            if f.attrs.get("records_fingerprint", "") != fingerprint["records"]:
                return False
            return ignore_code or f.attrs.get("code_fingerprint", "") == fingerprint["code"]
    except OSError:
        return False


def current_fingerprint(base_argv, env) -> dict:
    """Fingerprint of the selected records and the code, from run_pipeline itself."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
                        *base_argv, "--print-fingerprint"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines:
        raise SystemExit(f"run_pipeline.py --print-fingerprint failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return json.loads(lines[-1])


def run_one(pkg, dp, variant, base_argv, out_dir, force, env, fingerprint, ignore_code):
    out = out_dir / output_name(dp, variant)
    log = out.with_suffix(".log")
    argv = base_argv + ["--design-point", str(dp)] + (["--variant", variant] if variant else [])
    sig = source_signature(pkg, dp)
    if not force and sig and up_to_date(out, argv, sig, fingerprint, ignore_code):
        return dp, variant, "up-to-date", 0.0
    tmp = out.with_name(out.name + ".tmp")
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
                            *argv, "--output", str(tmp)],
                           cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return dp, variant, f"FAILED (exit {r.returncode}, see {log.name})", dt
    if source_signature(pkg, dp) != sig:
        tmp.unlink(missing_ok=True)
        return dp, variant, "DISCARDED (source changed during the run; rerun)", dt
    import h5py
    with h5py.File(tmp, "a") as f:
        f.attrs["campaign_argv"] = json.dumps(argv)
        f.attrs["campaign_sources"] = json.dumps(sig)
    os.replace(tmp, out)
    return dp, variant, "done", dt


def main():
    if "--" in sys.argv:
        i = sys.argv.index("--")
        own, extra = sys.argv[1:i], sys.argv[i + 1:]
    else:
        own, extra = sys.argv[1:], []
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-package", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--design-points", default="all",
                    help="'all' (default) or e.g. 1-100,205; only points with data run")
    ap.add_argument("--variants", default="all",
                    help="'all' (default) or a comma list")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel pipeline processes (each holds one design point "
                         "in memory, ~1-3 GB for 2000 events)")
    ap.add_argument("--no-default-selection", action="store_true",
                    help="ignore the manifest's selection; give it after --")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--force", action="store_true", help="rerun up-to-date outputs")
    ap.add_argument("--ignore-code-changes", action="store_true",
                    help="keep outputs made with older pipeline code (record "
                         "changes still trigger reruns)")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    a = ap.parse_args(own)

    try:
        pkg = SimPackage(a.sim_package, data_root=a.data_root, adapter_path=a.adapter_path)
    except SimPackageError as e:
        ap.error(str(e))
    points = parse_points(a.design_points, pkg.design_points())
    if a.variants == "all":
        variants = pkg.variants or [None]
    else:
        variants = [v.strip() for v in a.variants.split(",") if v.strip()]
        for v in variants:
            try:
                pkg.check_variant(v)
            except SimPackageError as e:
                ap.error(str(e))

    base = ["--sim-package", str(pkg.manifest_path.parent)]
    if a.data_root:
        base += ["--data-root", str(Path(a.data_root).resolve())]
    if a.adapter_path:
        base += ["--adapter-path", str(Path(a.adapter_path).resolve())]
    if not a.no_default_selection:
        base += selection_args(pkg.selection(), pkg.dir)
    base += extra

    out_dir = Path(a.out_dir)
    jobs = [(dp, v) for dp in points for v in variants]
    print(f"{pkg.name}: {len(points)} design points x {len(variants)} variant(s) "
          f"= {len(jobs)} runs -> {out_dir}")
    print("run_pipeline.py " + " ".join(base) + " --design-point N"
          + (" --variant V" if variants != [None] else ""))
    if a.dry_run or not jobs:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "1")      # parallelism comes from --jobs
    # numpy madvises transparent huge pages on large arrays; with THP defrag
    # 'madvise' and a full page cache every page fault then compacts memory
    # directly (O+O 40k: memory pressure 69.5% with 90 GB available, and
    # systemd-oomd kills at 50%; with this 0.00)
    env.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    fingerprint = current_fingerprint(base, env)
    print(f"selection: {fingerprint['n_records']} records (fingerprint {fingerprint['records']}), "
          f"code {fingerprint['code']}")

    counts = {}
    failed = []
    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        futs = [ex.submit(run_one, pkg, dp, v, base, out_dir, a.force, env, fingerprint,
                          a.ignore_code_changes) for dp, v in jobs]
        for k, fut in enumerate(futs, 1):
            dp, v, status, dt = fut.result()
            key = status.split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
            if key in ("FAILED", "DISCARDED"):
                failed.append((dp, v, status))
            if key != "up-to-date":
                print(f"[{k}/{len(jobs)}] {output_name(dp, v)}: {status} ({dt:.0f} s)", flush=True)
    print("summary: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    for dp, v, status in failed:
        print(f"  {output_name(dp, v)}: {status}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
