#!/usr/bin/env python3
"""Check a simulation package and summarize what the pipeline will see.

    python scripts/check_sim_package.py <package dir> [--design-point N] [--variant V]

Opens the manifest, imports the adapter, loads one design point (by default
the first available, every variant) through the same validation as
run_pipeline.py, and prints the container geometry, event statistics and
centrality anchor.  Exit status 1 if anything breaks the package contract.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.sim_package import SimPackage, SimPackageError  # noqa: E402
from analysis.weights import resolve_anchor  # noqa: E402


def _summary_2d(qv):
    counts = qv.Qn[:, :, 0, :].real.sum(axis=(1, 2)) / qv.nsamples
    e = qv.pt_bin_edges
    widths = np.unique(np.round(np.diff(e), 10))
    lines = [
        f"species      {qv.n_species}: {', '.join(qv.species_names)}",
        f"pT grid      {e[0]:g}-{e[-1]:g} GeV, {qv.n_pt} bins"
        + (f" of {widths[0]:g} GeV" if len(widths) == 1 else " (non-uniform)"),
        f"harmonics    Q0..Q{qv.n_harmonics - 1}",
        f"rapidity     flat over |y| < {qv.rap_max:g}",
        f"oversampling nsamples {qv.nsamples.min():g}-{qv.nsamples.max():g}",
        f"yield/event  all species, all pT: median {np.median(counts):.1f}, "
        f"max {counts.max():.1f}; {int((counts == 0).sum())} empty events",
        f"moments      " + ", ".join(n for n in ("sum_pt", "sum_pt2", "sum_mT", "cent_mult",
                                                    "event_weights")
                                     if getattr(qv, n, None) is not None),
    ]
    return lines, counts


def _summary_3d(g):
    grid = getattr(g, "combined", None) or g
    if hasattr(g, "grids"):
        head = [f"species      {len(g.grids)} grids: {sorted(g.grids)}"]
        grid = g.combined or next(iter(g.grids.values()))
    else:
        head = []
    counts = grid.Qn[..., 0].real.sum(axis=(1, 2)) / grid.nsamples
    return head + [
        f"eta grid     {grid.eta_edges[0]:g}-{grid.eta_edges[-1]:g}, {len(grid.eta_edges) - 1} bins",
        f"pT grid      {grid.pt_edges[0]:g}-{grid.pt_edges[-1]:g} GeV, {len(grid.pt_edges) - 1} bins",
        f"harmonics    Q0..Q{grid.Qn.shape[-1] - 1}",
        f"charges      {'positives only (x2 for charged)' if grid.positives_only else 'all'}",
        f"yield/event  median {np.median(counts):.1f}; {int((counts == 0).sum())} empty events",
    ], counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("--design-point", type=int, default=None)
    ap.add_argument("--variant", default=None, help="default: every variant")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--adapter-path", default=None)
    a = ap.parse_args()

    try:
        pkg = SimPackage(a.package, data_root=a.data_root, adapter_path=a.adapter_path)
        prov = pkg.provenance()
        print(f"package      {pkg.name}: {pkg.manifest.get('title', '')}")
        print(f"manifest     {prov['manifest']}")
        print(f"container    {pkg.container}")
        print(f"systems      " + "; ".join(pkg.system_labels))
        print(f"variants     {', '.join(pkg.variants) if pkg.variants else '(none)'}")
        print(f"adapter      {prov['adapter_module']} {prov['adapter_version']} "
              f"({prov['adapter_file']}; commit {prov['adapter_commit'] or '?'})")
        print(f"analysis     commit {prov['analysis_commit'] or '?'}")
        print(f"data root    {pkg.data_root}")
        dps = pkg.design_points()
        names, design = pkg.design_table()
        print(f"design       {len(design)} points x {len(names)} parameters; "
              f"{len(dps)} with data"
              + (f" ({dps[0]}-{dps[-1]})" if dps else ""))
        missing = sorted(set(dps) - set(design)) if design else []
        if missing:
            raise SimPackageError(f"design points with data but no design row: {missing[:10]}")
        sel = pkg.selection()
        if sel:
            print(f"selection    {sel}")
        if not dps:
            raise SimPackageError("no design point has data")
        dp = a.design_point if a.design_point is not None else dps[0]
        variants = [a.variant] if a.variant else (pkg.variants or [None])
        for v in variants:
            obj = pkg.load(dp, v)
            print(f"\n-- design point {dp}" + (f", variant {v}" if v else "")
                  + f": {obj.n_events} events")
            if design.get(dp):
                print("parameters   " + ", ".join(f"{n}={x:.4g}" for n, x in
                                                  zip(names, design[dp])))
            lines, _ = (_summary_2d(obj) if pkg.container == "2d" else _summary_3d(obj))
            for line in lines:
                print(line)
            meta = getattr(obj, "meta", {}) or {}
            if meta:
                print("meta         " + ", ".join(
                    f"{k}={val if np.ndim(val) == 0 else f'<{np.size(val)} values>'}"
                    for k, val in meta.items()))
            spec = pkg.anchor(dp)
            if spec is None:
                print("anchor       none from the package (full set = 100%, unless "
                      "--cent-anchor)")
            else:
                an = resolve_anchor(spec, obj.n_events)
                print(f"anchor       {spec} -> f={an['f']:.4f}, events cover "
                      f"0-{an['anchor_cent']:.1f}% centrality")
    except (SimPackageError, OSError, KeyError, ValueError) as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
