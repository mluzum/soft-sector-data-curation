"""Simulation packages (analysis/sim_package.py, run_pipeline --sim-package).

Covers the manifest, adapter discovery and API check, the container
validation, variants/systems/design, and an end-to-end check that the
package path and the legacy ``--sim 2d`` path give identical pipeline output
on the same synthetic data.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.pipeline_dispatch import PTCUTS  # noqa: E402
from analysis.sim_package import (SimPackage, SimPackageError,  # noqa: E402
                                  build_container)
from analysis.qvector import MultiSpeciesGrid, QGridData, QVectorData  # noqa: E402

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("  PASS " if cond else "  FAIL ") + name)


def raises(fn, exc=SimPackageError):
    try:
        fn()
    except exc:
        return True
    return False


# ── synthetic data ───────────────────────────────────────────────────────

def _arrays_2d(n_ev=4, n_sp=2, n_pt=3):
    rng = np.random.default_rng(1)
    qn = rng.normal(size=(n_ev, n_sp, 6, n_pt)) + 1j * rng.normal(size=(n_ev, n_sp, 6, n_pt))
    qn[:, :, 0, :] = rng.integers(0, 20, size=(n_ev, n_sp, n_pt))
    return {"Qn": qn, "pt_bin_edges": np.linspace(0, 3, n_pt + 1),
            "masses": [0.14, 0.94][:n_sp], "species_names": ["pi+", "p"][:n_sp],
            "nsamples": np.ones(n_ev), "rap_max": 1.0,
            "sum_pt": np.ones((n_ev, n_sp, n_pt)), "meta": {"event_ids": np.arange(n_ev)}}


def _arrays_3d(n_ev=3, n_eta=4, n_pt=2):
    qn = np.zeros((n_ev, n_eta, n_pt, 4), complex)
    qn[..., 0] = 5.0
    return {"Qn": qn, "eta_edges": np.linspace(-2, 2, n_eta + 1),
            "pt_edges": np.linspace(0, 2, n_pt + 1), "cent_mult": np.arange(n_ev, dtype=float),
            "nsamples": np.ones(n_ev), "positives_only": False}


def _jetscape_qns(n_ev=300, seed=3):
    """Q-vectors in the JETSCAPE layout (event, species, harmonic, pT)."""
    rng = np.random.default_rng(seed)
    n_pt = len(PTCUTS) - 1
    centres = 0.5 * (PTCUTS[:-1] + PTCUTS[1:])
    lam = 40.0 * np.exp(-centres / 0.5)
    scale = rng.uniform(0.1, 1.5, n_ev)
    qn = np.zeros((n_ev, 5, 6, n_pt), complex)
    for s in range(5):
        n = rng.poisson(np.outer(scale, lam * (1.0, 0.3, 0.1, 0.03, 0.01)[s]))
        qn[:, s, 0, :] = n
        for h in range(1, 6):
            psi = rng.uniform(0, 2 * np.pi, n_ev)[:, None]
            v = (0.0, 0.01, 0.06, 0.02, 0.01, 0.005)[h] * np.minimum(centres / 1.5, 1.0)
            noise = rng.normal(size=n.shape) + 1j * rng.normal(size=n.shape)
            qn[:, s, h, :] = n * v * np.exp(1j * h * psi) + np.sqrt(n / 2) * noise
    return qn, rng.integers(1, 5, n_ev).astype(float)


ADAPTER = '''
import numpy as np
from pathlib import Path
API_VERSION = {api}

class Sim:
    def __init__(self, root, scale):
        self.root, self.scale = Path(root), scale
    def design_points(self):
        return sorted(int(p.stem.split("_")[1]) for p in self.root.glob("Qns_*.npy"))
    def load(self, dp, variant):
        qn = np.load(self.root / f"Qns_{{dp}}.npy")
        if variant == "doubled":
            qn = qn * self.scale
        return {{"Qn": qn, "pt_bin_edges": np.load(self.root / "edges.npy"),
                "masses": [0.13957, 0.49368, 0.93827, 1.18937, 1.32132],
                "species_names": ["pion", "kaon", "proton", "Sigma", "Xi"],
                "nsamples": np.load(self.root / f"Nsamples_{{dp}}.npy").astype(float),
                "rap_max": 1.0}}
    def anchor(self, dp):
        return None
    def source_files(self, dp):
        return [self.root / f"Qns_{{dp}}.npy"]

def open_simulation(data_root, package_dir, scale=1.0):
    return Sim(data_root, scale)
'''

MANIFEST = '''
format_version: 1
name: toy
container: 2d
data_root: {data_root}
adapter:
  module: {module}
  repo: toy-adapters
  options: {{scale: 2.0}}
systems:
  - {{reaction: Au Au, energy_gev: 200}}
variants: {variants}
design:
  file: design.csv
  id_column: label
'''


def _package(tmp, module, *, api=1, variants="[plain, doubled]", adapter_dir=None,
             points=(0,)):
    tmp = Path(tmp)
    data = tmp / "data"
    data.mkdir(exist_ok=True)
    for dp in points:
        qn, ns = _jetscape_qns(seed=3 + dp)
        np.save(data / f"Qns_{dp}.npy", qn)
        np.save(data / f"Nsamples_{dp}.npy", ns)
    qn, ns = _jetscape_qns(seed=3 + points[0])
    np.save(data / "edges.npy", PTCUTS)
    pkg = tmp / "pkg"
    pkg.mkdir(exist_ok=True)
    adir = Path(adapter_dir) if adapter_dir else pkg / "software" / "toy-adapters"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / f"{module}.py").write_text(textwrap.dedent(ADAPTER.format(api=api)))
    (pkg / "simulation.yaml").write_text(MANIFEST.format(
        data_root="../data", module=module, variants=variants))
    (pkg / "design.csv").write_text("label,eta_over_s,mode\ndesign_0000,0.1,a\ndesign_0003,0.2,b\n")
    sys.modules.pop(module, None)
    return pkg, data, qn, ns


# ── tests ────────────────────────────────────────────────────────────────

def test_build_container():
    qv = build_container(_arrays_2d(), "2d")
    check("2d container built", isinstance(qv, QVectorData) and qv.Qn.shape == (4, 2, 6, 3))
    check("2d meta kept", list(qv.meta["event_ids"]) == [0, 1, 2, 3])
    for name, edit in [
        ("unknown field refused", lambda a: a.update(sumpt=a["sum_pt"])),
        ("missing field refused", lambda a: a.pop("rap_max")),
        ("NaN refused", lambda a: a["Qn"].__setitem__((0, 0, 1, 0), np.nan)),
        ("negative count refused", lambda a: a["Qn"].__setitem__((0, 0, 0, 0), -1)),
        ("complex count refused", lambda a: a["Qn"].__setitem__((0, 0, 0, 0), 1 + 1j)),
        ("n_pt edges refused", lambda a: a.update(pt_bin_edges=np.linspace(0, 3, 3))),
        ("species mismatch refused", lambda a: a.update(masses=[0.14])),
        ("zero nsamples refused", lambda a: a["nsamples"].__setitem__(0, 0)),
        ("bad sum_pt shape refused", lambda a: a.update(sum_pt=np.ones((4, 2)))),
    ]:
        a = _arrays_2d()
        edit(a)
        check(name, raises(lambda: build_container(a, "2d")))
    g = build_container(_arrays_3d(), "3d")
    check("3d container built", isinstance(g, QGridData) and not g.positives_only)
    a = _arrays_3d()
    a.pop("positives_only")
    check("3d needs positives_only", raises(lambda: build_container(a, "3d")))
    m = build_container({"species": {"211": _arrays_3d(), 2212: _arrays_3d()},
                         "combined": _arrays_3d()}, "3d")
    check("3d multi-species", isinstance(m, MultiSpeciesGrid) and sorted(m.grids) == [211, 2212])
    check("unknown container kind", raises(lambda: build_container(_arrays_2d(), "4d")))


def test_package():
    with tempfile.TemporaryDirectory() as tmp:
        pkg_dir, data, qn, ns = _package(tmp, "toy_adapter_a")
        pkg = SimPackage(pkg_dir)
        check("adapter found in software/", pkg.adapter.__name__ == "toy_adapter_a")
        check("data root resolved", pkg.data_root == data.resolve())
        check("design points", pkg.design_points() == [0])
        qv = pkg.load(0, "plain")
        check("container equals source", np.array_equal(qv.Qn, qn) and np.array_equal(qv.nsamples, ns))
        check("cache", pkg.load(0, "plain") is qv)
        check("variant reaches adapter", np.array_equal(pkg.load(0, "doubled").Qn, 2 * qn))
        check("variant required", raises(lambda: pkg.load(0, None)))
        check("unknown variant refused", raises(lambda: pkg.load(0, "CE")))
        check("serves Au Au 200", pkg.serves("Au Au", "200.0") and pkg.serves("AU  AU", 200))
        check("not Pb Pb", not pkg.serves("Pb Pb", 2760) and not pkg.serves("Au Au", 62.4))
        names, table = pkg.design_table()
        check("design table", names == ["eta_over_s"] and table == {0: [0.1], 3: [0.2]})
        check("anchor passthrough", pkg.anchor(0) is None)
        check("source files", [p.name for p in pkg.source_files(0)] == ["Qns_0.npy"])
        prov = pkg.provenance()
        check("provenance", prov["name"] == "toy" and prov["adapter_module"] == "toy_adapter_a")
        check("data root override",
              SimPackage(pkg_dir, data_root=data).data_root == data.resolve())
        check("bad data root", raises(lambda: SimPackage(pkg_dir, data_root=Path(tmp) / "nope")))

    with tempfile.TemporaryDirectory() as tmp:
        elsewhere = Path(tmp) / "checkout"
        pkg_dir, *_ = _package(tmp, "toy_adapter_b", adapter_dir=elsewhere)
        check("adapter not found -> error", raises(lambda: SimPackage(pkg_dir)))
        sys.modules.pop("toy_adapter_b", None)
        check("--adapter-path", SimPackage(pkg_dir, adapter_path=elsewhere).name == "toy")
        sys.modules.pop("toy_adapter_b", None)
        os.environ["SIM_ADAPTERS_PATH"] = str(elsewhere)
        try:
            check("SIM_ADAPTERS_PATH", SimPackage(pkg_dir).name == "toy")
        finally:
            del os.environ["SIM_ADAPTERS_PATH"]

    with tempfile.TemporaryDirectory() as tmp:
        pkg_dir, *_ = _package(tmp, "toy_adapter_c", api=2)
        check("adapter API version checked", raises(lambda: SimPackage(pkg_dir)))
    with tempfile.TemporaryDirectory() as tmp:
        pkg_dir, *_ = _package(tmp, "toy_adapter_d", variants="null")
        pkg = SimPackage(pkg_dir)
        check("no variants: None ok", pkg.load(0, None).n_events == 300)
        check("no variants: variant refused", raises(lambda: pkg.load(0, "plain")))
        text = (pkg_dir / "simulation.yaml").read_text()
        (pkg_dir / "simulation.yaml").write_text(text.replace("format_version: 1",
                                                              "format_version: 9"))
        check("format version checked", raises(lambda: SimPackage(pkg_dir)))


def _h5_groups(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for g in f:
            out[g] = {k: f[g][k][()] for k in ("values", "errors", "covariance", "labels")}
    return out


def test_pipeline_matches_legacy_2d():
    """--sim-package and --sim 2d on the same Q-vectors: identical output."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg_dir, data, *_ = _package(tmp, "toy_adapter_e")
        common = [sys.executable, "scripts/run_pipeline.py", "--type", "flow_vn",
                  "--reaction", "AU AU --> CHARGED X", "--energy", "200",
                  "--value-type-regex", r"^v2\{2\}$", "--design-point", "0",
                  "--n-jk-blocks", "10"]
        legacy = Path(tmp) / "legacy.h5"
        viapkg = Path(tmp) / "package.h5"
        r1 = subprocess.run(common + ["--sim", "2d", "--sim-path", str(data),
                                      "--output", str(legacy)],
                            cwd=ROOT, capture_output=True, text=True)
        r2 = subprocess.run(common + ["--sim-package", str(pkg_dir), "--variant", "plain",
                                      "--output", str(viapkg)],
                            cwd=ROOT, capture_output=True, text=True)
        ok = r1.returncode == 0 and r2.returncode == 0
        check("both pipeline runs succeed", ok)
        if not ok:
            print(r1.stderr[-2000:], r2.stderr[-2000:])
            return
        a, b = _h5_groups(legacy), _h5_groups(viapkg)
        n = sum(len(g["values"]) for g in a.values())
        check(f"same groups and {n} observables", sorted(a) == sorted(b) and n > 0)
        check("identical labels", all(np.array_equal(a[g]["labels"], b[g]["labels"]) for g in a))
        # The containers are identical (test_package); the pipeline itself
        # reproduces only to the last bit, because numpy's SIMD reductions
        # sum in an order that depends on each array's memory alignment
        # (the legacy path differs from itself by ~1e-16 between runs).
        check("same values/errors/covariance to 1e-12",
              all(np.allclose(a[g][k], b[g][k], rtol=1e-12, atol=1e-15, equal_nan=True)
                  for g in a for k in ("values", "errors", "covariance")))
        import h5py
        with h5py.File(viapkg, "r") as f:
            check("provenance attrs", f.attrs["sim_package_name"] == "toy"
                  and f.attrs["variant"] == "plain" and f.attrs["sim_type"] == "2d")
        r3 = subprocess.run(common + ["--sim-package", str(pkg_dir),
                                      "--output", str(Path(tmp) / "x.h5")],
                            cwd=ROOT, capture_output=True, text=True)
        check("missing --variant is a usage error",
              r3.returncode == 2 and "choose a variant" in r3.stderr)


def test_campaign_driver_and_collector():
    """run_campaign.py runs, skips up-to-date runs, reruns changed points; collect stacks them."""
    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        pkg_dir, data, *_ = _package(tmp, "toy_adapter_f", points=(0, 1))
        out = Path(tmp) / "runs"
        cmd = [sys.executable, "scripts/run_campaign.py", "--sim-package", str(pkg_dir),
               "--out-dir", str(out), "--jobs", "2", "--",
               "--type", "flow_vn", "--reaction", "AU AU --> CHARGED X", "--energy", "200",
               "--value-type-regex", r"^v2\{2\}$", "--n-jk-blocks", "10"]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        names = sorted(p.name for p in out.glob("*.h5"))
        check("campaign: 2 points x 2 variants",
              r.returncode == 0 and names == ["dp0000_doubled.h5", "dp0000_plain.h5",
                                              "dp0001_doubled.h5", "dp0001_plain.h5"])
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        check("campaign: second pass all up to date", "4 up-to-date" in r.stdout)
        os.utime(data / "Qns_1.npy", ns=(1, 1))
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        check("campaign: changed source reruns its point only",
              "2 done" in r.stdout and "2 up-to-date" in r.stdout)
        r = subprocess.run(cmd[:8] + ["--dry-run", "--design-points", "1", "--variants", "plain"],
                           cwd=ROOT, capture_output=True, text=True)
        check("campaign: dry run selects", "1 design points x 1 variant(s)" in r.stdout)

        # The selection/code fingerprint: stored in the outputs, and a change
        # in either makes an output stale (records changed => rerun).
        import importlib.util
        import json as _json
        spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts" / "run_campaign.py")
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)
        sel = cmd[9:]
        base = ["--sim-package", str(pkg_dir)] + sel
        fp = rc.current_fingerprint(base, dict(os.environ))
        other = [r"^v3\{2\}$" if x == r"^v2\{2\}$" else x for x in base]
        fp_other = rc.current_fingerprint(other, dict(os.environ))
        check("fingerprint follows the selection",
              fp["records"] != fp_other["records"] and fp["code"] == fp_other["code"])
        out0 = out / "dp0000_plain.h5"
        with h5py.File(out0, "r") as f:
            stored = (f.attrs["records_fingerprint"], f.attrs["code_fingerprint"])
            argv = _json.loads(f.attrs["campaign_argv"])
            sig = _json.loads(f.attrs["campaign_sources"])
        check("outputs carry the fingerprint", stored == (fp["records"], fp["code"]))
        check("up to date with the same fingerprint", rc.up_to_date(out0, argv, sig, fp))
        check("stale when the records change",
              not rc.up_to_date(out0, argv, sig, dict(fp, records="0" * 16)))
        check("stale when the code changes, unless ignored",
              not rc.up_to_date(out0, argv, sig, dict(fp, code="0" * 16))
              and rc.up_to_date(out0, argv, sig, dict(fp, code="0" * 16), ignore_code=True))

        table = Path(tmp) / "table.h5"
        r = subprocess.run([sys.executable, "scripts/collect_campaign.py", str(out),
                            "--output", str(table), "--csv", str(Path(tmp) / "long.csv")],
                           cwd=ROOT, capture_output=True, text=True)
        check("collect runs", r.returncode == 0)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return
        with h5py.File(table, "r") as f:
            vals = f["values"][:]
            dps = list(f["runs/design_point"][:])
            var = [v.decode() for v in f["runs/variant"][:]]
            keys = [k.decode() for k in f["observables/key"][:]]
            pars = [p.decode() for p in f["design/parameter_names"][:]]
            dvals = f["design/values"][:]
            expv = f["observables/exp_values"][:]
        check("collect: runs sorted", dps == [0, 0, 1, 1] and var == ["doubled", "plain"] * 2)
        with h5py.File(out / "dp0001_plain.h5", "r") as f:
            g = list(f)[0]
            ref = f[g]["values"][:]
            ref_keys = [f"{g}|{x.decode()}" for x in f[g]["labels"][:]]
            ref_exp = f[g]["exp_values"][:]
        check("collect: values match the run file",
              keys == ref_keys and np.array_equal(vals[3], ref))
        check("collect: aligned experimental values",
              np.array_equal(expv, ref_exp, equal_nan=True) and np.isfinite(expv).any())
        check("collect: design table", pars == ["eta_over_s"] and dvals[0, 0] == 0.1
              and np.isnan(dvals[2, 0]))
        n_rows = sum(1 for _ in open(Path(tmp) / "long.csv")) - 1
        check("collect: long CSV", n_rows == int(np.isfinite(vals).sum()))


if __name__ == "__main__":
    test_build_container()
    test_package()
    test_pipeline_matches_legacy_2d()
    test_campaign_driver_and_collector()
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
