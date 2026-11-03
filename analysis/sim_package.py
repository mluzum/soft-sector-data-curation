"""Simulation packages: simulation data that describe themselves.

A *simulation package* is a small directory distributed with a set of
simulation results.  Its manifest, ``simulation.yaml``, tells the pipeline
what the data are (container kind, collision systems, design points,
variants such as delta f models, design table, default record selection)
and names an *adapter*: a Python module, maintained outside this repository,
that reads the simulation's own output format and returns the arrays of one
of the two pipeline containers (:class:`analysis.qvector.QVectorData` for
2D, :class:`analysis.qvector.QGridData` for 3D).  The manifest format is
documented in ``docs/sim_interface.md``.

Adapters import numpy (and whatever reads their format) but never this
repository, so the analysis code and the adapters evolve independently.
The adapter contract, version ``ADAPTER_API = 1``::

    API_VERSION = 1
    def open_simulation(data_root, package_dir, **options) -> sim
    sim.design_points() -> list[int]
    sim.load(design_point, variant) -> dict   # container arrays, see below
    sim.anchor(design_point) -> anchor spec or None   (optional)
    sim.source_files(design_point) -> list[path]      (optional)

``load`` returns plain arrays keyed by the container's field names:

- 2D: ``Qn`` (n_events, n_species, n_harm, n_pt) complex with the counts
  in slot 0, ``pt_bin_edges`` (n_pt+1), ``masses``, ``species_names``,
  ``nsamples``, ``rap_max``; optional ``pdg_ids``, ``sum_pt``, ``sum_pt2``,
  ``sum_mT``, ``cent_mult``, ``event_weights``.
- 3D: ``Qn`` (n_events, n_eta, n_pt, n_harm) complex, ``eta_edges``,
  ``pt_edges``, ``cent_mult``, ``nsamples``, ``positives_only``; optional
  ``event_weights``, ``sum_pt``, ``sum_pt2``, ``sum_ET``.  Several species:
  ``{"species": {pdg: <3D dict>}, "combined": <3D dict> or None}``.  A
  species grid binned in rapidity gives ``y_edges`` instead of
  ``eta_edges`` (its mass comes from the PDG code); the combined grid is
  binned in eta.

Anything else goes under ``meta`` (kept on the container as ``.meta``).
Unknown top-level keys are refused, so a misspelt optional field cannot be
silently dropped.  ``anchor`` returns anything
:func:`analysis.weights.resolve_anchor` accepts (the coverage of the loaded
events in the total cross section).
"""

from __future__ import annotations

import csv
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from analysis.qvector import SPECIES_MASS, MultiSpeciesGrid, QGridData, QVectorData


MANIFEST_NAME = "simulation.yaml"
FORMAT_VERSION = 1
ADAPTER_API = 1

_REPO_ROOT = Path(__file__).resolve().parent.parent

_FIELDS_2D_REQUIRED = ("Qn", "pt_bin_edges", "masses", "species_names",
                       "nsamples", "rap_max")
_FIELDS_2D_OPTIONAL = ("pdg_ids", "sum_pt", "sum_pt2", "sum_mT",
                       "cent_mult", "event_weights")
_FIELDS_3D_REQUIRED = ("Qn", "eta_edges", "pt_edges", "cent_mult",
                       "nsamples", "positives_only")
_FIELDS_3D_OPTIONAL = ("event_weights", "sum_pt", "sum_pt2", "sum_ET")
_FIELDS_3D_AXIS = ("eta_edges", "y_edges")       # exactly one; y_edges for species grids only


class SimPackageError(Exception):
    """A package, its manifest or its adapter breaks the contract."""


def _norm_reaction(reaction: str) -> str:
    return " ".join(str(reaction).split()).upper()


def _git_commit(path: Path) -> str:
    """HEAD of the git checkout containing ``path`` ('' if none), '+dirty' if modified."""
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return ""
        # --no-optional-locks: plain `git status` may take .git/index.lock to
        # refresh the index; many concurrent pipeline processes on NFS left
        # stale locks that blocked later checkouts (isobar campaign, 2026-11-02).
        dirty = subprocess.run(["git", "--no-optional-locks", "-C", str(path), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
        return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return ""


def adapter_search_path(spec: dict, package_dir: Path,
                        adapter_path: str | os.PathLike | None = None) -> list[Path]:
    """Directories searched for the adapter module, highest priority first.

    1. ``adapter_path`` (CLI ``--adapter-path``);
    2. ``$SIM_ADAPTERS_PATH`` (os.pathsep-separated);
    3. ``<package>/software/<adapter.repo>`` (where the package's setup
       script clones it);
    4. ``<adapter.repo>`` next to this repository's checkout.

    After these, an installed module (``pip install``) is found as usual.
    """
    cands = []
    if adapter_path:
        cands.append(Path(adapter_path))
    env = os.environ.get("SIM_ADAPTERS_PATH")
    if env:
        cands += [Path(p) for p in env.split(os.pathsep) if p]
    repo = spec.get("repo")
    if repo:
        cands.append(package_dir / "software" / repo)
        cands.append(_REPO_ROOT.parent / repo)
    return [c.resolve() for c in cands if c.is_dir()]


def _import_adapter(spec: dict, package_dir: Path, adapter_path=None):
    module = spec.get("module")
    if not module:
        raise SimPackageError("manifest: adapter.module is required")
    paths = adapter_search_path(spec, package_dir, adapter_path)
    for p in reversed(paths):
        s = str(p)
        sys.path[:] = [x for x in sys.path if x != s]
        sys.path.insert(0, s)
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise SimPackageError(
            f"cannot import adapter {module!r} (searched "
            f"{[str(p) for p in paths] or 'installed modules only'}): {e}. "
            f"Run the package's setup script, pass --adapter-path, or set "
            f"SIM_ADAPTERS_PATH") from e
    api = getattr(mod, "API_VERSION", None)
    if api != ADAPTER_API:
        raise SimPackageError(
            f"adapter {module!r} implements API_VERSION {api!r}; this "
            f"repository speaks {ADAPTER_API}")
    if not callable(getattr(mod, "open_simulation", None)):
        raise SimPackageError(f"adapter {module!r} has no open_simulation()")
    return mod


# ── Container construction and validation ────────────────────────────────

def _take(arrays: dict, required, optional, where: str) -> dict:
    unknown = set(arrays) - set(required) - set(optional)
    if unknown:
        raise SimPackageError(
            f"{where}: unknown fields {sorted(unknown)} (allowed: "
            f"{sorted(required + optional)}; put metadata under 'meta')")
    missing = [k for k in required if arrays.get(k) is None]
    if missing:
        raise SimPackageError(f"{where}: missing fields {missing}")
    return {k: arrays[k] for k in required + optional if arrays.get(k) is not None}


def _check(cond, where: str, msg: str):
    if not cond:
        raise SimPackageError(f"{where}: {msg}")


def _edges(x, name: str, n: int, where: str) -> np.ndarray:
    e = np.asarray(x, dtype=float)
    _check(e.ndim == 1 and len(e) == n + 1, where,
           f"{name} must have {n + 1} entries (n_bins + 1), got shape {e.shape}")
    _check(np.all(np.isfinite(e)) and np.all(np.diff(e) > 0), where,
           f"{name} must be finite and strictly increasing")
    return e


def _per_event(x, name: str, n_ev: int, where: str, positive=False) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    _check(v.shape == (n_ev,), where, f"{name} shape {v.shape} != ({n_ev},)")
    _check(np.all(np.isfinite(v)), where, f"{name} has non-finite entries")
    if positive:
        _check(np.all(v > 0), where, f"{name} must be > 0")
    return v


def _moments(kw: dict, names, shape, where: str):
    for nm in names:
        if nm in kw:
            v = np.asarray(kw[nm], dtype=float)
            _check(v.shape == shape, where, f"{nm} shape {v.shape} != {shape}")
            _check(np.all(np.isfinite(v)), where, f"{nm} has non-finite entries")
            kw[nm] = v


def _complex_qn(x, where: str) -> np.ndarray:
    qn = np.asarray(x)
    if not np.iscomplexobj(qn):
        qn = qn.astype(np.complex128)
    _check(qn.ndim == 4, where, f"Qn must be 4D, got shape {qn.shape}")
    _check(np.all(np.isfinite(qn)), where, "Qn has non-finite entries")
    return qn


def _build_2d(arrays: dict, where: str) -> QVectorData:
    kw = _take(arrays, _FIELDS_2D_REQUIRED, _FIELDS_2D_OPTIONAL, where)
    qn = _complex_qn(kw["Qn"], where)
    n_ev, n_sp, _n_h, n_pt = qn.shape
    counts = qn[:, :, 0, :]
    _check(np.all(counts.imag == 0) and np.all(counts.real >= 0), where,
           "Qn[:, :, 0, :] must hold real non-negative counts")
    kw["Qn"] = qn
    kw["pt_bin_edges"] = _edges(kw["pt_bin_edges"], "pt_bin_edges", n_pt, where)
    kw["masses"] = np.asarray(kw["masses"], dtype=float)
    kw["species_names"] = [str(s) for s in kw["species_names"]]
    _check(len(kw["masses"]) == n_sp == len(kw["species_names"]), where,
           f"{n_sp} species in Qn, {len(kw['masses'])} masses, "
           f"{len(kw['species_names'])} species_names")
    if "pdg_ids" in kw:
        kw["pdg_ids"] = [int(p) for p in kw["pdg_ids"]]
        _check(len(kw["pdg_ids"]) == n_sp, where, "pdg_ids length != n_species")
    kw["nsamples"] = _per_event(kw["nsamples"], "nsamples", n_ev, where, positive=True)
    for nm in ("cent_mult", "event_weights"):
        if nm in kw:
            kw[nm] = _per_event(kw[nm], nm, n_ev, where)
    kw["rap_max"] = float(kw["rap_max"])
    _check(kw["rap_max"] > 0, where, "rap_max must be > 0")
    _moments(kw, ("sum_pt", "sum_pt2", "sum_mT"), (n_ev, n_sp, n_pt), where)
    return QVectorData(sim_type="2d", **kw)


def _build_3d(arrays: dict, where: str, pdg=None) -> QGridData:
    axis = [a for a in _FIELDS_3D_AXIS if a in arrays]
    _check(len(axis) == 1, where, f"a 3D grid gives exactly one of eta_edges / y_edges, got {axis}")
    axis = axis[0]
    if axis == "y_edges":
        _check(pdg is not None, where, "y_edges (a rapidity-binned grid) is for species grids only")
        _check(int(pdg) in SPECIES_MASS, where, f"no mass for PDG {pdg} (needed for eta <-> y)")
    req = tuple(f for f in _FIELDS_3D_REQUIRED if f != "eta_edges") + (axis,)
    kw = _take(arrays, req, _FIELDS_3D_OPTIONAL, where)
    qn = _complex_qn(kw["Qn"], where)
    n_ev, n_eta, n_pt, _n_h = qn.shape
    counts = qn[..., 0]
    _check(np.all(counts.imag == 0) and np.all(counts.real >= 0), where,
           "Qn[..., 0] must hold real non-negative counts")
    kw["Qn"] = qn
    kw[axis] = _edges(kw[axis], axis, n_eta, where)
    if axis == "y_edges":
        kw["eta_edges"] = None
        kw["y_mass"] = float(SPECIES_MASS[int(pdg)])
    kw["pt_edges"] = _edges(kw["pt_edges"], "pt_edges", n_pt, where)
    kw["nsamples"] = _per_event(kw["nsamples"], "nsamples", n_ev, where, positive=True)
    kw["cent_mult"] = _per_event(kw["cent_mult"], "cent_mult", n_ev, where)
    if "event_weights" in kw:
        kw["event_weights"] = _per_event(kw["event_weights"], "event_weights", n_ev, where)
    _check(isinstance(kw["positives_only"], (bool, np.bool_)), where,
           "positives_only must be True or False (it sets the charge factor)")
    kw["positives_only"] = bool(kw["positives_only"])
    _moments(kw, ("sum_pt", "sum_pt2", "sum_ET"), (n_ev, n_eta, n_pt), where)
    return QGridData(sim_type="3d", **kw)


def build_container(arrays: dict, kind: str, where: str = "adapter"):
    """Validate an adapter's arrays and wrap them in the pipeline container.

    ``kind`` is the manifest's ``container`` ("2d" or "3d").  Raises
    :class:`SimPackageError` on any violation of the contract (the
    container classes' own asserts are not relied on: they vanish under
    ``python -O``).
    """
    if not isinstance(arrays, dict):
        raise SimPackageError(f"{where}: load() must return a dict, got {type(arrays)}")
    arrays = dict(arrays)
    meta = dict(arrays.pop("meta", None) or {})
    if kind == "2d":
        obj = _build_2d(arrays, where)
    elif kind == "3d" and "species" in arrays:
        _check(set(arrays) <= {"species", "combined"}, where,
               f"multi-species 3D dict takes 'species' and 'combined', got {sorted(arrays)}")
        grids = {int(pdg): _build_3d(d, f"{where} species {pdg}", pdg=pdg)
                 for pdg, d in arrays["species"].items()}
        comb = arrays.get("combined")
        obj = MultiSpeciesGrid(
            grids, combined=_build_3d(comb, f"{where} combined") if comb is not None else None)
        n = {g.n_events for g in grids.values()}
        _check(len(n) <= 1, where, f"species grids disagree on n_events: {sorted(n)}")
    elif kind == "3d":
        obj = _build_3d(arrays, where)
    else:
        raise SimPackageError(f"{where}: container must be '2d' or '3d', got {kind!r}")
    obj.meta = meta
    return obj


def read_design_table(manifest: dict, package_dir) -> tuple[list[str], dict[int, list[float]]]:
    """(parameter names, {design point: values}) from a manifest's design file.

    The design point is the trailing integer of ``design.id_column``
    (e.g. ``design_0042`` -> 42); non-numeric columns are dropped.  Needs
    only the manifest, not the adapter or the data.
    """
    d = manifest.get("design") or {}
    if not d.get("file"):
        return [], {}
    id_col = d.get("id_column", "label")
    with open(Path(package_dir) / d["file"], newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return [], {}
    names = []
    for k in rows[0]:
        if k == id_col:
            continue
        try:
            [float(r[k]) for r in rows]
        except (TypeError, ValueError):
            continue
        names.append(k)
    table = {}
    for r in rows:
        mt = re.search(r"(\d+)\s*$", str(r[id_col]))
        if mt:
            table[int(mt.group(1))] = [float(r[k]) for k in names]
    return names, table


# ── The package ──────────────────────────────────────────────────────────

class SimPackage:
    """A simulation package opened for reading.

    Parameters
    ----------
    path : the package directory or its ``simulation.yaml``
    data_root : overrides the manifest's ``data_root`` (data hosted apart
        from the package)
    adapter_path : directory to import the adapter from, searched first
    """

    def __init__(self, path, *, data_root=None, adapter_path=None):
        p = Path(path)
        mpath = p if p.is_file() else p / MANIFEST_NAME
        if not mpath.is_file():
            raise SimPackageError(f"no {MANIFEST_NAME} at {p}")
        with open(mpath) as fh:
            m = yaml.safe_load(fh) or {}
        self.manifest_path = mpath.resolve()
        self.dir = self.manifest_path.parent
        self.manifest = m
        if m.get("format_version") != FORMAT_VERSION:
            raise SimPackageError(
                f"{mpath}: format_version {m.get('format_version')!r}, "
                f"this repository reads {FORMAT_VERSION}")
        self.name = str(m.get("name") or self.dir.name)
        self.container = m.get("container")
        if self.container not in ("2d", "3d"):
            raise SimPackageError(f"{mpath}: container must be 2d or 3d")
        variants = m.get("variants")
        self.variants = [str(v) for v in variants] if variants else None
        self.systems = []          # (normalized reaction, energy) for matching
        self.system_labels = []    # as written in the manifest
        self.frames = {}           # asymmetric systems: container orientation
        for s in m.get("systems") or []:
            self.systems.append((_norm_reaction(s["reaction"]), float(s["energy_gev"])))
            if s.get("forward_beam"):
                self.frames[self.systems[-1]] = {"frame": s.get("frame", "nn_cms"),
                                                 "forward_beam": str(s["forward_beam"])}
            self.system_labels.append(f"{s['reaction']} {float(s['energy_gev']):g} GeV")
        if not self.systems:
            raise SimPackageError(f"{mpath}: list the simulated systems under 'systems'")

        root = Path(data_root) if data_root else Path(m.get("data_root") or ".")
        if not root.is_absolute():
            root = (Path.cwd() if data_root else self.dir) / root
        self.data_root = root.resolve()
        if not self.data_root.is_dir():
            raise SimPackageError(f"data root {self.data_root} is not a directory "
                                  f"(set data_root in {MANIFEST_NAME} or pass --data-root)")

        spec = m.get("adapter") or {}
        self.adapter = _import_adapter(spec, self.dir, adapter_path)
        self.sim = self.adapter.open_simulation(self.data_root, self.dir,
                                                **(spec.get("options") or {}))
        self._cache = None

    # -- description -------------------------------------------------------

    def serves(self, reaction: str, energy) -> bool:
        """True if the package simulates ``reaction`` (normalized form, e.g.
        'Ru Ru') at ``energy`` GeV."""
        r, e = _norm_reaction(reaction), float(energy)
        return any(r == sr and abs(e - se) <= 1e-6 * max(1.0, se)
                   for sr, se in self.systems)

    def frame(self, reaction: str, energy):
        """Orientation of the container for an asymmetric system (the
        system entry's frame and forward_beam, the beam toward +eta), or
        None when the manifest does not declare it."""
        r, e = _norm_reaction(reaction), float(energy)
        for (sr, se), f in self.frames.items():
            if r == sr and abs(e - se) <= 1e-6 * max(1.0, se):
                return f
        return None

    def check_variant(self, variant):
        if self.variants is None:
            if variant is not None:
                raise SimPackageError(f"{self.name} has no variants; drop --variant")
        elif variant is None:
            raise SimPackageError(f"{self.name}: choose a variant with --variant "
                                  f"({', '.join(self.variants)})")
        elif variant not in self.variants:
            raise SimPackageError(f"{self.name}: no variant {variant!r} "
                                  f"(have {', '.join(self.variants)})")

    def design_points(self) -> list[int]:
        return sorted(int(d) for d in self.sim.design_points())

    def selection(self) -> dict:
        """The manifest's default record selection (run_pipeline options)."""
        return dict(self.manifest.get("selection") or {})

    def design_table(self) -> tuple[list[str], dict[int, list[float]]]:
        """(parameter names, {design point: values}); see :func:`read_design_table`."""
        return read_design_table(self.manifest, self.dir)

    def provenance(self) -> dict[str, str]:
        mod_file = Path(getattr(self.adapter, "__file__", "") or ".").resolve()
        return {
            "name": self.name,
            "manifest": str(self.manifest_path),
            "data_root": str(self.data_root),
            "adapter_module": self.adapter.__name__,
            "adapter_file": str(mod_file),
            "adapter_version": str(getattr(self.adapter, "__version__", "")),
            "adapter_commit": _git_commit(mod_file.parent),
            "analysis_commit": _git_commit(_REPO_ROOT),
        }

    # -- data --------------------------------------------------------------

    def load(self, design_point: int, variant: str | None = None):
        """The container of one design point (and variant); the last one is cached."""
        self.check_variant(variant)
        key = (int(design_point), variant)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        where = f"{self.name} design point {design_point}" + (
            f" variant {variant}" if variant else "")
        obj = build_container(self.sim.load(int(design_point), variant),
                              self.container, where)
        self._cache = (key, obj)
        return obj

    def anchor(self, design_point: int):
        """Centrality anchor of the loaded events (None: the adapter has none)."""
        fn = getattr(self.sim, "anchor", None)
        return fn(int(design_point)) if callable(fn) else None

    def source_files(self, design_point: int) -> list[Path]:
        """Files the design point's data come from ([] if the adapter does not say)."""
        fn = getattr(self.sim, "source_files", None)
        return [Path(p) for p in fn(int(design_point))] if callable(fn) else []
