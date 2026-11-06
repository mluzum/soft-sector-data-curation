"""The centrality estimator on 3D containers counts charged particles.

The same toy events (positive pi, K, p) stored four ways:
  P  a positives-only grid (the stored count is half the charged count),
  A  an all-charged grid (Q x 2, positives_only false),
  S  a species collection without a combined grid,
  SC the species collection with a positives-only combined grid.
The estimator multiplicity (analysis.pipeline_dispatch.
estimator_multiplicity_3d) must be the same physical charged count on all
four, and every percentile result with it: Poisson-weighted classes smear
eff x N, so a halved N (P before November 2026) or the pions alone (S)
gave other classes.  Checked through the toy harness and through
run_pipeline on 3D simulation packages (P and A).
Run: python3 tests/test_centrality_3d_estimator.py
"""
import copy
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.qvector import MultiSpeciesGrid, _scale_grid  # noqa: E402
from analysis.toy_mc import SPECIES_MASSES, ToyGenConfig, evaluate, generate_events  # noqa: E402

RECORD = dict(
    observable_type='particle_yields', value_type='dNch_deta', observable='dNch_deta', hepdata_file='toy.yaml',
    inspire_id='0', version=1, particle_id=dict(species='charged'),
    collision_system={'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
    kinematic_range=dict(eta=dict(min=-0.5, max=0.5), pT=dict(min=0.2, max=3.0)),
    centrality=dict(classes=['0-10%', '10-30%', '30-60%', '60-100%'],
                    estimator=dict(name='RefMult', observable='Nch', detector='TPC'),
                    eta=[dict(min=-1.0, max=-0.5), dict(min=0.5, max=1.0)], pT=dict(min=0.2, max=100.0)))
results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + ('' if ok else f'  {detail}'))


def containers():
    cfg = ToyGenConfig(n_events=1500, mult_mean=120, mult_sigma=60, rap_max=1.5, seed=303, pt_min=0.05,
                       pt_max=3.0, flow={2: dict(kind='const', value=0.05)})
    events, _ = generate_events(cfg)
    eta_edges = np.round(np.linspace(-1.5, 1.5, 31), 10)
    pt_edges = np.round(np.linspace(0.0, 3.0, 31), 10)
    mult = np.array([len(e['pt']) for e in events], float)

    def grid(slot=None):
        evs = events if slot is None else [{k: v[e['species_idx'] == slot] for k, v in e.items()} for e in events]
        return particles_to_qgrid_data(evs, eta_edges, pt_edges, n_harm=4, masses=SPECIES_MASSES, cent_mult=mult)
    P = grid()
    A = _scale_grid(P, 2.0)
    A.positives_only = False
    species = {211: grid(0), 321: grid(1), 2212: grid(2)}
    S = MultiSpeciesGrid(dict(species))
    SC = MultiSpeciesGrid(dict(species), combined=grid())
    N = np.array([2 * np.count_nonzero((np.abs(e['eta']) > 0.5) & (np.abs(e['eta']) < 1.0) & (e['pt'] >= 0.2))
                  for e in events], float)
    return events, {'P': P, 'A': A, 'S': S, 'SC': SC}, N


ADAPTER = '''
import numpy as np
from pathlib import Path
API_VERSION = 1

class Sim:
    def __init__(self, root):
        self.root = Path(root)
    def design_points(self):
        return [0]
    def load(self, dp, variant):
        z = np.load(self.root / "grid.npz")
        return {"Qn": z["Qn"], "eta_edges": z["eta_edges"], "pt_edges": z["pt_edges"],
                "cent_mult": z["cent_mult"], "nsamples": z["nsamples"],
                "positives_only": bool(z["positives_only"])}
    def anchor(self, dp):
        return None
    def source_files(self, dp):
        return [self.root / "grid.npz"]

def open_simulation(data_root, package_dir):
    return Sim(data_root)
'''

MANIFEST = '''
format_version: 1
name: toy3d
container: 3d
data_root: {data_root}
adapter:
  module: {module}
  repo: toy-adapters
systems:
  - {{reaction: Au Au, energy_gev: 200}}
variants: [plain]
design:
  file: design.csv
  id_column: label
'''


def package(tmp, name, g):
    tmp = Path(tmp) / name
    data, pkg = tmp / "data", tmp / "pkg"
    data.mkdir(parents=True)
    np.savez(data / "grid.npz", Qn=g.Qn, eta_edges=g.eta_edges, pt_edges=g.pt_edges, cent_mult=g.cent_mult,
             nsamples=g.nsamples, positives_only=g.positives_only)
    adir = pkg / "software" / "toy-adapters"
    adir.mkdir(parents=True)
    module = f"toy_adapter_3d_{name}"
    (adir / f"{module}.py").write_text(textwrap.dedent(ADAPTER))
    (pkg / "simulation.yaml").write_text(MANIFEST.format(data_root="../data", module=module))
    (pkg / "design.csv").write_text("label,x\ndesign_0000,0.1\n")
    return pkg


def main():
    from analysis.pipeline_dispatch import estimator_multiplicity_3d
    from analysis.weights import CentralityWeights
    events, conts, N = containers()
    eta = [(-1.0, -0.5), (0.5, 1.0)]
    m = {k: estimator_multiplicity_3d(c, eta, (0.2, None)) for k, c in conts.items()}
    check('estimator multiplicity = the charged count on every container (P, A, S, SC)',
          all(np.allclose(m[k], N, rtol=1e-12, atol=1e-9) for k in m),
          {k: float(np.abs(v - N).max()) for k, v in m.items()})
    edges = [(0, 10), (10, 30), (30, 60), (60, 100)]
    w_full = CentralityWeights('poisson', N, efficiency=0.9).weights(edges, raw=True)
    w_half = CentralityWeights('poisson', N / 2, efficiency=0.9).weights(edges, raw=True)
    check('a halved multiplicity changes the Poisson-weighted classes (what P had before)',
          np.abs(w_full - w_half).max() > 0.05, f'{np.abs(w_full - w_half).max():.3f}')

    vals = {k: evaluate(RECORD, grid=c, cent_method='poisson', with_jackknife=False)[0] for k, c in conts.items()}
    print('    ' + '  '.join(f"{k}: {v.round(3)}" for k, v in vals.items()))
    check('toy harness: identical classes and values on P, A, S, SC',
          all(np.allclose(vals[k], vals['A'], rtol=1e-10) for k in vals))
    rec = copy.deepcopy(RECORD)
    rec['centrality']['normalization'] = dict(estimator_threshold=60, fraction_above=0.3, efficiency=0.9)
    vn = {k: evaluate(rec, grid=c, cent_method='poisson', with_jackknife=False)[0] for k, c in conts.items()}
    check('threshold normalization (N_trk >= 60): identical on P, A, S, SC',
          all(np.allclose(vn[k], vn['A'], rtol=1e-10) for k in vn))

    import h5py
    with tempfile.TemporaryDirectory() as tmp:
        rdir = Path(tmp) / "records" / "data" / "particle_yields"
        rdir.mkdir(parents=True)
        yaml.safe_dump(rec, open(rdir / "rec.yaml", "w"))
        manifest = Path(tmp) / "manifest.yaml"           # this record only
        yaml.safe_dump({"records": [{"fpath": str(rdir / "rec.yaml"), "doc_index": 0}]}, open(manifest, "w"))
        got = {}
        for name in ('P', 'A'):
            pkg = package(tmp, name, conts[name])
            out = Path(tmp) / f"{name}.h5"
            r = subprocess.run([sys.executable, "scripts/run_pipeline.py", "--sim-package", str(pkg),
                                "--variant", "plain", "--design-point", "0", "--n-jk-blocks", "10",
                                "--manifest", str(manifest), "--output", str(out)],
                               cwd=ROOT, capture_output=True, text=True)
            check(f"run_pipeline on the {name} package succeeds", r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:])
            if r.returncode != 0:
                return
            with h5py.File(out, "r") as f:
                got[name] = np.concatenate([np.asarray(g["values"][:]) for g in f.values()
                                            if "values" in g and "labels" in g])
        print(f"    run_pipeline P {got['P'].round(3)}  A {got['A'].round(3)}  toy {vn['A'].round(3)}")
        check('run_pipeline: positives-only and all-charged packages give the same values, as the toy harness',
              len(got['P']) == len(vn['A']) and np.allclose(got['P'], got['A'], rtol=1e-9)
              and np.allclose(got['A'], vn['A'], rtol=1e-9))


if __name__ == '__main__':
    main()
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
