"""Plot simulation vs experiment from pipeline HDF5 output.

Categorizes observables as:
  - integrated vs centrality (% in label, no pT)
  - pT-differential with fixed centrality (pT + % in label)
  - yield/spectra vs centrality (label type starts with yield_ or spectra_)

Usage:
    python scripts/plot_comparison.py \\
        --sim-2d /tmp/auau200_2d_flow_vn.h5 \\
        --sim-3d /tmp/auau200_3d_flow_vn.h5 \\
        --label-pattern "vn2_single|vn4_single" \\
        --output /tmp/vn2_comparison.pdf

    python scripts/plot_comparison.py \\
        --sim-2d /tmp/auau200_2d_yields.h5 \\
        --label-pattern "dNch|dNdy" \\
        --output /tmp/dNch_comparison.pdf
"""

import argparse
import re
import numpy as np
import h5py

def _load_h5_groups(fpath):
    result = {}
    with h5py.File(fpath, 'r') as f:
        for gk in f.keys():
            g = f[gk]
            labs = [l.decode() if isinstance(l, bytes) else str(l) for l in g['labels'][:]]
            vals = g['values'][:]
            errs = g['errors'][:]
            exp_list = []
            for k in sorted(g.keys()):
                if k.startswith('exp_'):
                    eg = g[k]
                    exp_list.append({
                        'values': eg['values'][:],
                        'errors': eg['errors'][:],
                        'attrs': dict(eg.attrs),
                    })
            result[gk] = {
                'labels': labs,
                'values': vals,
                'errors': errs,
                'exp': exp_list,
            }
    return result


# ── Label parsing ──────────────────────────────────────────────────────

_CENT_RE = re.compile(r'(?P<lo>[\d.]+)\s*-\s*(?P<hi>[\d.]+)\s*%')
_PT_RE = re.compile(r'pT\s+(?P<lo>[\d.]+)\s*-\s*(?P<hi>[\d.]+)')

def _parse_cent(label):
    m = _CENT_RE.search(label)
    if m:
        return (float(m.group('lo')), float(m.group('hi')))
    if re.search(r'\bMB\b', label):
        return (0.0, 100.0)
    return None

def _parse_pT(label):
    m = _PT_RE.search(label)
    if m:
        return (float(m.group('lo')), float(m.group('hi')))
    return None

def _parse_label(label):
    info = {'type': '', 'pT': None, 'cent': None}
    parts = label.split()
    if not parts:
        return info
    info['type'] = parts[0]
    info['pT'] = _parse_pT(label)
    info['cent'] = _parse_cent(label)
    return info

def _cent_mid(info):
    if info['cent']:
        return (info['cent'][0] + info['cent'][1]) / 2
    return 50.0

def _cent_str(info):
    if info['cent']:
        if info['cent'][0] == 0.0 and info['cent'][1] == 100.0:
            return 'MB'
        return f"{info['cent'][0]:.0f}-{info['cent'][1]:.0f}%"
    return '?'

def _cent_str_label(info):
    if info['cent']:
        if info['cent'][0] == 0.0 and info['cent'][1] == 100.0:
            return 'MB'
        c1, c2 = info['cent']
        if c1 == int(c1) and c2 == int(c2):
            return f"{int(c1)}-{int(c2)}%"
        return f"{c1:.1f}-{c2:.1f}%"
    return '?'


def _match_observables(data_2d, data_3d, label_pattern):
    """Match observables by label pattern across 2D and 3D.

    Returns list of dicts with shared label, 2D/3D values+errors, exp data.
    """
    def _flatten(data, tag):
        rows = []
        for gk, gd in data.items():
            reaction = gk.rsplit('_', 1)[0]
            for i, lab in enumerate(gd['labels']):
                rows.append({
                    'tag': tag,
                    'group': gk,
                    'reaction': reaction,
                    'label': lab,
                    'value': gd['values'][i],
                    'error': gd['errors'][i],
                    'exp_list': gd['exp'],
                    'idx_in_group': i,
                })
        return rows

    all_2d = _flatten(data_2d, '2d')
    all_3d = _flatten(data_3d, '3d')

    matched = {}
    for r in all_2d:
        if re.search(label_pattern, r['label']):
            key = (r['reaction'], r['label'])
            if key not in matched:
                matched[key] = {'label': r['label'], 'reaction': r['reaction'], '2d': None, '3d': None, 'exp': None}
            matched[key]['2d'] = r

    for r in all_3d:
        if re.search(label_pattern, r['label']):
            key = (r['reaction'], r['label'])
            if key not in matched:
                matched[key] = {'label': r['label'], 'reaction': r['reaction'], '2d': None, '3d': None, 'exp': None}
            matched[key]['3d'] = r

    for m in matched.values():
        src = m['2d'] if m['2d'] else m['3d']
        if src and src['exp_list']:
            m['exp'] = src['exp_list']
        if not m['exp'] and m['3d'] and m['3d']['exp_list']:
            m['exp'] = m['3d']['exp_list']

    return list(matched.values())


# ── Plotting ───────────────────────────────────────────────────────────

def _draw_sim_band(ax, x_vals, sim_vals, sim_errs, color, marker, ls, label):
    """Draw sim values with error bars."""
    valid = [(x, v, e) for x, v, e in zip(x_vals, sim_vals, sim_errs) if v is not None and np.isfinite(v) and np.isfinite(e)]
    if not valid:
        return
    xv, vv, ev = zip(*valid)
    ax.errorbar(list(xv), list(vv), yerr=list(ev),
                fmt=marker + ls, color=color, label=label, capsize=3, alpha=0.8)

def _draw_exp(ax, x_vals, exp_list, x_err=None):
    """Draw experimental data from first matching exp entry."""
    if not exp_list:
        return
    for exp in exp_list:
        ev = exp['values']
        ee = exp['errors']
        if len(ev) == len(x_vals):
            ax.errorbar(x_vals, ev, yerr=ee, xerr=x_err,
                        fmt='D-', color='k', label='exp',
                        alpha=0.7, capsize=3, markersize=4)
            break


def plot_observables(matched_list, output_path, plot_title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Classify
    integrated = []
    diff_pT_cent = []
    diff_pT_only = []  # pT-diff without cent (%)
    yields_vs_cent = []  # yield/spectra vs cent (no pT in label)

    for m in matched_list:
        info = _parse_label(m['label'])
        has_cent = info['cent'] is not None
        has_pT = info['pT'] is not None
        is_yield = info['type'].startswith('yield_') or info['type'].startswith('spectra_')

        if is_yield and has_cent and not has_pT:
            yields_vs_cent.append(m)
        elif has_pT and has_cent:
            diff_pT_cent.append(m)
        elif has_pT and not has_cent:
            diff_pT_only.append(m)
        elif has_cent:
            integrated.append(m)

    n_panels = sum([len(integrated) > 0, len(diff_pT_cent) > 0,
                    len(diff_pT_only) > 0, len(yields_vs_cent) > 0])
    if n_panels == 0:
        print("No matching observables found")
        if matched_list:
            print("  Sample labels:")
            for m in matched_list[:5]:
                print(f"    {m['label']}")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)
    ax_idx = 0

    # ── Integrated vs centrality ──────────────────────────────────
    if integrated:
        ax = axes[0][ax_idx]
        by_rxn_type = defaultdict(list)
        for m in integrated:
            info = _parse_label(m['label'])
            by_rxn_type[(m['reaction'], info['type'])].append(m)

        for (rxn, otype), obs_list in sorted(by_rxn_type.items()):
            obs_list = [m for m in obs_list if _parse_cent(m['label']) is not None]
            if not obs_list:
                continue
            obs_list.sort(key=lambda x: _cent_mid(_parse_label(x['label'])))
            info0 = _parse_label(obs_list[0]['label'])
            cents = [info0['cent']]
            for m in obs_list[1:]:
                cents.append(_parse_label(m['label'])['cent'])

            x_cent = [(c[0] + c[1]) / 2 for c in cents]
            x_err = [(c[1] - c[0]) / 2 for c in cents]

            for tag, color, marker, ls in [('2d', '#1f77b4', 'o', '-'), ('3d', '#ff7f0e', 's', '--')]:
                vals = [m[tag]['value'] if m[tag] else None for m in obs_list]
                errs = [m[tag]['error'] if m[tag] else None for m in obs_list]
                valid_pts = [(x, v, e) for x, v, e in zip(x_cent, vals, errs) if v is not None and np.isfinite(v) and np.isfinite(e)]
                if valid_pts:
                    xv, vv, ev = zip(*valid_pts)
                    ax.errorbar(list(xv), list(vv), yerr=list(ev),
                                fmt=marker + ls, color=color, label=f'{tag} sim',
                                capsize=3)

            _draw_exp(ax, x_cent, obs_list[0]['exp'], x_err=x_err)

            ax.set_xlabel('Centrality [%]')
            ax.set_ylabel(otype)
            ax.set_title(f'{otype} vs centrality' + (f'\n{rxn[:50]}' if rxn else ''))
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    # ── Differential vs pT (fixed centrality) ────────────────────
    if diff_pT_cent:
        ax = axes[0][ax_idx] if ax_idx < n_panels else axes[0][0]
        by_cent = defaultdict(list)
        for m in diff_pT_cent:
            info = _parse_label(m['label'])
            cent_label = _cent_str_label(info)
            by_cent[(m['reaction'], cent_label, info['type'])].append(m)

        for (rxn, cstr, otype), obs_list in sorted(by_cent.items()):
            obs_list.sort(key=lambda x: _parse_pT(x['label'])[0] if _parse_pT(x['label']) else 0)
            pT_vals = []
            for m in obs_list:
                pt = _parse_pT(m['label'])
                if pt:
                    pT_vals.append((pt[0] + pt[1]) / 2)
                else:
                    pT_vals.append(None)

            for tag, color, marker, ls in [('2d', '#1f77b4', 'o', '-'), ('3d', '#ff7f0e', 's', '--')]:
                vals = [m[tag]['value'] if m[tag] else None for m in obs_list]
                errs = [m[tag]['error'] if m[tag] else None for m in obs_list]
                valid = [(x, v, e) for x, v, e in zip(pT_vals, vals, errs) if x is not None and v is not None and np.isfinite(v) and e is not None and np.isfinite(e)]
                if valid:
                    xv, vv, ev = zip(*valid)
                    ax.errorbar(list(xv), list(vv), yerr=list(ev),
                                fmt=marker + ls, color=color, label=f'{tag} sim',
                                capsize=3, alpha=0.8)

            _draw_exp(ax, [x for x in pT_vals if x is not None], obs_list[0]['exp'])

            ax.set_xlabel(r'$p_T$ [GeV]')
            ax.set_ylabel(otype)
            ax.set_title(f'{otype} vs $p_T$, {cstr}' + (f'\n{rxn[:40]}' if rxn else ''))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    # ── pT-differential without cent ───────────────────────────
    if diff_pT_only:
        ax = axes[0][ax_idx] if ax_idx < n_panels else axes[0][0]
        by_rxn = defaultdict(list)
        for m in diff_pT_only:
            by_rxn[m['reaction']].append(m)

        for rxn, obs_list in sorted(by_rxn.items()):
            obs_list.sort(key=lambda x: _parse_pT(x['label'])[0] if _parse_pT(x['label']) else 0)
            pT_vals = []
            for m in obs_list:
                pt = _parse_pT(m['label'])
                pT_vals.append((pt[0] + pt[1]) / 2 if pt else None)

            for tag, color, marker, ls in [('2d', '#1f77b4', 'o', '-'), ('3d', '#ff7f0e', 's', '--')]:
                vals = [m[tag]['value'] if m[tag] else None for m in obs_list]
                errs = [m[tag]['error'] if m[tag] else None for m in obs_list]
                valid = [(x, v, e) for x, v, e in zip(pT_vals, vals, errs) if x is not None and v is not None and np.isfinite(v) and e is not None and np.isfinite(e)]
                if valid:
                    xv, vv, ev = zip(*valid)
                    ax.errorbar(list(xv), list(vv), yerr=list(ev),
                                fmt=marker + ls, color=color, label=f'{tag} sim',
                                capsize=3, alpha=0.8)

            _draw_exp(ax, [x for x in pT_vals if x is not None], obs_list[0]['exp'])

            otype = _parse_label(obs_list[0]['label'])['type']
            ax.set_xlabel(r'$p_T$ [GeV]')
            ax.set_ylabel(otype)
            ax.set_title(f'{otype} vs $p_T$' + (f'\n{rxn[:40]}' if rxn else ''))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    # ── Yields vs centrality (no pT in label) ─────────────────
    if yields_vs_cent:
        ax = axes[0][ax_idx] if ax_idx < n_panels else axes[0][0]
        by_rxn_type = defaultdict(list)
        for m in yields_vs_cent:
            info = _parse_label(m['label'])
            by_rxn_type[(m['reaction'], info['type'])].append(m)

        for (rxn, otype), obs_list in sorted(by_rxn_type.items()):
            obs_list.sort(key=lambda x: _cent_mid(_parse_label(x['label'])))
            cents = []
            for m in obs_list:
                ci = _parse_label(m['label'])['cent']
                if ci:
                    cents.append(ci)
            if not cents:
                continue
            x_cent = [(c[0] + c[1]) / 2 for c in cents]
            x_err = [(c[1] - c[0]) / 2 for c in cents]
            cent_labels = [_cent_str_label({'cent': c}) for c in cents]

            for tag, color, marker, ls in [('2d', '#1f77b4', 'o', '-'), ('3d', '#ff7f0e', 's', '--')]:
                vals = [m[tag]['value'] if m[tag] else None for m in obs_list]
                errs = [m[tag]['error'] if m[tag] else None for m in obs_list]
                valid = [(x, v, e) for x, v, e in zip(x_cent, vals, errs) if v is not None and np.isfinite(v) and np.isfinite(e)]
                if valid:
                    xv, vv, ev = zip(*valid)
                    ax.errorbar(list(xv), list(vv), yerr=list(ev),
                                fmt=marker + ls, color=color, label=f'{tag} sim',
                                capsize=3)

            _draw_exp(ax, x_cent, obs_list[0]['exp'], x_err=x_err)

            ax.set_xlabel('Centrality [%]')
            ax.set_ylabel(r'dN/d\eta  or  dN/dy')
            ax.set_title(f'{otype} vs centrality' + (f'\n{rxn[:50]}' if rxn else ''))
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    if plot_title:
        fig.suptitle(plot_title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot sim vs experiment from pipeline HDF5")
    ap.add_argument("--sim-2d", default="/tmp/auau200_2d_flow_vn.h5", help="2D sim HDF5")
    ap.add_argument("--sim-3d", default=None, help="3D sim HDF5 (optional)")
    ap.add_argument("--label-pattern", default="vn2_single", help="Regex for label matching")
    ap.add_argument("--output", default="/tmp/comparison.pdf", help="Output plot file")
    ap.add_argument("--title", default=None, help="Optional plot title")
    args = ap.parse_args()

    d2 = _load_h5_groups(args.sim_2d)
    d3 = _load_h5_groups(args.sim_3d) if args.sim_3d else {}
    matched = _match_observables(d2, d3, args.label_pattern)

    print(f"Matched {len(matched)} observables with pattern '{args.label_pattern}'")
    n_2d = sum(1 for m in matched if m['2d'] is not None)
    n_3d = sum(1 for m in matched if m['3d'] is not None)
    print(f"  {n_2d} have 2D data, {n_3d} have 3D data")

    if not matched:
        print("No matching observables found — try a different --label-pattern")
        return

    plot_observables(matched, args.output, args.title)


if __name__ == "__main__":
    main()
