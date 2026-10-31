"""Transcribe STAR arXiv:0808.2041 Table VII (AuAu 200 GeV mean-pT section).

Upstream HEPData ins793126 (v1, 68 tables) does NOT contain the
mean-pT-vs-centrality table, but curated records
(star_mean_pT_systematic_measurements..., TableVII.yaml) reference it.
This script writes a HEPData-shaped TableVII.yaml from the paper source
(paper_v31h_final_eprint.tex tab:meanpt) so load_experimental works.

Values: extrapolated <pT> in GeV/c, errors = quoted quadratic sums.
Six dependent variables (pi+, pi-, K+, K-, p, pbar) matching the
PHENIX fig3-1.yaml layout; curated docs average charge pairs via
dv_average, mirroring the JETSCAPE calibration usage.
"""
import yaml

# (cent, pi-, pi+, K-, K+, pbar, p) values and symmetric errors,
# AuAu 200 GeV section of Table VII, arXiv:0808.2041.
ROWS = [
    # cent,        pi-              pi+              K-               K+               pbar             p
    (('0-5%'),     (0.422, 0.022), (0.427, 0.022), (0.719, 0.074), (0.720, 0.074), (1.103, 0.114), (1.104, 0.110)),
    (('5-10%'),    (0.418, 0.021), (0.422, 0.021), (0.704, 0.064), (0.703, 0.064), (1.070, 0.098), (1.071, 0.098)),
    (('10-20%'),   (0.416, 0.021), (0.421, 0.021), (0.680, 0.055), (0.681, 0.055), (1.017, 0.082), (1.017, 0.082)),
    (('20-30%'),   (0.408, 0.021), (0.411, 0.021), (0.668, 0.047), (0.668, 0.047), (0.989, 0.071), (0.989, 0.071)),
    (('30-40%'),   (0.402, 0.021), (0.404, 0.021), (0.643, 0.042), (0.643, 0.042), (0.939, 0.062), (0.939, 0.062)),
    (('40-50%'),   (0.395, 0.020), (0.395, 0.020), (0.619, 0.037), (0.619, 0.037), (0.897, 0.055), (0.897, 0.055)),
    (('50-60%'),   (0.389, 0.020), (0.389, 0.020), (0.609, 0.036), (0.608, 0.036), (0.863, 0.052), (0.864, 0.052)),
    (('60-70%'),   (0.377, 0.019), (0.377, 0.019), (0.583, 0.033), (0.583, 0.033), (0.814, 0.047), (0.817, 0.047)),
    (('70-80%'),   (0.363, 0.018), (0.367, 0.018), (0.550, 0.035), (0.553, 0.035), (0.746, 0.049), (0.749, 0.049)),
]
NAMES = ['pi+', 'pi-', 'K+', 'K-', 'p', 'pbar']
# table columns are (pi-, pi+, K-, K+, pbar, p); map to NAMES order
COL = [1, 0, 3, 2, 5, 4]

OUT = ('hepdata_raw/records/ins793126/HEPData-ins793126-v1-yaml/'
       'TableVII.yaml')

HEADER = ('# Transcribed from Table "tab:meanpt" (AuAu 200 GeV section) of '
          'arXiv:0808.2041 by the soft-sector-data-curation project. '
          'Upstream HEPData ins793126 v1 (68 tables) does not contain this '
          'table; curated records reference TableVII.yaml. Values are '
          'extrapolated <pT> in GeV/c; errors are the quoted quadratic sums.\n')


def main():
    indep = {'header': {'name': 'Centrality', 'units': '%'},
             'values': [{'value': c} for c, *_ in ROWS]}
    dvs = []
    for j, name in enumerate(NAMES):
        vals = []
        for row in ROWS:
            v, e = row[1 + COL[j]]
            vals.append({'value': v, 'errors': [{'symerror': e}]})
        dvs.append({'header': {'name': f'Mean pT {name} GeV/c'},
                    'qualifiers': [{'name': 'particle type', 'value': name}],
                    'values': vals})
    doc = {'dependent_variables': dvs, 'independent_variables': [indep]}
    with open(OUT, 'w') as f:
        f.write(HEADER)
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
    print(f"wrote {OUT} ({len(ROWS)} cent bins x {len(NAMES)} series)", flush=True)


if __name__ == '__main__':
    main()
