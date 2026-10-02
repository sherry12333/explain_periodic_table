# Verification report

Completed 2026-10-02. Tests and CLI calculations were run locally; no MATLAB parity or experimental-accuracy claim.

## Environment

Python 3.11.5; NumPy 1.26.2; SciPy 1.14.0.

## Automated tests

11 unittest tests passed, including all six neutral/ion examples. Coverage includes spline derivatives/endpoints, quadrature, hydrogen levels, Poisson uniform-sphere analytic solution, exchange energy derivative, invalid inputs, explicit nonconvergence failure and helium mesh stability.

## Nuclear-only hydrogen convergence

| Intervals | Ground energy / Ha | Absolute error against -0.5 Ha |
| --- | --- | --- |
| 24 | -0.499999874788 | 1.252e-07 |
| 48 | -0.499999998199 | 1.801e-09 |
| 96 | -0.499999999969 | 3.070e-11 |

## Atomic examples: default alpha=1

Default grid: 100 intervals, R=35 Bohr, cubic splines, 8 Gaussian points per interval. Ionisation energy = E(cation)-E(neutral). These numbers belong to the specified approximate functional, not measured atom energies.

| Atom | Neutral E / Ha | Cation E / Ha | Model ionisation / Ha | Neutral / ion SCF iterations |
| --- | --- | --- | --- | --- |
| He | -3.170112244 | -2.013386636 | 1.156725608 | 43 / 41 |
| Ne | -133.066783589 | -132.018476592 | 1.048306997 | 53 / 50 |
| K | -611.915551639 | -611.736285681 | 0.179265958 | 58 / 52 |

## Discretisation sensitivity

Fine grid: 140 intervals. Larger-box check: R=50 with 140 intervals versus R=35 with 140 intervals (both box extent and physical knot spacing change). Quadrature check: 12 versus 8 points per interval at 100 intervals.

| Atom | Neutral grid ΔE / Ha | Ion grid ΔE / Ha | Ionisation grid Δ / Ha | Ionisation box Δ / Ha | Ionisation quadrature Δ / Ha |
| --- | --- | --- | --- | --- | --- |
| He | 1.309e-10 | 1.238e-10 | 7.081e-12 | 3.189e-12 | 4.130e-14 |
| Ne | 5.782e-07 | 5.797e-07 | 1.526e-09 | 4.370e-10 | 2.274e-13 |
| K | 5.113e-05 | 5.113e-05 | 5.526e-09 | 3.482e-09 | 4.547e-13 |

All checked changes passed the bounds saved in results.json (2e-3 Ha for individual total-energy grid changes, 2e-4 Ha for ionisation changes). These thresholds assess numerical stability only.

## Remaining limitations

- No original MATLAB calculation or course report has been reproduced.
- Only the stated atoms, occupations and numerical configurations were exercised.
- Fixed occupations, spherical averaging, no spin polarisation, no correlation and no relativistic effects.
- Default alpha=1 differs from standard Dirac exchange; systematic physical errors can remain even when numerical residuals are small.
- Clean-environment dependency installation and other operating systems were not tested.
- Passing these tests is not a guarantee that no bugs remain.

See METHOD.md for intentional changes and results.json for full machine-readable evidence.

## Notebook check

All four code cells executed in order in a fresh Python process and the notebook passed schema validation. Saved outputs are from that run. A Jupyter kernel could not launch in this sandbox because local socket binding is disallowed, so interactive Jupyter execution was not verified here.
