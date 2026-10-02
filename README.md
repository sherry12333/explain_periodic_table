# Atomic Electronic-Structure Calculations

A MATLAB coursework project with a tested Python numerical reimplementation.
The Python version uses radial B-splines, Gaussian quadrature and self-consistent
atomic X-alpha calculations. **It is an educational, exchange-only model, not a
general-purpose or experimentally validated DFT package.**

## Two implementations, one project

| Directory | Role |
| --- | --- |
| `matlab/` | Original uploaded coursework code, unchanged; not a verified runnable benchmark |
| `python/` | New solver, command-line example and notebook |
| `tests/` | Analytic, input-validation and end-to-end tests |
| `validation/` | Reproducible mesh/box/quadrature study and measured results |
| `archive/` | Historical, unvalidated AI-assisted translation; not used by the solver |

This version replaces the earlier Python draft; it does not preserve its function
API. The original MATLAB work and the subsequent Python reimplementation should
be described separately in a portfolio. The new implementation and tests were
prepared with AI assistance. Master's thesis code is not included.

## Quick start

Python 3.10 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python python/run_atom.py He --ion
python python/run_atom.py Ne --ion
python python/run_atom.py K --ion
python -m unittest discover -s tests -v
python validation/run_validation.py
python -m jupyter lab python/periodic_table.ipynb
```

On Windows, activate with `.venv\Scripts\activate`. Only NumPy and SciPy are
needed for the CLI and tests; Matplotlib/JupyterLab are for the notebook.
The CLI prints energies in Hartree, occupations, electron count, convergence
residuals and iteration count. `--output file.json` saves the same report.
An unconverged calculation raises an error rather than silently reporting success.

## Model and deliberate changes

- Atomic units: lengths in Bohr, energies in Hartree.
- Spherical density and spin-unpolarised exchange; fixed subshell occupations.
- No correlation, spin polarisation, relativistic corrections or automatic
  ground-state configuration search. Open-shell ions and K are spherical-average
  approximations, not spin-resolved ground-state calculations.
- `alpha=1` matches the exchange-potential coefficient in the uploaded MATLAB
  expression. `--alpha 0.6666666666666666` selects the standard spin-unpolarised
  Dirac exchange coefficient. These choices have different energies.
- A variational B-spline Poisson solve replaces the fragile collocation/handwritten
  LU path. SciPy's symmetric generalised eigensolver supplies sorted, overlap-
  normalised eigenvectors. Gaussian nodes avoid division by zero at the origin.
- Total energy is evaluated as `sum(occupation * <P|H0|P>) + EH + Ex`, where
  `EH = 1/2 integral(q*VH dr)` and `Ex = 3/4 integral(q*vx dr)`.
  The old code subtracted half of the combined Hartree and exchange potential
  from the occupied eigenvalue sum; that exchange correction is inconsistent
  with this functional. Thus new total energies need not match the old formula.
- K uses `...3p6 4s1`; K+ uses `...3p6`. The ambiguous old example's occupation
  array is not treated as a validated configuration.

See [method details](validation/METHOD.md) and [verification report](validation/REPORT.md).
Tests establish the checks described there, not absence of all possible bugs.
No successful original MATLAB reference run has been supplied or reproduced.

## Python use

When working inside the `python/` directory:

```python
from modules import AtomicSolver
solver = AtomicSolver(Z=2)
neutral = solver.solve({(1, 0): 2})  # n=1, l=0, two electrons
ion = solver.solve({(1, 0): 1})
print(ion.total_energy - neutral.total_energy)
```

`result.radius` contains positive quadrature nodes, `result.density` is rho(r),
and `result.radial_density` is q(r)=4*pi*r^2*rho(r). Integrate q using
`result.weights @ result.radial_density` to obtain electron number.

## Provenance

Original project: https://github.com/sherry12333/explain_periodic_table .
Original files are retained for transparency; author/contributor permissions
should be resolved before assigning a repository-wide licence.
No new licence is assigned by this reorganisation.

## 中文摘要

这次是真正重写并测试了 Python 数值核心，不只是整理目录。提供 He、Ne、K
及一价阳离子的运行入口、单元测试和结果报告。它是球对称、不区分自旋、仅含
交换的教学模型；未声称与原 MATLAB 数值逐项一致，也不声称能准确预测实验电离能。
