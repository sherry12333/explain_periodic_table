# Repair and validation plan

## Scope of this review

Static review of the four downloaded files from https://github.com/sherry12333/explain_periodic_table on 2026-10-01. No complete MATLAB/Python numerical calculation has been executed. Syntax and notebook-structure checks are distinct from numerical validation.

## Known blockers

1. `matlab/explain_periodic_table.m`: `N40=0` is a scalar, but the ion call indexes `N40(1,2)`. `N32=[1,0]` is passed as a whole array. Determine the intended configuration from the coursework/report instead of guessing a fix. `Z=19` also has an inconsistent `He` comment.
2. `python/modules.py`: `eig`, `csc_matrix`, and `splu` are referenced without their imports. Adding imports alone is insufficient.
3. `cal_eigen` / `cal_eigen1` return eigenvalue vectors, whereas `cal_ryo_E` reads diagonal entries as if they were matrices. Establish one representation, explicit eigenpair ordering, and overlap-matrix normalisation.
4. `bsplgen`: the function has no explicit return statement; portions of basis output/derivative handling are outside the point loop. Repair by comparison with the mathematics and original MATLAB routine, including repeated knots and endpoints.
5. `l_u`: the manual sparse-factor solve needs to account correctly for permutations. Validate on a matrix requiring pivoting, with the residual `A @ x - b`, before using it inside the solver.
6. The historical notebook gives neutral and ion calculations identical inputs, passes an occupation array where a scalar is expected, and indexes the density as a 2-D array. Establish a documented scalar occupation format and density shape first.
7. The radial grid includes zero, while density expressions divide by radius. Handle the origin from the mathematical boundary conditions; do not simply hide numerical warnings.
8. The self-consistent loop uses a fixed iteration count. Before reporting converged energies, define a convergence criterion and expose failure to converge.

## Small, ordered milestones

### A. Establish the reference

- Retrieve one original MATLAB input/result that was known to run successfully. A simple helium case is a candidate only if it exists in the coursework records.
- Record nuclear charge, electron occupations, units, radial grid, spline order, quadrature, mixing parameter, stopping criterion and outputs.
- Distinguish fixes to the uploaded example from changes to the physical method.

### B. Verify numerical building blocks

- B-splines: shape, partition of unity in the applicable interior, boundary handling and derivative consistency.
- Quadrature: known polynomial integrals with predefined tolerances.
- Linear systems: residual check including pivoting.
- Generalised eigenproblem: known small system, residual and overlap normalisation; keep eigenvalues and eigenvectors paired when sorting.

### C. Reproduce one atom

- Compare matching MATLAB/Python inputs and outputs.
- Verify integrated electron number, finite outputs and iteration convergence.
- Record absolute/relative differences and explain tolerances; agreement between implementations alone does not establish physical accuracy.

### D. Expand and publish results

- Add neutral/ion cases with independently specified occupations.
- Compare total energies and ionisation energy using consistent units and sign convention.
- Add grid/basis convergence evidence and compare with an appropriate documented reference, accounting for the approximation used.
- Save environment versions, commands and result tables. Update README status only after evidence exists.

## Needed from the owner

The original successful MATLAB configuration and output, if available, and the coursework model description. No thesis code is needed. If those records are unavailable, reconstruct and independently validate a small reference problem before claiming reproduction of the coursework results.
