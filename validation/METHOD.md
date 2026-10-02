# Numerical method and verification boundary

With P_nl=r R_nl and integral(P_nl^2 dr)=1, define
q(r)=sum(f_nl P_nl^2), rho=q/(4*pi*r^2). The radial operator is
H_l=-1/2 d²/dr²+l(l+1)/(2r²)-Z/r+VH+vx.

Clamped cubic B-splines (order=4, degree=3) span an exponentially stretched finite
radial grid. The first and last orbital basis functions are excluded to enforce
zero orbital boundary values. Interval-by-interval Gauss-Legendre quadrature
assembles the overlap and weak-form kinetic matrices. The routine `bsplgen`
evaluates the basis and derivatives through SciPy BSpline; the handwritten draft
recursion is not retained. `eigh(H,S)` sorts eigenvalues and normalises in S.

Poisson is solved as -u''=q/r, u=r*VH, u(0)=0, u(R)=N, with the same spline basis
and weak derivative matrix. Boundary coefficients are imposed explicitly, and
the symmetric positive-definite interior system is factored once. The external
Coulomb potential fixes the finite-box boundary. The matrix is solved by SciPy
Cholesky routines rather than manually manipulating permuted LU factors.

Exchange is vx=-(3/2)*alpha*(3*rho/pi)^(1/3), Ex=(3/4)*integral(q*vx dr).
The default alpha=1 reproduces the coefficient in the original uploaded code.
This is stronger than Dirac exchange (alpha=2/3); a helium energy below the exact
interacting ground-state energy in this approximation is not evidence of an
exact physical prediction or a variational bound to the real many-body system.
The energy and potential here are functional derivatives of the same model.

SCF starts from nuclear-only orbitals and mixes the new density with the previous
density. Successful return requires both weighted L1 density residual per electron
<1e-8 and absolute total-energy change <1e-9 Hartree. Eigen residual is reported
for orbitals solved in the last input potential; at convergence the returned
density agrees with that input density to the stated tolerance. No convergence
guarantee is made for arbitrary atoms/configurations/mixing values.

Independent tests use nuclear-only analytic hydrogen levels (not a one-electron
X-alpha atom), the analytic electrostatic potential of a uniformly charged sphere,
polynomial quadrature and finite-difference derivative checks. Grid/box/quadrature
studies establish selected numerical stability, not exchange-functional accuracy.

Sources for implementation APIs:

- [SciPy BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
- [SciPy eigh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html)

The X-alpha coefficient comparison is derived directly from the original MATLAB
`cal_integral1` expression. No external experimental benchmark is claimed.
