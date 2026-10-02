"""Spherical, spin-unpolarised atomic X-alpha solver in atomic units.

Numerical reimplementation of the coursework approach, NOT a drop-in MATLAB
translation. Fixed occupations; exchange only; no correlation or relativity.
alpha=1 reproduces the exchange-potential coefficient in the uploaded MATLAB
source; alpha=2/3 gives standard spin-unpolarised Dirac exchange.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh, cho_factor, cho_solve


class ConvergenceError(RuntimeError):
    """The requested SCF tolerance was not reached; no success is reported."""


def _positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def bsplgen(points, knots, order):
    """Return basis values and first/second derivatives; order = degree + 1."""
    _integer(order, "order", 3)
    t = np.asarray(knots, dtype=float)
    x = np.atleast_1d(np.asarray(points, dtype=float))
    if t.ndim != 1 or len(t) < 2 * order or not np.all(np.isfinite(t)):
        raise ValueError("invalid knot vector")
    if np.any(np.diff(t) < 0) or t[order - 1] >= t[-order]:
        raise ValueError("knots must be nondecreasing with a nonempty domain")
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("points must be a finite 1-D array")
    if np.any(x < t[order - 1]) or np.any(x > t[-order]):
        raise ValueError("points lie outside the spline domain")
    spline = BSpline(t, np.eye(len(t) - order), order - 1, extrapolate=False)
    return spline(x), spline(x, nu=1), spline(x, nu=2)


@dataclass
class Result:
    Z: int
    occupations: dict
    alpha: float
    total_energy: float
    electron_count: float
    iterations: int
    density_residual: float
    energy_change: float
    eigen_residual: float
    orbital_energies: dict
    radius: np.ndarray
    weights: np.ndarray
    radial_density: np.ndarray
    density: np.ndarray
    history: list

    def summary(self):
        return {"Z": self.Z, "occupations": {f"{n}{'spdf'[l]}": v
                for (n, l), v in self.occupations.items()}, "alpha": self.alpha,
                "total_energy_hartree": self.total_energy,
                "electron_count": self.electron_count, "iterations": self.iterations,
                "density_residual": self.density_residual,
                "energy_change_hartree": self.energy_change,
                "relative_eigen_residual": self.eigen_residual,
                "orbital_energies_hartree": self.orbital_energies,
                "converged": True}


class AtomicSolver:
    """Finite-box radial B-spline Galerkin discretisation with Gaussian quadrature.

    Orbitals P=rR obey P(0)=P(rmax)=0. Hartree potential u=r*VH obeys
    u(0)=0, u(rmax)=N. The origin is excluded from quadrature nodes.
    """
    def __init__(self, Z, *, rmax=35.0, intervals=100, order=4,
                 quadrature=8, stretch=5.0, alpha=1.0):
        _integer(Z, "Z")
        _integer(intervals, "intervals", 8)
        _integer(order, "order", 3)
        _integer(quadrature, "quadrature", order)
        _positive(rmax, "rmax")
        _positive(stretch, "stretch")
        if stretch > 20:
            raise ValueError("stretch > 20 creates numerically degenerate knots")
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.Z, self.alpha, self.rmax, self.order = int(Z), float(alpha), float(rmax), order
        grid = rmax * np.expm1(stretch * np.linspace(0, 1, intervals + 1)) / np.expm1(stretch)
        self.knots = np.r_[np.repeat(0.0, order - 1), grid,
                           np.repeat(rmax, order - 1)]
        g, w = np.polynomial.legendre.leggauss(quadrature)
        widths = np.diff(grid)
        self.r = (grid[:-1, None] + widths[:, None] * (g + 1) / 2).ravel()
        self.w = (widths[:, None] * w / 2).ravel()
        full, dfull, _ = bsplgen(self.r, self.knots, order)
        self.full = full
        self.B, self.dB = full[:, 1:-1], dfull[:, 1:-1]
        self.S = self.B.T @ (self.w[:, None] * self.B)
        self.T = 0.5 * self.dB.T @ (self.w[:, None] * self.dB)
        K = dfull.T @ (self.w[:, None] * dfull)
        self._poisson = cho_factor(K[1:-1, 1:-1])
        self._boundary = K[1:-1, -1]
        self._core = {}

    def matrix(self, potential):
        v = np.asarray(potential, dtype=float)
        if v.shape != self.r.shape or not np.all(np.isfinite(v)):
            raise ValueError("potential must be finite and match the quadrature grid")
        return self.B.T @ ((self.w * v)[:, None] * self.B)

    def core(self, l):
        _integer(l, "l", 0)
        if l not in self._core:
            self._core[l] = self.T + self.matrix(l * (l + 1) / (2 * self.r**2) - self.Z / self.r)
        return self._core[l]

    def solve_channel(self, l, potential, count):
        _integer(count, "count")
        if count > self.S.shape[0]:
            raise ValueError("more orbitals requested than basis functions")
        H = self.core(l) + self.matrix(potential)
        energy, coefficients = eigh(H, self.S, subset_by_index=(0, count - 1))
        residual = H @ coefficients - (self.S @ coefficients) * energy
        relative = np.linalg.norm(residual) / max(1.0, np.linalg.norm(H @ coefficients))
        return energy, coefficients, float(relative)

    def hydrogenic(self, l=0, count=3):
        """Nuclear-only benchmark; no Hartree or exchange contribution."""
        return self.solve_channel(l, np.zeros_like(self.r), count)

    def hartree(self, radial_density):
        """Solve -u'' = q/r, q=4*pi*r^2*rho, with finite-box Coulomb boundary."""
        q = np.asarray(radial_density, dtype=float)
        if q.shape != self.r.shape or not np.all(np.isfinite(q)) or np.any(q < 0):
            raise ValueError("radial density must be finite, nonnegative and match grid")
        electrons = float(self.w @ q)
        rhs = self.B.T @ (self.w * q / self.r) - self._boundary * electrons
        coeff = np.r_[0.0, cho_solve(self._poisson, rhs), electrons]
        return (self.full @ coeff) / self.r

    def exchange(self, radial_density):
        """X-alpha potential; exchange energy is (3/4) integral rho*vx d^3r."""
        q = np.asarray(radial_density, dtype=float)
        if q.shape != self.r.shape or not np.all(np.isfinite(q)) or np.any(q < 0):
            raise ValueError("invalid radial density")
        rho = q / (4 * np.pi * self.r**2)
        vx = -1.5 * self.alpha * np.cbrt(3 * rho / np.pi)
        return vx, float(0.75 * self.w @ (q * vx))

    def _occupations(self, occupations):
        if not isinstance(occupations, dict) or not occupations:
            raise ValueError("occupations must be a nonempty {(n,l): count} dictionary")
        clean = {}
        for key, value in occupations.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("orbital keys must be (n,l) tuples")
            n, l = key
            _integer(n, "n")
            _integer(l, "l", 0)
            if l >= n or l > 3:
                raise ValueError("require 0 <= l < n and l <= 3")
            _positive(value, "occupation")
            if value > 2 * (2 * l + 1):
                raise ValueError("occupation exceeds subshell capacity")
            clean[(int(n), int(l))] = float(value)
        return clean

    def _orbitals(self, occupations, potential):
        q = np.zeros_like(self.r)
        one_body = 0.0
        levels = {}
        residual = 0.0
        for l in sorted({l for n, l in occupations}):
            count = max(n - l for n, ll in occupations if ll == l)
            energies, coefficients, err = self.solve_channel(l, potential, count)
            residual = max(residual, err)
            for (n, ll), occupation in occupations.items():
                if ll != l:
                    continue
                c = coefficients[:, n - l - 1]
                orbital = self.B @ c
                q += occupation * orbital**2
                one_body += occupation * float(c @ self.core(l) @ c)
                levels[f"{n}{'spdf'[l]}"] = float(energies[n - l - 1])
        return q, one_body, levels, residual

    def solve(self, occupations, *, mixing=0.3, tolerance=1e-8,
              energy_tolerance=1e-9, max_iterations=400):
        """Converge density and total energy, or raise ConvergenceError.

        mixing is the fraction of NEW density. Fixed subshell occupations are
        supplied by the caller; this is not an automatic ground-state search.
        """
        occ = self._occupations(occupations)
        if not np.isscalar(mixing) or not np.isfinite(mixing) or not 0 < mixing <= 1:
            raise ValueError("mixing must be in (0,1]")
        _positive(tolerance, "tolerance")
        _positive(energy_tolerance, "energy_tolerance")
        _integer(max_iterations, "max_iterations")
        expected = sum(occ.values())
        q, _, _, _ = self._orbitals(occ, np.zeros_like(self.r))
        previous_energy = None
        history = []
        for iteration in range(1, max_iterations + 1):
            potential = self.hartree(q) + self.exchange(q)[0]
            new_q, one_body, levels, eigen_residual = self._orbitals(occ, potential)
            vh = self.hartree(new_q)
            energy = float(one_body + 0.5 * self.w @ (new_q * vh) + self.exchange(new_q)[1])
            residual = float(self.w @ np.abs(new_q - q) / expected)
            change = float(abs(energy - previous_energy)) if previous_energy is not None else None
            if not np.isfinite(energy) or not np.isfinite(residual):
                raise FloatingPointError("non-finite SCF result")
            history.append({"iteration": iteration, "energy_hartree": energy,
                            "density_residual": residual, "energy_change": change})
            if change is not None and residual < tolerance and change < energy_tolerance:
                count = float(self.w @ new_q)
                if abs(count - expected) > 1e-7 * max(1, expected):
                    raise ArithmeticError("electron normalisation failed")
                return Result(self.Z, occ, self.alpha, energy, count, iteration,
                              residual, change, eigen_residual, levels,
                              self.r.copy(), self.w.copy(), new_q,
                              new_q / (4 * np.pi * self.r**2), history)
            q = (1 - mixing) * q + mixing * new_q
            previous_energy = energy
        raise ConvergenceError(f"SCF did not converge after {max_iterations} iterations; "
                               f"density residual={residual:.3g}, energy change={change}")


CONFIGURATIONS = {
    "He": (2, {(1, 0): 2}),
    "He+": (2, {(1, 0): 1}),
    "Ne": (10, {(1, 0): 2, (2, 0): 2, (2, 1): 6}),
    "Ne+": (10, {(1, 0): 2, (2, 0): 2, (2, 1): 5}),
    "K": (19, {(1, 0): 2, (2, 0): 2, (2, 1): 6, (3, 0): 2, (3, 1): 6, (4, 0): 1}),
    "K+": (19, {(1, 0): 2, (2, 0): 2, (2, 1): 6, (3, 0): 2, (3, 1): 6}),
}
