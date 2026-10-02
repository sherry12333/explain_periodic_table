"""Independent analytic checks and end-to-end regression tests; stdlib unittest."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from modules import AtomicSolver, CONFIGURATIONS, ConvergenceError, bsplgen


class NumericalTests(unittest.TestCase):
    def test_spline_partition_derivatives_and_endpoints(self):
        t = np.r_[np.zeros(3), np.linspace(0, 4, 12), np.repeat(4., 3)]
        x = np.linspace(0, 4, 103)
        b, d, dd = bsplgen(x, t, 4)
        np.testing.assert_allclose(b.sum(axis=1), 1, atol=1e-13)
        np.testing.assert_allclose(d.sum(axis=1), 0, atol=1e-12)
        np.testing.assert_allclose(dd.sum(axis=1), 0, atol=1e-11)
        self.assertEqual(b[0, 0], 1)
        self.assertEqual(b[-1, -1], 1)
        x = np.array([0.25, 1.35, 3.75]); h = 1e-5
        b, d, dd = bsplgen(x, t, 4)
        bp, dp, _ = bsplgen(x + h, t, 4)
        bm, dm, _ = bsplgen(x - h, t, 4)
        np.testing.assert_allclose((bp-bm)/(2*h), d, atol=1e-8)
        np.testing.assert_allclose((dp-dm)/(2*h), dd, atol=1e-8)

    def test_polynomial_quadrature(self):
        s = AtomicSolver(1, intervals=20, rmax=3)
        for power in range(6):
            self.assertAlmostEqual(s.w @ s.r**power, 3**(power+1)/(power+1), places=10)

    def test_hydrogen_analytic_levels_and_normalisation(self):
        s = AtomicSolver(1, intervals=100, rmax=60)
        for l in [0, 1, 2]:
            energies, c, residual = s.hydrogenic(l, 2)
            n = np.arange(l+1, l+3)
            np.testing.assert_allclose(energies, -0.5/n**2, atol=3e-6, rtol=0)
            np.testing.assert_allclose(c.T @ s.S @ c, np.eye(2), atol=1e-11)
            self.assertLess(residual, 1e-8)

    def test_hydrogen_grid_refinement(self):
        errors = []
        for size in [24, 48, 96]:
            s = AtomicSolver(1, intervals=size)
            errors.append(abs(s.hydrogenic(0, 1)[0][0] + 0.5))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[2], 1e-7)

    def test_uniform_sphere_poisson_analytic_solution(self):
        s = AtomicSolver(2, intervals=30, rmax=4)
        N, R = 2., 4.
        q = 3*N*s.r**2/R**3
        exact = N/(2*R) * (3-s.r**2/R**2)
        np.testing.assert_allclose(s.hartree(q), exact, atol=2e-10, rtol=0)

    def test_exchange_energy_functional_derivative(self):
        s = AtomicSolver(2, intervals=25)
        q = s.r**2 * np.exp(-2*s.r)
        direction = q * (0.3 + np.sin(s.r))
        delta = 1e-5
        finite = (s.exchange(q+delta*direction)[1]-s.exchange(q-delta*direction)[1])/(2*delta)
        exact = s.w @ (s.exchange(q)[0] * direction)
        self.assertAlmostEqual(finite, exact, places=9)

    def test_exchange_coefficient_matches_original_formula(self):
        s = AtomicSolver(2, intervals=20, alpha=1)
        q = s.r**2 * np.exp(-s.r)
        rho = q/(4*np.pi*s.r**2)
        np.testing.assert_allclose(s.exchange(q)[0], -3*np.cbrt(3*rho/(8*np.pi)), atol=1e-14)

    def test_invalid_inputs(self):
        for options in [dict(Z=0),dict(Z=2,alpha=float('nan')),dict(Z=2,intervals=2),
                        dict(Z=2,rmax=-1),dict(Z=2,stretch=100)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                AtomicSolver(**options)
        s = AtomicSolver(2, intervals=20)
        for occ in [{}, {(1,0):3}, {(1,1):1}, {(1,0):[1,0]}, {(1,0):float('nan')}]:
            with self.subTest(occ=occ), self.assertRaises(ValueError):
                s.solve(occ)
        with self.assertRaises(ValueError):
            s.solve({(1,0):2}, mixing=0)
        with self.assertRaises(ValueError):
            s.hartree(-np.ones_like(s.r))
        with self.assertRaises(ValueError):
            bsplgen([-1], s.knots, 4)

    def test_unconverged_calculation_raises(self):
        with self.assertRaises(ConvergenceError):
            AtomicSolver(2, intervals=25).solve({(1,0):2}, max_iterations=1)


class AtomicExamples(unittest.TestCase):
    def test_neutral_and_ion_cases(self):
        for name in ['He','Ne','K']:
            with self.subTest(atom=name):
                Z, occ = CONFIGURATIONS[name]
                s = AtomicSolver(Z, intervals=100)
                neutral = s.solve(occ)
                ion = s.solve(CONFIGURATIONS[name+'+'][1])
                self.assertAlmostEqual(neutral.electron_count, Z, places=8)
                self.assertAlmostEqual(ion.electron_count, Z-1, places=8)
                self.assertTrue(np.all(np.isfinite(neutral.density)))
                self.assertTrue(np.all(neutral.density >= 0))
                self.assertLess(neutral.density_residual, 1e-8)
                self.assertLess(ion.density_residual, 1e-8)
                self.assertGreater(ion.total_energy - neutral.total_energy, 0)

    def test_helium_energy_stability(self):
        values=[]
        for size in [60,100,140]:
            result=AtomicSolver(2,intervals=size).solve({(1,0):2})
            values.append(result.total_energy)
        self.assertLess(abs(values[2]-values[1]), 2e-5)
        self.assertLess(abs(values[2]-values[1]), abs(values[1]-values[0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
