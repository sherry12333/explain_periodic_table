"""CLI examples: python python/run_atom.py He --ion --output validation/he.json"""
import argparse
import json
from pathlib import Path
from modules import AtomicSolver, CONFIGURATIONS


def main():
    parser = argparse.ArgumentParser(description="Educational spherical atomic X-alpha solver")
    parser.add_argument("atom", choices=["He", "Ne", "K"])
    parser.add_argument("--ion", action="store_true", help="also calculate singly charged cation")
    parser.add_argument("--intervals", type=int, default=100)
    parser.add_argument("--rmax", type=float, default=35.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    Z, occupation = CONFIGURATIONS[args.atom]
    solver = AtomicSolver(Z, intervals=args.intervals, rmax=args.rmax, alpha=args.alpha)
    neutral = solver.solve(occupation)
    report = {"model": "spherical spin-unpolarised exchange-only X-alpha",
              "grid": {"intervals": args.intervals, "rmax_bohr": args.rmax,
                       "order": 4, "quadrature_per_interval": 8, "stretch": 5.0},
              "neutral": neutral.summary()}
    if args.ion:
        ion = solver.solve(CONFIGURATIONS[args.atom + "+"][1])
        report["ion"] = ion.summary()
        report["ionisation_energy_hartree"] = ion.total_energy - neutral.total_energy
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
