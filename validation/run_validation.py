"""Reproduce analytic and mesh/box/quadrature evidence, without MATLAB claims."""
import json
import platform
import sys
from pathlib import Path
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from modules import AtomicSolver, CONFIGURATIONS


def main():
    report = {'environment': {'python': platform.python_version(),
                              'numpy': np.__version__, 'scipy': scipy.__version__},
              'method': 'spin-unpolarised spherical X-alpha, alpha=1, fixed occupations',
              'not_claimed': ['MATLAB parity', 'experimental accuracy',
                              'spin-polarised open-shell ground states'],
              'hydrogen': [], 'atoms': {}}
    for intervals in [24, 48, 96]:
        s = AtomicSolver(1, intervals=intervals)
        e = float(s.hydrogenic(0, 1)[0][0])
        report['hydrogen'].append({'intervals': intervals, 'energy': e,
                                   'absolute_error_vs_minus_half': abs(e+0.5)})
    for atom in ['He','Ne','K']:
        records={}
        for label, settings in [('base',{}), ('fine',{'intervals':140}),
                                ('larger_box',{'rmax':50.,'intervals':140}),
                                ('higher_quadrature',{'quadrature':12})]:
            Z, occ = CONFIGURATIONS[atom]
            s = AtomicSolver(Z, **settings)
            neutral=s.solve(occ)
            ion=s.solve(CONFIGURATIONS[atom+'+'][1])
            records[label]={'settings':{'intervals':100,'rmax':35.,'quadrature':8,**settings},
                            'neutral':neutral.summary(), 'ion':ion.summary(),
                            'ionisation_energy_hartree':ion.total_energy-neutral.total_energy}
        base=records['base']; fine=records['fine']
        changes={}
        for key in ['neutral','ion']:
            changes[key+'_grid_delta_hartree']=abs(fine[key]['total_energy_hartree']-base[key]['total_energy_hartree'])
        changes['ionisation_grid_delta_hartree']=abs(fine['ionisation_energy_hartree']-base['ionisation_energy_hartree'])
        changes['ionisation_box_delta_hartree']=abs(records['larger_box']['ionisation_energy_hartree']-fine['ionisation_energy_hartree'])
        changes['ionisation_quadrature_delta_hartree']=abs(records['higher_quadrature']['ionisation_energy_hartree']-base['ionisation_energy_hartree'])
        # Fixed acceptance bounds for this educational implementation, not experimental accuracy.
        limits={k:(2e-3 if k.startswith(('neutral','ion_grid')) else 2e-4) for k in changes}
        for key,value in changes.items():
            if value >= limits[key]:
                raise AssertionError(f'{atom}: {key}={value} exceeds {limits[key]}')
        report['atoms'][atom]={'runs':records,'changes':changes,'acceptance_limits':limits}
        print(atom, changes, flush=True)
    standard=AtomicSolver(2,alpha=2/3).solve({(1,0):2})
    report['helium_dirac_exchange_example']=standard.summary()
    (ROOT/'validation/results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Wrote validation/results.json',flush=True)


if __name__ == '__main__':
    main()
