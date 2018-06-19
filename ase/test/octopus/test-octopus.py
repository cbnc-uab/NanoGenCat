from __future__ import print_function

import numpy as np

from ase import Atoms
from ase.calculators.interfacechecker import check_interface
from ase.calculators.octopus import Octopus, OctopusIOError
from ase.build import molecule

def getkwargs(**kwargs):
    kwargs0 = dict(FromScratch=True,
                   RestartWrite=False,
                   BoxShape='parallelepiped',
                   stdout='"stdout.txt"',
                   Spacing=0.15)
    kwargs0.update(kwargs)
    return kwargs0

# Verify that axes defined by ASE correspond correctly to axes
# in Octopus, i.e., that input/output arrays are not rotated
# incorrectly or similar due to C/Fortran conventions or other errors.
def test_axis_layout():
    system = Atoms('H')
    a = 3.
    system.cell = (a, a, a)
    system.pbc = 1

    for axis in range(3):
        system.center()
        system.positions[0, axis] = 0.0
        calc = Octopus(**getkwargs(label='ink-%s' % 'xyz'[axis],
                                   Output='density + potential + wfs'))
        system.set_calculator(calc)
        system.get_potential_energy()
        rho = calc.get_pseudo_density(pad=False)
        #for dim in rho.shape:
        #    assert dim % 2 == 1, rho.shape

        maxpoint = np.unravel_index(rho.argmax(), rho.shape)
        print('axis=%d: %s/%s' % (axis, maxpoint, rho.shape))

        expected_max = [dim // 2 for dim in rho.shape]
        expected_max[axis] = 0
        assert maxpoint == tuple(expected_max), '%s vs %s' % (maxpoint,
                                                              expected_max)

    errs = check_interface(calc)

    for err in errs:
        if err.code == 'not implemented':
            continue

        if err.methname == 'get_dipole_moment':
            assert isinstance(err.error, OctopusIOError)
        else:
            raise AssertionError(err.error)


# Test that that density and wavefunctions are normalized properly.
# Also does some tests of the energy; they should correspond to
# certain reference values.  Probably this depends a bit too much on
# octopus version though, so we will see if we need to revise this.
def test_integrals(pbc=True):
    system = molecule('H2O')
    a = 2.6006  # So the spacing does not divide exactly
    system.cell = (a, a, a)
    system.center()
    system.pbc = pbc
    spacing = 0.2
    calc = Octopus(**getkwargs(label='ink-integrals-pbc-%s' % pbc,
                               # Restart destroys normalization of output
                               # ...........sometimes.
                               RestartWrite=False,
                               Output='density + potential + wfs',
                               OutputFormat='cube + xcrysden',
                               ExtraStates=0,
                               Spacing=spacing,
                               SCFCalculateDipole=True))
    system.set_calculator(calc)
    E = system.get_potential_energy()

    if pbc:
        Eref = -496.98663392
    else:
        Eref = -451.05348602
    err = abs(E - Eref)
    #print('Energy=%f :: err=%e' % (E, err))
    # The reference has changed between version 5 and trunk,
    # so we will not check the total energy.
    # Checks of individual contributions that are physical and therefore
    # trustworthy will have to be sufficient.
    #assert err < 5e-3

    rho = calc.get_pseudo_density(pad=False)
    v = calc.get_effective_potential(pad=False)

    if pbc:  # spacing adjusted but cell constant
        dv = system.get_volume() / rho.size
    else:  # cell adjusted but spacing constant
        dv = spacing**3

    ne = rho.sum() * dv
    err = abs(ne - 8.0)
    print('nelectrons: %f, err: %e' % (ne, err))
    assert err < 1e-12

    for n in range(calc.get_number_of_bands()):
        psi = calc.get_pseudo_wave_function(band=n, pad=True)
        norm = (np.abs(psi)**2).sum() * dv
        err = abs(norm - 1)
        print('norm=%f :: err=%e' % (norm, err))
        assert err < 1e-12

    eps = calc.get_eigenvalues()
    f = calc.get_occupation_numbers()
    E_band = (eps * f).sum()
    E_nv = (rho * v).sum() * dv
    if pbc:
        E_nl = -155.16024733 # Reference value from output.
        E_kin_ref = 377.899824
    else:
        E_nl = -151.28372313 # Ref from output.
        E_kin_ref = 396.66270596
    E_kin_ours = E_band - E_nv - E_nl
    err = abs(E_kin_ours - E_kin_ref)
    print('E_band=%f :: E_nv=%f :: E_kin_ours=%f :: err=%e'
          % (E_band, E_nv, E_kin_ours, err))
    assert err < 5e-3  # Orig err: 6.8e-05 (pbc=False) and 3.47e-07 (pbc=True)

    errs = check_interface(calc)
    for err in errs:
        if err.code == 'not implemented':
            continue
        else:
            raise AssertionError(err.error)

def main():
    #try:
    #    proc = Popen(['octopus', '--version'], stdout=PIPE)
    #    version_text = proc.stdout.read()
    #except OSError:
    #    raise NotAvailable
    #else:
    #    print('Octopus version: %s' % version_text)
    #    assert version_text.startswith('octopus tetricus')

    test_axis_layout()
    test_integrals(pbc=False)
    test_integrals(pbc=True)

main()
