"""Example, in order to run you must place a pseudopotential 'Na.psf' in
the folder"""

from ase.units import Ry, eV
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.siesta_raman import SiestaRaman
from ase import Atoms
import numpy as np

# Define the systems
# example of Raman calculation for CO2 molecule,
# comparison with QE calculation can be done from
# https://github.com/maxhutch/quantum-espresso/blob/master/PHonon/examples/example15/README

CO2 = Atoms('CO2',
            positions=[[-0.009026, -0.020241, 0.026760],
                       [1.167544, 0.012723, 0.071808],
                       [-1.185592, -0.053316, -0.017945]],
            cell=[20, 20, 20])

# enter siesta input
siesta = Siesta(
    mesh_cutoff=150 * Ry,
    basis_set='DZP',
    pseudo_qualifier='',
    energy_shift=(10 * 10**-3) * eV,
    fdf_arguments={
        'SCFMustConverge': False,
        'COOP.Write': True,
        'WriteDenchar': True,
        'PAO.BasisType': 'split',
        'DM.Tolerance': 1e-4,
        'DM.MixingWeight': 0.01,
        'MaxSCFIterations': 300,
        'DM.NumberPulay': 4})

mbpt_inp = {'prod_basis_type': 'MIXED',
            'solver_type': 1,
            'gmres_eps': 0.001,
            'gmres_itermax': 256,
            'gmres_restart': 250,
            'gmres_verbose': 20,
            'xc_ord_lebedev': 14,
            'xc_ord_gl': 48,
            'nr': 512,
            'akmx': 100,
            'eigmin_local': 1e-06,
            'eigmin_bilocal': 1e-08,
            'freq_eps_win1': 0.15,
            'd_omega_win1': 0.05,
            'dt': 0.1,
            'omega_max_win1': 5.0,
            'ext_field_direction': 2,
            'dr': np.array([0.3, 0.3, 0.3]),
            'para_type': 'MATRIX',
            'chi0_v_algorithm': 14,
            'format_output': 'text',
            'comp_dens_chng_and_polarizability': 1,
            'store_dens_chng': 1,
            'enh_given_volume_and_freq': 0,
            'diag_hs': 0,
            'do_tddft_tem': 0,
            'do_tddft_iter': 1,
            'plot_freq': 3.02,
            'gwa_initialization': 'SIESTA_PB'}


CO2.set_calculator(siesta)

ram = SiestaRaman(CO2, siesta, mbpt_inp)
ram.run()
ram.summary(intensity_unit_ram='A^4 amu^-1')

ram.write_spectra(start=200)
