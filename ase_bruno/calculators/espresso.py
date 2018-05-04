"""Quantum ESPRESSO Calculator

export ASE_ESPRESSO_COMMAND="/path/to/pw.x -in PREFIX.pwi > PREFIX.pwo"

Run pw.x jobs.
"""


from ase import io
from ase.calculators.calculator import FileIOCalculator


class Espresso(FileIOCalculator):
    """
    """
    implemented_properties = ['energy', 'forces', 'stress', 'magmoms']
    command = 'pw.x -in PREFIX.pwi > PREFIX.pwo'

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='espresso', atoms=None, **kwargs):
        """
        All options for pw.x are copied verbatim to the input file, and put
        into the correct section. Use ``input_data`` for parameters that are
        already in a dict, all other ``kwargs`` are passed as parameters.

        Accepts all the options for pw.x as given in the QE docs, plus some
        additional options:

        input_data: dict
            A flat or nested dictionary with input parameters for pw.x
        pseudopotentials: dict
            A filename for each atomic species, e.g.
            ``{'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}``.
            A dummy name will be used if none are given.
        kspacing: float
            Generate a grid of k-points with this as the minimum distance,
            in A^-1 between them in reciprocal space. If set to None, kpts
            will be used instead.
        kpts:
            Number of kpoints in each dimension for automatic kpoint generation.
        koffset: (int, int, int)
            Offset of kpoints in each direction. Must be 0 (no offset) or
            1 (half grid offset). Setting to True is equivalent to (1, 1, 1).


        .. note::
           Set ``tprnfor=True`` and ``tstress=True`` to calculate forces and
           stresses.


        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        io.write(self.label + '.pwi', atoms, **self.parameters)

    def read_results(self):
        output = io.read(self.label + '.pwo')
        self.results = output.calc.results
