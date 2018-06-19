# -*- coding: utf-8 -*-
from __future__ import print_function
"""This module defines an interface to CASTEP for
    use by the ASE (Webpage: http://wiki.fysik.dtu.dk/ase)

Authors:
    Max Hoffmann, max.hoffmann@ch.tum.de
    Joerg Meyer, joerg.meyer@ch.tum.de

Contributors:
    Juan M. Lorenzi, juan.lorenzi@tum.de
    Georg S. Michelitsch, georg.michelitsch@tch.tum.de
    Reinhard J. Maurer, reinhard.maurer@yale.edu
    Simon P. Rittmeyer, simon.rittmeyer@tum.de
"""

from copy import deepcopy
import difflib
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.constraints import FixCartesian
from ase.parallel import paropen
from ase.utils import basestring

__all__ = [
    'Castep',
    'CastepCell',
    'CastepParam',
    'create_castep_keywords']

contact_email = 'simon.rittmeyer@tum.de'

# A convenient table to avoid the previously used "eval"
_tf_table = {
    '': True,  # Just the keyword is equivalent to True
    'True': True,
    'False': False}


def _self_getter(getf):
    # A decorator that makes it so that if no 'atoms' argument is passed to a
    # getter function, self.atoms is used instead

    def decor_getf(self, atoms=None, *args, **kwargs):

        if atoms is None:
            atoms = self.atoms

        return getf(self, atoms, *args, **kwargs)

    return decor_getf


class Castep(Calculator):

    r"""

    CASTEP Interface Documentation

Introduction
============


CASTEP_ [1]_ W_ is a software package which uses density functional theory to
provide a good atomic-level description of all manner of materials and
molecules. CASTEP can give information about total energies, forces and
stresses on an atomic system, as well as calculating optimum geometries, band
structures, optical spectra, phonon spectra and much more. It can also perform
molecular dynamics simulations.

The CASTEP calculator interface class offers intuitive access to all CASTEP
settings and most results. All CASTEP specific settings are accessible via
attribute access (*i.e*. ``calc.param.keyword = ...`` or
``calc.cell.keyword = ...``)


Getting Started:
================

Set the environment variables appropriately for your system.

>>> export CASTEP_COMMAND=' ... '
>>> export CASTEP_PP_PATH=' ... '

Note: alternatively to CASTEP_PP_PATH one can set PSPOT_DIR
      as CASTEP consults this by default, i.e.

>>> export PSPOT_DIR=' ... '



Running the Calculator
======================

The default initialization command for the CASTEP calculator is

.. class:: Castep(directory='CASTEP', label='castep')

To do a minimal run one only needs to set atoms, this will use all
default settings of CASTEP, meaning LDA, singlepoint, etc..

With a generated castep_keywords.py in place all options are accessible
by inspection, *i.e.* tab-completion. This works best when using ``ipython``.
All options can be accessed via ``calc.param.<TAB>`` or ``calc.cell.<TAB>``
and documentation is printed with ``calc.param.<keyword> ?`` or
``calc.cell.<keyword> ?``. All options can also be set directly
using ``calc.keyword = ...`` or ``calc.KEYWORD = ...`` or even
``ialc.KeYwOrD`` or directly as named arguments in the call to the constructor
(*e.g.* ``Castep(task='GeometryOptimization')``).

All options that go into the ``.param`` file are held in an ``CastepParam``
instance, while all options that go into the ``.cell`` file and don't belong
to the atoms object are held in an ``CastepCell`` instance. Each instance can
be created individually and can be added to calculators by attribute
assignment, *i.e.* ``calc.param = param`` or ``calc.cell = cell``.

All internal variables of the calculator start with an underscore (_).
All cell attributes that clearly belong into the atoms object are blocked.
Setting ``calc.atoms_attribute`` (*e.g.* ``= positions``) is sent directly to
the atoms object.


Arguments:
==========

=========================  ====================================================
Keyword                    Description
=========================  ====================================================
``directory``              The relative path where all input and output files
                           will be placed. If this does not exist, it will be
                           created.  Existing directories will be moved to
                           directory-TIMESTAMP unless self._rename_existing_dir
                           is set to false.

``label``                  The prefix of .param, .cell, .castep, etc. files.

=========================  ====================================================


Additional Settings
===================

=========================  ====================================================
Internal Setting           Description
=========================  ====================================================
``_castep_command``        (``=castep``): the actual shell command used to
                           call CASTEP.

``_check_checkfile``       (``=True``): this makes write_param() only
                           write a continue or reuse statement if the
                           addressed .check or .castep_bin file exists in the
                           directory.

``_copy_pspots``           (``=False``): if set to True the calculator will
                           actually copy the needed pseudo-potential (\*.usp)
                           file, usually it will only create symlinks.

``_link_pspots``           (``=True``): if set to True the calculator will
                           actually will create symlinks to the needed pseudo
                           potentials. Set this option (and ``_copy_pspots``)
                           to False if you rather want to access your pseudo
                           potentials using the PSPOT_DIR environment variable
                           that is read by CASTEP.
                           *Note:* This option has no effect if ``copy_pspots``
                           is True..

``_export_settings``       (``=True``): if this is set to
                           True, all calculator internal settings shown here
                           will be included in the .param in a comment line (#)
                           and can be read again by merge_param. merge_param
                           can be forced to ignore this directive using the
                           optional argument ``ignore_internal_keys=True``.

``_force_write``           (``=True``): this controls wether the \*cell and
                           \*param will be overwritten.

``_prepare_input_only``    (``=False``): If set to True, the calculator will
                           create \*cell und \*param file but not
                           start the calculation itself.
                           If this is used to prepare jobs locally
                           and run on a remote cluster it is recommended
                           to set ``_copy_pspots = True``.

``_castep_pp_path``        (``='.'``) : the place where the calculator
                           will look for pseudo-potential files.

``_rename_existing_dir``   (``=True``) : when using a new instance
                           of the calculator, this will move directories out of
                           the way that would be overwritten otherwise,
                           appending a date string.

``_set_atoms``             (``=False``) : setting this to True will overwrite
                           any atoms object previously attached to the
                           calculator when reading a \.castep file.  By de-
                           fault, the read() function will only create a new
                           atoms object if none has been attached and other-
                           wise try to assign forces etc. based on the atom's
                           positions.  ``_set_atoms=True`` could be necessary
                           if one uses CASTEP's internal geometry optimization
                           (``calc.param.task='GeometryOptimization'``)
                           because then the positions get out of sync.
                           *Warning*: this option is generally not recommended
                           unless one knows one really needs it. There should
                           never be any need, if CASTEP is used as a
                           single-point calculator.

``_track_output``          (``=False``) : if set to true, the interface
                           will append a number to the label on all input
                           and output files, where n is the number of calls
                           to this instance. *Warning*: this setting may con-
                           sume a lot more disk space because of the additio-
                           nal \*check files.

``_try_reuse``             (``=_track_output``) : when setting this, the in-
                           terface will try to fetch the reuse file from the
                           previous run even if _track_output is True. By de-
                           fault it is equal to _track_output, but may be
                           overridden.

                           Since this behavior may not always be desirable for
                           single-point calculations. Regular reuse for *e.g.*
                           a geometry-optimization can be achieved by setting
                           ``calc.param.reuse = True``.
``_pedantic``              (``=False``) if set to true, the calculator will
                           inform about settings probably wasting a lot of CPU
                           time or causing numerical inconsistencies.

=========================  ====================================================

Special features:
=================


``.dryrun_ok()``
  Runs ``castep_command seed -dryrun`` in a temporary directory return True if
  all variables initialized ok. This is a fast way to catch errors in the
  input. Afterwards _kpoints_used is set.

``.merge_param()``
  Takes a filename or filehandler of a .param file or CastepParam instance and
  merges it into the current calculator instance, overwriting current settings

``.keyword.clear()``
  Can be used on any option like ``calc.param.keyword.clear()`` or
  ``calc.cell.keyword.clear()`` to return to the CASTEP default.

``.initialize()``
  Creates all needed input in the ``_directory``. This can then copied to and
  run in a place without ASE or even python.

``.set_pspot('<library>')``
  This automatically sets the pseudo-potential for all present species to
  *<Species>_<library>.usp*. Make sure that ``_castep_pp_path`` is set
  correctly.

``print(calc)``
  Prints a short summary of the calculator settings and atoms.

``ase.io.castep.read_seed('path-to/seed')``
  Given you have a combination of seed.{param,cell,castep} this will return an
  atoms object with the last ionic positions in the .castep file and all other
  settings parsed from the .cell and .param file. If no .castep file is found
  the positions are taken from the .cell file. The output directory will be
  set to the same directory, only the label is preceded by 'copy_of\_'  to
  avoid overwriting.


Notes/Issues:
==============

* Currently *only* the FixAtoms *constraint* is fully supported for reading and
  writing. There is some experimental support for the FixCartesian constraint.

* There is no support for the CASTEP *unit system*. Units of eV and Angstrom
  are used throughout. In particular when converting total energies from
  different calculators, one should check that the same CODATA_ version is
  used for constants and conversion factors, respectively.

.. _CASTEP: http://www.castep.org/

.. _W: http://en.wikipedia.org/wiki/CASTEP

.. _CODATA: http://physics.nist.gov/cuu/Constants/index.html

.. [1] S. J. Clark, M. D. Segall, C. J. Pickard, P. J. Hasnip, M. J. Probert,
       K. Refson, M. C. Payne Zeitschrift für Kristallographie 220(5-6)
       pp.567- 570 (2005) PDF_.

.. _PDF: http://goo.gl/wW50m


End CASTEP Interface Documentation
    """

    # Class attributes !
    # keys set through atoms object
    atoms_keys = [
        'charge',
        'ionic_constraints',
        'lattice_abs',
        'lattice_cart',
        'positions_abs',
        'positions_abs_final',
        'positions_abs_intermediate',
        'positions_frac',
        'positions_frac_final',
        'positions_frac_intermediate']

    atoms_obj_keys = [
        'dipole',
        'energy_free',
        'energy_zero',
        'fermi',
        'forces',
        'nbands',
        'positions',
        'stress']

    internal_keys = [
        '_castep_command',
        '_check_checkfile',
        '_copy_pspots',
        '_link_pspots',
        '_directory',
        '_export_settings',
        '_force_write',
        '_label',
        '_prepare_input_only',
        '_castep_pp_path',
        '_rename_existing_dir',
        '_set_atoms',
        '_track_output',
        '_try_reuse',
        '_pedantic']

    def __init__(self, directory='CASTEP', label='castep',
                 castep_command=None, check_castep_version=False,
                 castep_pp_path=None,
                 **kwargs):

        self.__name__ = 'Castep'

        # initialize the ase.calculators.general calculator
        Calculator.__init__(self)

        from ase.io.castep import write_cell
        self._write_cell = write_cell

        castep_keywords = import_castep_keywords(castep_command)
        self.param = CastepParam(castep_keywords)
        self.cell = CastepCell(castep_keywords)

        ###################################
        # Calculator state variables      #
        ###################################
        self._calls = 0
        self._castep_version = castep_keywords.castep_version

        # collects warning from .castep files
        self._warnings = []
        # collects content from *.err file
        self._error = None
        # warnings raised by the ASE interface
        self._interface_warnings = []

        # store to check if recalculation is necessary
        self._old_atoms = None
        self._old_cell = None
        self._old_param = None

        ###################################
        # Internal keys                   #
        # Allow to tweak the behavior     #
        ###################################
        self._opt = {}
        self._castep_command = get_castep_command(castep_command)
        self._castep_pp_path = get_castep_pp_path(castep_pp_path)
        self._check_checkfile = True
        self._copy_pspots = False
        self._link_pspots = True
        self._directory = os.path.abspath(directory)
        self._export_settings = True
        self._force_write = True
        self._label = label
        self._prepare_input_only = False
        self._rename_existing_dir = True
        self._set_atoms = False
        self._track_output = False
        self._try_reuse = False

        # turn off the pedantic user warnings
        self._pedantic = False

        # will be set on during runtime
        self._seed = None

        ###################################
        # (Physical) result variables     #
        ###################################
        self.atoms = None
        # initialize result variables
        self._forces = None
        self._energy_total = None
        self._energy_free = None
        self._energy_0K = None

        # dispersion corrections
        self._dispcorr_energy_total = None
        self._dispcorr_energy_free = None
        self._dispcorr_energy_0K = None

        # spins and hirshfeld volumes
        self._spins = None
        self._hirsh_volrat = None

        self._number_of_cell_constraints = None
        self._output_verbosity = None
        self._stress = None
        self._unit_cell = None
        self._kpoints = None

        # pointers to other files used at runtime
        self._check_file = None
        self._castep_bin_file = None

        # check version of CASTEP options module against current one
        if check_castep_version:
            local_castep_version = get_castep_version(self._castep_command)
            if not hasattr(self, '_castep_version'):
                print('No castep version found')
                return
            if not local_castep_version == self._castep_version:
                print(('The options module was generated from version %s\n'
                       'while your are currently using CASTEP version %s') %
                      (self._castep_version,
                       get_castep_version(self._castep_command)))
                self._castep_version = local_castep_version

        # processes optional arguments in kw style
        for keyword, value in kwargs.items():
            # first fetch special keywords issued by ASE CLI
            if keyword == 'kpts':
                self.__setattr__('kpoint_mp_grid', '%s %s %s' % tuple(value))
            elif keyword == 'xc':
                self.__setattr__('xc_functional', str(value))
            elif keyword == 'ecut':
                self.__setattr__('cut_off_energy', str(value))
            else:  # the general case
                self.__setattr__(keyword, value)

    def _castep_find_last_record(self, castep_file):
        """Checks wether a given castep file has a regular
        ending message following the last banner message. If this
        is the case, the line number of the last banner is message
        is return, otherwise False.

        returns (record_start, record_end, end_found, last_record_complete)
        """
        if isinstance(castep_file, basestring):
            castep_file = paropen(castep_file, 'r')
            file_opened = True
        else:
            file_opened = False
        record_starts = []
        while True:
            line = castep_file.readline()
            if 'Welcome' in line and 'CASTEP' in line:
                record_starts = [castep_file.tell()] + record_starts
            if not line:
                break

        if record_starts == []:
            print('Could not find CASTEP label in result file: %s'
                  % castep_file)
            print('Are you sure this is a .castep file?')
            return

        # search for regular end of file
        end_found = False
        # start to search from record beginning from the back
        # and see if
        record_end = -1
        for record_nr, record_start in enumerate(record_starts):
            castep_file.seek(record_start)
            while True:
                line = castep_file.readline()
                if not line:
                    break
                if 'warn' in line.lower():
                    self._warnings.append(line)

                # HOTFIX: This string appears twice from CASTEP 7 on and thus
                # prevents reading forces. So, better go for another keyword
                # to indicate the regular end of a run.
                # 'Initialization time' seems to do the job.
                # if 'Writing analysis data to' in line:
                # if 'Writing model to' in line:
                if 'Initialisation time' in line:
                    end_found = True
                    record_end = castep_file.tell()
                    break

            if end_found:
                break

        if file_opened:
            castep_file.close()

        if end_found:
            # record_nr == 0 corresponds to the last record here
            if record_nr == 0:
                return (record_start, record_end, True, True)
            else:
                return (record_start, record_end, True, False)
        else:
            return (0, record_end, False, False)

    def read(self, castep_file=None):
        """Read a castep file into the current instance."""

        _close = True

        if castep_file is None:
            if self._castep_file:
                castep_file = self._castep_file
                out = paropen(castep_file, 'r')
            else:
                print('No CASTEP file specified')
                return
            if not os.path.exists(castep_file):
                print('No CASTEP file found')

        elif isinstance(castep_file, basestring):
            out = paropen(castep_file, 'r')

        else:
            # in this case we assume that we have a fileobj already, but check
            # for attributes in order to avoid extended EAFP blocks.
            out = castep_file

            # look before you leap...
            attributes = ['name',
                          'seek',
                          'close',
                          'readline',
                          'tell']

            for attr in attributes:
                if not hasattr(out, attr):
                    raise TypeError(
                        '"castep_file" is neither str nor valid fileobj')

            castep_file = out.name
            _close = False

        if self._seed is None:
            self._seed = os.path.splitext(os.path.basename(castep_file))[0]

        err_file = '%s.0001.err' % self._seed
        if os.path.exists(err_file):
            err_file = paropen(err_file)
            self._error = err_file.read()
            err_file.close()
            # we return right-away because it might
            # just be here from a previous run
        # look for last result, if several CASTEP
        # run are appended

        record_start, record_end, end_found, _\
            = self._castep_find_last_record(out)
        if not end_found:
            print('No regular end found in %s file' % castep_file)
            print(self._error)
            if _close:
                out.close()
            return
            # we return here, because the file has no a regular end

        # now iterate over last CASTEP output in file to extract information
        # could be generalized as well to extract trajectory from file
        # holding several outputs
        n_cell_const = 0
        forces = []

        # HOTFIX:
        # we have to initialize the _stress variable as a zero array
        # otherwise the calculator crashes upon pickling trajectories
        # Alternative would be to raise a NotImplementedError() which
        # is also kind of not true, since we can extract stresses if
        # the user configures CASTEP to print them in the outfile
        # stress = []
        stress = np.zeros([3, 3])
        hirsh_volrat = []

        # Two flags to check whether spin-polarized or not, and whether
        # Hirshfeld volumes are calculated
        spin_polarized = False
        calculate_hirshfeld = False

        positions_frac_list = []

        out.seek(record_start)
        while True:
            # TODO: add a switch if we have a geometry optimization: record
            # atoms objects for intermediate steps.
            try:
                # in case we need to rewind back one line, we memorize the bit
                # position of this line in the file.
                # --> see symops problem below
                _line_start = out.tell()
                line = out.readline()
                if not line or out.tell() > record_end:
                    break
                elif 'output verbosity' in line:
                    iprint = int(line.split()[-1][1])
                    if int(iprint) != 1:
                        self.param.iprint = iprint
                elif 'treating system as spin-polarized' in line:
                    spin_polarized = True
                elif 'treating system as non-spin-polarized' in line:
                    spin_polarized = False
                elif 'Unit Cell' in line:
                    lattice_real = []
                    lattice_reci = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 6:
                            break
                    for i in range(3):
                        lattice_real.append([float(f) for f in fields[0:3]])
                        lattice_reci.append([float(f) for f in fields[3:7]])
                        line = out.readline()
                        fields = line.split()
                elif 'Cell Contents' in line:
                    while True:
                        line = out.readline()
                        if 'Total number of ions in cell' in line:
                            n_atoms = int(line.split()[7])
                        if 'Total number of species in cell' in line:
                            int(line.split()[7])
                        fields = line.split()
                        if len(fields) == 0:
                            break
                elif 'Fractional coordinates of atoms' in line:
                    species = []
                    custom_species = None  # A CASTEP special thing
                    positions_frac = []
                    # positions_cart = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 7:
                            break
                    for n in range(n_atoms):
                        spec_custom = fields[1].split(':', 1)
                        elem = spec_custom[0]
                        if len(spec_custom) > 1 and custom_species is None:
                            # Add it to the custom info!
                            custom_species = list(species)
                        species.append(elem)
                        if custom_species is not None:
                            custom_species.append(fields[1])
                        positions_frac.append([float(s) for s in fields[3:6]])
                        line = out.readline()
                        fields = line.split()
                    positions_frac_list.append(positions_frac)
                elif 'Files used for pseudopotentials' in line:
                    while True:
                        line = out.readline()
                        if 'Pseudopotential generated on-the-fly' in line:
                            continue
                        fields = line.split()
                        if (len(fields) >= 2):
                            elem, pp_file = fields
                            self.cell.species_pot = (elem, pp_file)
                        else:
                            break
                elif 'k-Points For BZ Sampling' in line:
                    # TODO: generalize for non-Monkhorst Pack case
                    # (i.e. kpoint lists) -
                    # kpoints_offset cannot be read this way and
                    # is hence always set to None
                    while True:
                        line = out.readline()
                        if not line.strip():
                            break
                        if 'MP grid size for SCF calculation' in line:
                            # kpoints =  ' '.join(line.split()[-3:])
                            # self.kpoints_mp_grid = kpoints
                            # self.kpoints_mp_offset = '0. 0. 0.'
                            # not set here anymore because otherwise
                            # two calculator objects go out of sync
                            # after each calculation triggering unnecessary
                            # recalculation
                            break
                elif 'Symmetry and Constraints' in line:
                    # this is a bit of a hack, but otherwise the read_symops
                    # would need to re-read the entire file. --> just rewind
                    # back by one line, so the read_symops routine can find the
                    # start of this block.
                    out.seek(_line_start)
                    self.read_symops(castep_castep=out)
                elif 'Number of cell constraints' in line:
                    n_cell_const = int(line.split()[4])
                elif 'Final energy' in line:
                    self._energy_total = float(line.split()[-2])
                elif 'Final free energy' in line:
                    self._energy_free = float(line.split()[-2])
                elif 'NB est. 0K energy' in line:
                    self._energy_0K = float(line.split()[-2])

                # Add support for dispersion correction
                # filtering due to SEDC is done in get_potential_energy
                elif 'Dispersion corrected final energy' in line:
                    self._dispcorr_energy_total = float(line.split()[-2])
                elif 'Dispersion corrected final free energy' in line:
                    self._dispcorr_energy_free = float(line.split()[-2])
                elif 'dispersion corrected est. 0K energy' in line:
                    self._dispcorr_energy_0K = float(line.split()[-2])

                # remember to remove constraint labels in force components
                # (lacking a space behind the actual floating point number in
                # the CASTEP output)
                elif '******************** Forces *********************'\
                     in line or\
                     '************** Symmetrised Forces ***************'\
                     in line or\
                     '************** Constrained Symmetrised Forces ****'\
                     '**********'\
                     in line or\
                     '******************** Constrained Forces **********'\
                     '**********'\
                     in line or\
                     '******************* Unconstrained Forces *********'\
                     '**********'\
                     in line:
                    fix = []
                    fix_cart = []
                    forces = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 7:
                            break
                    for n in range(n_atoms):
                        consd = np.array([0, 0, 0])
                        fxyz = [0, 0, 0]
                        for (i, force_component) in enumerate(fields[-4:-1]):
                            if force_component.count("(cons'd)") > 0:
                                consd[i] = 1
                            fxyz[i] = float(force_component.replace(
                                "(cons'd)", ''))
                        if consd.all():
                            fix.append(n)
                        elif consd.any():
                            fix_cart.append(FixCartesian(n, consd))
                        forces.append(fxyz)
                        line = out.readline()
                        fields = line.split()

                # add support for Hirshfeld analysis
                elif 'Hirshfeld / free atomic volume :' in line:
                    # if we are here, then params must be able to cope with
                    # Hirshfeld flag (if castep_keywords.py matches employed
                    # castep version)
                    calculate_hirshfeld = True
                    hirsh_volrat = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 1:
                            break
                    for n in range(n_atoms):
                        hirsh_atom = float(fields[0])
                        hirsh_volrat.append(hirsh_atom)
                        while True:
                            line = out.readline()
                            if 'Hirshfeld / free atomic volume :' in line or\
                               'Hirshfeld Analysis' in line:
                                break
                        line = out.readline()
                        fields = line.split()

                elif '***************** Stress Tensor *****************'\
                     in line or\
                     '*********** Symmetrised Stress Tensor ***********'\
                     in line:
                    stress = []
                    while True:
                        line = out.readline()
                        fields = line.split()
                        if len(fields) == 6:
                            break
                    for n in range(3):
                        stress.append([float(s) for s in fields[2:5]])
                        line = out.readline()
                        fields = line.split()
                elif ('BFGS: starting iteration' in line or
                      'BFGS: improving iteration' in line):
                    if n_cell_const < 6:
                        lattice_real = []
                        lattice_reci = []
                    species = []
                    positions_frac = []
                    forces = []

                    # HOTFIX:
                    # Same reason for the stress initialization as before
                    # stress = []
                    stress = np.zeros([3, 3])

                elif 'BFGS: Final Configuration:' in line:
                    break
                elif 'warn' in line.lower():
                    self._warnings.append(line)
            except Exception as exception:
                print(line, end=' ')
                print('|-> line triggered exception: ' + str(exception))
                raise

        # get the spins in a separate run over the file as we
        # do not want to break the BFGS-break construct
        # probably one can implement it in a more convenient
        # way, but this constructon does the job.

        if spin_polarized:
            spins = []
            out.seek(record_start)
            while True:
                try:
                    line = out.readline()
                    if not line or out.tell() > record_end:
                        break
                    elif 'Atomic Populations' in line:
                        # skip the separating line
                        line = out.readline()
                        # this is the headline
                        line = out.readline()
                        if 'Spin' in line:
                            # skip the next separator line
                            line = out.readline()
                            while True:
                                line = out.readline()
                                fields = line.split()
                                if len(fields) == 1:
                                    break
                                spins.append(float(fields[-1]))
                        break

                except Exception as exception:
                    print(line + '|-> line triggered exception: ' +
                          str(exception))
                    raise
        else:
            # set to zero spin if non-spin polarized calculation
            spins = np.zeros(len(positions_frac))

        if _close:
            out.close()

        positions_frac_atoms = np.array(positions_frac)
        forces_atoms = np.array(forces)
        spins_atoms = np.array(spins)

        if calculate_hirshfeld:
            hirsh_atoms = np.array(hirsh_volrat)
        else:
            hirsh_atoms = np.zeros_like(spins)

        if self.atoms and not self._set_atoms:
            # compensate for internal reordering of atoms by CASTEP
            # using the fact that the order is kept within each species

            # positions_frac_ase = self.atoms.get_scaled_positions(wrap=False)
            atoms_assigned = [False] * len(self.atoms)

            # positions_frac_castep_init = np.array(positions_frac_list[0])
            positions_frac_castep = np.array(positions_frac_list[-1])

            # species_castep = list(species)
            forces_castep = np.array(forces)
            hirsh_castep = np.array(hirsh_volrat)
            spins_castep = np.array(spins)

            # go through the atoms position list and replace
            # with the corresponding one from the
            # castep file corresponding atomic number
            for iase in range(n_atoms):
                for icastep in range(n_atoms):
                    if (species[icastep] == self.atoms[iase].symbol and
                            not atoms_assigned[icastep]):
                        positions_frac_atoms[iase] = \
                            positions_frac_castep[icastep]
                        forces_atoms[iase] = np.array(forces_castep[icastep])
                        if iprint > 1 and calculate_hirshfeld:
                            hirsh_atoms[iase] = np.array(hirsh_castep[icastep])
                        if spin_polarized:
                            # reordering not necessary in case all spins == 0
                            spins_atoms[iase] = np.array(spins_castep[icastep])
                        atoms_assigned[icastep] = True
                        break

            if not all(atoms_assigned):
                not_assigned = [i for (i, assigned)
                                in zip(range(len(atoms_assigned)),
                                       atoms_assigned) if not assigned]
                print('%s atoms not assigned.' % atoms_assigned.count(False))
                print('DEBUGINFO: The following atoms where not assigned: %s' %
                      not_assigned)
            else:
                self.atoms.set_scaled_positions(positions_frac_atoms)

        else:
            # If no atoms, object has been previously defined
            # we define it here and set the Castep() instance as calculator.
            # This covers the case that we simply want to open a .castep file.

            # The next time around we will have an atoms object, since
            # set_calculator also set atoms in the calculator.
            if self.atoms:
                constraints = self.atoms.constraints
            else:
                constraints = []
            atoms = ase.atoms.Atoms(species,
                                    cell=lattice_real,
                                    constraint=constraints,
                                    pbc=True,
                                    scaled_positions=positions_frac,
                                    )
            if custom_species is not None:
                atoms.new_array('castep_custom_species',
                                np.array(custom_species))

            if self.param.spin_polarized:
                # only set magnetic moments if this was a spin polarized
                # calculation
                atoms.set_initial_magnetic_moments(magmoms=spins_atoms)

            atoms.set_calculator(self)

        self._forces = forces_atoms
        # stress in .castep file is given in GPa:
        self._stress = np.array(stress) * units.GPa
        self._hirsh_volrat = hirsh_atoms
        self._spins = spins_atoms

        if self._warnings:
            print('WARNING: %s contains warnings' % castep_file)
            for warning in self._warnings:
                print(warning)
        # reset
        self._warnings = []

    def read_symops(self, castep_castep=None):
        # TODO: check that this is really backwards compatible
        # with previous routine with this name...
        """Read all symmetry operations used from a .castep file."""

        if castep_castep is None:
            castep_castep = self._seed + '.castep'

        if isinstance(castep_castep, basestring):
            if not os.path.isfile(castep_castep):
                print('Warning: CASTEP file %s not found!' % castep_castep)
            f = paropen(castep_castep, 'a')
            _close = True
        else:
            # in this case we assume that we have a fileobj already, but check
            # for attributes in order to avoid extended EAFP blocks.
            f = castep_castep

            # look before you leap...
            attributes = ['name',
                          'readline',
                          'close']

            for attr in attributes:
                if not hasattr(f, attr):
                    raise TypeError('read_castep_castep_symops: castep_castep '
                                    'is not of type str nor valid fileobj!')

            castep_castep = f.name
            _close = False

        while True:
            line = f.readline()
            if not line:
                return
            if 'output verbosity' in line:
                iprint = line.split()[-1][1]
                # filter out the default
                if int(iprint) != 1:
                    self.param.iprint = iprint
            if 'Symmetry and Constraints' in line:
                break

        if self.param.iprint is None or self.param.iprint < 2:
            self._interface_warnings.append(
                'Warning: No symmetry'
                'operations could be read from %s (iprint < 2).' % f.name)
            return

        while True:
            line = f.readline()
            if not line:
                break
            if 'Number of symmetry operations' in line:
                nsym = int(line.split()[5])
                # print "nsym = %d" % nsym
                # information about symmetry related atoms currently not read
                symmetry_operations = []
                for _ in range(nsym):
                    rotation = []
                    displacement = []
                    while True:
                        if 'rotation' in f.readline():
                            break
                    for _ in range(3):
                        line = f.readline()
                        rotation.append([float(r) for r in line.split()[1:4]])
                    while True:
                        if 'displacement' in f.readline():
                            break
                    line = f.readline()
                    displacement = [float(d) for d in line.split()[1:4]]
                    symop = {'rotation': rotation,
                             'displacement': displacement}
                    self.symmetry_ops = symop
                self.symmetry = symmetry_operations
                print('Symmetry operations successfully read from %s' % f.name)
                print(self.cell.symmetry_ops)
                break

        # only close if we opened the file in this routine
        if _close:
            f.close()

    def get_hirsh_volrat(self):
        """
        Return the Hirshfeld volumes.
        """
        return self._hirsh_volrat

    def get_spins(self):
        """
        Return the spins from a plane-wave Mulliken analysis.
        """
        return self._spins

    def set_label(self, label):
        """The label is part of each seed, which in turn is a prefix
        in each CASTEP related file.
        """
        self._label = label

    def set_pspot(self, pspot, elems=None,
                  notelems=None,
                  clear=True,
                  suffix='usp'):
        """Quickly set all pseudo-potentials: Usually CASTEP psp are named
        like <Elem>_<pspot>.<suffix> so this function function only expects
        the <LibraryName>. It then clears any previous pseudopotential
        settings apply the one with <LibraryName> for each element in the
        atoms object. The optional elems and notelems arguments can be used
        to exclusively assign to some species, or to exclude with notelemens.

        Parameters ::

            - elems (None) : set only these elements
            - notelems (None): do not set the elements
            - clear (True): clear previous settings
            - suffix (usp): PP file suffix



        """

        if clear and not elems and not notelems:
            self.cell.species_pot.clear()
        for elem in set(self.atoms.get_chemical_symbols()):
            if elems is not None and elem not in elems:
                continue
            if notelems is not None and elem in notelems:
                continue
            self.cell.species_pot = (elem, '%s_%s.%s' % (elem, pspot, suffix))

    @_self_getter
    def get_forces(self, atoms):
        """Run CASTEP calculation if needed and return forces."""
        self.update(atoms)
        return np.array(self._forces)

    @_self_getter
    def get_total_energy(self, atoms):
        """Run CASTEP calculation if needed and return total energy."""
        self.update(atoms)
        return self._energy_total

    @_self_getter
    def get_free_energy(self, atoms):
        """Run CASTEP calculation if needed and return free energy.
           Only defined with smearing."""
        self.update(atoms)
        return self._energy_free

    @_self_getter
    def get_0K_energy(self, atoms):
        """Run CASTEP calculation if needed and return 0K energy.
           Only defined with smearing."""
        self.update(atoms)
        return self._energy_0K

    @_self_getter
    def get_potential_energy(self, atoms, force_consistent=False):
        # here for compatibility with ase/calculators/general.py
        # but accessing only _name variables
        """Return the total potential energy."""
        self.update(atoms)
        if force_consistent:
            # Assumption: If no dispersion correction is applied, then the
            # respective value will default to None as initialized.
            if self._dispcorr_energy_free is not None:
                return self._dispcorr_energy_free
            else:
                return self._energy_free
        else:
            if self._energy_0K is not None:
                if self._dispcorr_energy_0K is not None:
                    return self._dispcorr_energy_0K
                else:
                    return self._energy_0K
            else:
                if self._dispcorr_energy_total is not None:
                    return self._dispcorr_energy_total
                else:
                    return self._energy_total

    @_self_getter
    def get_stress(self, atoms):
        """Return the stress."""
        self.update(atoms)
        return self._stress

    @_self_getter
    def get_unit_cell(self, atoms):
        """Return the unit cell."""
        self.update(atoms)
        return self._unit_cell

    @_self_getter
    def get_kpoints(self, atoms):
        """Return the kpoints."""
        self.update(atoms)
        return self._kpoints

    @_self_getter
    def get_number_cell_constraints(self, atoms):
        """Return the number of cell constraints."""
        self.update(atoms)
        return self._number_of_cell_constraints
    
    def set_atoms(self, atoms):
        """Sets the atoms for the calculator and vice versa."""
        atoms.pbc = [True, True, True]
        self.__dict__['atoms'] = atoms.copy()
        self.atoms._calc = self

    def update(self, atoms):
        """Checks if atoms object or calculator changed and
        runs calculation if so.
        """
        if self.calculation_required(atoms):
            self.calculate(atoms)

    def calculation_required(self, atoms, _=None):
        """Checks wether anything changed in the atoms object or CASTEP
        settings since the last calculation using this instance.
        """
        # SPR: what happens with the atoms parameter here? Why don't we use it?
        # from all that I can tell we need to compare against atoms instead of
        # self.atoms
        # if not self.atoms == self._old_atoms:
        if not atoms == self._old_atoms:
            return True
        if self._old_param is None or self._old_cell is None:
            return True
        if not self.param._options == self._old_param._options:
            return True
        if not self.cell._options == self._old_cell._options:
            return True
        return False

    def calculate(self, atoms):
        """Write all necessary input file and call CASTEP."""
        self.prepare_input_files(atoms, force_write=self._force_write)
        if not self._prepare_input_only:
            self.run()
            self.read()

            # we need to push the old state here!
            # although run() pushes it, read() may change the atoms object
            # again.
            # yet, the old state is supposed to be the one AFTER read()
            self.push_oldstate()

    def push_oldstate(self):
        """This function pushes the current state of the (CASTEP) Atoms object
        onto the previous state. Or in other words after calling this function,
        calculation_required will return False and enquiry functions just
        report the current value, e.g. get_forces(), get_potential_energy().
        """
        # make a snapshot of all current input
        # to be able to test if recalculation
        # is necessary
        self._old_atoms = self.atoms.copy()
        self._old_param = deepcopy(self.param)
        self._old_cell = deepcopy(self.cell)

    def initialize(self, *args, **kwargs):
        """Just an alias for prepar_input_files to comply with standard
        function names in ASE.
        """
        self.prepare_input_files(*args, **kwargs)

    def prepare_input_files(self, atoms=None, force_write=None):
        """Only writes the input .cell and .param files and return
        This can be useful if one quickly needs to prepare input files
        for a cluster where no python or ASE is available. One can than
        upload the file manually and read out the results using
        Castep().read().
        """

        if self.param.reuse.value is None:
            if self._pedantic:
                print('You have not set e.g. calc.param.reuse = True')
                print('Reusing a previous calculation may save CPU time!\n')
                print(
                    'The interface will make sure by default, a .check exists')
                print(
                    'file before adding this statement to the .param file.\n')
        if self.param.num_dump_cycles.value is None:
            if self._pedantic:
                print('You have not set e.g. calc.param.num_dump_cycles = 0.')
                print('This can save you a lot of disk space. One only needs')
                print('*wvfn* if electronic convergence is not achieved.\n')
        from ase.io.castep import write_param

        if atoms is None:
            atoms = self.atoms
        else:
            self.atoms = atoms

        if force_write is None:
            force_write = self._force_write

        # if we have new instance of the calculator,
        # move existing results out of the way, first
        if (os.path.isdir(self._directory) and
                self._calls == 0 and
                self._rename_existing_dir):
            if os.listdir(self._directory) == []:
                os.rmdir(self._directory)
            else:
                # rename appending creation date of the directory
                ctime = time.localtime(os.lstat(self._directory).st_ctime)
                os.rename(self._directory, '%s.bak-%s' %
                          (self._directory,
                           time.strftime('%Y%m%d-%H%M%S', ctime)))

        # create work directory
        if not os.path.isdir(self._directory):
            os.makedirs(self._directory, 0o775)

        # we do this every time, not only upon first call
        # if self._calls == 0:
        self._fetch_pspots()

        cwd = os.getcwd()
        os.chdir(self._directory)

        # if _try_reuse is requested and this
        # is not the first run, we try to find
        # the .check file from the previous run
        # this is only necessary if _track_output
        # is set to true
        if self._try_reuse and self._calls > 0:
            if os.path.exists(self._check_file):
                self.param.reuse = self._check_file
            elif os.path.exists(self._castep_bin_file):
                self.param.reuse = self._castep_bin_file
        self._seed = self._build_castep_seed()
        self._check_file = '%s.check' % self._seed
        self._castep_bin_file = '%s.castep_bin' % self._seed
        self._castep_file = os.path.abspath('%s.castep' % self._seed)

        # write out the input file
        self._write_cell('%s.cell' % self._seed,
                         self.atoms, castep_cell=self.cell,
                         force_write=force_write)

        if self._export_settings:
            interface_options = self._opt
        else:
            interface_options = None
        write_param('%s.param' % self._seed, self.param,
                    check_checkfile=self._check_checkfile,
                    force_write=force_write,
                    interface_options=interface_options,)
        os.chdir(cwd)

    def _build_castep_seed(self):
        """Abstracts to construction of the final castep <seed>
        with and without _tracking_output.
        """
        if self._track_output:
            return '%s-%06d' % (self._label, self._calls)
        else:
            return '%s' % (self._label)

    def run(self):
        """Simply call castep. If the first .err file
        contains text, this will be printed to the screen.
        """
        # change to target directory
        cwd = os.getcwd()
        os.chdir(self._directory)
        self._calls += 1

        # run castep itself
        stdout, stderr = shell_stdouterr('%s %s' % (self._castep_command,
                                                    self._seed))
        if stdout:
            print('castep call stdout:\n%s' % stdout)
        if stderr:
            print('castep call stderr:\n%s' % stderr)

        # shouldn't it be called after read()???
        # self.push_oldstate()

        # check for non-empty error files
        err_file = '%s.0001.err' % self._seed
        if os.path.exists(err_file):
            err_file = open(err_file)
            self._error = err_file.read()
            err_file.close()
        os.chdir(cwd)
        if self._error:
            raise RuntimeError(self._error)

    def __repr__(self):
        """Returns generic, fast to capture representation of
        CASTEP settings along with atoms object.
        """
        expr = ''
        expr += '-----------------Atoms--------------------\n'
        if self.atoms is not None:
            expr += str('%20s\n' % self.atoms)
        else:
            expr += 'None\n'

        expr += '-----------------Param keywords-----------\n'
        expr += str(self.param)
        expr += '-----------------Cell keywords------------\n'
        expr += str(self.cell)
        expr += '-----------------Internal keys------------\n'
        for key in self.internal_keys:
            expr += '%20s : %s\n' % (key, self._opt[key])

        return expr

    def __getattr__(self, attr):
        """___getattr___ gets overloaded to reroute the internal keys
        and to be able to easily store them in in the param so that
        they can be read in again in subsequent calls.
        """
        if attr in self.internal_keys:
            return self._opt[attr]
        if attr in ['__repr__', '__str__']:
            raise AttributeError
        elif attr not in self.__dict__:
            raise AttributeError
        else:
            return self.__dict__[attr]

    def __setattr__(self, attr, value):
        """We overload the settattr method to make value assignment
        as pythonic as possible. Internal values all start with _.
        Value assigment is case insensitive!
        """

        if attr.startswith('_'):
            # internal variables all start with _
            # let's check first if they are close but not identical
            # to one of the switches, that the user accesses directly
            similars = difflib.get_close_matches(attr, self.internal_keys,
                                                 cutoff=0.9)
            if attr not in self.internal_keys and similars:
                print('Warning: You probably tried one of: %s' % similars)
                print('but typed %s' % attr)
            if attr in self.internal_keys:
                self._opt[attr] = value
                if attr == '_track_output':
                    if value:
                        self._try_reuse = True
                        if self._pedantic:
                            print('You switched _track_output on. This will')
                            print('consume a lot of disk-space. The interface')
                            print('also switched _try_reuse on, which will')
                            print('try to find the last check file. Set')
                            print('_try_reuse = False, if you need')
                            print('really separate calculations')
                    elif '_try_reuse' in self._opt and self._try_reuse:
                        self._try_reuse = False
                        if self._pedantic:
                            print('_try_reuse is set to False, too')
            else:
                self.__dict__[attr] = value
            return
        elif attr in ['atoms', 'cell', 'param']:
            if value is not None:
                if attr == 'atoms' and not isinstance(value, ase.atoms.Atoms):
                    raise TypeError(
                        '%s is not an instance of ase.atoms.Atoms.' % value)
                elif attr == 'cell' and not isinstance(value, CastepCell):
                    raise TypeError('%s is not an instance of CastepCell.' %
                                    value)
                elif attr == 'param' and not isinstance(value, CastepParam):
                    raise TypeError('%s is not an instance of CastepParam.' %
                                    value)
            # These 3 are accepted right-away, no matter what
            self.__dict__[attr] = value
            return
        elif attr in self.atoms_obj_keys:
            # keywords which clearly belong to the atoms object are
            # rerouted to go there
            self.atoms.__dict__[attr] = value
            return
        elif attr in self.atoms_keys:
            # CASTEP keywords that should go into the atoms object
            # itself are blocked
            print('Ignoring setings of "%s", since this has to be set\n'
                  'through the atoms object' % attr)
            return

        attr = attr.lower()
        if attr not in (list(self.cell._options.keys()) +
                        list(self.param._options.keys())):
            # what is left now should be meant to be a castep keyword
            # so we first check if it defined, and if not offer some error
            # correction
            similars = difflib.get_close_matches(
                attr,
                self.cell._options.keys() + self.param._options.keys())
            if similars:
                raise UserWarning('Option "%s" not known! You mean "%s"?' %
                                  (attr, similars[0]))
            else:
                raise UserWarning('Option "%s" is not known!' % attr)

        # here we know it must go into one of the component param or cell
        # so we first determine which one
        if attr in self.param._options.keys():
            comp = 'param'
        elif attr in self.cell._options.keys():
            comp = 'cell'
        else:
            raise UserWarning('Programming error: could not attach '
                              'the keyword to an input file')

        self.__dict__[comp].__setattr__(attr, value)

    def merge_param(self, param, overwrite=True, ignore_internal_keys=False):
        """Parse a param file and merge it into the current parameters."""
        INT_TOKEN = 'ASE_INTERFACE'
        if isinstance(param, CastepParam):
            for key, option in param._options.items():
                if option.value is not None:
                    self.param.__setattr__(key, option.value)
            return

        elif isinstance(param, basestring):
            param_file = open(param, 'r')
            _close = True

        else:
            # in this case we assume that we have a fileobj already, but check
            # for attributes in order to avoid extended EAFP blocks.
            param_file = param

            # look before you leap...
            attributes = ['name',
                          'close'
                          'readlines']

            for attr in attributes:
                if not hasattr(param_file, attr):
                    raise TypeError('"param" is neither CastepParam nor str '
                                    'nor valid fileobj')

            param = param_file.name
            _close = False

        # ok, we need to load the file beforehand into memory, seems like the
        # easiest way to do the BLOCK handling.
        lines = param_file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # note that i will point to the next line from now on
            i += 1

            # remove comments
            for comment_char in ['#', ';', '!']:
                if comment_char in line:
                    if INT_TOKEN in line:
                        # This block allows to read internal settings from
                        # a *param file
                        iline = line[line.index(INT_TOKEN) + len(INT_TOKEN):]
                        if (iline.split()[0] in self.internal_keys and
                                not ignore_internal_keys):
                            value = ' '.join(iline.split()[1:])
                            if value in _tf_table:
                                self._opt[iline.split()[0]] = _tf_table[value]
                            else:
                                self._opt[iline.split()[0]] = value
                    line = line[:line.index(comment_char)]

            # if nothing remains
            if not line.strip():
                continue

            if line == 'reuse':
                self.param.reuse.value = 'default'
                continue
            if line == 'continuation':
                self.param.continuation.value = 'default'
                continue

            # here comes the handling of the devel block (the only block so far
            # I know to be in the param file)
            if line.upper() == '%BLOCK DEVEL_CODE':
                key = 'devel_code'
                value = ''
                while True:
                    line = lines[i].strip()
                    i += 1
                    if line.upper() == '%ENDBLOCK DEVEL_CODE':
                        break
                    value += '\n{0}'.format(line)
                value = value.strip()

                if (not overwrite and
                        getattr(self.param, key).value is not None):
                    continue

                self.__setattr__(key, value)
                continue

            try:
                # we go for the regex split here
                key, value = [s.strip() for s in re.split(r'[:=]+', line)]
            except:
                print('Could not parse line %s of your param file: %s'
                      % (i, line))
                raise UserWarning('Seems to me malformed')

            if not overwrite and getattr(self.param, key).value is not None:
                continue
            self.__setattr__(key, value)

        if _close:
            param_file.close()

    def dryrun_ok(self, dryrun_flag='-dryrun'):
        """Starts a CASTEP run with the -dryrun flag [default]
        in a temporary and check wether all variables are initialized
        correctly. This is recommended for every bigger simulation.
        """
        from ase.io.castep import write_param

        temp_dir = tempfile.mkdtemp()
        curdir = os.getcwd()
        self._fetch_pspots(temp_dir)
        os.chdir(temp_dir)
        self._fetch_pspots(temp_dir)
        seed = 'dryrun'

        self._write_cell('%s.cell' % seed, self.atoms,
                         castep_cell=self.cell)
        # This part needs to be modified now that we rely on the new formats.py
        # interface
        if not os.path.isfile('%s.cell' % seed):
            print('%s.cell not written - aborting dryrun' % seed)
            return
        write_param('%s.param' % seed, self.param, )

        stdout, stderr = shell_stdouterr(('%s %s %s' % (self._castep_command,
                                                        seed,
                                                        dryrun_flag)))

        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        result_file = open('%s.castep' % seed)

        txt = result_file.read()
        ok_string = r'.*DRYRUN finished.*No problems found with input files.*'
        match = re.match(ok_string, txt, re.DOTALL)

        try:
            self._kpoints_used = int(
                re.search(
                    r'Number of kpoints used = *([0-9]+)', txt).group(1))
        except:
            print('Couldn\'t fetch number of kpoints from dryrun CASTEP file')

        err_file = '%s.0001.err' % seed
        if match is None and os.path.exists(err_file):
            err_file = open(err_file)
            self._error = err_file.read()
            err_file.close()

        result_file.close()
        os.chdir(curdir)
        shutil.rmtree(temp_dir)

        # re.match return None is the string does not match
        return match is not None

    # this could go into the Atoms() class at some point...
    def _get_number_in_species(self, at, atoms=None):
        """Return the number of the atoms within the set of it own
        species. If you are an ASE commiter: why not move this into
        ase.atoms.Atoms ?"""
        if atoms is None:
            atoms = self.atoms
        numbers = atoms.get_atomic_numbers()
        n = numbers[at]
        nis = numbers.tolist()[:at + 1].count(n)
        return nis

    def _get_absolute_number(self, species, nic, atoms=None):
        """This is the inverse function to _get_number in species."""
        if atoms is None:
            atoms = self.atoms
        ch = atoms.get_chemical_symbols()
        ch.reverse()
        total_nr = 0
        assert nic > 0, 'Number in species needs to be 1 or larger'
        while True:
            if ch.pop() == species:
                if nic == 1:
                    return total_nr
                nic -= 1
            total_nr += 1

    def _fetch_pspots(self, directory=None):
        """Put all specified pseudo-potentials into the working directory.
        """
        # should be a '==' right? Otherwise setting _castep_pp_path is not
        # honored.
        if (not os.environ.get('PSPOT_DIR', None) and
                self._castep_pp_path == os.path.abspath('.')):
            # By default CASTEP consults the environment variable
            # PSPOT_DIR. If this contains a list of colon separated
            # directories it will check those directories for pseudo-
            # potential files if not in the current directory.
            # Thus if PSPOT_DIR is set there is nothing left to do.
            # If however PSPOT_DIR was been accidentally set
            # (e.g. with regards to a different program)
            # setting CASTEP_PP_PATH to an explicit value will
            # still be honored.
            return

        if directory is None:
            directory = self._directory
        if not os.path.isdir(self._castep_pp_path):
            print('PSPs directory %s not found' % self._castep_pp_path)
        pspots = {}
        if self.cell.species_pot.value is not None:
            for line in self.cell.species_pot.value.split('\n'):
                line = line.split()
                if line:
                    pspots[line[0]] = line[1]
        for species in self.atoms.get_chemical_symbols():
            if not pspots or species not in pspots.keys():
                if self._pedantic:
                    print('Warning: you have no PP specified for %s.' %
                          species)
                    print('CASTEP will now generate an on-the-fly potentials.')
                    print('For sake of numerical consistency and efficiency')
                    print('this is discouraged.')
        if self.cell.species_pot.value:
            for (species, pspot) in pspots.items():
                orig_pspot_file = os.path.join(self._castep_pp_path, pspot)
                cp_pspot_file = os.path.join(directory, pspot)
                if (os.path.exists(orig_pspot_file) and
                        not os.path.exists(cp_pspot_file)):
                    if self._copy_pspots:
                        shutil.copy(orig_pspot_file, directory)
                    elif self._link_pspots:
                        os.symlink(orig_pspot_file, cp_pspot_file)
                    else:
                        if self._pedantic:
                            print("""\
Warning: PP files have neither been linked nor copied
to the working directory. Make sure to set the evironment
variable PSPOT_DIR accordingly!""")


def get_castep_version(castep_command):
    """This returns the version number as printed in the CASTEP banner.
       For newer CASTEP versions ( > 6.1) the --version command line option
       has been added; this will be attempted first.
    """
    temp_dir = tempfile.mkdtemp()
    jname = 'dummy_jobname'
    stdout, stderr = '', ''
    fallback_version = 16.  # CASTEP 16.0 and 16.1 report version wrongly
    try:
        stdout, stderr = subprocess.Popen(
            castep_command.split() + ['--version'],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE, cwd=temp_dir).communicate()
        if 'CASTEP version' not in stdout:
            stdout, stderr = subprocess.Popen(
                castep_command.split() + [jname],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE, cwd=temp_dir).communicate()
    except:
        msg = ''
        msg += 'Could not determine the version of your CASTEP binary \n'
        msg += 'This usually means one of the following \n'
        msg += '   * you do not have CASTEP installed \n'
        msg += '   * you have not set the CASTEP_COMMAND to call it \n'
        msg += '   * you have provided a wrong CASTEP_COMMAND. \n'
        msg += '     Make sure it is in your PATH\n\n'
        msg += stdout
        msg += stderr
        raise Exception(msg)
    if 'CASTEP version' in stdout:
        output_txt = stdout.split('\n')
        version_re = re.compile(r'CASTEP version:\s*([0-9\.]*)')
    else:
        output = open(os.path.join(temp_dir, '%s.castep' % jname))
        output_txt = output.readlines()
        output.close()
        version_re = re.compile(r'(?<=CASTEP version )[0-9.]*')
    shutil.rmtree(temp_dir)
    for line in output_txt:
        if 'CASTEP version' in line:
            try:
                return float(version_re.findall(line)[0])
            except ValueError:
                # Fallback for buggy --version on CASTEP 16.0, 16.1
                return fallback_version


def create_castep_keywords(castep_command, filename='castep_keywords.py',
                           force_write=True, path='.', fetch_only=None):
    """This function allows to fetch all available keywords from stdout
    of an installed castep binary. It furthermore collects the documentation
    to harness the power of (ipython) inspection and type for some basic
    type checking of input. All information is stored in two 'data-store'
    objects that are not distributed by default to avoid breaking the license
    of CASTEP.
    """
    # Takes a while ...
    # Fetch all allowed parameters
    # fetch_only : only fetch that many parameters (for testsuite only)
    code = {}
    suffixes = ['cell', 'param']
    for suffix in suffixes:
        code[suffix] = ''

    if os.path.exists(filename) and not force_write:
        print('CASTEP Options Module file exists.')
        print('You can overwrite it by calling')
        print('python castep.py -f [CASTEP_COMMAND].')
        return False

    # Not saving directly to file her to prevent half-generated files
    # which will cause problems on future runs

    from StringIO import StringIO

    fh = StringIO()
    fh.write('"""This file is generated by')
    fh.write('ase/calculators/castep.py\n')
    fh.write('and is not distributed with ASE to avoid breaking')
    fh.write('CASTEP copyright\n"""\n')
    fh.write('class Opt:\n')
    fh.write('    """"A CASTEP option"""\n')
    fh.write("""    def __init__(self):
        self.keyword = None
        self.level = None
        self.value = None
        self.type = None
    def clear(self):
        \"\"\"Reset the value of the option to None again\"\"\"
        self.value = None\n""")
    fh.write('    def __repr__(self):\n')
    fh.write('        expr = \'\'\n')
    fh.write('        if self.value:\n')
    fh.write('            expr += \'Option: %s(%s, %s):\\n%s\\n\'' +
             '% (self.keyword, self.type, self.level, self.value)\n')
    fh.write('        else:\n')
    fh.write('            expr += \'Option: %s[unset]\' % self.keyword\n')
    fh.write('            expr += \'(%s, %s)\' % (self.type, self.level)\n')
    fh.write('        return expr\n\n')
    fh.write("""class ComparableDict(dict):
    \"\"\"Extends a dict to make to sets of options comparable\"\"\"
    def __init__(self):
        dict.__init__(self)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        if not isinstance(other, ComparableDict):
            return False
        if set(self) - set(other):
            return False
        for key in sorted(self):
            if self[key].value != other[key].value:
                return False
        return True\n""")

    code['cell'] += '\n\nclass CastepCellDict(object):\n'
    code['param'] += '\n\nclass CastepParamDict(object):\n'

    types = []
    levels = []

    for suffix in suffixes:
        code[suffix] += '    """A flat object that holds %s options"""\n'\
            % suffix
        code[suffix] += '    def __init__(self):\n'
        code[suffix] += '        object.__init__(self)\n'
        code[suffix] += '        self._options = ComparableDict()\n'
    castep_version = get_castep_version(castep_command)

    help_all, _ = shell_stdouterr('%s -help all' % castep_command)

    # Filter out proper keywords
    try:
        # The old pattern does not math properly as in CASTEP as of v8.0 there
        # are some keywords for the semi-empircal dispersion correction (SEDC)
        # which also include numbers.
        if castep_version < 7.0:
            pattern = r'((?<=^ )[A-Z_]{2,}|(?<=^)[A-Z_]{2,})'
        else:
            pattern = r'((?<=^ )[A-Z_\d]{2,}|(?<=^)[A-Z_\d]{2,})'

        raw_options = re.findall(pattern, help_all, re.MULTILINE)
    except:
        print('Problem parsing: %s' % help_all)
        raise

    processed_options = 0
    for option in raw_options[:fetch_only]:
        doc, _ = shell_stdouterr('%s -help %s' % (castep_command, option))

        # Stand Back! I know regular expressions (http://xkcd.com/208/) :-)
        match = re.match(r'(?P<before_type>.*)Type: (?P<type>.+?)\s+' +
                         r'Level: (?P<level>[^ ]+)\n\s*\n' +
                         r'(?P<doc>.*?)(\n\s*\n|$)', doc, re.DOTALL)

        if match is not None:
            match = match.groupdict()
            processed_options += 1

            # JM: uncomment lines in following block to debug issues
            #     with keyword assignment during extraction process from CASTEP
            suffix = None
            if re.findall(r'PARAMETERS keywords:\n\n\s?None found', doc):
                suffix = 'cell'
            if re.findall(r'CELL keywords:\n\n\s?None found', doc):
                suffix = 'param'
            if suffix is None:
                print('%s -> not assigned to either'
                      ' CELL or PARAMETERS keywords' % option)

            sys.stdout.write('.')
            sys.stdout.flush()

            code[suffix] += '        opt_obj = Opt()\n'

            code[suffix] += ('        opt_obj.keyword = \'%s\'\n'
                             % option.lower())
            if 'type' in match:
                code[suffix] += ('        opt_obj.type = \'%s\'\n'
                                 % match['type'])
                if match['type'] not in types:
                    types.append(match['type'])
            else:
                raise Exception('Found no type for %s' % option)

            if 'level' in match:
                code[suffix] += ('        opt_obj.level = \'%s\'\n'
                                 % match['level'])
                if match['level'] not in levels:
                    levels.append(match['level'])
            else:
                raise Exception('Found no level for %s' % option)

            if 'doc' in match:
                code[suffix] += ('        opt_obj.__doc__ = """%s\n"""\n'
                                 % match['doc'])
            else:
                raise Exception('Found no doc string for %s' % option)
            code[suffix] += ('        opt_obj.value = None\n')

            code[suffix] += ('        self._options[\'%s\'] = opt_obj\n\n'
                             % option.lower())
            code[suffix] += ('        self.__dict__[\'%s\'] = opt_obj\n\n'
                             % option.lower())
        else:
            sys.stdout.write(doc)
            sys.stdout.flush()

            raise Exception('create_castep_keywords: Could not process %s'
                            % option)

    # write classes out
    for suffix in suffixes:
        fh.write(code[suffix])

    fh.write('types = %s\n' % types)
    fh.write('levels = %s\n' % levels)
    fh.write('castep_version = %s\n\n' % castep_version)

    fh_disk = open(os.path.join(path, filename), 'w')
    fh_disk.write(fh.getvalue())

    fh.close()
    fh_disk.close()

    print('\nCASTEP v%s, fetched %s keywords'
          % (castep_version, processed_options))
    return True


class CastepParam(object):

    """CastepParam abstracts the settings that go into the .param file"""

    def __init__(self, castep_keywords):
        object.__init__(self)
        castep_param_dict = castep_keywords.CastepParamDict()
        self._options = castep_param_dict._options
        self.__dict__.update(self._options)

    def __repr__(self):
        expr = ''
        if [x for x in self._options.values() if x.value is not None]:
            for key, option in sorted(self._options.items()):
                if option.value is not None:
                    expr += ('%20s : %s\n' % (key, option.value))
        else:
            expr += 'Default\n'
        return expr

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            self.__dict__[attr] = value
            return
        if attr not in self._options.keys():
            similars = difflib.get_close_matches(attr, self._options.keys())
            if similars:
                raise UserWarning(('Option "%s" not known! You mean "%s"?')
                                  % (attr, similars[0]))
            else:
                raise UserWarning('Option "%s" is not known!' % attr)
        attr = attr.lower()
        opt = self._options[attr]
        if not opt.type == 'Block' and isinstance(value, basestring):
            value = value.replace(':', ' ')
        if opt.type in ['Boolean (Logical)', 'Defined']:
            if False:
                pass
            else:
                try:
                    value = _tf_table[str(value).title()]
                except:
                    raise ConversionError('bool', attr, value)
                self._options[attr].value = value
        elif opt.type == 'String':
            if attr == 'reuse':
                if self._options['continuation'].value:
                    print('Cannot set reuse if continuation is set, and')
                    print('vice versa. Set the other to None, if you want')
                    print('this setting.')
                else:
                    if value is True:
                        self._options['reuse'].value = 'default'
                    else:
                        self._options['reuse'].value = str(value)
            elif attr == 'continuation':
                if self._options['reuse'].value:
                    print('Cannot set continuation if reuse is set, and')
                    print('vice versa. Set the other to None, if you want')
                    print('this setting.')
                else:
                    if value is True:
                        self._options['continuation'].value = 'default'
                    else:
                        self._options['continuation'].value = str(value)
            else:
                try:
                    value = str(value)
                except:
                    raise ConversionError('str', attr, value)
                self._options[attr].value = value
        elif opt.type == 'Integer':
            if False:
                pass
            else:
                try:
                    value = int(value)
                except:
                    raise ConversionError('int', attr, value)
                self._options[attr].value = value
        elif opt.type == 'Real':
            try:
                value = float(value)
            except:
                raise ConversionError('float', attr, value)
            self._options[attr].value = value
        # Newly added "Vector" options
        elif opt.type == 'Integer Vector':
            # crashes if value is not a string
            if isinstance(value, basestring):
                if ',' in value:
                    value = value.replace(',', ' ')
            if isinstance(value, basestring) and len(value.split()) == 3:
                try:
                    [int(x) for x in value.split()]
                except:
                    raise ConversionError('int vector', attr, value)
                opt.value = value
            else:
                print('Wrong format for Integer Vector: expected I I I')
                print('and you said %s' % value)
        elif opt.type == 'Real Vector':
            if ',' in value:
                value = value.replace(',', ' ')
            if isinstance(value, basestring) and len(value.split()) == 3:
                try:
                    [float(x) for x in value.split()]
                except:
                    raise ConversionError('float vector', attr, value)
                opt.value = value
            else:
                print('Wrong format for Real Vector: expected R R R')
                print('and you said %s' % value)
        elif opt.type == 'Physical':
            # Usage of the CASTEP unit system is not fully implemented
            # for now.
            # We assume, that the user is happy with setting/getting the
            # CASTEP default units refer to http://goo.gl/bqYf2
            # page 13, accessed Apr 6, 2011

            # However if a unit is present it will be dealt with

            # this crashes if non-string types are passed
            if isinstance(value, basestring):
                if len(value.split()) > 1:
                    value = value.split(' ', 1)[0]
            try:
                value = float(value)
            except:
                raise ConversionError('float', attr, value)
            self._options[attr].value = value

        elif opt.type in ['Block']:
            self._options[attr].value = value
        else:
            raise RuntimeError('Caught unhandled option: %s = %s'
                               % (attr, value))


class CastepCell(object):

    """CastepCell abstracts all setting that go into the .cell file"""

    def __init__(self, castep_keywords):
        object.__init__(self)
        castep_cell_dict = castep_keywords.CastepCellDict()
        self._options = castep_cell_dict._options
        self.__dict__.update(self._options)

    def __repr__(self):
        expr = ''
        if [x for x in self._options.values() if x.value is not None]:
            for key, option in sorted(self._options.items()):
                if option.value is not None:
                    expr += ('%20s : %s\n' % (key, option.value))
        else:
            expr += 'Default\n'

        return expr

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            self.__dict__[attr] = value
            return

        if attr not in self._options.keys():
            similars = difflib.get_close_matches(attr, self._options.keys())
            if similars:
                raise UserWarning(('Option "%s" not known! You mean "%s"?')
                                  % (attr, similars[0]))
            else:
                raise UserWarning('Option "%s" is not known!' % attr)
            return
        attr = attr.lower()
        # Handling the many cases where kpoint_ and kpoints_ are treated
        # equivalently
        if 'kpoint_' in attr:
            if attr.replace('kpoint_', 'kpoints_') in self._options:
                attr = attr.replace('kpoint_', 'kpoints_')

        # CASTEP < 16 lists kpoints_mp_grid as type "Integer" -> convert to
        # "Integer Vector"
        if attr == 'kpoints_mp_grid':
            self._options[attr].type = 'Integer Vector'

        opt = self._options[attr]
        if not opt.type == 'Block' and isinstance(value, basestring):
            value = value.replace(':', ' ')
        if opt.type in ['Boolean (Logical)', 'Defined']:
            try:
                value = _tf_table[str(value).title()]
            except:
                raise ConversionError('bool', attr, value)
            self._options[attr].value = value
        elif opt.type == 'String':
            if False:
                pass
            else:
                try:
                    value = str(value)
                except:
                    raise ConversionError('str', attr, value)
            self._options[attr].value = value
        elif opt.type == 'Integer':
            try:
                value = int(value)
            except:
                raise ConversionError('int', attr, value)
            self._options[attr].value = value
        elif opt.type == 'Real':
            try:
                value = float(value)
            except:
                raise ConversionError('float', attr, value)
            self._options[attr].value = value
        # Newly added "Vector" options
        elif opt.type == 'Integer Vector':
            if ',' in value:
                value = value.replace(',', ' ')
            if isinstance(value, basestring) and len(value.split()) == 3:
                try:
                    [int(x) for x in value.split()]
                except:
                    raise ConversionError('int vector', attr, value)
                opt.value = value
            else:
                print('Wrong format for Integer Vector: expected I I I')
                print('and you said %s' % value)
        elif opt.type == 'Real Vector':
            if ',' in value:
                value = value.replace(',', ' ')
            if isinstance(value, basestring) and len(value.split()) == 3:
                try:
                    [float(x) for x in value.split()]
                except:
                    raise ConversionError('float vector', attr, value)
                opt.value = value
            else:
                print('Wrong format for Real Vector: expected R R R')
                print('and you said %s' % value)
        elif opt.type == 'Physical':
            # Usage of the CASTEP unit system is not fully implemented
            # for now.
            # We assume, that the user is happy with setting/getting the
            # CASTEP default units refer to http://goo.gl/bqYf2
            # page 13, accessed Apr 6, 2011

            # However if a unit is present it will be dealt with

            # this crashes if non-string types are passed
            if isinstance(value, basestring):
                if len(value.split()) > 1:
                    value = value.split(' ', 1)[0]
            try:
                value = float(value)
            except:
                raise ConversionError('float', attr, value)
            self._options[attr].value = value
        elif opt.type == 'Block':
            if attr == 'species_pot':
                if not isinstance(value, tuple) or len(value) != 2:
                    print('Please specify pseudopotentials in python as')
                    print('a tuple, like:')
                    print('(species, file), e.g. ("O", "path-to/O_OTFG.usp")')
                    print('Anything else will be ignored')
                else:
                    if self.__dict__['species_pot'].value is None:
                        self.__dict__['species_pot'].value = ''
                    self.__dict__['species_pot'].value = \
                        re.sub(r'\n?\s*%s\s+.*' % value[0], '',
                               self.__dict__['species_pot'].value)
                    if value[1]:
                        self.__dict__['species_pot'].value += '\n%s %s' % value

                    # now sort lines as to match the CASTEP output
                    pspots = self.__dict__['species_pot'].value.split('\n')
                    # throw out empty lines
                    pspots = [x for x in pspots if x]

                    # sort based on atomic numbers
                    pspots.sort(key=lambda x: ase.data.atomic_numbers[
                        re.split('[\s:]', x, 1)[0]])

                    # rejoin; the first blank-line
                    # makes the print(calc) output look prettier
                    self.__dict__['species_pot'].value = \
                        '\n' + '\n'.join(pspots)
                    return

            # probably we will support this at some point...
#            elif attr == 'nonlinear_constraints':
#                if type(value) is not str:
#                    print("Please specify nonlinear constraint in python as "
#                          "a string")
#                    print("Anything else will be ignored")
#                else:
#                    if self.__dict__['nonlinear_constraints'].value is None:
#                        self._options['nonlinear_constraints'].value = ''
#                    self.__dict__['nonlinear_constraints'].value = (
#                        str(self.__dict__['nonlinear_constraints'].value) +
#                        ' \n')
#                    self._options[attr].value += value
#                    return

            elif attr == 'symmetry_ops':
                if not isinstance(value, tuple) \
                   or not len(value) == 2 \
                   or not value[0].shape[1:] == (3, 3) \
                   or not value[1].shape[1:] == (3,) \
                   or not value[0].shape[0] == value[1].shape[0]:
                    print('Invalid symmetry_ops block, skipping')
                    return
                # Now on to print...
                text_block = ''
                for op_i, (op_rot, op_tranls) in enumerate(zip(*value)):
                    text_block += '\n'.join([' '.join([str(x) for x in row])
                                             for row in op_rot])
                    text_block += '\n'
                    text_block += ' '.join([str(x) for x in op_tranls])
                    text_block += '\n'
                value = text_block

            elif attr in ['positions_abs_intermediate',
                          'positions_abs_product']:
                if not isinstance(value, ase.atoms.Atoms):
                    raise UserWarning('castep.cell.%s expects Atoms object' %
                                      attr)
                target = self.__dict__[attr]
                target.value = ''
                for elem, pos in zip(value.get_chemical_symbols(),
                                     value.get_positions()):
                    target.value += ('%4s %9.6f %9.6f %9.6f\n' % (elem,
                                                                  pos[0],
                                                                  pos[1],
                                                                  pos[2]))
                return
            else:
                # For generic, non-implemented blocks all we want is to
                # store the lines and reprint them without any changes later
                value = '\n'.join(value)
            self._options[attr].value = value
        else:
            raise RuntimeError('Caught unhandled option: %s = %s'
                               % (attr, value))


class ConversionError(Exception):

    """Print customized error for options that are not converted correctly
    and point out that they are maybe not implemented, yet"""

    def __init__(self, key_type, attr, value):
        Exception.__init__(self)
        self.key_type = key_type
        self.value = value
        self.attr = attr

    def __str__(self):
        return 'Could not convert %s = %s to %s\n' \
            % (self.attr, self.value, self.key_type) \
            + 'This means you either tried to set a value of the wrong\n'\
            + 'type or this keyword needs some special care. Please feel\n'\
            + 'to add it to the corresponding __setattr__ method and send\n'\
            + 'the patch to %s, so we can all benefit.' % (contact_email)


def get_castep_pp_path(castep_pp_path=''):
    """Abstract the quest for a CASTEP PSP directory."""
    if castep_pp_path:
        return os.path.abspath(os.path.expanduser(castep_pp_path))
    elif 'CASTEP_PP_PATH' in os.environ:
        return os.environ['CASTEP_PP_PATH']
    else:
        return os.path.abspath('.')


def get_castep_command(castep_command=''):
    """Abstract the quest for a castep_command string."""
    if castep_command:
        return castep_command
    elif 'CASTEP_COMMAND' in os.environ:
        return os.environ['CASTEP_COMMAND']
    else:
        return 'castep'


def shell_stdouterr(raw_command):
    """Abstracts the standard call of the commandline, when
    we are only interested in the stdout and stderr
    """
    stdout, stderr = subprocess.Popen(raw_command,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True,
                                      shell=True).communicate()
    return stdout.strip(), stderr.strip()


def import_castep_keywords(castep_command=''):
    try:
        # Adapt import path to give local versions of castep_keywords
        # a higher priority, assuming that personal folder will be
        # standardized at ~/.ase, watch [ase-developers]
        sys.path[:0] = ['',
                        os.path.expanduser('~/.ase'),
                        os.path.join(ase.__path__[0], 'calculators')]
        import castep_keywords
    except ImportError:
        print("""    Generating castep_keywords.py ... hang on.
    The castep_keywords.py contains abstractions for CASTEP input
    parameters (for both .cell and .param input files), including some
    format checks and descriptions. The latter are extracted from the
    internal online help facility of a CASTEP binary, thus allowing to
    easily keep the calculator synchronized with (different versions of)
    the CASTEP code. Consequently, avoiding licensing issues (CASTEP is
    distributed commercially by accelrys), we consider it wise not to
    provide castep_keywords.py in the first place.
""")
        create_castep_keywords(get_castep_command(castep_command))
        print("""\n\n    Stored castep_keywords.py in %s.
                 Copy castep_keywords.py to your
    ASE installation under ase/calculators for system-wide installation
""" % os.path.abspath(os.path.curdir))
        print("""\n\n    Using a *nix OS this can be a simple as\nmv %s %s""" %
              (os.path.join(os.path.abspath(os.path.curdir),
                            'castep_keywords.py'),
               os.path.join(os.path.dirname(ase.__file__),
                            'calculators')))

        import castep_keywords
    finally:
        del sys.path[:3]
    return castep_keywords


if __name__ == '__main__':
    print('When called directly this calculator will fetch all available')
    print('keywords from the binarys help function into a castep_keywords.py')
    print('in the current directory %s' % os.getcwd())
    print('For system wide usage, it can be copied into an ase installation')
    print('at ASE/calculators.\n')
    print('This castep_keywords.py usually only needs to be generated once')
    print('for a CASTEP binary/CASTEP version.')

    import optparse
    parser = optparse.OptionParser()
    parser.add_option(
        '-f', '--force-write', dest='force_write',
        help='Force overwriting existing castep_keywords.py', default=False,
        action='store_true')
    (options, args) = parser.parse_args()

    if args:
        opt_castep_command = ''.join(args)
    else:
        opt_castep_command = ''
    generated = create_castep_keywords(get_castep_command(opt_castep_command),
                                       force_write=options.force_write)

    if generated:
        try:
            exec(compile(open('castep_keywords.py').read(),
                         'castep_keywords.py', 'exec'))
        except Exception as e:
            print(e)
            print('Ooops, something went wrong with the CASTEP keywords')
        else:
            print('Import works. Looking good!')
