"""Module to read and write atoms in cif file format.

See http://www.iucr.org/resources/cif/spec/version1.1/cifsyntax for a
description of the file format.  STAR extensions as save frames,
global blocks, nested loops and multi-data values are not supported.
"""

import re
import shlex
import warnings

import numpy as np

from ase.parallel import paropen
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data
from ase.utils import basestring


# Old conventions:
old_spacegroup_names = {'Abm2': 'Aem2',
                        'Aba2': 'Aea2',
                        'Cmca': 'Cmce',
                        'Cmma': 'Cmme',
                        'Ccca': 'Ccc1'}


def convert_value(value):
    """Convert CIF value string to corresponding python type."""
    value = value.strip()
    if re.match('(".*")|(\'.*\')$', value):
        return value[1:-1]
    elif re.match(r'[+-]?\d+$', value):
        return int(value)
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$', value):
        return float(value)
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\(\d+\)$',
                  value):
        return float(value[:value.index('(')])  # strip off uncertainties
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\(\d+$',
                  value):
        warnings.warn('Badly formed number: "{0}"'.format(value))
        return float(value[:value.index('(')])  # strip off uncertainties
    else:
        return value


def parse_multiline_string(lines, line):
    """Parse semicolon-enclosed multiline string and return it."""
    assert line[0] == ';'
    strings = [line[1:].lstrip()]
    while True:
        line = lines.pop().strip()
        if line[:1] == ';':
            break
        strings.append(line)
    return '\n'.join(strings).strip()


def parse_singletag(lines, line):
    """Parse a CIF tag (entries starting with underscore). Returns
    a key-value pair."""
    kv = line.split(None, 1)
    if len(kv) == 1:
        key = line
        line = lines.pop().strip()
        while not line or line[0] == '#':
            line = lines.pop().strip()
        if line[0] == ';':
            value = parse_multiline_string(lines, line)
        else:
            value = line
    else:
        key, value = kv
    return key, convert_value(value)


def parse_loop(lines):
    """Parse a CIF loop. Returns a dict with column tag names as keys
    and a lists of the column content as values."""
    header = []
    line = lines.pop().strip()
    while line.startswith('_'):
        header.append(line.lower())
        line = lines.pop().strip()
    columns = dict([(h, []) for h in header])

    tokens = []
    while True:
        lowerline = line.lower()
        if (not line or
            line.startswith('_') or
            lowerline.startswith('data_') or
            lowerline.startswith('loop_')):
            break
        if line.startswith('#'):
            line = lines.pop().strip()
            continue
        if line.startswith(';'):
            t = [parse_multiline_string(lines, line)]
        else:
            if len(header) == 1:
                t = [line]
            else:
                t = shlex.split(line, posix=False)

        line = lines.pop().strip()

        tokens.extend(t)
        if len(tokens) < len(columns):
            continue
        if len(tokens) == len(header):
            for h, t in zip(header, tokens):
                columns[h].append(convert_value(t))
        else:
            warnings.warn('Wrong number of tokens: {0}'.format(tokens))
        tokens = []
    if line:
        lines.append(line)
    return columns


def parse_items(lines, line):
    """Parse a CIF data items and return a dict with all tags."""
    tags = {}
    while True:
        if not lines:
            break
        line = lines.pop()
        if not line:
            break
        line = line.strip()
        lowerline = line.lower()
        if not line or line.startswith('#'):
            continue
        elif line.startswith('_'):
            key, value = parse_singletag(lines, line)
            tags[key.lower()] = value
        elif lowerline.startswith('loop_'):
            tags.update(parse_loop(lines))
        elif lowerline.startswith('data_'):
            if line:
                lines.append(line)
            break
        elif line.startswith(';'):
            parse_multiline_string(lines, line)
        else:
            raise ValueError('Unexpected CIF file entry: "{0}"'.format(line))
    return tags


def parse_block(lines, line):
    """Parse a CIF data block and return a tuple with the block name
    and a dict with all tags."""
    assert line.lower().startswith('data_')
    blockname = line.split('_', 1)[1].rstrip()
    tags = parse_items(lines, line)
    return blockname, tags


def parse_cif(fileobj):
    """Parse a CIF file. Returns a list of blockname and tag
    pairs. All tag names are converted to lower case."""
    if isinstance(fileobj, basestring):
        fileobj = open(fileobj)
    lines = [''] + fileobj.readlines()[::-1]  # all lines (reversed)
    blocks = []
    while True:
        if not lines:
            break
        line = lines.pop()
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        blocks.append(parse_block(lines, line))
    return blocks


def tags2atoms(tags, store_tags=False, primitive_cell=False,
               subtrans_included=True):
    """Returns an Atoms object from a cif tags dictionary.  See read_cif()
    for a description of the arguments."""
    if primitive_cell and subtrans_included:
        raise RuntimeError(
            'Primitive cell cannot be determined when sublattice translations '
            'are included in the symmetry operations listed in the CIF file, '
            'i.e. when `subtrans_included` is True.')

    a = tags['_cell_length_a']
    b = tags['_cell_length_b']
    c = tags['_cell_length_c']
    alpha = tags['_cell_angle_alpha']
    beta = tags['_cell_angle_beta']
    gamma = tags['_cell_angle_gamma']

    scaled_positions = np.array([tags['_atom_site_fract_x'],
                                 tags['_atom_site_fract_y'],
                                 tags['_atom_site_fract_z']]).T

    symbols = []
    if '_atom_site_type_symbol' in tags:
        labels = tags['_atom_site_type_symbol']
    else:
        labels = tags['_atom_site_label']
    for s in labels:
        # Strip off additional labeling on chemical symbols
        m = re.search(r'([A-Z][a-z]?)', s)
        symbol = m.group(0)
        symbols.append(symbol)

    # Symmetry specification, see
    # http://www.iucr.org/resources/cif/dictionaries/cif_sym for a
    # complete list of official keys.  In addition we also try to
    # support some commonly used depricated notations
    no = None
    if '_space_group.it_number' in tags:
        no = tags['_space_group.it_number']
    elif '_space_group_it_number' in tags:
        no = tags['_space_group_it_number']
    elif '_symmetry_int_tables_number' in tags:
        no = tags['_symmetry_int_tables_number']

    symbolHM = None
    if '_space_group.Patterson_name_h-m' in tags:
        symbolHM = tags['_space_group.patterson_name_h-m']
    elif '_symmetry_space_group_name_h-m' in tags:
        symbolHM = tags['_symmetry_space_group_name_h-m']
    elif '_space_group_name_h-m_alt' in tags:
        symbolHM = tags['_space_group_name_h-m_alt']

    if symbolHM is not None:
        symbolHM = old_spacegroup_names.get(symbolHM.strip(), symbolHM)

    for name in ['_space_group_symop_operation_xyz',
                 '_space_group_symop.operation_xyz',
                 '_symmetry_equiv_pos_as_xyz']:
        if name in tags:
            sitesym = tags[name]
            break
    else:
        sitesym = None

    spacegroup = 1
    if sitesym is not None:
        subtrans = [(0.0, 0.0, 0.0)] if subtrans_included else None
        spacegroup = spacegroup_from_data(
            no=no, symbol=symbolHM, sitesym=sitesym, subtrans=subtrans)
    elif no is not None:
        spacegroup = no
    elif symbolHM is not None:
        spacegroup = symbolHM
    else:
        spacegroup = 1

    if store_tags:
        kwargs = {'info': tags.copy()}
    else:
        kwargs = {}

    if 'D' in symbols:
        deuterium = [symbol == 'D' for symbol in symbols]
        symbols = [symbol if symbol != 'D' else 'H' for symbol in symbols]
    else:
        deuterium = False

    atoms = crystal(symbols, basis=scaled_positions,
                    cellpar=[a, b, c, alpha, beta, gamma],
                    spacegroup=spacegroup, primitive_cell=primitive_cell,
                    **kwargs)
    if deuterium:
        masses = atoms.get_masses()
        masses[atoms.numbers == 1] = 1.00783
        masses[deuterium] = 2.01355
        atoms.set_masses(masses)

    return atoms


def read_cif(fileobj, index, store_tags=False, primitive_cell=False,
             subtrans_included=True):
    """Read Atoms object from CIF file. *index* specifies the data
    block number or name (if string) to return.

    If *index* is None or a slice object, a list of atoms objects will
    be returned. In the case of *index* is *None* or *slice(None)*,
    only blocks with valid crystal data will be included.

    If *store_tags* is true, the *info* attribute of the returned
    Atoms object will be populated with all tags in the corresponding
    cif data block.

    If *primitive_cell* is true, the primitive cell will be built instead
    of the conventional cell.

    If *subtrans_included* is true, sublattice translations are
    assumed to be included among the symmetry operations listed in the
    CIF file (seems to be the common behaviour of CIF files).
    Otherwise the sublattice translations are determined from setting
    1 of the extracted space group.  A result of setting this flag to
    true, is that it will not be possible to determine the primitive
    cell.
    """
    blocks = parse_cif(fileobj)
    # Find all CIF blocks with valid crystal data
    images = []
    for name, tags in blocks:
        try:
            atoms = tags2atoms(tags, store_tags, primitive_cell,
                               subtrans_included)
            images.append(atoms)
        except KeyError:
            pass
    for atoms in images[index]:
        yield atoms


def split_chem_form(comp_name):
    """Returns e.g. AB2  as ['A', '1', 'B', '2']"""
    split_form = re.findall(r'[A-Z][a-z]*|\d+',
                            re.sub('[A-Z][a-z]*(?![\da-z])',
                                   r'\g<0>1', comp_name))
    return split_form


def write_cif(fileobj, images, format='default'):
    """Write *images* to CIF file."""
    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'w')

    if hasattr(images, 'get_positions'):
        images = [images]

    for i, atoms in enumerate(images):
        fileobj.write('data_image%d\n' % i)

        a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()

        if format == 'mp':

            comp_name = atoms.get_chemical_formula(mode='reduce')
            sf = split_chem_form(comp_name)
            formula_sum = ''
            ii = 0
            while ii < len(sf):
                formula_sum = formula_sum + ' ' + sf[ii] + sf[ii + 1]
                ii = ii + 2

            formula_sum = str(formula_sum)
            fileobj.write('_chemical_formula_structural       %s\n' %
                          atoms.get_chemical_formula(mode='reduce'))
            fileobj.write('_chemical_formula_sum      "%s"\n' % formula_sum)

        fileobj.write('_cell_length_a       %g\n' % a)
        fileobj.write('_cell_length_b       %g\n' % b)
        fileobj.write('_cell_length_c       %g\n' % c)
        fileobj.write('_cell_angle_alpha    %g\n' % alpha)
        fileobj.write('_cell_angle_beta     %g\n' % beta)
        fileobj.write('_cell_angle_gamma    %g\n' % gamma)
        fileobj.write('\n')

        if atoms.pbc.all():
            fileobj.write('_symmetry_space_group_name_H-M    %s\n' % '"P 1"')
            fileobj.write('_symmetry_int_tables_number       %d\n' % 1)
            fileobj.write('\n')

            fileobj.write('loop_\n')
            fileobj.write('  _symmetry_equiv_pos_as_xyz\n')
            fileobj.write("  'x, y, z'\n")
            fileobj.write('\n')

        fileobj.write('loop_\n')

        if format == 'mp':
            fileobj.write('  _atom_site_type_symbol\n')
            fileobj.write('  _atom_site_label\n')
            fileobj.write('   _atom_site_symmetry_multiplicity\n')
            fileobj.write('  _atom_site_fract_x\n')
            fileobj.write('  _atom_site_fract_y\n')
            fileobj.write('  _atom_site_fract_z\n')
            fileobj.write('  _atom_site_occupancy\n')
        else:
            fileobj.write('  _atom_site_label\n')
            fileobj.write('  _atom_site_occupancy\n')
            fileobj.write('  _atom_site_fract_x\n')
            fileobj.write('  _atom_site_fract_y\n')
            fileobj.write('  _atom_site_fract_z\n')
            fileobj.write('  _atom_site_thermal_displace_type\n')
            fileobj.write('  _atom_site_B_iso_or_equiv\n')
            fileobj.write('  _atom_site_type_symbol\n')

        scaled = atoms.get_scaled_positions()
        no = {}
        for i, atom in enumerate(atoms):
            symbol = atom.symbol
            if symbol in no:
                no[symbol] += 1
            else:
                no[symbol] = 1
            if format == 'mp':
                fileobj.write(
                    '  %-2s  %4s  %4s  %7.5f  %7.5f  %7.5f  %6.1f\n' %
                    (symbol, symbol + str(no[symbol]), 1,
                     scaled[i][0], scaled[i][1], scaled[i][2], 1.0))
            else:
                fileobj.write(
                    '  %-8s %6.4f %7.5f  %7.5f  %7.5f  %4s  %6.3f  %s\n' %
                    ('%s%d' % (symbol, no[symbol]),
                     1.0,
                     scaled[i][0],
                     scaled[i][1],
                     scaled[i][2],
                     'Biso',
                     1.0,
                     symbol))
