"""
Extended XYZ support

Read/write files in "extended" XYZ format, storing additional
per-configuration information as key-value pairs on the XYZ
comment line, and additional per-atom properties as extra columns.

See http://jrkermode.co.uk/quippy/io.html#extendedxyz for a full
description of the Extended XYZ file format.

Contributed by James Kermode <james.kermode@gmail.com>
"""

from __future__ import print_function

from itertools import islice
import re
import numpy as np

from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spacegroup.spacegroup import Spacegroup
from ase.parallel import paropen
from ase.utils import basestring

__all__ = ['read_xyz', 'write_xyz', 'iread_xyz']

PROPERTY_NAME_MAP = {'positions': 'pos',
                     'numbers': 'Z',
                     'charges': 'charge',
                     'symbols': 'species'}

REV_PROPERTY_NAME_MAP = dict(zip(PROPERTY_NAME_MAP.values(),
                                 PROPERTY_NAME_MAP.keys()))

KEY_QUOTED_VALUE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_-]*)' +
                              r'\s*=\s*["\{\}]([^"\{\}]+)["\{\}]\s*')
KEY_VALUE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_]*)\s*=' +
                       r'\s*([^\s]+)\s*')
KEY_RE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_-]*)\s*')

UNPROCESSED_KEYS = ['uid']


def key_val_str_to_dict(s):
    """
    Parse strings in the form 'key1=value1 key2="quoted value"'
    """
    d = {}
    s = s.strip()
    while True:
        # Match quoted string first, then fall through to plain key=value
        m = KEY_QUOTED_VALUE.match(s)
        if m is None:
            m = KEY_VALUE.match(s)
            if m is not None:
                s = KEY_VALUE.sub('', s, 1)
            else:
                # Just a key with no value
                m = KEY_RE.match(s)
                if m is not None:
                    s = KEY_RE.sub('', s, 1)
        else:
            s = KEY_QUOTED_VALUE.sub('', s, 1)

        if m is None:
            break        # No more matches

        key = m.group(1)
        try:
            value = m.group(2)
        except IndexError:
            # default value is 'T' (True)
            value = 'T'

        if key.lower() not in UNPROCESSED_KEYS:
            # Try to convert to (arrays of) floats, ints
            try:
                numvalue = []
                for x in value.split():
                    if x.find('.') == -1:
                        numvalue.append(int(float(x)))
                    else:
                        numvalue.append(float(x))
                if len(numvalue) == 1:
                    numvalue = numvalue[0]         # Only one number
                elif len(numvalue) == 9:
                    # special case: 3x3 matrix, fortran ordering
                    numvalue = np.array(numvalue).reshape((3, 3), order='F')
                else:
                    numvalue = np.array(numvalue)  # vector
                value = numvalue
            except (ValueError, OverflowError):
                pass

            # Parse boolean values: 'T' -> True, 'F' -> False,
            #                       'T T F' -> [True, True, False]
            if isinstance(value, basestring):
                str_to_bool = {'T': True, 'F': False}

                if len(value.split()) > 1:
                    if all([x in str_to_bool.keys() for x in value.split()]):
                        value = [str_to_bool[x] for x in value.split()]
                elif value in str_to_bool:
                    value = str_to_bool[value]

        d[key] = value

    return d


def key_val_dict_to_str(d, sep=' '):
    """
    Convert atoms.info dictionary to extended XYZ string representation
    """
    if len(d) == 0:
        return ''
    s = ''
    type_val_map = {(bool, True): 'T',
                    (bool, False): 'F',
                    (np.bool_, True): 'T',
                    (np.bool_, False): 'F'}

    s = ''
    for key in d.keys():
        val = d[key]
        if isinstance(val, dict):
            continue
        if hasattr(val, '__iter__'):
            val = np.array(val)
            val = ' '.join(str(type_val_map.get((type(x), x), x))
                           for x in val.reshape(val.size, order='F'))
            val.replace('[', '')
            val.replace(']', '')
        elif isinstance(val, Spacegroup):
            val = val.symbol
        else:
            val = type_val_map.get((type(val), val), val)

        if val is None:
            s = s + '%s%s' % (key, sep)
        elif isinstance(val, basestring) and ' ' in val:
            s = s + '%s="%s"%s' % (key, val, sep)
        else:
            s = s + '%s=%s%s' % (key, str(val), sep)

    return s.strip()


def parse_properties(prop_str):
    """
    Parse extended XYZ properties format string

    Format is "[NAME:TYPE:NCOLS]...]", e.g. "species:S:1:pos:R:3".
    NAME is the name of the property.
    TYPE is one of R, I, S, L for real, integer, string and logical.
    NCOLS is number of columns for that property.
    """

    properties = {}
    properties_list = []
    dtypes = []
    converters = []

    fields = prop_str.split(':')

    def parse_bool(x):
        """
        Parse bool to string
        """
        return {'T': True, 'F': False,
                'True': True, 'False': False}.get(x)

    fmt_map = {'R': ('d', float),
               'I': ('i', int),
               'S': (object, str),
               'L': ('bool', parse_bool)}

    for name, ptype, cols in zip(fields[::3],
                                 fields[1::3],
                                 [int(x) for x in fields[2::3]]):
        if ptype not in ('R', 'I', 'S', 'L'):
            raise ValueError('Unknown property type: ' + ptype)
        ase_name = REV_PROPERTY_NAME_MAP.get(name, name)

        dtype, converter = fmt_map[ptype]
        if cols == 1:
            dtypes.append((name, dtype))
            converters.append(converter)
        else:
            for c in range(cols):
                dtypes.append((name + str(c), dtype))
                converters.append(converter)

        properties[name] = (ase_name, cols)
        properties_list.append(name)

    dtype = np.dtype(dtypes)
    return properties, properties_list, dtype, converters


def _read_xyz_frame(lines, natoms):
    # comment line
    line = next(lines)
    info = key_val_str_to_dict(line)

    pbc = None
    if 'pbc' in info:
        pbc = info['pbc']
        del info['pbc']
    elif 'Lattice' in info:
        # default pbc for extxyz file containing Lattice
        # is True in all directions
        pbc = [True, True, True]

    cell = None
    if 'Lattice' in info:
        # NB: ASE cell is transpose of extended XYZ lattice
        cell = info['Lattice'].T
        del info['Lattice']

    if 'Properties' not in info:
        # Default set of properties is atomic symbols and positions only
        info['Properties'] = 'species:S:1:pos:R:3'
    properties, names, dtype, convs = parse_properties(info['Properties'])
    del info['Properties']

    data = []
    for ln in range(natoms):
        line = next(lines)
        vals = line.split()
        row = tuple([conv(val) for conv, val in zip(convs, vals)])
        data.append(row)

    try:
        data = np.array(data, dtype)
    except TypeError:
        raise IOError('Badly formatted data, ' +
                      'or end of file reached before end of frame')

    arrays = {}
    for name in names:
        ase_name, cols = properties[name]
        if cols == 1:
            value = data[name]
        else:
            value = np.vstack([data[name + str(c)]
                              for c in range(cols)]).T
        arrays[ase_name] = value

    symbols = None
    if 'symbols' in arrays:
        symbols = arrays['symbols']
        del arrays['symbols']

    numbers = None
    duplicate_numbers = None
    if 'numbers' in arrays:
        if symbols is None:
            numbers = arrays['numbers']
        else:
            duplicate_numbers = arrays['numbers']
        del arrays['numbers']

    charges = None
    if 'charges' in arrays:
        charges = arrays['charges']
        del arrays['charges']

    positions = None
    if 'positions' in arrays:
        positions = arrays['positions']
        del arrays['positions']

    atoms = Atoms(symbols=symbols,
                  positions=positions,
                  numbers=numbers,
                  charges = charges,
                  cell=cell,
                  pbc=pbc,
                  info=info)

    for name, array in arrays.items():
        atoms.new_array(name, array)

    if duplicate_numbers is not None:
        atoms.set_atomic_numbers(duplicate_numbers)

    # Load results of previous calculations into SinglePointCalculator
    results = {}
    for key in list(atoms.info.keys()):
        if key in all_properties:
            results[key] = atoms.info[key]
            # special case for stress- convert to Voigt 6-element form
            if key.startswith('stress') and results[key].shape == (3, 3):
                stress = results[key]
                stress = np.array([stress[0, 0],
                                   stress[1, 1],
                                   stress[2, 2],
                                   stress[1, 2],
                                   stress[0, 2],
                                   stress[0, 1]])
                results[key] = stress
            del atoms.info[key]
    for key in list(atoms.arrays.keys()):
        if key in all_properties:
            results[key] = atoms.arrays[key]
            del atoms.arrays[key]
    if results != {}:
        calculator = SinglePointCalculator(atoms, **results)
        atoms.set_calculator(calculator)
    return atoms


class XYZError(IOError):
    pass


class XYZChunk:
    def __init__(self, lines, natoms):
        self.lines = lines
        self.natoms = natoms

    def build(self):
        """Convert unprocessed chunk into Atoms."""
        return _read_xyz_frame(iter(self.lines), self.natoms)


def ixyzchunks(fd):
    """Yield unprocessed chunks (header, lines) for each xyz image."""
    while True:
        line = next(fd).strip()  # Raises StopIteration on empty file
        try:
            natoms = int(line)
        except ValueError:
            raise XYZError('Expected integer, found "{0}"'.format(line))
        try:
            lines = [next(fd) for _ in range(1 + natoms)]
        except StopIteration:
            raise XYZError('Incomplete XYZ chunk')
        yield XYZChunk(lines, natoms)


class ImageIterator:
    """"""
    def __init__(self, ichunks):
        self.ichunks = ichunks

    def __call__(self, fd, indices=-1):
        if not hasattr(indices, 'start'):
            if indices < 0:
                indices = slice(indices - 1, indices)
            else:
                indices = slice(indices, indices + 1)

        for chunk in self._getslice(fd, indices):
            yield chunk.build()

    def _getslice(self, fd, indices):
        try:
            iterator = islice(self.ichunks(fd), indices.start, indices.stop,
                              indices.step)
        except ValueError:
            # Negative indices.  Go through the whole thing to get the length,
            # which allows us to evaluate the slice, and then read it again
            startpos = fd.tell()
            nchunks = 0
            for chunk in self.ichunks(fd):
                nchunks += 1
            fd.seek(startpos)
            indices_tuple = indices.indices(nchunks)
            iterator = islice(self.ichunks(fd), *indices_tuple)
        return iterator


iread_xyz = ImageIterator(ixyzchunks)


def read_xyz(fileobj, index=-1):
    """
    Read from a file in Extended XYZ format

    index is the frame to read, default is last frame (index=-1).
    """
    if isinstance(fileobj, basestring):
        fileobj = open(fileobj)

    if not isinstance(index, int) and not isinstance(index, slice):
        raise TypeError('Index argument is neither slice nor integer!')

    # If possible, build a partial index up to the last frame required
    last_frame = None
    if isinstance(index, int) and index >= 0:
        last_frame = index
    elif isinstance(index, slice):
        if index.stop is not None and index.stop >= 0:
            last_frame = index.stop

    # scan through file to find where the frames start
    fileobj.seek(0)
    frames = []
    while fileobj:
        frame_pos = fileobj.tell()
        line = fileobj.readline()
        if line.strip() == '':
            break
        natoms = int(line)
        frames.append((frame_pos, natoms))
        if last_frame is not None and len(frames) > last_frame:
            break
        fileobj.readline()  # read comment line
        for i in range(natoms):
            fileobj.readline()

    if isinstance(index, int):
        if index < 0:
            tmpsnp = len(frames) + index
            trbl = range(tmpsnp, tmpsnp + 1, 1)
        else:
            trbl = range(index, index + 1, 1)
    elif isinstance(index, slice):
        start = index.start
        stop = index.stop
        step = index.step

        if start is None:
            start = 0
        elif start < 0:
            start = len(frames) + start

        if step is None:
            step = 1

        if stop is None:
            stop = len(frames)
        elif stop < 0:
            stop = len(frames) + stop

        trbl = range(start, stop, step)
        if step < 0:
            trbl.reverse()

    for index in trbl:
        frame_pos, natoms = frames[index]
        fileobj.seek(frame_pos)
        # check for consistency with frame index table
        assert int(fileobj.readline()) == natoms
        yield _read_xyz_frame(fileobj, natoms)


def output_column_format(atoms, columns, arrays,
                         write_info=True, results=None):
    """
    Helper function to build extended XYZ comment line
    """
    fmt_map = {'d': ('R', '%16.8f '),
               'f': ('R', '%16.8f '),
               'i': ('I', '%8d '),
               'O': ('S', '%s '),
               'S': ('S', '%s '),
               'U': ('S', '%s '),
               'b': ('L', ' %.1s ')}

    # NB: Lattice is stored as tranpose of ASE cell,
    # with Fortran array ordering
    lattice_str = ('Lattice="' +
                   ' '.join([str(x) for x in np.reshape(atoms.cell.T,
                                                        9, order='F')]) +
                   '"')

    property_names = []
    property_types = []
    property_ncols = []
    dtypes = []
    formats = []

    for column in columns:
        array = arrays[column]
        dtype = array.dtype

        property_name = PROPERTY_NAME_MAP.get(column, column)
        property_type, fmt = fmt_map[dtype.kind]
        property_names.append(property_name)
        property_types.append(property_type)

        if (len(array.shape) == 1 or
                (len(array.shape) == 2 and array.shape[1] == 1)):
            ncol = 1
            dtypes.append((column, dtype))
        else:
            ncol = array.shape[1]
            for c in range(ncol):
                dtypes.append((column + str(c), dtype))

        formats.extend([fmt] * ncol)
        property_ncols.append(ncol)

    props_str = ':'.join([':'.join(x) for x in
                          zip(property_names,
                              property_types,
                              [str(nc) for nc in property_ncols])])

    comment_str = ''
    if atoms.cell.any():
        comment_str += lattice_str + ' '
    comment_str += 'Properties={}'.format(props_str)

    info = {}
    if write_info:
        info.update(atoms.info)
    if results is not None:
        info.update(results)
    info['pbc'] = atoms.get_pbc()  # always save periodic boundary conditions
    comment_str += ' ' + key_val_dict_to_str(info)

    dtype = np.dtype(dtypes)
    fmt = ''.join(formats) + '\n'

    return comment_str, property_ncols, dtype, fmt


def write_xyz(fileobj, images, comment='', columns=None, write_info=True,
              write_results=True, plain=False):
    """
    Write output in extended XYZ format

    Optionally, specify which columns (arrays) to include in output,
    and whether to write the contents of the Atoms.info dict to the
    XYZ comment line (default is True) and the results of any
    calculator attached to this Atoms.
    """
    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'w')

    if hasattr(images, 'get_positions'):
        images = [images]

    for atoms in images:
        natoms = len(atoms)

        if columns is None:
            fr_cols = None
        else:
            fr_cols = columns[:]

        if fr_cols is None:
            fr_cols = (['symbols', 'positions'] +
                       [key for key in atoms.arrays.keys() if
                        key not in ['symbols', 'positions',
                                    'species', 'pos']])

        if plain:
            fr_cols = ['symbols', 'positions']
            write_info = False
            write_results = False

        per_frame_results = {}
        per_atom_results = {}
        if write_results:
            calculator = atoms.get_calculator()
            if (calculator is not None and
                    isinstance(calculator, Calculator)):
                for key in all_properties:
                    value = calculator.results.get(key, None)
                    if value is None:
                        # skip missing calculator results
                        continue
                    if (isinstance(value, np.ndarray) and
                            value.shape[0] == len(atoms)):
                        # per-atom quantities (forces, energies, stresses)
                        per_atom_results[key] = value
                    else:
                        # per-frame quantities (energy, stress)
                        # special case for stress, which should be converted
                        # to 3x3 matrices before writing
                        if key.startswith('stress'):
                            xx, yy, zz, yz, xz, xy = value
                            value = np.array([(xx, xy, xz),
                                              (xy, yy, yz),
                                              (xz, yz, zz)])
                        per_frame_results[key] = value

        # Move symbols and positions to first two properties
        if 'symbols' in fr_cols:
            i = fr_cols.index('symbols')
            fr_cols[0], fr_cols[i] = fr_cols[i], fr_cols[0]

        if 'positions' in fr_cols:
            i = fr_cols.index('positions')
            fr_cols[1], fr_cols[i] = fr_cols[i], fr_cols[1]

        # Check first column "looks like" atomic symbols
        if fr_cols[0] in atoms.arrays:
            symbols = atoms.arrays[fr_cols[0]]
        else:
            symbols = atoms.get_chemical_symbols()
        if not isinstance(symbols[0], basestring):
            raise ValueError('First column must be symbols-like')

        # Check second column "looks like" atomic positions
        pos = atoms.arrays[fr_cols[1]]
        if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
            raise ValueError('Second column must be position-like')

        # Collect data to be written out
        arrays = {}
        for column in fr_cols:
            if column in atoms.arrays:
                arrays[column] = atoms.arrays[column]
            elif column == 'symbols':
                arrays[column] = np.array(symbols)
            else:
                raise ValueError('Missing array "%s"' % column)

        if write_results:
            fr_cols += per_atom_results.keys()
            arrays.update(per_atom_results)

        comm, ncols, dtype, fmt = output_column_format(atoms,
                                                       fr_cols,
                                                       arrays,
                                                       write_info,
                                                       per_frame_results)
        if plain or comment != '':
            # override key/value pairs with user-speficied comment string
            comm = comment

        # Pack fr_cols into record array
        data = np.zeros(natoms, dtype)
        for column, ncol in zip(fr_cols, ncols):
            value = arrays[column]
            if ncol == 1:
                data[column] = np.squeeze(value)
            else:
                for c in range(ncol):
                    data[column + str(c)] = value[:, c]

        # Write the output
        fileobj.write('%d\n' % natoms)
        fileobj.write('%s\n' % comm)
        for i in range(natoms):
            fileobj.write(fmt % tuple(data[i]))


# create aliases for read/write functions
read_extxyz = read_xyz
write_extxyz = write_xyz
