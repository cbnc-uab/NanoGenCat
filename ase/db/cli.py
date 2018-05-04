from __future__ import print_function
import collections
import optparse
import sys
from random import randint

import ase.io
from ase.db import connect
from ase.db.core import convert_str_to_int_float_or_str
from ase.db.summary import Summary
from ase.db.table import Table, all_columns
from ase.calculators.calculator import get_calculator
from ase.utils import plural, basestring

try:
    input = raw_input  # Python 2+3 compatibility
except NameError:
    pass

description = """Selecton is a comma-separated list of
selections where each selection is of the type "ID", "key" or
"key=value".  Instead of "=", one can also use "<", "<=", ">=", ">"
and  "!=" (these must be protected from the shell by using quotes).
Special keys: id, user, calculator, age, natoms, energy, magmom,
and charge.  Chemical symbols can also be used to select number of
specific atomic species (H, He, Li, ...)."""

examples = ['calculator=nwchem',
            'age<1d',
            'natoms=1',
            'user=alice',
            '2.2<bandgap<4.1',
            'Cu>=10']


def main(args=sys.argv[1:]):
    if isinstance(args, basestring):
        args = args.split(' ')
    parser = optparse.OptionParser(
        usage='Usage: %prog db-name [selection] [options]',
        description=description,
        epilog='Selection examples: ' + ', '.join(examples) + '.')

    add = parser.add_option
    add('-v', '--verbose', action='store_true', default=False)
    add('-q', '--quiet', action='store_true', default=False)
    add('-n', '--count', action='store_true',
        help='Count number of selected rows.')
    add('-l', '--long', action='store_true',
        help='Long description of selected row')
    add('-i', '--insert-into', metavar='db-name',
        help='Insert selected rows into another database.')
    add('-a', '--add-from-file', metavar='[type:]filename',
        help='Add results from file.')
    add('-k', '--add-key-value-pairs', metavar='key1=val1,key2=val2,...',
        help='Add key-value pairs to selected rows.  Values must be numbers '
        'or strings and keys must follow the same rules as keywords.')
    add('-L', '--limit', type=int, default=20, metavar='N',
        help='Show only first N rows (default is 20 rows).  Use --limit=0 '
        'to show all.')
    add('--offset', type=int, default=0, metavar='N',
        help='Skip first N rows.  By default, no rows are skipped')
    add('--delete', action='store_true',
        help='Delete selected rows.')
    add('--delete-keys', metavar='key1,key2,...',
        help='Delete keys for selected rows.')
    add('-y', '--yes', action='store_true',
        help='Say yes.')
    add('--explain', action='store_true',
        help='Explain query plan.')
    add('-c', '--columns', metavar='col1,col2,...',
        help='Specify columns to show.  Precede the column specification '
        'with a "+" in order to add columns to the default set of columns.  '
        'Precede by a "-" to remove columns.  Use "++" for all.')
    add('-s', '--sort', metavar='column', default='id',
        help='Sort rows using column.  Use -column for a descendin sort.  '
        'Default is to sort after id.')
    add('--cut', type=int, default=35, help='Cut keywords and key-value '
        'columns after CUT characters.  Use --cut=0 to disable cutting. '
        'Default is 35 characters')
    add('-p', '--plot', metavar='[a,b:]x,y1,y2,...',
        help='Example: "-p x,y": plot y row against x row. Use '
        '"-p a:x,y" to make a plot for each value of a.')
    add('-P', '--plot-data', metavar='name',
        help="Show plot from data['name'] from the selected row.")
    add('--csv', action='store_true',
        help='Write comma-separated-values file.')
    add('-w', '--open-web-browser', action='store_true',
        help='Open results in web-browser.')
    add('--no-lock-file', action='store_true', help="Don't use lock-files")
    add('--analyse', action='store_true',
        help='Gathers statistics about tables and indices to help make '
        'better query planning choices.')
    add('-j', '--json', action='store_true',
        help='Write json representation of selected row.')
    add('--unique', action='store_true',
        help='Give rows a new unique id when using --insert-into.')
    opts, args = parser.parse_args(args)

    if not args:
        parser.error('No database given')

    verbosity = 1 - opts.quiet + opts.verbose

    try:
        run(opts, args, verbosity)
    except Exception as x:
        if verbosity < 2:
            print('{0}: {1}'.format(x.__class__.__name__, x))
            sys.exit(1)
        else:
            raise


def run(opts, args, verbosity):
    filename = args.pop(0)
    query = ','.join(args)

    if query.isdigit():
        query = int(query)

    add_key_value_pairs = {}
    if opts.add_key_value_pairs:
        for pair in opts.add_key_value_pairs.split(','):
            key, value = pair.split('=')
            add_key_value_pairs[key] = convert_str_to_int_float_or_str(value)

    if opts.delete_keys:
        delete_keys = opts.delete_keys.split(',')
    else:
        delete_keys = []

    con = connect(filename, use_lock_file=not opts.no_lock_file)

    def out(*args):
        if verbosity > 0:
            print(*args)

    if opts.analyse:
        con.analyse()
        return

    if opts.add_from_file:
        filename = opts.add_from_file
        if ':' in filename:
            calculator_name, filename = filename.split(':')
            atoms = get_calculator(calculator_name)(filename).get_atoms()
        else:
            atoms = ase.io.read(filename)
        con.write(atoms, key_value_pairs=add_key_value_pairs)
        out('Added {0} from {1}'.format(atoms.get_chemical_formula(),
                                        filename))
        return

    if opts.count:
        n = con.count(query)
        print('%s' % plural(n, 'row'))
        return

    if opts.explain:
        for row in con.select(query, explain=True,
                              verbosity=verbosity,
                              limit=opts.limit, offset=opts.offset):
            print(row['explain'])
        return

    if opts.insert_into:
        nkvp = 0
        nrows = 0
        with connect(opts.insert_into,
                     use_lock_file=not opts.no_lock_file) as con2:
            for row in con.select(query):
                kvp = row.get('key_value_pairs', {})
                nkvp -= len(kvp)
                kvp.update(add_key_value_pairs)
                nkvp += len(kvp)
                if opts.unique:
                    row['unique_id'] = '%x' % randint(16**31, 16**32 - 1)
                con2.write(row, data=row.get('data'), **kvp)
                nrows += 1

        out('Added %s (%s updated)' %
            (plural(nkvp, 'key-value pair'),
             plural(len(add_key_value_pairs) * nrows - nkvp, 'pair')))
        out('Inserted %s' % plural(nrows, 'row'))
        return

    if add_key_value_pairs or delete_keys:
        ids = [row['id'] for row in con.select(query)]
        m, n = con.update(ids, delete_keys, **add_key_value_pairs)
        out('Added %s (%s updated)' %
            (plural(m, 'key-value pair'),
             plural(len(add_key_value_pairs) * len(ids) - m, 'pair')))
        out('Removed', plural(n, 'key-value pair'))

        return

    if opts.delete:
        ids = [row['id'] for row in con.select(query)]
        if ids and not opts.yes:
            msg = 'Delete %s? (yes/No): ' % plural(len(ids), 'row')
            if input(msg).lower() != 'yes':
                return
        con.delete(ids)
        out('Deleted %s' % plural(len(ids), 'row'))
        return

    if opts.plot_data:
        from ase.db.plot import dct2plot
        dct2plot(con.get(query).data, opts.plot_data)
        return

    if opts.plot:
        if ':' in opts.plot:
            tags, keys = opts.plot.split(':')
            tags = tags.split(',')
        else:
            tags = []
            keys = opts.plot
        keys = keys.split(',')
        plots = collections.defaultdict(list)
        X = {}
        labels = []
        for row in con.select(query, sort=opts.sort):
            name = ','.join(str(row[tag]) for tag in tags)
            x = row.get(keys[0])
            if x is not None:
                if isinstance(x, basestring):
                    if x not in X:
                        X[x] = len(X)
                        labels.append(x)
                    x = X[x]
                plots[name].append([x] + [row.get(key) for key in keys[1:]])
        import matplotlib.pyplot as plt
        for name, plot in plots.items():
            xyy = zip(*plot)
            x = xyy[0]
            for y, key in zip(xyy[1:], keys[1:]):
                plt.plot(x, y, label=name + ':' + key)
        if X:
            plt.xticks(range(len(labels)), labels, rotation=90)
        plt.legend()
        plt.show()
        return

    if opts.long:
        row = con.get(query)
        summary = Summary(row)
        summary.write()
    elif opts.json:
        row = con.get(query)
        con2 = connect(sys.stdout, 'json', use_lock_file=False)
        kvp = row.get('key_value_pairs', {})
        con2.write(row, data=row.get('data'), **kvp)
    else:
        if opts.open_web_browser:
            import ase.db.app as app
            app.db = con
            app.app.run(host='0.0.0.0', debug=True)
        else:
            columns = list(all_columns)
            c = opts.columns
            if c and c.startswith('++'):
                keys = set()
                for row in con.select(query,
                                      limit=opts.limit, offset=opts.offset):
                    keys.update(row._keys)
                columns.extend(keys)
                if c[2:3] == ',':
                    c = c[3:]
                else:
                    c = ''
            if c:
                if c[0] == '+':
                    c = c[1:]
                elif c[0] != '-':
                    columns = []
                for col in c.split(','):
                    if col[0] == '-':
                        columns.remove(col[1:])
                    else:
                        columns.append(col.lstrip('+'))

            table = Table(con, verbosity, opts.cut)
            table.select(query, columns, opts.sort, opts.limit, opts.offset)
            if opts.csv:
                table.write_csv()
            else:
                table.write(query)
