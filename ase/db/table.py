from __future__ import print_function

import numpy as np

from ase.db.core import float_to_time_string, now


all_columns = ['id', 'age', 'user', 'formula', 'calculator',
               'energy', 'fmax', 'pbc', 'volume',
               'charge', 'mass', 'smax', 'magmom']


def plural(n, word):
    if n == 1:
        return '1 ' + word
    return '%d %ss' % (n, word)

    
def cut(txt, length):
    if len(txt) <= length or length == 0:
        return txt
    return txt[:length - 3] + '...'


def cutlist(lst, length):
    if len(lst) <= length or length == 0:
        return lst
    return lst[:9] + ['... ({0} more)'.format(len(lst) - 9)]

    
class Table:
    def __init__(self, connection, verbosity=1, cut=35):
        self.connection = connection
        self.verbosity = verbosity
        self.cut = cut
        self.rows = []
        self.columns = None
        self.id = None
        self.right = None
        self.keys = None
        
    def select(self, query, columns, sort, limit, offset):
        self.limit = limit
        self.offset = offset
        
        self.rows = [Row(d, columns)
                     for d in self.connection.select(
                         query, verbosity=self.verbosity,
                         limit=limit, offset=offset, sort=sort)]

        delete = set(range(len(columns)))
        for row in self.rows:
            for n in delete.copy():
                if row.values[n] is not None:
                    delete.remove(n)
        delete = sorted(delete, reverse=True)
        for row in self.rows:
            for n in delete:
                del row.values[n]
                
        self.columns = list(columns)
        for n in delete:
            del self.columns[n]
                
    def format(self, subscript=None):
        right = set()  # right-adjust numbers
        allkeys = set()
        for row in self.rows:
            numbers = row.format(self.columns, subscript)
            right.update(numbers)
            allkeys.update(row.dct.get('key_value_pairs', {}))
            
        right.add('age')
        self.right = [column in right for column in self.columns]
        
        self.keys = sorted(allkeys)

    def write(self, query=None):
        self.format()
        L = [[len(s) for s in row.strings]
             for row in self.rows]
        L.append([len(c) for c in self.columns])
        N = np.max(L, axis=0)

        fmt = '{0:{align}{width}}'
        print('|'.join(fmt.format(c, align='<>'[a], width=w)
                       for c, a, w in zip(self.columns, self.right, N)))
        for row in self.rows:
            print('|'.join(fmt.format(c, align='<>'[a], width=w)
                           for c, a, w in
                           zip(row.strings, self.right, N)))

        if self.verbosity == 0:
            return
            
        nrows = len(self.rows)
        
        if self.limit and nrows == self.limit:
            n = self.connection.count(query)
            print('Rows:', n, '(showing first {0})'.format(self.limit))
        else:
            print('Rows:', nrows)

        if self.keys:
            print('Keys:', ', '.join(cutlist(self.keys, self.cut)))
            
    def write_csv(self):
        print(', '.join(self.columns))
        for row in self.rows:
            print(', '.join(str(val) for val in row.values))

            
class Row:
    def __init__(self, dct, columns):
        self.dct = dct
        self.values = None
        self.strings = None
        self.more = False
        self.set_columns(columns)
        
    def set_columns(self, columns):
        self.values = []
        for c in columns:
            if c == 'age':
                value = float_to_time_string(now() - self.dct.ctime)
            elif c == 'pbc':
                value = ''.join('FT'[p] for p in self.dct.pbc)
            else:
                value = getattr(self.dct, c, None)
            self.values.append(value)
            
    def toggle(self):
        self.more = not self.more
        
    def format(self, columns, subscript=None):
        self.strings = []
        numbers = set()
        for value, column in zip(self.values, columns):
            if column == 'formula' and subscript:
                value = subscript.sub(r'<sub>\1</sub>', value)
            elif isinstance(value, int):
                value = str(value)
                numbers.add(column)
            elif isinstance(value, float):
                numbers.add(column)
                value = '{0:.3f}'.format(value)
            elif value is None:
                value = ''
            self.strings.append(value)
        
        return numbers
