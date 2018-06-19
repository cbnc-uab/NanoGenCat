import re
import os

from ase.db.core import default_key_descriptions


functions = []


def creates(*filenames):
    def decorator(func):
        functions.append((func, filenames))
        return func
    return decorator


def process_metadata(db, html=True):
    meta = db.metadata

    mod = {}
    if db.python:
        with open(db.python) as fd:
            code = fd.read()
        path = os.path.dirname(db.python)
        code = 'import sys; sys.path[:0] = ["{}"]; {}'.format(path, code)

        # We use eval here instead of exec because it works on both
        # Python 2 and 3.
        eval(compile(code, db.python, 'exec'), mod, mod)

    for key, default in [('title', 'ASE database'),
                         ('default_columns', []),
                         ('special_keys', []),
                         ('key_descriptions', {}),
                         ('layout', [])]:
        meta[key] = mod.get(key, meta.get(key, default))

    if not meta['default_columns']:
        meta['default_columns'] = ['id', 'formula']

    # Also fill in default key-descriptions:
    kd = default_key_descriptions.copy()
    kd.update(meta['key_descriptions'])
    meta['key_descriptions'] = kd
    for key, value in kd.items():
        if not value[1]:
            kd[key] = (value[0], value[0], value[2])

    sk = []
    for special in meta['special_keys']:
        kind = special[0]
        if kind == 'SELECT':
            key = special[1]
            choises = sorted({row.get(key) for row in db.select(key)})
            if key in kd:
                longkey = kd[key][0]
            else:
                longkey = key
            special = ['SELECT', key, longkey, choises]
        elif kind == 'BOOL':
            key = special[1]
            if key in kd:
                longkey = kd[key][0]
            else:
                longkey = key
            special = ['BOOL', key, longkey]
        else:
            # RANGE
            pass
        sk.append(special)
    meta['special_keys'] = sk

    if not meta['layout']:
        keys = ['id', 'formula', 'age']
        meta['layout'] = [
            ('Basic properties',
             [['ATOMS', 'CELL'],
              [('Key Value Pairs', keys), 'FORCES']])]

    if mod:
        meta['functions'] = functions[:]
        functions[:] = []

    sub = re.compile(r'`(.)_(.)`')
    sup = re.compile(r'`(.*)\^\{?(.*?)\}?`')

    # Convert LaTeX to HTML or raw text:
    for key, value in meta['key_descriptions'].items():
        short, long, unit = value
        if html:
            unit = sub.sub(r'\1<sub>\2</sub>', unit)
            unit = sup.sub(r'\1<sup>\2</sup>', unit)
            unit = unit.replace(r'\text{', '').replace('}', '')
        else:
            unit = sub.sub(r'\1_\2', unit)
            unit = sup.sub(r'\1^\2', unit)
        meta['key_descriptions'][key] = (short, long, unit)

    all_keys1 = set(meta['key_descriptions'])
    for row in db.select():
        all_keys1.update(row._keys)
    all_keys2 = []
    for key in all_keys1:
        short, long, unit = meta['key_descriptions'].get(key, ('', '', ''))
        all_keys2.append((key, long, unit))
    meta['all_keys'] = sorted(all_keys2)

    return meta
