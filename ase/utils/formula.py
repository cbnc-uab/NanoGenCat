from ase.data import chemical_symbols


def formula_hill(numbers):
    """Convert list of atomic numbers to a chemical formula as a string.

    Elements are alphabetically ordered with C and H first."""

    if isinstance(numbers, dict):
        count = dict(numbers)
    else:
        count = {}
        for Z in numbers:
            symb = chemical_symbols[Z]
            count[symb] = count.get(symb, 0) + 1
    result = [(s, count.pop(s)) for s in 'CH' if s in count]
    result += [(s, count[s]) for s in sorted(count)]
    return ''.join('{0}{1}'.format(symbol, n) if n > 1 else symbol
                   for symbol, n in result)


def formula_metal(numbers):
    """Convert list of atomic numbers to a chemical formula as a string.

    Elements are alphabetically ordered with metals first."""

    # non metals, half-metals/metalloid, halogen, noble gas
    non_metals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                  'Si', 'P', 'S', 'Cl', 'Ar',
                  'Ge', 'As', 'Se', 'Br', 'Kr',
                  'Sb', 'Te', 'I', 'Xe',
                  'Po', 'At', 'Rn']

    if isinstance(numbers, dict):
        count = dict(numbers)
    else:
        count = {}
        for Z in numbers:
            symb = chemical_symbols[Z]
            count[symb] = count.get(symb, 0) + 1
    result2 = [(s, count.pop(s)) for s in non_metals if s in count]
    result = [(s, count[s]) for s in sorted(count)]
    result += sorted(result2)
    return ''.join('{0}{1}'.format(symbol, n) if n > 1 else symbol
                   for symbol, n in result)
