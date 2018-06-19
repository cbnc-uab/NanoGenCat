import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixBondLength
from ase.db import connect
from ase.io import read
from ase.build import molecule
from ase.test import must_raise


for name in ['y2.json', 'y2.db']:
    c = connect(name)
    print(name, c)

    id = c.reserve(abc=7)
    c.delete([d.id for d in c.select(abc=7)])
    id = c.reserve(abc=7)
    assert c[id].abc == 7

    a = c.get_atoms(id)
    c.write(Atoms())
    ch4 = molecule('CH4', calculator=EMT())
    ch4.constraints = [FixAtoms(indices=[1]),
                       FixBondLength(0, 2)]
    f1 = ch4.get_forces()
    print(f1)

    c.delete([d.id for d in c.select(C=1)])
    chi = np.array([1 + 0.5j, 0.5])
    id = c.write(ch4, data={'1-butyne': 'bla-bla', 'chi': chi})

    row = c.get(id)
    print(row.data['1-butyne'], row.data.chi)
    assert (row.data.chi == chi).all()

    assert len(c.get_atoms(C=1).constraints) == 2

    f2 = c.get(C=1).forces
    assert abs(f2.sum(0)).max() < 1e-14
    f3 = c.get_atoms(C=1).get_forces()
    assert abs(f1 - f3).max() < 1e-14
    a = read(name + '@id=' + str(id))[0]
    f4 = a.get_forces()
    assert abs(f1 - f4).max() < 1e-14

    with must_raise(ValueError):
        c.update(id, abc={'a': 42})

    c.update(id, grr='hmm')
    row = c.get(C=1)
    assert row.id == id
    assert (row.data.chi == chi).all()
    print(row)

    for row in c.select(include_data=False):
        with must_raise(AttributeError):
            row.data

    with must_raise(ValueError):
        c.write(ch4, foo=['bar', 2])  # not int, bool, float or str

    with must_raise(ValueError):
        c.write(Atoms(), pi='3.14')  # number as a string

    with must_raise(ValueError):
        c.write(Atoms(), S=42)  # chemical symbol as key

    id = c.write(Atoms(), b=np.bool_(True))
    assert isinstance(c[id].b, bool)

    # Make sure deleting a single sey works:
    id = c.write(Atoms(), key=7)
    c.update(id, delete_keys=['key'])
    assert 'key' not in c[id]
