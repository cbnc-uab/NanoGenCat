import os

import ase.db
from ase import Atoms
from ase.io import read


def dcdft():
    os.environ['USER'] = 'ase'
    con = ase.db.connect('dcdft.json')
    with open('WIEN2k.txt') as fd:
        lines = fd.readlines()
    for line in lines[2:73]:
        words = line.split()
        symbol = words.pop(0)
        vol, B, Bp = (float(x) for x in words)
        filename = 'cif/' + symbol + '.cif'
        atoms = read(filename)
        M = {'Fe': 2.3,
             'Co': 1.2,
             'Ni': 0.6,
             'Cr': 1.5,
             'O': 1.5,
             'Mn': 2.0}.get(symbol)
        if M is not None:
            magmoms = [M] * len(atoms)
            if symbol in ['Cr', 'O', 'Mn']:
                magmoms[len(atoms) // 2:] = [-M] * (len(atoms) // 2)
            atoms.set_initial_magnetic_moments(magmoms)
        con.write(atoms, name=symbol, w2k_B=B, w2k_Bp=Bp, w2k_volume=vol)
        filename = 'pcif/' + symbol + '.cif'
        p = read(filename, primitive_cell=True)
        v = atoms.get_volume() / len(atoms)
        dv = v - p.get_volume() / len(p)
        p2 = read(filename)
        dv2 = v - p2.get_volume() / len(p2)
        print(symbol, vol - atoms.get_volume() / len(atoms),
              len(atoms), len(p), dv, dv2)
        print(p.info)
        # assert dv < 0.0001
  
        
def g2():
    from ase.data.g2 import data
    os.environ['USER'] = 'ase'
    con = ase.db.connect('g2.json')
    for name, d in data.items():
        kwargs = {}
        if d['magmoms']:
            kwargs['magmoms'] = d['magmoms']
        atoms = Atoms(d['symbols'], d['positions'], **kwargs)
        con.write(atoms, name=name)
