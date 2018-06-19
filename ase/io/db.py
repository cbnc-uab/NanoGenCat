import ase.db
from ase.utils import basestring


def read_db(filename, index, **kwargs):
    con = ase.db.connect(filename, serial=True, **kwargs)
    if index == slice(-1, None):
        yield con.get().toatoms()
    else:
        if index == slice(None, None, None):
            index = None
        elif isinstance(index, basestring) and index.isdigit():
            index = int(index)
        for row in con.select(index):
            yield row.toatoms()


def write_db(filename, images, **kwargs):
    con = ase.db.connect(filename, serial=True, **kwargs)
    for atoms in images:
        con.write(atoms)


read_json = read_db
write_json = write_db
read_postgresql = read_db
write_postgresql = write_db
