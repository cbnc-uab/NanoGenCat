"""Check reading of a sample v_sim .ascii file, and I/O consistency"""

try:
    from urllib.request import urlretrieve
    from urllib.error import URLError
except ImportError:
    from urllib import urlretrieve
    from urllib2 import URLError
from socket import error as SocketError

from ase.test import NotAvailable
from ase.io import read

dest = 'demo.ascii'
src = 'http://inac.cea.fr/L_Sim/V_Sim/files/' + dest
copy = 'demo2.ascii'

try:
    urlretrieve(src, filename=dest)
except (IOError, URLError, SocketError):
    raise NotAvailable('Retrieval of ' + src + ' failed')

atoms = read(dest, format='v-sim')

atoms.write(copy)
atoms2 = read(copy)

tol = 1e-6
assert sum(abs((atoms.positions - atoms2.positions).ravel())) < tol
assert sum(abs((atoms.cell - atoms2.cell).ravel())) < tol
