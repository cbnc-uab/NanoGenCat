import sys

from ase import Atoms
from ase.io import write
from ase.gui.ag import main


write('x.json', Atoms('X'))

# Make sure ase-gui can run in terminal mode without $DISPLAY and tkinter:
main(['--verbose', '--terminal', 'x.json@id=1'])
assert 'tkinter' not in sys.modules
assert 'Tkinter' not in sys.modules  # legacy Python
