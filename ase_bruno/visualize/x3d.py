"""Inline viewer for jupyter notebook using X3D."""

from tempfile import NamedTemporaryFile
from IPython.display import HTML


def view_x3d(atoms):
    """View atoms inline in a jupyter notbook. This command
    should only be used within a jupyter/ipython notebook."""
    with NamedTemporaryFile('r+', suffix='.html') as ntf:
        atoms.write(ntf.name, format='html')
        ntf.seek(0)
        html_atoms = ntf.read()
    return HTML(html_atoms)
