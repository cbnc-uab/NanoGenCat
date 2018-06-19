import numpy as np

from ase.dft.kpoints import labels_from_kpts
from ase.io.jsonio import encode, decode
from ase.parallel import paropen


class BandStructure:
    def __init__(self, atoms=None, calc=None, filename=None):
        """Band-structure object.

        Create a band-structure object from an Atoms object, a calculator or
        from a pickle file.  Labels for special points will be automatically
        added.
        """
        if filename:
            self.read(filename)
        else:
            atoms = atoms or calc.atoms
            calc = calc or atoms.calc

            self.cell = atoms.cell
            self.kpts = calc.get_ibz_k_points()
            self.fermilevel = calc.get_fermi_level()

            energies = []
            for s in range(calc.get_number_of_spins()):
                energies.append([calc.get_eigenvalues(kpt=k, spin=s)
                                 for k in range(len(self.kpts))])
            self.energies = np.array(energies)

            x, X, labels = labels_from_kpts(self.kpts, self.cell)
            self.xcoords = x
            self.label_xcoords = X
            self.labels = labels

    def todict(self):
        return dict((key, getattr(self, key))
                    for key in
                    ['cell', 'kpts', 'energies', 'fermilevel',
                     'xcoords', 'label_xcoords', 'labels'])

    def write(self, filename):
        """Write to json file."""
        with paropen(filename, 'w') as f:
            f.write(encode(self))

    def read(self, filename):
        """Read from json file."""
        with open(filename, 'r') as f:
            dct = decode(f.read())
        self.__dict__.update(dct)

    def plot(self, spin=None, emax=None, filename=None, ax=None, show=None):
        """Plot band-structure.

        spin: int or None
            Spin channel.  Default behaviour is to plot both spi up and down
            for spin-polarized calculations.
        emax: float
            Maximum energy above fermi-level.
        filename: str
            Write imagee to a file.
        ax: Axes
            MatPlotLib Axes object.  Will be created if not supplied.
        show: bool
            Show the image.
        """

        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        def pretty(kpt):
            if kpt == 'G':
                kpt = r'\Gamma'
            elif len(kpt) == 2:
                kpt = kpt[0] + '_' + kpt[1]
            return '$' + kpt + '$'

        if spin is None:
            e_skn = self.energies
        else:
            e_skn = self.energies[spin, None]

        emin = e_skn.min()
        if emax is not None:
            emax = emax + self.fermilevel

        labels = [pretty(name) for name in self.labels]
        i = 1
        while i < len(labels):
            if self.label_xcoords[i - 1] == self.label_xcoords[i]:
                labels[i - 1] = labels[i - 1][:-1] + ',' + labels[i][1:]
                labels[i] = ''
            i += 1

        for spin, e_kn in enumerate(e_skn):
            color = 'br'[spin]
            for e_k in e_kn.T:
                ax.plot(self.xcoords, e_k, color=color)

        for x in self.label_xcoords[1:-1]:
            ax.axvline(x, color='0.5')

        ax.set_xticks(self.label_xcoords)
        ax.set_xticklabels(labels)
        ax.axis(xmin=0, xmax=self.xcoords[-1], ymin=emin, ymax=emax)
        ax.set_ylabel('eigenvalues [eV]')
        ax.axhline(self.fermilevel, color='k')
        try:
            plt.tight_layout()
        except AttributeError:
            pass
        if filename:
            plt.savefig(filename)

        if show is None:
            show = not filename

        if show:
            plt.show()

        return ax
