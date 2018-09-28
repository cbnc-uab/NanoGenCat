import os
import sys
import math
import numpy as np
import pickle

from ase import Atoms
from ase.utils import basestring

class bcn_Cluster(Cluster):
    def get_layers(self):
        """Return number of atomic layers in stored surfaces directions."""
        layers = []

        for s in self.surfaces:
            n = self.miller_to_direction(s)
            c = self.get_positions().mean(axis=0)
            r = np.dot(self.get_positions() - c, n).max()
            d = self.bcn_get_layers_distance(s, 2)
            l = 2 * np.round(r / d).astype(int)

            ls = np.arange(l - 1, l + 2)
            ds = np.array([self.bcn_get_layer_distance(s, i) for i in ls])

            mask = (np.abs(ds - r) < 1e-10)

            layers.append(ls[mask][0])

        return np.array(layers, int)



