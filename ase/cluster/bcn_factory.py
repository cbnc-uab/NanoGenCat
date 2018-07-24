from __future__ import print_function
import numpy as np

from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.cluster import Cluster
from ase.cluster.factory import ClusterFactory, reduce_miller
from ase.utils import basestring
from re import findall
from ase.io import write

class ClusterFactory(ClusterFactory):
    directions = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]

    atomic_basis = np.array([[0., 0., 0.]])

    element_basis = None

    Cluster = Cluster   # Make it possible to change the class of the object returned.
    cl_cut = False
    
    def __call__(self, symbols, surfaces, layers, distances, center=[0.0,0.0,0.0],
                 latticeconstant=None, vacuum=0.0, debug=0):
        self.debug = 1
        if self.cl_cut == True:
            atoms = symbols
            # Interpret symbol
            self.atomic_numbers = atoms.get_atomic_numbers().tolist()
            self.chemical_formula = atoms.get_chemical_formula()
            self.atomic_basis = atoms.get_scaled_positions()
            self.lattice_basis = atoms.get_cell()
            self.resiproc_basis = atoms.get_reciprocal_cell()
            
            self.spacegroup = int(str(atoms.info['spacegroup'])[0:3])
            
            self.set_surfaces_layers(surfaces, layers)
            self.set_lattice_size(center)
            self.distances = distances
            if distances is not None:
                self.set_surfaces_distances(surfaces, distances)
            cluster = self.make_cluster(vacuum)
            cluster.surfaces = self.surfaces.copy()
            cluster.lattice_basis = self.lattice_basis.copy()
            cluster.atomic_basis = self.atomic_basis.copy()
            cluster.resiproc_basis = atoms.get_reciprocal_cell()
            if distances is not None:
                cluster.distances = self.distances.copy()
            else:
                cluster.distances = None
            
        return cluster

    def make_cluster(self, vacuum):
        size = np.array(self.size)
        translations = np.zeros((size.prod(), 3))
        for h in range(size[0]):
            for k in range(size[1]):
                for l in range(size[2]):
                    i = h * (size[1] * size[2]) + k * size[2] + l
                    translations[i] = np.dot([h, k, l], self.lattice_basis)
        atomic_basis = np.dot(self.atomic_basis, self.lattice_basis)
        positions = np.zeros((len(translations) * len(atomic_basis), 3))
        
        numbers = np.zeros(len(positions))
        n = len(atomic_basis)
        for i, trans in enumerate(translations):
            
            positions[n*i:n*(i+1)] = atomic_basis + trans
            numbers[n*i:n*(i+1)] = self.atomic_numbers

        index = -1
        for s, l in zip(self.surfaces, self.layers):
            index += 1
            n = self.miller_to_direction(s)
            if self.distances is not None:
                rmax = self.distances[index]
                #print("RMAX",s,rmax)
            else:
                rmax = self.get_layer_distance(s, l + 0.2)
            r = np.dot(positions - self.center, n)
            mask = np.less(r, rmax)
            if self.debug > 1:
                print("Cutting %s at %i layers ~ %.3f A" % (s, l, rmax))
            
            positions = positions[mask]
            numbers = numbers[mask]
        # Fit the cell, so it only just consist the atoms
       
        min = np.zeros(3)
        max = np.zeros(3)
        for i in range(3):
            v = self.directions[i]
            r = np.dot(positions, v)
            min[i] = r.min()
            max[i] = r.max()

        cell = max - min + vacuum
        positions = positions - min + vacuum / 2.0
        self.center = self.center - min + vacuum / 2.0
        self.Cluster.unit_cell_formula = self.chemical_formula
        return self.Cluster(symbols=numbers, positions=positions, cell=cell)
                
    def set_lattice_size(self, center):
        if center is None:
            offset = np.zeros(3)
        else:
            offset = np.array(center)
            if (offset > 1.0).any() or (offset < 0.0).any():
                raise ValueError("Center offset must lie within the lattice unit \
                                  cell.")
        # print('holiiiii')
        max = np.ones(3)
        min = -np.ones(3)
        v = np.linalg.inv(self.lattice_basis.T)
        for s, l in zip(self.surfaces, self.layers):
            n = self.miller_to_direction(s) * self.get_layer_distance(s, l*4)
            k = np.round(np.dot(v, n), 2)
            for i in range(3):
                if k[i] > 0.0:
                    k[i] = np.ceil(k[i])
                elif k[i] < 0.0:
                    k[i] = np.floor(k[i])

            if self.debug > 1:

                print("Spaning %i layers in %s in lattice basis ~ %s" % (l, s, k))
            max[k > max] = k[k > max]
            min[k < min] = k[k < min]
            if l % 2 != 0:
                self.center = np.dot(offset - min, self.lattice_basis)
            else:
                self.center = np.dot(offset - min, self.lattice_basis)
        self.size = (max - min + np.ones(3)).astype(int)

    def set_surfaces_distances(self, surfaces, distances):
        if len(surfaces) != len(distances):
            raise ValueError("Improper size of surface and layer arrays: %i != %i"
                             % (len(surfaces), len(distances)))
        
        sg = Spacegroup(self.spacegroup)
        surfaces = np.array(surfaces)
        distances = np.array(distances)
        
        for i, s in enumerate(surfaces):
            s = reduce_miller(s)
            surfaces[i] = s

        surfaces_full = surfaces.copy()
        distances_full = distances.copy()

        for s, l in zip(surfaces, distances):
            equivalent_surfaces = sg.equivalent_reflections(s.reshape(-1, 3))

            for es in equivalent_surfaces:
                # If the equivalent surface (es) is not in the surface list,
                # then append it.
                if not np.equal(es, surfaces_full).all(axis=1).any():
                    surfaces_full = np.append(surfaces_full, es.reshape(1, 3), axis=0)
                    distances_full = np.append(distances_full, l)
                    
        

        self.surfaces = surfaces_full.copy()
        self.distances = distances_full.copy()



