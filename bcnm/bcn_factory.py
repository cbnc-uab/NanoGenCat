from __future__ import print_function
import numpy as np

from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.factory import ClusterFactory, reduce_miller
from bcn_wulff import interplanarDistance
from ase.utils import basestring
from re import findall
from ase.io import write
from cluster import Cluster

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
            # Interpret symbols
            self.atomic_numbers = atoms.get_atomic_numbers().tolist()
            self.chemical_formula = atoms.get_chemical_formula()
            self.atomic_basis = atoms.get_scaled_positions()
            self.lattice_basis = atoms.get_cell()
            self.resiproc_basis = atoms.get_reciprocal_cell()
            
            self.spacegroup = int(str(atoms.info['spacegroup'])[0:3])
            
            self.set_surfaces_layers(surfaces, layers)
            # print('center pre self center',center)
            self.set_lattice_size(center)
            # print('selfcenterbitch\n',self.center)
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

    def make_cluster(self, vacuum,debug=0):
        size = np.array(self.size)
        if debug>1:
            print('size\n',size)
            print(size.prod())
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
                rmax = self.bcn_get_layer_distance(s, l + 0.2)
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
        """
        Routine that change the center 
        in function of how much layers of materials
        must be added.Also defines the size
        Args:
            self(CutCluster):
            center(list): initial centering
        """
        ##Understanding this routine
        ##Check if the center is inside the cell
        # print('.........')
        if center is None:
            offset = np.zeros(3)
        else:
            offset = np.array(center)
            if (offset > 1.0).any() or (offset < 0.0).any():
                raise ValueError("Center offset must lie within the lattice unit \
                                  cell.")
        max = np.ones(3)
        min = -np.ones(3)
        #Calculate the reciprocal latice, why?
        v = self.resiproc_basis

        #For surface and layers
        for s, l in zip(self.surfaces, self.layers):
            # print('surface',s)
            # n is the miler to direction of each surface times the interplanarDistance times the layers
            # its like how much do you have to grow in each direction
            n = self.miller_to_direction(s) * interplanarDistance(self.resiproc_basis,[s])*l
            # print('n',n)
            # dot product between inverse latice and n vector(3) and round to 2 decimal to give the k vector(3) 
            k = np.round(np.dot(v, n), 2)
            # print('k',k)
            #Round to the smallest integer if i is larger than 0 (ceil) or the largest integer (floor) if i
            # is smallest than 0
            for i in range(3):
                if k[i] > 0.0:
                    k[i] = np.ceil(k[i])
                elif k[i] < 0.0:
                    k[i] = np.floor(k[i])
            # print('k post ceil of floor',k)
            if self.debug > 1:
                print("Spaning %i layers in %s direction in lattice basis ~ %s" % (l, s, k))
            # print("Spaning %i layers in %s direction in lattice basis ~ %s" % (l, s, k))

            #Update max and min values for every surface with k
            max[k > max] = k[k > max]
            min[k < min] = k[k < min]
            if self.debug>1:
                print('max ',max)
                print('min ',min)
            #I think that is not necesary
            if l % 2 != 0:
                self.center = np.dot(offset - min, self.lattice_basis)
            else:
                self.center = np.dot(offset - min, self.lattice_basis)
            if self.debug>1:
                print('self.center set lattice size',self.center)
                print('............................................')
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




