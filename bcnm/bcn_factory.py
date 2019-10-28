from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from re import findall

from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.factory import ClusterFactory, reduce_miller
from ase.utils import basestring
from ase.io import write
from ase.visualize import view
from ase import Atom
from bcnm.bcn_wulff import interplanarDistance
from bcnm.cluster import Cluster

from mpl_toolkits.mplot3d import Axes3D

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

            # At the beggining we add initial distances, but not
            # all the symmetry equivalent ones
            if distances is not None:
                self.set_surfaces_distances(surfaces, distances)
            ##keep the default surfaces
            cluster = self.make_cluster(vacuum)
            # print(realSize)
            # self.test(realSize)
            # self.wulffDistanceBased(surfaces,realSize)
            # print(realSize)
            # view(cluster)
            cluster.surfaces = self.surfaces.copy()
            cluster.lattice_basis = self.lattice_basis.copy()
            cluster.atomic_basis = self.atomic_basis.copy()
            cluster.resiproc_basis = atoms.get_reciprocal_cell()
            if distances is not None:
                cluster.distances = self.distances.copy()
            else:
                cluster.distances = None
            # print(cluster.get_layers())
        return cluster

    def make_cluster(self, vacuum,debug=0):
        """
        Function that creates the cluster
        firstly, reproducing the bulk material
        to get a large enough bulk to cut the
        nanoparticle.
        To cut the nanoparticle measure
        the distance from the ion
        translated to the center in direction
        to the plane of interest and 
        compare it with the distance
        of layers. If the distance
        is larger than the distance of layers
        the ion is removed.
        Return:
            Cluster(atoms): cluster structure of the atoms type
        """
        size = np.array(self.size)
        # print('size original\n',size)
        # if debug>1:
        # size = np.asarray([1,1,1])
        # print('size\n',size)
        # print(size.prod())
        #Construct a list of empty translations
        translations = np.zeros((size.prod(), 3))
        # print(translations,len(translations))


        #Get a set of points in the range of the size

        # Build a  set of points using the number of cells
        for h in range(size[0]):
            for k in range(size[1]):
                for l in range(size[2]):
                    # print('h,k,l',h,k,l)
                    i = h * (size[1] * size[2]) + k * size[2] + l
                    # print('index',i)
                    # keep in mind that size is the number of cells that has to be
                    # replied, when we multiply by lattice_basis(unit cell), we get
                    # the translation in cartesian units
                    translations[i] = np.dot([h, k, l], self.lattice_basis)
                    
                    
        # print('self.atomic_basis,self.lattice_basis',self.atomic_basis,'\n',self.lattice_basis)
        # print('lenselfatomicbasis\n',len(self.atomic_basis))

        # Transform the atomic positions in cartesian ones by multiplying it for the cell parameters
        # print(self.atomic_basis)
        atomic_basis = np.dot(self.atomic_basis, self.lattice_basis)
        # print('atomic_basis\n',atomic_basis)
        #positions is an empty array the product between transitions and atomic basis
        positions = np.zeros((len(translations) * len(atomic_basis), 3))
        #numbers as the len of positions
        numbers = np.zeros(len(positions))
        # n is the len of atomic basis 
        n = len(atomic_basis)
        # print('n\n',n)

        #per each translation add the value to the atomic basis
        # and save as the position

        for i, trans in enumerate(translations):
            # print('--------------------------------')
            # print(i,trans)
            positions[n*i:n*(i+1)] = atomic_basis + trans
            # print(*positions, sep='\n')
            # print(positions)
            numbers[n*i:n*(i+1)] = self.atomic_numbers
            # print(numbers)
            # break

        #Up to here we have the positions of the replied cell by translations
        # replicatedBulk=Cluster(positions=positions,numbers=numbers) 
        # replicatedBulk.append(Atom('Cs',position=self.center))
        # view(replicatedBulk)
        # exit(1)
        # For each suface get the interlayer distances
        # to get the final size


        realSize=[]

        index = -1
        for s, l in zip(self.surfaces, self.layers):
            index += 1
            n = self.miller_to_direction(s)
            if self.distances is not None:
                rmax = self.distances[index]
                # print("RMAX",s,rmax)

            # r value is the distance from the position 
            # previously translated to the center to the plane
            
            r = np.dot(positions - self.center, n)
            
            # print('r\n',r)
            # print(l)
            # print(s,np.dot(l*self.center,n))

            # by using less_equal function, only keep the positions
            # that has a lower or equal distance to rmax,
            # so the largest of the keeped r values
            # has to be on the surface, so i can use it
            # as a criteria to evaluate the growing

            mask = np.less(r, rmax)
            # print('mask',mask)
            if self.debug > 1:
                print("Cutting %s at %i layers ~ %.3f A" % (s, l, rmax))
            # rkeeped=r[mask]
            ##Getting the real size as the average of the values that are larger
            # or equal than 80% of rmax
            # rselected=np.max(rkeeped)
            # print('rselected',s,rselected)
            # realSize.append([s,np.max(rselected)])
            # print('plane,largestpos',',',s,',',np.max(rkeeped))
            
            #-------------------
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.plot(rkeeped)
            # plt.show()
            #---------------------

            positions = positions[mask]
            # print(positions)
            numbers = numbers[mask]
            # break
        # print('..............')
        # Fit the cell, so it only just consist the atoms
        min = np.zeros(3)
        max = np.zeros(3)
        for i in range(3):
            v = self.directions[i]
            r = np.dot(positions, v)
            min[i] = r.min()
            max[i] = r.max()

        cell = max - min + vacuum
        # print(np.mean(cell,axis=0))
        positions = positions - min + vacuum / 2.0
        self.center = self.center - min + vacuum / 2.0
        self.Cluster.cut_origin=self.center
        self.Cluster.unit_cell_formula = self.chemical_formula
        # exit(1)
        # cuted=Cluster(positions=positions,numbers=numbers) 
        # cuted.append(Atom('Cs',position=self.center))
        # view(cuted) 
        # exit(1) 
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
            # print('max ',max)
            # print('min ',min)
            if l % 2 != 0:
                self.center = np.dot(offset - min, self.lattice_basis)
            else:
                self.center = np.dot(offset - min, self.lattice_basis)
            if self.debug >1:
                print('self.center set lattice size',self.center)
                print('............................................')
        self.size = (max - min + np.ones(3)).astype(int)
        # print('self.center')
        # # print(self.size)
        # print(self.center)
    def set_surfaces_distances(self, surfaces, distances):
        """
        Function that add all the distances to all
        equivalent surfaces
        """
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



