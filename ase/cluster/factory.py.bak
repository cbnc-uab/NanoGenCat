from __future__ import print_function
import numpy as np

from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.base import ClusterBase
from ase.cluster.cluster import Cluster
from ase.utils import basestring
from re import findall
from ase.io import write

class ClusterFactory(ClusterBase):
    ##print('Call to ClusterFactory')
    directions = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]

    atomic_basis = np.array([[0., 0., 0.]])

    element_basis = None

    Cluster = Cluster   # Make it possible to change the class of the object returned.
    cl_cut = False
    ##print('End of call to ClusterFactory',cl_cut)
    
    def __call__(self, bulk, surfaces, layers, distances, centering=False, center=[0.0,0.0,0.0],
                 latticeconstant=None, vacuum=0.0, debug=0):
                      #center=[0.5,0.5,0.5], vacuum=0.0, debug=0):
        self.debug = debug
        self.centering = centering
        #print("self.centering",self.centering)
        
        if self.cl_cut == True:
            # print (distances)
            #print("How many times here?")
            atoms = bulk
            #print("2layers",layers)
            # Interpret symbol
            self.atomic_numbers = atoms.get_atomic_numbers().tolist()
            self.chemical_formula = atoms.get_chemical_formula()
            if self.centering ==True:
                atoms.center()
            self.atomic_basis = atoms.get_scaled_positions()
            #print(np.array(findall('\d+',atoms.get_chemical_formula())))
            ##self.atomic_basis =  cart_to_frac(atoms.positions,atoms.cell)          
            # Interpret lattice constant
            ## do I need this?
            #give the cell from atoms object as lattice basis, because this are no defined in
            #database
            self.lattice_basis = atoms.get_cell()
            self.resiproc_basis = atoms.get_reciprocal_cell()
            
            self.spacegroup = int(str(atoms.info['spacegroup'])[0:3])
            # print ('surfaces,layers')
            # print (surfaces,layers) 
            self.set_surfaces_layers(surfaces, layers)
            ##self.set_distances(surfaces, max_distances)
            print("FIRST CALL IN FACTORY")
            self.set_lattice_size(center)
            self.distances = distances
            if distances is not None:
                # print('surfaces,distances',surfaces,distances)
                self.set_surfaces_distances(surfaces, distances)
                #print("self.distances2",self.distances)
            ##HERE IS WHERE THE MAGIC HAPPENS
            cluster = self.make_cluster(vacuum)
            ##cluster.symmetry = self.xtal_name
            cluster.surfaces = self.surfaces.copy()
            cluster.lattice_basis = self.lattice_basis.copy()
            ##THINK ABOUT THIS
            cluster.atomic_basis = self.atomic_basis.copy()
            ##print("cluster.atomic_basis",cluster.atomic_basis)
            cluster.resiproc_basis = atoms.get_reciprocal_cell()
            if distances is not None:
                cluster.distances = self.distances.copy()
            else:
                cluster.distances = None
            #print("DISTANCES", distances)
            
            
        else:
            self.debug = debug
            # Interpret symbol
            self.set_atomic_numbers(symbols)
            ##print('atomic_numbers',self.atomic_numbers)
            # Interpret lattice constant
            if latticeconstant is None:
                if self.element_basis is None:
                    self.lattice_constant = self.get_lattice_constant()
                else:
                    raise ValueError("A lattice constant must be specified for a compound")
            else:
                self.lattice_constant = latticeconstant
            ##print("element_basis",self.element_basis)
            self.set_basis()
            ##print('self.set_basis()',self.lattice_basis)
    
            if self.debug:
                print("Lattice constant(s):", self.lattice_constant)
                print("Lattice basis:\n", self.lattice_basis)
                print("Resiprocal basis:\n", self.resiproc_basis)
                print("Atomic basis:\n", self.atomic_basis)
    
            self.set_surfaces_layers(surfaces, layers)
            self.set_lattice_size(center)
            
            if self.debug:
                print("Center position:", self.center.round(2))
                print("Base lattice size:", self.size)
            ##HERE IS WHERE THE MAGIC HAPPENS
            cluster = self.make_cluster(vacuum)
            cluster.symmetry = self.xtal_name
            cluster.surfaces = self.surfaces.copy()
            cluster.lattice_basis = self.lattice_basis.copy()
            cluster.atomic_basis = self.atomic_basis.copy()
            cluster.resiproc_basis = self.resiproc_basis.copy()
        return cluster

    def make_cluster(self, vacuum):
        # Make the base crystal by repeating the unit cell
        size = np.array(self.size)
        print("size",size)##
        ##ALL THE POSSIBLE TRANSLATIONS
        translations = np.zeros((size.prod(), 3))
        for h in range(size[0]):
            for k in range(size[1]):
                for l in range(size[2]):
                    i = h * (size[1] * size[2]) + k * size[2] + l
                    #print(i, h,k,l)
                    ##THE LATTICE VECTORS ARE COVOLUTED TO EVERY TRANSLATION
                    translations[i] = np.dot([h, k, l], self.lattice_basis)
                    ##print("translations",translations[i])         
        ##POSITION OF ATOMS IN THE SUPERCELL BEFORE CUTTING
        atomic_basis = np.dot(self.atomic_basis, self.lattice_basis)
        positions = np.zeros((len(translations) * len(atomic_basis), 3))
        
        numbers = np.zeros(len(positions))
        n = len(atomic_basis)
        for i, trans in enumerate(translations):
            
            positions[n*i:n*(i+1)] = atomic_basis + trans
            ##print("PROBLEM:self.atomic_numbers",self.atomic_numbers)
            numbers[n*i:n*(i+1)] = self.atomic_numbers
        ##print("positions")
        ##print(numbers, positions)
        # Remove all atoms that is outside the defined surfaces
        ##print(self.layers)
        
        """The following lines are to print the cluster before the cutting
        
        atoms = self.Cluster(symbols=numbers, positions=positions)
        atom_position = atoms.get_positions()
        print(len(numbers))
        print("Title")
        for j, i in enumerate(atom_position):
            print(int(numbers[j]),"x",*i)
        print("******************************************")
        """
        
        index = -1
        for s, l in zip(self.surfaces, self.layers):
            index += 1
            ##print("SURFACES",self.surfaces,self.layers)
            n = self.miller_to_direction(s)
            ##MAX DISTANCE FROM THE CENTRE
            if self.distances is not None:
                rmax = self.distances[index]
                #print("RMAX",s,rmax)
            else:
                rmax = self.get_layer_distance(s, l + 0.2)
            ##print("FIRST POSITIONS \n",positions)
            r = np.dot(positions - self.center, n)
            #print("positions \n",positions,
            #      "positions - self.center \n",positions - self.center,
            #      "n \n", n,
            #      "r \n", r)
            #print("self.center",self.center)
            ##THIS CUTS OUT THE ATOMS OUTSIDE THE WULFF CONSTRUCTION (MAGIC)
            mask = np.less(r, rmax)
            if self.debug > 1:
                print("Cutting %s at %i layers ~ %.3f A" % (s, l, rmax))
            
            ##AT EACH CYCLE THE POSITION OF THE ATOMS IS UPDATED
            positions = positions[mask]
            numbers = numbers[mask]
            """
            atomss = self.Cluster(symbols=numbers, positions=positions)
            name="test_NP_"+str(index)+".xyz"
            write(name,atomss,format="xyz")
            """
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
        #print(positions.size)
        self.center = self.center - min + vacuum / 2.0
        self.Cluster.unit_cell_formula = self.chemical_formula
        #atoms = self.Cluster(symbols=numbers, positions=positions, cell=cell)
        #self.make_stoich(atoms)
        #print("*****************************************")
        return self.Cluster(symbols=numbers, positions=positions, cell=cell)
        
        """Check on stoichiometry: THIS WILL GO INTO A FUNCTION
        
    def make_stoich(self,atoms):                
        stoich_unit = np.array(findall('\d+',self.chemical_formula))
        #print(self.chemical_formula)
        stoich_unit = stoich_unit.astype(np.int)
        #rint("Stoich_U",stoich_unit)
        divisor_unit = GCD(stoich_unit[0],stoich_unit[1])
        ##print("divisor_unit",divisor_unit)
        unit_formula_unit = stoich_unit/divisor_unit
        ##print("unit_formula_unit",unit_formula_unit)
        """
        """
        atoms = self.Cluster(symbols=numbers, positions=positions, cell=cell)
        """
        """
        ##print("MYB",atoms.get_chemical_formula())
        stoich_cl = np.array(findall('\d+',atoms.get_chemical_formula()))
        stoich_cl = stoich_cl.astype(np.int)
        #print("Cluster composition",stoich_cl)
        #print("Stoich",(stoich_cl/np.amin(stoich_cl))/unit_formula_unit)
        #print("IDEAL",ideal_cl,"DELTA",stoich_cl-ideal_cl)
        #print("MIN where?", np.argmin((stoich_cl/np.amin(stoich_cl))/unit_formula_unit))
        if stoich_cl.size != unit_formula_unit.size:
            #print("NO!")
            self.Cluster.stoichiometry = False
            #self.Cluster.excess = 
        else:
            divisor_cl = GCD(stoich_cl[0],stoich_cl[1])
            unit_formula_cluster = stoich_cl/divisor_cl
            ##print("unit_formula_cluster",unit_formula_cluster)
            ##print("DIFF",-unit_formula_unit+unit_formula_cluster)
            ##print("ID",stoich_cl[0]/unit_formula_unit[0])
            if sum(abs((unit_formula_cluster - unit_formula_unit))) == 0:
                self.Cluster.stoichiometry = True
            else:
                self.Cluster.stoichiometry = False
                #a = min(stoich_cl)
                #print(stoich_cl)
                t= np.argmin((stoich_cl/np.amin(stoich_cl))/unit_formula_unit)
                ideal_cl =unit_formula_unit/t * stoich_cl[t] 
                self.Cluster.excess = stoich_cl - ideal_cl
                
        ##THIS IS THE FINAL CLUSTER
        #print("AA",self.Cluster(symbols=numbers, positions=positions, cell=cell))
    """    
        
        

    def set_atomic_numbers(self, symbols):
        "Extract atomic number from element"
        # The types that can be elements: integers and strings
        atomic_numbers = []
        ##print("self.element_basis",self.element_basis)
        if self.element_basis is None:
            if isinstance(symbols, basestring):
                atomic_numbers.append(ref_atomic_numbers[symbols])
            elif isinstance(symbols, int):
                atomic_numbers.append(symbols)
            else:
                raise TypeError("The symbol argument must be a " +
                                "string or an atomic number.")
            element_basis = [0] * len(self.atomic_basis)
        else:
            if isinstance(symbols, (list, tuple)):
                nsymbols = len(symbols)
            else:
                nsymbols = 0

            nelement_basis = max(self.element_basis) + 1
            if nsymbols != nelement_basis:
                raise TypeError("The symbol argument must be a sequence " +
                                "of length %d" % (nelement_basis,) +
                                " (one for each kind of lattice position")

            for s in symbols:
                if isinstance(s, basestring):
                    atomic_numbers.append(ref_atomic_numbers[s])
                elif isinstance(s, int):
                    atomic_numbers.append(s)
                else:
                    raise TypeError("The symbol argument must be a " +
                                    "string or an atomic number.")
            element_basis = self.element_basis

        self.atomic_numbers = [atomic_numbers[n] for n in element_basis]
        assert len(self.atomic_numbers) == len(self.atomic_basis)

    def set_lattice_size(self, center):
        if center is None:
            offset = np.zeros(3)
        else:
            offset = np.array(center)
            if (offset > 1.0).any() or (offset < 0.0).any():
                raise ValueError("Center offset must lie within the lattice unit \
                                  cell.")

        max = np.ones(3)
        min = -np.ones(3)
        v = np.linalg.inv(self.lattice_basis.T)
        # print (self.lattice_basis.T,'matrix inicial')
        # print (v,'matrix final')
        # print('surfaces and layers')
        # print(self.surfaces,self.layers)
        for s, l in zip(self.surfaces, self.layers):
            n = self.miller_to_direction(s) * self.get_layer_distance(s, l*4)
            ##print('s',s,'l',l,'n',n)
            k = np.round(np.dot(v, n), 2)
            ##print('s',s,'l',l,'n',n,'k',k)
            for i in range(3):
                if k[i] > 0.0:
                    k[i] = np.ceil(k[i])
                elif k[i] < 0.0:
                    k[i] = np.floor(k[i])

            if self.debug > 1:
                print("Spaning %i layers in %s in lattice basis ~ %s" % (l, s, k))
            #print('k',k)
            max[k > max] = k[k > max]
            min[k < min] = k[k < min]
            #a=(0.0,0.0,0.0)
            ##print('max',max,'min',min,'TOTAL',max - min)
            if l % 2 != 0:
                #print('l is even!')
                self.center = np.dot(offset - min, self.lattice_basis)
            else:
                self.center = np.dot(offset - min, self.lattice_basis)
                ##self.center = np.dot(offset - min, self.lattice_basis)
        ##SAVEself.center = np.dot(offset - min, self.lattice_basis)
        ##print('offset - min',offset - min,'\n self.lattice_basis',self.lattice_basis)
        ##print('center',np.dot(offset - min, self.lattice_basis))
        #print("CENTER IS DEFINED HERE",self.center)
        self.size = (max - min + np.ones(3)).astype(int)

    def set_surfaces_layers(self, surfaces, layers):
        ##THE INPUT ONES ARE THE IRREDUCIBLE ONES
        if len(surfaces) != len(layers):
            raise ValueError("Improper size of surface and layer arrays: %i != %i"
                             % (len(surfaces), len(layers)))
        
        ##GETS THE SPACEGROUP SYMMETRY 
        sg = Spacegroup(self.spacegroup)
        # print(sg)
        surfaces = np.array(surfaces)
        layers = np.array(layers)
        # print (layers,'layers')
        
        for i, s in enumerate(surfaces):
            s = reduce_miller(s)
            surfaces[i] = s

        surfaces_full = surfaces.copy()
        layers_full = layers.copy()

        for s, l in zip(surfaces, layers):
            ##RETURN THE EQUIVALENT SURFACES
            equivalent_surfaces = sg.equivalent_reflections(s.reshape(-1, 3))
            # print("s",s,"\n",equivalent_surfaces)

            # print(equivalent_surfaces,'equivalent_surfaces')
         
        

            for es in equivalent_surfaces:
                # If the equivalent surface (es) is not in the surface list,
                # then append it.
                if not np.equal(es, surfaces_full).all(axis=1).any():
                    surfaces_full = np.append(surfaces_full, es.reshape(1, 3), axis=0)
                    layers_full = np.append(layers_full, l)
                    
        
        self.surfaces = surfaces_full.copy()
        self.layers = layers_full.copy()
        # print(len(layers_full),'layers_full')
        # print(len(surfaces_full),'surfaces_full')
        
    def set_surfaces_distances(self, surfaces, distances):
        ##THE INPUT ONES ARE THE IRREDUCIBLE ONES
        if len(surfaces) != len(distances):
            raise ValueError("Improper size of surface and layer arrays: %i != %i"
                             % (len(surfaces), len(distances)))
        
        ##GETS THE SPACEGROUP SYMMETRY 
        sg = Spacegroup(self.spacegroup)
        surfaces = np.array(surfaces)
        distances = np.array(distances)
        # print('distances',distances)
        
        for i, s in enumerate(surfaces):
            s = reduce_miller(s)
            surfaces[i] = s

        surfaces_full = surfaces.copy()
        distances_full = distances.copy()

        for s, l in zip(surfaces, distances):
            ##RETURN THE EQUIVALENT SURFACES
            equivalent_surfaces = sg.equivalent_reflections(s.reshape(-1, 3))

            for es in equivalent_surfaces:
                # If the equivalent surface (es) is not in the surface list,
                # then append it.
                if not np.equal(es, surfaces_full).all(axis=1).any():
                    surfaces_full = np.append(surfaces_full, es.reshape(1, 3), axis=0)
                    distances_full = np.append(distances_full, l)
                    
        

        self.surfaces = surfaces_full.copy()
        # print("self.surfaces",self.surfaces)
        self.distances = distances_full.copy()
        print("self.distances",self.distances)

    def get_resiproc_basis(self, basis):
        """Returns the resiprocal basis to a given lattice (crystal) basis"""
        k = 1 / np.dot(basis[0], cross(basis[1], basis[2]))

        # The same as the inversed basis matrix transposed
        return k * np.array([cross(basis[1], basis[2]),
                             cross(basis[2], basis[0]),
                             cross(basis[0], basis[1])])

# Helping functions
def cross(a, b):
    """The cross product of two vectors."""
    return np.array([a[1]*b[2] - b[1]*a[2],
                     a[2]*b[0] - b[2]*a[0],
                     a[0]*b[1] - b[0]*a[1]])


def GCD(a,b):
    """Greatest Common Divisor of a and b."""
    #print "--"
    while a != 0:
        #print a,b,">",
        a,b = b%a,a
        #print a,b
    return b



def GCD1(a):
    for n in a:
        #print(n)
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
            #print(factors)
            #return factors

    """
    divisors = [ d for d in range(2,a//2+1) if a % d == 0 ]
    print("divisors",divisors)
    return [d for d in divisors if all( d % od != 0 for od in divisors if od != d )]
    """


def reduce_miller(hkl):
    """Reduce Miller index to the lowest equivalent integers."""
    hkl = np.array(hkl)
    old = hkl.copy()

    d = GCD(GCD(hkl[0], hkl[1]), hkl[2])
    while d != 1:
        hkl = hkl // d
        d = GCD(GCD(hkl[0], hkl[1]), hkl[2])

    if np.dot(old, hkl) > 0:
        return hkl
    else:
        return -hkl
    
def cart_to_frac(positions,cell):
    """Convert the atomic positions from cartesian to fractional."""
    atomic_basis = np.transpose(np.dot(np.linalg.inv(np.transpose(cell)),
                                       np.transpose(positions)))
    return atomic_basis
