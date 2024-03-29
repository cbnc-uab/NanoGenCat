from __future__ import print_function
import os, time
import subprocess
import copy
import numpy as np
import pandas as pd
import glob

from os import remove
from re import findall
from random import seed,shuffle,choice
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull
import scipy.constants as constants
from itertools import combinations,product
from math import sqrt

from ase import Atoms,Atom
from ase.atoms import symbols2numbers
from ase.neighborlist import NeighborList
from ase.utils import basestring
from ase.cluster.factory import GCD
from ase.visualize import view
from ase.io import write,read
from ase.data import chemical_symbols,covalent_radii
from ase.spacegroup import Spacegroup
from ase.build import surface as slabBuild

from pymatgen.analysis.wulff import WulffShape
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import IMolecule
from pymatgen.core.surface import SlabGenerator,generate_all_slabs
from pymatgen import Molecule

nonMetals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                  'Si', 'P', 'S', 'Cl', 'Ar',
                  'Ge', 'As', 'Se', 'Br', 'Kr',
                  'Sb', 'Te', 'I', 'Xe',
                  'Po', 'At', 'Rn']
nonMetalsNumbers=symbols2numbers(nonMetals)

delta = 1e-10
_debug = False
seed(42)

def bcn_wulff_construction(symbol, surfaces, energies, size, structure,
    rounding='closest',latticeconstant=None, maxiter=100,
    center=[0.,0.,0.],stoichiometryMethod=1,np0=False,wl_method='hybridMethod',
    sampleSize=1000,totalReduced=False,coordinationLimit='half',polar=False,
    termNature='non-metal',neutralize=False,inertiaMoment=False,debug=0):
    """Function that build a Wulff-like nanoparticle.
    That can be bulk-cut, stoichiometric and reduced
    
    Args:
        symbol(Atom):Crystal structure

        surfaces[lists]: A list of list surfaces index. 

        energies[float]: A list of surface energies for the surfaces.

        size(float): The desired aproximate size.

        structure: The desired crystal structure.  Either one of the strings
        "fcc", "bcc", "sc", "hcp", "graphite"; or one of the cluster factory
        objects from the ase.cluster.XXX modules.

        rounding (optional): Specifies what should be done if no Wulff
        construction corresponds to exactly the requested size.
        Should be a string, either "above", "below" or "closest" (the
        default), meaning that the nearest cluster above or below - or the
        closest one - is created instead.

        latticeconstant (optional): The lattice constant.  If not given,
        extracted from ase.data.

        debug (optional): If non-zero, information about the iteration towards
        the right cluster size is printed.

        center((tuple)): The origin of coordinates

        stoichiometryMethod: Method to transform Np0 in Np stoichometric 0 Bruno, 1 Danilo

        np0(bool): Only gets the Np0, by means, the one that is build by plane replication

        wl_method(string): Method to calculate the plane contributuion. Two options are
        available by now, surfaceBased and distanceBased being the first one the most
        robust solution.

        sampleSize(float): Number of selected combinations

        totalReduced(bool): Removes all unbounded and singly coordinated atoms

        coordinationLimit(int): fathers minimum coordination 

        polar(bool): Reduce polarity of the Np0

        termNature(str): Terminations, could be 'metal' or 'non-metal'

        neutralize(bool): True if hydrogen or OH is added, else False

    """
    global _debug
    _debug = debug

    if debug:
        if type(size) == float:
            print('Wulff: Aiming for cluster with radius %i Angstrom (%s)' %
                  (size, rounding))
        elif type(size) == int:
            print('Wulff: Aiming for cluster with %i atoms (%s)' %
                  (size, rounding))

        if rounding not in ['above', 'below', 'closest']:
            raise ValueError('Invalid rounding: %s' % rounding)
    # Interpret structure, if it is a string.
    if isinstance(structure, basestring):
        if structure == 'fcc':
            ##STRUCTURE INHERITS FROM CLASSES IN FACTORY
            from ase.cluster.cubic import FaceCenteredCubic as structure
        elif structure == 'bcc':
            from ase.cluster.cubic import BodyCenteredCubic as structure
        elif structure == 'sc':
            from ase.cluster.cubic import SimpleCubic as structure
        elif structure == 'hcp':
            from ase.cluster.hexagonal import HexagonalClosedPacked as structure
        elif structure == 'graphite':
            from ase.cluster.hexagonal import Graphite as structure
        elif structure == 'ext':
            from bcnm.bcn_cut_cluster import CutCluster as structure
        else:
            error = 'Crystal structure %s is not supported.' % structure
            raise NotImplementedError(error)

    # Check if the number of surfaces and the number of energies are equal
    nsurf = len(surfaces)
    if len(energies) != nsurf:
        raise ValueError('The energies array should contain %d values.'
                         % (nsurf,))

    #Calculate the interplanar distance
    recCell=symbol.get_reciprocal_cell()
    dArray=interplanarDistance(recCell,surfaces)

    # Get the equivalent surfaces
    eq=equivalentSurfaces(symbol,surfaces)
    
    #Calculate the normal normalized vectors for each surface
    norms=planesNorms(eq,recCell)

    # Get the ideal wulffPercentages
    ideal_wulff_fractions=idealWulffFractions(symbol,surfaces,energies)
    #Array for the np0 properties
    np0Properties=[]

    #This is the loop to get the NP closest to the desired size

    if len(energies) == 1:
        #For systems with only one surface energy, we dont evaluate
        #too much parameters, only chemical formula and min coord 
        scale_f = np.array([0.5])
        distances = scale_f*size
        # print('distances from bcn_wulff_construction',distances)
        layers = np.array(distances/dArray)
    else:
        small = np.array(energies)/((max(energies)*2.))
        large = np.array(energies)/((min(energies)*2.))
        midpoint = (large+small)/2.
        distances = midpoint*size
        layers= distances/dArray
        # print(layers)
    if debug>0:
        print('interplanarDistances\n',dArray)
        print('layers\n',layers)
        print('distances\n',distances)
        print('surfaces\n',surfaces)
    # print('interplanarDistances\n',dArray)
    # print('layers\n',layers)
    # print('distances\n',distances)
    # print('surfaces\n',surfaces)

    # Construct the np0
    atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                        structure, center, latticeconstant,debug)
    # Remove uncordinated atoms
    removeUnbounded(symbol,atoms_midpoint)
    # Check the minimum coordination on metallic centers
    minCoord=check_min_coord(symbol,atoms_midpoint,coordinationLimit)
    # Save midpoint
    name = atoms_midpoint.get_chemical_formula()+str(center)+"_NP0.xyz"
    write(name,atoms_midpoint,format='xyz',columns=['symbols', 'positions'])

    # Check symmetry
    # pymatgenMolecule=IMolecule(species=atoms_midpoint.get_chemical_symbols(),coords=atoms_midpoint.get_positions())
    # pga=PointGroupAnalyzer(pymatgenMolecule)
    # centrosym=pga.is_valid_op(pga.inversion_op)

    # Calculate the Wulff-like index
    if wl_method=='surfaceBased':
        areasIndex=areaCalculation(atoms_midpoint,norms)
        plane_area=planeArea(symbol,areasIndex,surfaces)
        wulff_like=wulffLike(symbol,ideal_wulff_fractions,plane_area[1])
    
        # np0Properties.extend(plane_area[0])
        if np0==True:
            np0Properties=[atoms_midpoint.get_chemical_formula()]
            np0Properties.extend(minCoord)
            np0Properties.append(plane_area[0])
            np0Properties.extend(wulff_like)
            # np0Properties.extend(centrosym)
            return np0Properties
        else:

            reduceNano(symbol,atoms_midpoint,size,sampleSize,coordinationLimit,debug)
            
        if debug>0:
            print('--------------')
            print(atoms_midpoint.get_chemical_formula())
            print('areasIndex',areasIndex)
            print('plane_area',plane_area[0])
            print('--------------')
    #################################
    elif wl_method=='distancesBased':
        wulff_like=wulffDistanceBased(symbol,atoms_midpoint,surfaces,distances)
        np0Properties.extend(wulff_like)
        # plane_area=planeArea(symbol,areasIndex,surfaces)
        if debug>0:
            print('areasIndex',areasIndex)
            print('--------------')
    
        if np0==True:
            np0Properties=[atoms_midpoint.get_chemical_formula()]
            np0Properties.extend(minCoord)
            np0Properties.append(0)
            np0Properties.extend(wulff_like)

            return np0Properties
    ######################################################################
    ######################################################################
    elif wl_method=='wulfflikeLayerBased':
        wulff_like=wulfflikeLayerBased(symbol,surfaces,layers,dArray,ideal_wulff_fractions)
        np0Properties.extend(wulff_like)
        # plane_area=planeArea(symbol,areasIndex,surfaces)
        if debug>0:
            print('areasIndex',areasIndex)
            print('--------------')
    
        if np0==True:
            np0Properties=[atoms_midpoint.get_chemical_formula()]
            np0Properties.extend(minCoord)
            np0Properties.append(0)
            np0Properties.extend(wulff_like)

    #* #####################################################################
    #*#####################################################################
    #* Definitive method
    elif wl_method=='hybridMethod':
        wulff_like=hybridMethod(symbol,atoms_midpoint,surfaces,layers,distances,dArray,ideal_wulff_fractions)
        # print('------------------------------------------------------')
        # exit(1)
        # plane_area=planeArea(symbol,areasIndex,surfaces)
        # if debug>0:
        #     print('areasIndex',areasIndex)
        #     print('--------------')
        # print('polar aqui',polar) 
        if np0==True:
            np0Properties=[atoms_midpoint.get_chemical_formula()]
            np0Properties.extend(minCoord)
            np0Properties.extend(wulff_like)
            # exit(1)
            # np0Properties.append(centrosym)
            return np0Properties
        elif totalReduced==True:
            totalReduce(symbol,atoms_midpoint)
        # elif polar==True:
        #     decoratedNano=forceTermination3(symbol,atoms_midpoint,
        #     surfaces,distances,termNature)
        #     name = decoratedNano.get_chemical_formula()+str(center)+"_NP_"+str(termNature)+".xyz"
        #     write(name,decoratedNano,format='xyz',columns=['symbols', 'positions'])
        elif polar==True:
            models=[]
            finalSize=[]
            # get the polarity indexes 
            ions=[]
            charges=[]
            for element,charge in zip(symbol.get_chemical_symbols(),symbol.get_initial_charges()):
                if element not in ions:
                    ions.append(element)
                    if charge not in charges:
                        charges.append(charge)
            polarity=evaluateSurfPol(symbol,surfaces,ions,charges)
            # print(polarity)
            polarSurfacesIndex=[i for i,pol in enumerate(polarity) if not 'non Polar' in pol]
            # print(polarSurfacesIndex)
            # exit(1)

            # Add the NP0
            # models.append(atoms_midpoint)

            cutoffSets=forceTermination2(symbol,atoms_midpoint,surfaces,distances,dArray,termNature)
            finalSize=[]
            # print(len(cutoffSets))
            # counter=0
            for bunch in cutoffSets:
                # counter+=1
                # print(counter)
                layers=bunch/dArray
                # get the nanoparticle for the new set of distances 
                atoms_midpoint = make_atoms_dist(symbol, surfaces, layers,bunch, 
                                    structure, center, latticeconstant,debug)
                # add each midpoint
                models.append(atoms_midpoint)
                # evauate terminations only on polar surfaces
                termEva=[]
                for s in polarSurfacesIndex:
                    termEva.append(terminationStoichiometry(symbol,atoms_midpoint,[surfaces[s]]))
                # print(polarSurfacesIndex)
                # print(termEvaperSurf)
                # exit(1) 
                    # if len(list(set(termEvaperSurf)))>1:
                    #     print('symmetry equivalent surfaces does not have the same termination')
                    # else:
                    #     termEva.append(termEvaperSurf[0])
                # print(termEva)
                # print(bunch)
                # print(termEva)
                if len(set(termEva))==1:
                    if termEva[0]=="nonMetalRich" and termNature=='non-metal':
                        # view(atoms_midpoint)
                        # print(termEva)
                        # view(models)
                        # exit(1)
                        finalSize.append([layers,bunch,len(atoms_midpoint)])
                    elif termEva[0]=="metalRich"  and termNature=='metal':
                        finalSize.append([layers,bunch,len(atoms_midpoint)])
            # view(models)
            # exit(1)
            if len(finalSize)==0:
                print('Not possible to get the desired termination for this size and centering')
            else:
                orderedSizes=sorted(finalSize,key=lambda x:x[2])

                # keep the smallest nanoparticle
                atoms_midpoint=make_atoms_dist(symbol,surfaces,orderedSizes[0][0].tolist(),orderedSizes[0][1].tolist(),
                                structure,center,latticeconstant,debug)
                removeUnbounded(symbol,atoms_midpoint)
                # name = atoms_midpoint.get_chemical_formula()+str(center)+"_NP_0"+str(termNature)+"_f.xyz"
                # write(name,atoms_midpoint,format='xyz',columns=['symbols', 'positions'])
                models.append(atoms_midpoint)
                # view(atoms_midpoint)
                
                atoms=orientedReduction(symbol,atoms_midpoint,surfaces,orderedSizes[0][1])
                # view(atoms)
                # exit(1)
                models.append(atoms) 

                name = atoms.get_chemical_formula()+str(center)+"_NP_"+str(termNature)+"_f.xyz"
                write(name,atoms,format='xyz',columns=['symbols', 'positions'])
                # view(models)
                if neutralize==True:
                    polarSurf=[surfaces[s] for s in polarSurfacesIndex]
                    adsorbed=addSpecies(symbol,atoms,polarSurf,termNature)
                    write(adsorbed.get_chemical_formula()+str('neutralized_f.xyz'),adsorbed,format='xyz')

        else:
            #* Stoichiometric NPS
            reduceNano(symbol,atoms_midpoint,size,sampleSize,coordinationLimit,inertiaMoment,debug)
            
def make_atoms_dist(symbol, surfaces, layers, distances, structure, center, latticeconstant,debug):
    """
    Function that use the structure to get the nanoparticle.
    All surface related arguments has the same order
    Args:
        symbol(Atoms): crystal structure
        surfaces([list]): list of list of miller surface index
        layers([float]): list of number of layers
        distances([float]): list of distances to cut in each direction.
        structure(ClusterFactory): Class that contains the functions
        to build the nanoparticle
        center([floats]): list of origins
        latticeconstant: latticeconstant
    Return:
        atoms(Atoms): Nanoparticle cut
    """
    # print("here")
    # print(layers)
    layers = np.ceil(layers).astype(int)
    # print(layers)
    # exit(1)
    # print("1layers",layers)
    cluster = structure(symbol, surfaces, layers, distances, center= center,                   
                      latticeconstant=latticeconstant,debug=1)
    
    atoms=Atoms(symbols=cluster.symbols,positions=cluster.positions,cell=cluster.cell)
    # view(atoms)
    atoms.cut_origin=cluster.cut_origin
    return (atoms)

def coordination(atoms,debug,size,n_neighbour):
    #time_0_coord = time.time()
    """Now find how many atoms have the first coordination shell
    """
    
    newpath = './{}'.format(str(int(size)))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    name=atoms.get_chemical_formula()+'_NP0.xyz'
    write(name,atoms,format='xyz',columns=['symbols', 'positions'])

    nearest_neighbour= []
    if n_neighbour is None:
        for i in range(len(atoms.get_atomic_numbers())): 
            nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))
        nearest_neighbour_av = np.average(nearest_neighbour)
        for i in nearest_neighbour:
            if i > nearest_neighbour_av*1.5:
                print("EXITING: there is something strange with the distances, check NP0 for size", int(size))
                return None

    else:
        nearest_neighbour = [n_neighbour]*len(atoms.get_atomic_numbers())


    final = False
    while final == False:
        final = True
        
        C = make_C(atoms,nearest_neighbour)
        # atomicNumbers=atoms.get_atomic_numbers()
        # atomsCoordination=zip(atomicNumbers,C)

        # # for i in atomsCoordination:
        # #     if 
        
        coord=np.empty((0,5))
        for d in set(atoms.get_atomic_numbers()):
            a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        np.max([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        np.min([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        chemical_symbols[d]])
            coord = np.append(coord,[a],axis=0)
        coord = coord[coord[:,4].argsort()]
        print("coord \n",coord)
        
        if check_stoich(atoms,coord) is 'stop':
            print("Exiting because the structure is nonmetal defective")
            return None

        E=[]
        for i in atoms.get_atomic_numbers():
            if int(i) in nonMetalsNumbers:
                E.append(1)
            else:
                for n in coord:
                    if i == int(n[0]):
                        E.append((int(n[2]))/2)
        # print('E',E)
        D = []
        print('atoms pre pop\n',atoms.get_chemical_formula())
        for i,j in enumerate(C):
            if j < E[i]:
                D.append(i)
        for i,j in enumerate(D):
            atoms.pop(j-i)
            nearest_neighbour.pop(j-i)
            C = np.delete(C,j-i)
        print('atoms post pop\n',atoms.get_chemical_formula())
        check_stoich(atoms)
        
        atoms_only_metal = copy.deepcopy(atoms)

        del atoms_only_metal[[atom.index for atom in atoms if atom.symbol in nonMetals ]]
        
        # del atoms_only_metal[[atom.index for atom in atoms if atom.symbol=='O']]
        # del atoms_only_metal[[atom.index for atom in atoms if atom.symbol=='S']]
        center_of_metal = atoms_only_metal.get_center_of_mass()
        
        S = None
        
        dev_s = 100000000.0
        index=0
        if debug == 1:
            print("Atoms to be removed",atoms.excess)
        if atoms.excess is not None:
            """
            atoms.excess is an atribute vector that contains the atom excess to be 
            stoichiometric. e.g. a (IrO2)33 initial nano contains 33 Ir and 80 O atoms
            so to be stoichiometric we have to remove 18 O, this is representated
            by the vector atom.excess [0 14].
            From here until final the stuff is:
            Calculate coordination aka C=make_C
            Generate the an array of shufled list aka S=make_F
            For every element of S recover j non metal atoms.
            in this case 14 metal atoms, the first 14 in tmp_l.

            Kepping in mind that eliminating atoms change the numeration order
            is mandatory to include the changes, this is done by tot.
            in the first elimination, tot =0, so eliminate the atom k
            in the second cycle, tot=1, so eliminate the k-1 atom, that 
            is equivalent to the k atom in the initial atoms object.

            """
            print('int(coord[i,0])',int(coord[0,0]))
            # time.sleep(10)
            for i,j in enumerate(atoms.excess):
                # print(i,j,'i,j')
                if j > 0:
                    C = make_C(atoms,nearest_neighbour)
                    S = make_F(atoms,C,nearest_neighbour,debug)
                    # print('s\n',S)
                    # S =make_F_Test(atoms,coordination_testing)
                    # break
                    for h in range(len(S)):
                        atoms1 = copy.deepcopy(atoms)
                        C1 = copy.deepcopy(C)
                        E1 = copy.deepcopy(E)
                        nearest_neighbour1 = copy.deepcopy(nearest_neighbour)
                        ind=0
                        tmp_l = []
                        for m in S[h]:
                            """
                            This part could be made more pythonic
                            """
                            if ind < int(j):
                                if atoms1.get_atomic_numbers()[m] == int(coord[i,0]):
                                    tmp_l.append(m)
                                    ind = ind + 1
                        # print('tmp_l',tmp_l)
                        tot = 0
                        for k in sorted(tmp_l):
                            # print('tot',tot)
                            atoms1.pop(k-tot)
                            nearest_neighbour1.pop(k-tot)
                            
                            C1 = np.delete(C1,k-tot)
                            E1 = np.delete(E1,k-tot)
                            tot += 1
                        # time.sleep(10)
                        atoms_only_oxygen = copy.deepcopy(atoms1)
                        del atoms_only_oxygen[[atom.index for atom in atoms1 if atom.symbol not in nonMetals]]
                        center_of_oxygen = atoms_only_oxygen.get_center_of_mass()
                        dev = np.std(abs(center_of_metal-atoms_only_oxygen.get_center_of_mass()))
                        dev_p = float("{:.7f}".format(round(float(dev*100),7)))
                        """
                        THIS WRITE IS FOR TESTING PURPOSES
                        
                        if dev_p == 0.0:
                            index += 1
                            name = atoms1.get_chemical_formula()+'_NPtest'+str(index)+".xyz"
                            write(name,atoms1,format='xyz',columns=['symbols', 'positions'])
                        """
                        if debug == True:
                            # comment = ("DC = "+str(dev*100)+' Lattice="' +
                            #            ' '.join([str(x) for x in np.reshape(atoms.cell.T,
                            #                             9, order='F')]) +
                            #            '"')
                            comment = ("DC = "+str(dev*100))
                            name = atoms1.get_chemical_formula()+'_'+str(dev_p)+".xyz"
                            write(name,atoms1,format='xyz',comment=comment,columns=['symbols', 'positions'])
                        if dev < dev_s:
                            dev_s = dev
                            atoms_f = copy.deepcopy(atoms1) 
                            nearest_neighbour_f = copy.deepcopy(nearest_neighbour1)
                            C_f = copy.deepcopy(C1)
                            E_f = copy.deepcopy(E1)
                            if debug == False:
                                if round(dev_s,7) == 0.:
                                    break
            atoms = copy.deepcopy(atoms_f)
            nearest_neighbour = copy.deepcopy(nearest_neighbour_f)
            C = copy.deepcopy(C_f)
            E = copy.deepcopy(E_f)

        C = make_C(atoms,nearest_neighbour)

        for i in range(len(atoms.get_atomic_numbers())):
            if C[i] < E[i]:
                final = False
                print("final",final)
                break
          

    coord_final=np.empty((0,2))
    for d in set(atoms.get_atomic_numbers()):
        a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d])])
        coord_final = np.append(coord_final,[a],axis=0)
    
    check_stoich(atoms)
    if atoms.stoichiometry == True:
        
        if S == None:
            if debug == 1:
                print("It did't go into the coordination loop, check why")
            atoms_only_oxygen = copy.deepcopy(atoms)
            del atoms_only_oxygen[[atom.index for atom in atoms if atom.symbol not in nonMetals]]
            # del atoms_only_oxygen[[atom.index for atom in atoms if atom.symbol!='O']]
            center_of_oxygen = atoms_only_oxygen.get_center_of_mass()
            
            dev_s = np.std(abs(center_of_metal-center_of_oxygen))
            dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))

        else:
            if debug == 1:
                print(len(S)," different combinations were tried resulting in", 
                      len([name for name in os.listdir('.') if os.path.isfile(name)])-1,"final NPs")
                """
                Identify the equal models with sprint coordinates
                """
                # print (os.listdir('.'))
                singulizator(glob.glob('*.xyz'),debug)


        dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))
        name = atoms.get_chemical_formula()+'_NPf_'+str(dev_p)+".xyz"
        # comment = ("DC = "+ str(dev_s*100) +
        #        str(np.reshape(coord_final,(1,4))) +
        #        ' Lattice="' +
        #        'a= '+str(atoms.cell[0,0])+
        #        ' b= '+str(atoms.cell[1,1])+
        #        ' c= '+str(atoms.cell[2,2]) +
        #        '"')
        comment = ("DC = "+ str(dev_s*100)) 

        print("Final NP", atoms.get_chemical_formula(), "| DC =", dev_p, "| coord", coord_final[:,0], coord_final[:,1])
        write(name,atoms,format='xyz',comment=comment,columns=['symbols', 'positions'])
        return atoms
    
def check_stoich(atoms,coord=None):
    stoich_unit = np.array(findall('\d+',atoms.unit_cell_formula))
    stoich_unit = stoich_unit.astype(np.int)
    divisor_unit = GCD(stoich_unit[0],stoich_unit[1])
    unit_formula_unit = stoich_unit/divisor_unit
    stoich_cl = np.array(findall('\d+',atoms.get_chemical_formula()))
    stoich_cl = stoich_cl.astype(np.int)
    a = 0
    b = 0
    if coord is not None:
        for i in range(len(stoich_cl)):
            # if coord[i][4] == 'S' or coord[i][4] == 'O':
            if coord[i][4] in nonMetals:
                a = stoich_cl[i]/unit_formula_unit[i]
            else:
                b = stoich_cl[i]/unit_formula_unit[i]
        if (b == 0) or (a == 0):
            return 'stop'
        if b > a:
            return 'stop'
    atoms.excess = None
    if stoich_cl.size != unit_formula_unit.size:
        atoms.stoichiometry = False
    else:
        divisor_cl = GCD(stoich_cl[0],stoich_cl[1])
        unit_formula_cluster = stoich_cl/divisor_cl
        if sum(abs((unit_formula_cluster - unit_formula_unit))) == 0:
            atoms.stoichiometry = True
        else:
            atoms.stoichiometry = False
            t= np.argmin((stoich_cl/np.amin(stoich_cl))/unit_formula_unit)
            ideal_cl = stoich_cl[t]//unit_formula_unit[t] * unit_formula_unit
            atoms.excess = stoich_cl - ideal_cl
  
def make_C(atoms,nearest_neighbour):
    C=[]
    """Let's test this cause I am not 100% sure"""
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append(len(indices))
    # print('C\n',C)
    return C
    
def make_F(atoms,C,nearest_neighbour,debug):
    #time_0_make_F = time.time()
    """
    Create the F vector, which contains the index of the atom to which the 
    singly coordinated oxygen atoms are linked to.
    The other values are set =len(C) for the other oxygen atoms and
    =len(C)+1 for the other atom types
    """
    # print('c',C)
    # for i,j in enumerate(C):
        # print('element,index, coordination ',atoms[i].symbol,i,j)
    if debug == 1:
        print("Start calculating combinations")
        time_F0 = time.time()
    # F_test=make_F_Test(atoms)
    # print('F_test\n',F_test)

    F=[]
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(C)):
            if atoms.get_atomic_numbers()[i] in nonMetalsNumbers:
                # print(atoms.get_atomic_numbers()[i],i,'holi')
            # if atoms.get_atomic_numbers()[i] == (8 or 16):
                if C[i] == 1:
                    # print('i',i)
                    indices, offsets = nl.get_neighbors(i)
                    # print(indices,i,'indices,i')
                    if len(indices) == 1:
                        F.append(indices[0])
                        # print(i,indices[0],'join')
                    else:
                        print("There is a problema....",indices,C[i])
                        # exit()
                else:

                    F.append(len(C))
            else:
                F.append(len(C)+1)
    # print('aquiiii')
    # print('len(F)\n',len(F))
    # print('F\n',F)
    """
    A nex coordination list is created by adding the values 10 and 11
    for the higher coordination oxygen atoms and other atom types, respectively.
    This is to avoid having an error in the following loop, where the coordination
    is checked and the values len(C) and len(C)+1 would be out of bound
    """
    c = list(C)
    c.append(11)
    c.append(12)
    # print('len(c)\n',len(c),'\n',c)

    K=[]
    n_tests = 1000
    for i in range(n_tests):
        K.append([])
        # break
    # print ('K\n',K)
    """
    l=list of allowed coordinations

    NOT SURE THIS IS STILL VALID
    In the first part of the loop I generate the following lists:
    g: it is a list of indices grouping together the atoms
    linked to singly coordinated atoms, which have the same coordination
    a: gives a list of indices of the order to pick from the atoms.whatever
    in order to select the atoms in g
    In the second loop the G and A lists are equivalent to the g and a lists,
    but the indices g all appear once before appearing a second time. This way
    first all O atoms linked to 6 coordinated atoms, once these lost one O
    they are 5 coordinated atoms and should go after.
    """
    l = [10,9,8,7,6,5,4,3,2,1,11,12]

    F_safe = copy.deepcopy(F)
    c_safe = copy.deepcopy(c)
    for y in range(n_tests):
        a=[]
        F = copy.deepcopy(F_safe)
        c = copy.deepcopy(c_safe)
        for j in l:
            # print('j ',j)
            g = []
            r = list(range(len(F)))
            # print(len(r),'lenr')
            shuffle(r)
            # print(r)
            for i in r:
                if F[i] != None:
                    if c[F[i]] == j:
                        # print('c[F[i]]',c[F[i]],i)
                        # time.sleep(3)
                        if j < 11:
                            if F[i] not in g:
                                g.append(F[i])
                                a.append(i)
                                F[i] = None
                            else:
                                c[F[i]] = j-1
                        else:
                            g.append(F[i])
                            a.append(i)
                            F[i] = None
                # print (i,j,c[F[i]])
                # time.sleep(5)
        # print('a',a)
        # print('g',g)
        K[y].extend(a)

    if debug == 1:
        time_F1 = time.time()
        print("Total time to calculate combinations", round(time_F1-time_F0,5)," s")
    return K

def check_min_coord(symbol,atoms,coordinationLimit):
    """Function that return information that allows to characterize the
    NP0 like undercoordinated metals, the number of metals, non metals
    and also calculates the global coordination.
    Args:
        symbol(Atoms): Crystal structure
        atoms(Atoms): Nanoparticle
        coordinationLimit(str): half or minus2
    Return: 
        characterization([metals,nonmetals,undercoordinated,globalCoord])
            list of information of np0.

    """
    
    characterization=[]

    C=coordinationv2(symbol,atoms)
    indexes=[[c[0],len(c[1])] for c in C]
    # print(indexes)
    ##Sum the coordination of all elements
    globalCoord=np.sum(indexes,axis=0)[1]
    # print('globalCoord',globalCoord)
    # identify the relative abundance of atoms and sort it
    elements=list(set(symbol.get_chemical_symbols()))
    if elements[0] in nonMetals:
        elements.reverse()
    
    abundance=[]
    for e in elements:
        abundance.append(symbol.get_chemical_symbols().count(e))
    
    lessAbundantSpecie=elements[np.argmin(abundance)]
    moreAbundantSpecie=elements[np.argmax(abundance)]

    # Get the less abundant and more abundant in NP
    lessAbIndex=[atom.index for atom in atoms if atom.symbol==lessAbundantSpecie]
    moreAbIndex=[atom.index for atom in atoms if atom.symbol==moreAbundantSpecie]

    #Get the less abundant coordination also the max and min
    lessAbcoord=[i[1] for i in indexes if i[0] in lessAbIndex]
    # print(metalsCoordinations)

    maxCoord=np.amax(lessAbcoord)
    minCoord=np.min(lessAbcoord)
    # print('maxCoord:',maxCoord)

    ##Filling characterization list

    characterization.append(len(lessAbIndex))
    characterization.append(len(moreAbIndex))
    #Evaluate if less abundant have coordination larger than
    #the half of maximium coordination
    # print('minCoord',minCoord)
    if coordinationLimit=='half':
        if minCoord>=maxCoord/2:
            coordTest=True
        else:
            coordTest=False
    elif coordinationLimit=='minus2':
        if minCoord>=maxCoord-2:
            coordTest=True
        else:
            coordTest=False

    characterization.append(coordTest)

    characterization.append(globalCoord)

    return characterization

def singulizator(nanoList,debug):
    """
    Function that eliminates the nanoparticles
    that are equivalent by SPRINT coordinates
    """

    print('Enter in the singulizator')
    time_F0 = time.time()

    sprintCoordinates=[]
    results=[]
    sprintTime=time.time()
    for i in nanoList:
        # print (i)
        sprintCoordinates.append(sprint(i))
        # break
    print('end sprints',np.round((time.time()-sprintTime),2), 's')
    convStart=time.time()
    for c in combinations(range(len(sprintCoordinates)),2):
    #     # print (c[0],c[1],'c')
        if compare(sprintCoordinates[c[0]],sprintCoordinates[c[1]]) ==True:
            results.append(c)
    print('end conv',np.round((time.time()-convStart),2), 's')

    # print(results)
    
    # keeping in mind that a repeated structure can appear
    # on both columns, I just take the first
    
    if debug>0:
        for i in results:
            print('NP '+nanoList[i[0]]+' and '+nanoList[i[1]]+ ' are equal')

    
    results1=[i[0] for i in results]
    # print (results1)
    toRemove=list(set(results1))

    for i in toRemove:
        # print(i)
        # print('NP '+nanoList[results[i][0]]+' and '+nanoList[results[i][1]]+ ' are equal')
        # print('Removing '+nanoList[results[i][0]])
        remove(nanoList[i])
        # pass
    finalModels=len(nanoList)-len(toRemove)
    print('Removed NPs:',len(toRemove))
    print('Final models:',finalModels)

    time_F1 = time.time()
    print("Total time singulizator", round(time_F1-time_F0,5)," s\n")

def sprint(nano):
    """
    Function that calculates the sprint coordinates matrix for a nanoparticle.

    First calculate the coordination, then build the adjacency
    matrix. To calculate the coordination firstly generates a 
    nearest_neighbour cutoffs for NeighborList.

    The C list contains the atom and the binding atoms indices.
    From C list we build the adjMatrix. The idea is translate from
    coordination list to adjacency matrix.

    Then, calculate the sprint coordinates
    Args:
        nano(file): xyz file
    Return:
        sFormated([float]): SPRINT coordinates
    """
    atoms=read(nano,format='xyz')
    atoms.center(vacuum=20)
    adjacencyName=nano+'dat'

    # print (nano)
    nearest_neighbour=[]
    C=[]

    for i in range(len(atoms.get_atomic_numbers())):
        nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))

    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)

    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append([i,indices])
        # print(i,indices) 

    m=len(C)
    adjMatrix=np.zeros((m,m))
    for i in C:
        for j in i[1]:
            adjMatrix[i[0],j]=1.0
    # np.savetxt('adjMatrix',adjMatrix,newline='\n',fmt='%.1f')

    # Diagonal elements defined by 1+zi/10 if i is non metal
    # and 1+zi/100 if is metal

    numbers=symbols2numbers(atoms.get_atomic_numbers())
    # print(numbers)
    for i in range(len(adjMatrix)):
        if numbers[i] <=99 :
            adjMatrix[i][i]=1+float(numbers[i])/10
        else:
            adjMatrix[i][i]=1+float(numbers[i])/100

    # np.savetxt(adjacencyName,adjMatrix,newline='\n',fmt='%.3f')

    # Calculating the largest algebraic eigenvalues and their 
    # correspondent eigenvector
    val,vec=eigsh(adjMatrix,k=1,which='LA')

    # print(val,'val')
    # print(vec)
    
    # Sorting and using positive eigenvector values (by definition)
    # to calculate the sprint coordinates

    vecAbs=[abs(i)for i in vec]
    vecAbsSort=sorted(vecAbs)
    s=[sqrt(len(adjMatrix))*val[0]*i[0] for i in vecAbsSort]
    sFormated=['{:.3f}'.format(i) for i in s]
    # print (s)
    # print(sFormated)
    # print (len(s))
    return sFormated

def compare(sprint0,sprint1):

    """
    compare the SPRINT coordinates between two nanoparticles.
    If two NP has the same sprint coordinates, both are equally
    connected.
    Args:
        sprint0(list): sprint coordinates list
        sprint1(list): sprint coordinates list
    Return:
        (Bool)
    """
    # print(sprint0,'\n',sprint1) 
    # diff=(list(set(sprint0) - set(sprint1)))
    if len(sprint0)==len(sprint1):
        diff=(list(set(sprint0) - set(sprint1)))
        if len(diff)==0:
            return True

def reduceNano(symbol,atoms,size,sampleSize,coordinationLimit,inertiaMoment,debug=0):
    """
    Function that make the nano stoichiometric
    by removing dangling atoms. It is element
    insensitive
    Args:
        symbol(Atoms): Atoms object of bulk material
        atoms(Atoms): Atoms object of selected Np0
        size(float): size of nanoparticle
        sampleSize: number of replicas
        coordinationLimit(str): half or minus2
        debug(int): for print stuff

    """
    print('Enter to reduceNano')
    time_F0 = time.time()


    # Check the stoichiometry of NP0
    if check_stoich_v2(symbol,atoms,debug) is 'stop':
        print("Exiting because the structure can not achieve stoichiometry by removing just one type of ions")
        print("-------------------------")
        return None
    if check_stoich_v2(symbol,atoms,debug) is 'stoichiometric':
        print("NP0 is stoichiometric")
        print("-------------------------")
        name=atoms.get_chemical_formula()+'stoich_f.xyz'
        write(name,atoms,format='xyz',columns=['symbols', 'positions'])
        return None

    # Save as np0_f to distinguish between them and the others 
    name=atoms.get_chemical_formula()+'_NP0_f.xyz'
    write(name,atoms,format='xyz',columns=['symbols', 'positions'])

    ##Recalculate coordination after removal
    C=coordinationv2(symbol,atoms)
    # print('C',C)


    # if debug>0:
    #     atomsBeta=copy.deepcopy(atoms)
    #     for j in C:
    #         if atomsBeta[j[0]].symbol=='Cu':
    #             if len(j[1])==1:
    #                 atomsBeta[j[0]].symbol='Mg'
    #     write('coordinationEvaluation.xyz',atomsBeta)
    #     # print(*C, sep='\n')

    
    #* 4 lists:
    #* singly: contains the indexes of singly coordinated atoms
    #* father: contains the heavy metal atoms which singly
    #* coordinated atoms are bounded
    #* coordFather: contains the coordination of father atoms
    #* fatherFull: contains the combination of father and their coordination.
    
    singly=[i for i in range(len(atoms)) if len(C[i][1])==1]
    # print('singly test')
    # for i in singly:
    #     print(atoms[i].symbol)

    father=list(set([C[i][1][0] for i in singly]))

    coordFather=[len(C[i][1]) for i in father]

    fatherFull=[[i,coordFather[n]] for n,i in enumerate(father)]

    #* Add the excess attribute to atoms object
    #* and checking if the dangling atoms belong
    #* to the excess element. If not, stop
    #* and removing this nps_f
    danglingElement=check_stoich_v2(symbol,atoms,singly,debug)
    if danglingElement=='stop it':
        remove(name)
        return None

    #* if the nano does not have dangling and not stoichiometric, discard 
    #* the model
    
    # if len(singly)==0:
    #     print('NP0 does not have singly coordinated atoms to remove','\n',
    #         'to achive the stoichiometry')
    #     return None 

    if debug>0:
        print('singly:',singly)
        print('father:',father)
        print('coordFather:',coordFather)
        print('fatherFull:',fatherFull)
    #* allowedCoordination must to be generalized
    #* the upper limit is maximum coordination -2
    #* and the inferior limit is the maximum
    #* coordination. i.e. for fluorite, the maximum coordination
    #* is 8, so using list(range(8,6,-1)) we obtain the list
    #* [8, 7, 6, 5, 4] that is fully functional.
    
    maxCord=int(np.max(coordFather))
    # if maxCord == 2:
    #     mid=int(maxCord-2)
    # else:
    # print ('maxCord',maxCord)

    # coordinationLimit Evaluation
    # Default value
    if coordinationLimit=='half':
        mid=int(maxCord/2)
    elif coordinationLimit=='minus2': 
        # User value
        mid=int(maxCord-3)
    # Control the value of coordinationLimit
    # if mid > maxCord or mid<=0:
    #     print('reduction limit must be lower than the maximum coordination,',
    #     'positive, and larger than 0')
    #     return None
    # print('mid',mid)
    # exit(1)

    allowedCoordination=list(range(maxCord,mid,-1))
    print('allowedCoordination',allowedCoordination)
    # exit(1)
    if debug>0:
        print('allowedCoordinations')
        print('excess:',atoms.excess)
        print('sampleSize:',sampleSize)
    # Discard models where can not remove singly coordinated
    if np.min(coordFather) < np.min(allowedCoordination):
        print('We can not remove dangling atoms with the available coordination limits')
        print("-------------------------")
        # exit(1)
        return None
    # To have a large amounth of conformation we generate
    # 1000 replicas for removing atoms. 
    # To make the selection random we use shuffle and 
    # choice. 
    # S=xaviSingulizator(C,singly,father,fatherFull,atoms.excess,allowedCoordination)
    S=daniloSingulizator(C,singly,father,fatherFull,atoms.excess,allowedCoordination,sampleSize)
    if S==None: 
        return None
    # Build the nanoparticles removing the s atom list. Then, calculate the DC

    atomsOnlyMetal=copy.deepcopy(atoms)
    del atomsOnlyMetal[[atom.index for atom in atomsOnlyMetal if atom.symbol in nonMetals]]
    centerOfMetal = atomsOnlyMetal.get_center_of_mass()
    # print('centerOfMetal',centerOfMetal)

    #Calculate the size as the maximum distance between cations
    # npFinalSize=np.amax(atomsOnlyMetal.get_all_distances())
    #Calculate the size as the maximum distance between atoms
    npFinalSize=np.amax(atoms.get_all_distances())

    print('stoichiometric NPs:',len(S))

    nanoList=[]
    for n,s in enumerate(S):
        NP=copy.deepcopy(atoms)
        s.sort(reverse=True)
        del NP[[s]]

        #DC calculation
        atomsOnlyNotMetal = copy.deepcopy(NP)
        del atomsOnlyNotMetal[[atom.index for atom in atomsOnlyNotMetal if atom.symbol not in nonMetals]]
        centerOfNonMetal = atomsOnlyNotMetal.get_center_of_mass()
        # print('centerOfNotMetal',centerOfNonMetal)
        dev = np.std(abs(centerOfMetal-centerOfNonMetal))
        dev_p = float("{:.7f}".format(round(float(dev*100),7)))
        name=str(NP.get_chemical_formula(mode='hill'))+'_'+str(dev_p)+'_'+str(n)+'_f.xyz'
        # print('name',name)
        nanoList.append(name)
        #Saving NP
        write(name,NP,format='xyz')
        #calculating coulomb energy
        #calculating real dipole moment
        # coulomb_energy=coulombEnergy(symbol,NP)
        # # print('coulomb_energy',coulomb_energy)
        # dipole_moment=dipole(NP)
        # size as the maximum distance between cations
        # comment='E:'+str(coulomb_energy)+',mu:'+str(dipole_moment)+'size:'+str(npFinalSize)
        #replace the ase standard comment by our comment
        # command='sed -i \'2s/.*/'+comment+'/\' '+name
        # print(command)
        # subprocess.run(command,shell=True)
        # view(NP)
        # break

    time_F1 = time.time()
    print("Total time reduceNano", round(time_F1-time_F0,5)," s\n")
    #Calling the singulizator function
    if len (nanoList) >1:
        if inertiaMoment==True:
            intertiaTensorSing(atoms,S,C,nanoList) 
        else:
            if npFinalSize<20.0:
                singulizator(nanoList,debug)
            else:
                pass
    else:
        pass
    
def interplanarDistance(recCell,millerIndexes): 
    """Function that calculates the interplanar distances
    using 1/d_hkl^2 = hkl .dot. Gstar .dot. hkl equation.
    A Journey into Reciprocal Space: A Crystallographer's Perspective
    2-7
    Args:
        recCell(list): reciprocal cell of crystal structure
        millerIndexes(list): miller indexes of relevant surfaces
    Returns:
        distances(list): interplanar distance
    """
    # print(type(recCell))
    G=recCell
    # print(G)
    Gstar = np.dot(G, G.T)
    # print(Gstar)
    d=[]
    for indexes in millerIndexes:
        id2 = np.dot(indexes, np.dot(Gstar, indexes))
        d.append(np.sqrt(1/id2))
    # for n,i in enumerate(d):
    #     print(millerIndexes[n],d[n])

    return d

def equivalentSurfaces(symbols,millerIndexes):
    """Function that get the equivalent surfaces for a set of  millerIndexes

    Args:
        symbols(Atoms):Crystal structure
        millerIndexes(list([])): list of miller Indexes
    Return:
        equivalentSurfaces(list([])): list of all equivalent miller indexes
    """
    sg=Spacegroup((int(str(symbols.info['spacegroup'])[0:3])))
    equivalent_surfaces=[]
    for s in millerIndexes:
        equivalent_surfaces.extend(sg.equivalent_reflections(s))
        # print('-----------------------------------')
        # print('s',s)
        # print('equivalent_surfaces',sg.equivalent_reflections(s))
        # print('-----------------------------------')
    return equivalent_surfaces

def planesNorms(millerIndexes,recCell):
    """Function that calculates the normalized
    normal vector of the miller indexes  
    Args:
        miller indexes(list)
    Return:
        norm(list)
    """
    norms=[]
    for millerIndex in millerIndexes:
        normal=np.dot(millerIndex,recCell)
        normal /=np.linalg.norm(normal)
        norms.append([millerIndex,normal])
        # norms[millerIndex]=normal
    # print(norms)
    return norms

def areaCalculation(atoms,norms):
    """Function that calculates the real areas of the faces
    of Np0 and their percentages.
    Args:
        atoms(Atoms): atoms object of NP0
        norms(List): list of norms of miller planes from
        the crystal
    Return:
        percentage(list): surfaces and their percentage
    
    Warning:
        Keep in mind that using the criteria of
    vector·normal=0  to asign the plane, you know
    that you can have two normal planes that are orthogonal
    to each vector, formaly axb, bxa.

    Solution:
        To ovecome the two normals problem,we have
        to evaluate if the normals are pointing
        inwards of outwards of the surface, 
        if the center of the simplex and the normal
        have the same direction, both are pointing
        outwards, else, inwards

    """
    # Steps:

    #     Use the convexHull to get the simplex

    #     evaluate if vectors that form the simplex are
    #     ortogonal to normal vectors of the miller index
    #     related plane.

    #     get the area of the simplexes and sum by plane

    #Reading the atoms object

    #Only get the metal positions, just to make it easy
    positions=np.array([atom.position[:] for atom in atoms if atom.symbol not in nonMetals])
    #Calculate the centroid
    centroid=positions.mean(axis=0)


    #Create the ConvexHull  object
    hull=ConvexHull(positions)
    # print(hull.area)

    #Identify the miller index of the simplices and calculate the area per miller index
    simplices=[]
    for n,simplex in enumerate(hull.simplices):
        u=positions[simplex[0]]-positions[simplex[1]]
        v=positions[simplex[1]]-positions[simplex[2]]
        area=np.abs(np.linalg.norm(np.cross(u, v)) / 2)
        # print(area)
        #Calculate the centroid of the simplex
        tempPos=np.asarray([positions[i] for i in simplex])
        simplexCentroid=tempPos.mean(axis=0)
        # print(tempPos)
        # print(simplexCentroid)
        #Calculate the normalized normal
        normalVector=np.cross(u, v)
        # print(normalVector)
        norm2=normalVector/np.linalg.norm(normalVector)
        # print(n2)
        #identify if is pointing inwards of outwards
        #1 change the origin of pmid to the centroid
        simplexCentroid2=simplexCentroid-centroid
        #2 Calculate the dot product between normalized normal and
        #simplexCentroid2 vector, if the dot product is positive, the normal
        #and the simplexCentroid2 have the same direction. By definition
        #the simplexCentroid2 is pointing outwards of centroid always.
        if np.dot(simplexCentroid2,normalVector)<0:
            norm2*=-1
        #3 compare the norms of each simplex with the crystal norms
        #The threshold of 1e-2 just has been tested for IrO2 and works well.
        #If the diference of between elements of the norm are less than threshold
        #evaluator is True, so that means that the simplex and the 
        #plane has the same normal so they are parallel.
        for i in norms:
            test=i[1]-norm2
            evaluator=all(np.abs(j)< 1e-2 for j in test)
            # print(evaluator)
            if evaluator==True:

                # print('normal of simplex',norm2)
                # print('normal of crystal',i[1])
                # print(np.sum(i[1]-norm2))
                # print (n,i[0])
                # print('--')

                simplices.append([str(i[0]),area])

    areasPerMiller=[]
    areasIndex=[]
    for i in norms:
        millerArea=0
        for j in simplices:
            if str(i[0])==str(j[0]):
                # print(j[0],i[0])
                # print('---')
                millerArea+=j[1]

        areasPerMiller.append(millerArea)
        percentage=millerArea/hull.area
        areasIndex.append([str(i[0]),percentage])
        # print(i[0],percentage)
    # print(areasIndex)
    return areasIndex

def planeArea(atoms,areasIndex,millerIndexes):
    """Function that get the areas per miller index
    and evaluates the symmetry.
    Args:
        atoms(Atoms): atoms object
        areasIndex([index,area]): list of indexes and areas per each index
        millerIndexes([millerIndex]): list of initial miller indexes
    Return:
        symmetric(Boolean): True if symmetric, False else.
        areasPerInitialIndex([index,area]): list of initial indexes and areas per each index

    """
    #get the spacegroup object
    surfaces = np.array(millerIndexes)
    sg=Spacegroup((int(str(atoms.info['spacegroup'])[0:3])))
    
    #For each miller index get the equivalent reflexions and append
    # as strings in equivalent_strings
    areasPerInitialIndex=[]
    symmetric=[]
    for s in surfaces:
        areaPerSurface=[]
        equivalentSurfaces=[]
        equivalentStrings=[]
        equivalentSurfaces.append(sg.equivalent_reflections(s))
        for indexes in equivalentSurfaces:
            for index in indexes:
                equivalentStrings.append(str(index))
        #Make a comparison between the index of area Index and the equivalent_strings lists
        #and accumulate the areas
        for area in areasIndex:
            if area[0] in equivalentStrings:
                areaPerSurface.append(area[1])
        # To evaluate if the nano faces are symmetrical
        # we compare the areas per normal. If the areas are not
        # equal we discard the model
        if np.sum(areaPerSurface)==0.0:
            # print(s,np.sum(areaPerSurface),'\n----')
            areasPerInitialIndex.append([s,np.sum(areaPerSurface)])
        else:
            areasPerInitialIndex.append([s,np.sum(areaPerSurface)])
            # print(s,np.sum(areaPerSurface),'\n----')
            # print(areaPerSurface)
            # print('-------------')
            tempArea=["%.4f"%i for i in areaPerSurface if i!=0.0]
            # print(s,tempArea)
            # print('-------------')
            # temp=np.asarray(tempArea)
            unique=list(set((tempArea)))
            if len(unique)>1:
                # print('not symmetric')
                symmetric.append(0)
            else:
                # print('symmetric')
                symmetric.append(1)
    if 0 in symmetric:
        # print('Non symmetric grow')
        # print(areasPerInitialIndex)
        return False,areasPerInitialIndex
    else:
        # print('Symmetric grow')
        # print(areasPerInitialIndex)
        return True,areasPerInitialIndex

def wulffLike(atoms,idealWulffAreasFraction,areasPerInitialIndex):
    """Function that calculates the wulff-like index,
    defined as the absolute value of the diference of 
    areas normalized by equivalent planes.
    Args:
        idealWulffAreasFraction([index,areas]):list of areas fraction present and areas per each index
        areasPerInitialIndex:([index,area]): list of present and areas per each index
    return: 
        order(bool): true if the planes have the same area contribution in ideal and real np, false if not 
        wli(float) wulff-like index. Close to 0 means that NP are close to wulffShape
    """
    sg=Spacegroup((int(str(atoms.info['spacegroup'])[0:3])))

    wli=[]
    idealAreasPerEquivalent=[]
    realAreasPerEquivalent=[]
    ##Calculate the area per equivalent faces and sort
    for n,i in enumerate(idealWulffAreasFraction):

        # number of equivalent faces
        numberOfEquivalentFaces=len(sg.equivalent_reflections(i[0]))
        # print('equivalentFaces',numberOfEquivalentFaces)

        #Ideal
        indexStringIdeal=''.join(map(str,i[0])) 
        idealAreaPerEquivalent=i[1]/numberOfEquivalentFaces
        idealAreasPerEquivalent.append([indexStringIdeal,idealAreaPerEquivalent])

        #Real
        indexStringReal=''.join(map(str,tuple(areasPerInitialIndex[n][0].tolist())))
        realAreaPerEquivalent=areasPerInitialIndex[n][1]/numberOfEquivalentFaces
        realAreasPerEquivalent.append([indexStringReal,realAreaPerEquivalent])

    #Sorting
    idealAreasPerEquivalentSort=sorted(idealAreasPerEquivalent,key=lambda x:x[1],reverse=True)
    realAreasPerEquivalentSort=sorted(realAreasPerEquivalent,key=lambda x:x[1],reverse=True)

    # print('idealAreasPerEquivalentSort',idealAreasPerEquivalentSort)
    # print('realAreasPerEquivalentSort',realAreasPerEquivalentSort)

    for n,indexArea in enumerate(idealAreasPerEquivalentSort):
        if indexArea[0]==realAreasPerEquivalentSort[n][0]:
            sameOrder=True
        else:
            # print('notEqual')
            sameOrder=False
        break

    #Calculate the index
    wlindex=0
    for n,indexArea in enumerate(idealAreasPerEquivalent):
        wlindex+=abs((indexArea[1]-realAreasPerEquivalentSort[n][1])/numberOfEquivalentFaces)

    return sameOrder,"%.4f"%wlindex

def idealWulffFractions(atoms,surfaces,energies):
    """Function that calculate the ideal wulff Areas Fraction
    using the pymatgen WulffShape.
    Args:
        atoms(atoms):atoms object
        surfaces([(index)]): list of index tuples
        energy([energy]): list of energy floats
    return:
        idealWulffAreasFraction([index,areas]):list of areas fraction present and areas per each index
    """
   
    lattice=atoms.get_cell()
    tupleMillerIndexes=[]
    for index in surfaces:
        tupleMillerIndexes.append(tuple(index))
    # print(type(energies))
    # print(len(energies))
    # print(energies)
    # print(energies[0])
    
    # if type(energies)==np.array:
    #     print('holaaaaa')
    # print('lattice,tupleMillerIndexes,energies',lattice,tupleMillerIndexes,energies)
    idealWulffShape = WulffShape(lattice,tupleMillerIndexes, energies)
    areas=idealWulffShape.area_fraction_dict

    idealWulffAreasFraction=[]
    for millerIndex,areaFraction in areas.items():
        idealWulffAreasFraction.append([millerIndex,areaFraction])
    # print(idealWulffAreasFraction)

    return idealWulffAreasFraction

def coulombEnergy(symbol,atoms):
    """
    Function that calculate the coulomb like energy
    E=sum((qiqj/d(i,j)))for np.
    Args:
        symbol(Atoms): Crystal atoms object
        atoms(Atoms): Atoms object
    Return:
        coulombLikeEnergy(float): coulomb like energy in atomic units
    """
    #Add the charges
    for iatom in atoms:
        for jatom in symbol:
            if iatom.symbol==jatom.symbol:
                iatom.charge=jatom.charge

    #Calculating the energy
    coulombLikeEnergy=0
    for c in combinations(atoms,2):
        #coulomb
        tempCoulomb=(c[0].charge*c[1].charge)/euclidean(c[0].position,c[1].position)
        # print('tempCoulomb',tempCoulomb)
        coulombLikeEnergy+=0.529*tempCoulomb

    return coulombLikeEnergy

def dipole_np(atoms):
    """
    Function that calculate the dipole moment
    E=sum(qi/ri))for np.
    Args:
        atoms(Atoms): Atoms object
    Return:
        dipole(float): dipole in atomic units
    """
    dipole=0
    for atom in atoms:
        dipoleTemp=np.sum(atom.charge*atom.position*1.88973/8.478353552e-30)
        dipole+=dipoleTemp

    return dipole

def specialCenterings(spaceGroupNumber):
    """
    Function that returns the centerings that Xavi proposes on
    the draft
    Args:
        spaceGroupNumber(int):Standard space group number
    Return:
        special_centerings([tuples]): List of tuples with special centerings
        per spacegroup

    """
    data=[
    [[
    22, 42, 43, 69, 70, 196, 202, 203, 209, 210, 216, 219, 225, 
    226, 227, 228],
    [(0.25, 0.25, 0.25),(0.75, 0.25, 0.25),(0.25, 0.75, 0.25),
    (0.25, 0.25, 0.75),(0.5, 0.0, 0.0),(0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),(0.5, 0.5, 0.5)
    ]],
    [[
    23, 24, 44, 45, 46, 71, 72, 73, 74,79, 80, 82, 87, 88, 97, 98,
    107, 108, 109, 110, 119, 120, 121, 122, 139, 140, 141, 142,
    197, 199, 204, 206, 211, 214, 217, 220, 229, 230],
    [(0.5, 0.0, 0.0),(0.0, 0.5, 0.0),(0.0, 0.0, 0.5),(0.5, 0.5, 0.0),
    (0.0, 0.5, 0.5),(0.5, 0.0, 0.5)
    ]],
    [[
    5, 8, 9, 12, 15, 20, 21, 35, 36, 37, 63, 64, 65, 66, 67, 68],
    [(0.5, 0.0, 0.0),(0.0, 0.5, 0.0),(0.0, 0.0, 0.5),
    (0.0, 0.5, 0.5),(0.5, 0.0, 0.5),(0.5, 0.5, 0.5)
    ]],
    [[
    38, 39, 40, 41],
    [(0.5, 0.0, 0.0),(0.0, 0.5, 0.0),(0.0, 0.0, 0.5),
    (0.5, 0.5, 0.0),(0.5, 0.0, 0.5),(0.5, 0.5, 0.5)
    ]],
    [[
    1, 2,3, 4, 6, 7, 10, 11, 13, 14,16, 17, 18, 19, 25, 26, 27,
    28, 29, 30, 31, 32, 33,34, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62,75, 76, 77, 78, 81, 83, 84,
    85, 86, 89, 90, 91, 82, 93, 94, 95, 96, 99, 100, 101, 102,
    103, 104, 105, 106, 111, 112, 113, 114, 115, 116, 117, 118,
    123, 124, 125, 126, 127, 128 ,129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 195, 198, 200, 201, 205, 207, 208, 212,
    213, 215, 218, 221, 222, 223, 224.],
    [(0.5, 0.0, 0.0),(0.0, 0.5, 0.0),(0.0, 0.0, 0.5),(0.5, 0.5, 0.0),
    (0.0, 0.5, 0.5),(0.5, 0.0, 0.5),(0.5, 0.5, 0.5)
    ]],
    [[138,139,140,141,142,143,144,145,146,147,148,149,150,
    151,152,153,154,155,156,157,158,159,160,161,162,163,
    164,165,166,167,168, 169, 170, 171, 172, 173, 174, 175, 
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 
    188, 189, 190, 191, 192, 193, 194],
    [(0.50, 0.50, 0.50), (0.00, 0.00, 0.50),(0.333, 0.333, 0.00),
    (0.50, 0.00, 0.00), (0.50, 0.50, 0.00),(0.333, 0.333, 0.50),
    (0.50, 0.00, 0.50),(0.333, 0.667, 0.50) 
    ]]
    ]

    for familly in data:
        if spaceGroupNumber in familly[0]:
            # print('match')
            # print(familly[1])
            special_centerings=copy.deepcopy(familly[1])
    # # print(centerings)
    return special_centerings

def check_stoich_v2(Symbol,atoms,singly=0,debug=0):
    """
    Function that evaluates the stoichiometry
    of a np.
    To do it compare calculate the excess
    of majoritary element. If not excess
    the Np is stoichiometric,
    else  add the excess atribute to atoms
    object.
    Args:
        Atoms(atoms): Nanoparticle
        Symbol(atoms): unit cell atoms structure
        singly(list): List of single coordinated atoms
        debug(int): debug to print stuff
    """

    #Get the stoichiometry of the unit cell

    #Get the symbols inside the cell
    listOfChemicalSymbols=Symbol.get_chemical_symbols()
    
    #Count and divide by the greatest common divisor
    chemicalElements=list(set(listOfChemicalSymbols))

    # put the stuff in order, always metals first

    if chemicalElements[0] in nonMetals:
        chemicalElements.reverse()

    counterCell=[]
    for e in chemicalElements:
        counterCell.append(listOfChemicalSymbols.count(e))

    gcd=np.gcd.reduce(counterCell)

    cellStoichiometry=counterCell/gcd
    # Compare the cell stoichiometry with the np stoichiometry
    
    # Get the nano stoichiometry
    listOfChemicalSymbolsNp=atoms.get_chemical_symbols()
    
    #Count and divide by the greatest common divisor
    counterNp=[]
    for e in chemicalElements:
        # print(e)
        counterNp.append(listOfChemicalSymbolsNp.count(e))
    # print(counterNp)
    gcdNp=np.gcd.reduce(counterNp)

    nanoStoichiometry=counterNp/gcdNp
    # print('nanoStoichiometry:',nanoStoichiometry)
    
    # ###
    # # The nanoparticle must respect the same ratio of ions in the crystal
    # # Just one of the ions can be excesive
    # ## Test one, just the largest in proportion is in excess

    # Get the index of the maximum value of nano stoichiometry
    # that is the element that is in excess
    excesiveIonIndex=np.argmax(nanoStoichiometry)


    ## calculate how many atoms has to be removed
    excess=np.max(counterNp)-np.min(counterNp)*(np.max(cellStoichiometry)/np.min(cellStoichiometry))

    ## verify that the number of excess are larger or equal to singly
    if debug>0:
        print('values')
        print(np.max(counterNp),np.min(counterNp),np.max(cellStoichiometry))
        print(chemicalElements[excesiveIonIndex])
        print('cellStoichiometry',cellStoichiometry)
        print('nanoStoichiometry',nanoStoichiometry)
        print(excess)

    if excess==0:
        return 'stoichiometric'
    if singly !=0:
        if len([i for i in singly if atoms[i].symbol==chemicalElements[excesiveIonIndex]])<excess:
            print('NP0 does not have enough singly coordinated excess atoms to remove','\n',
                'to achive the stoichiometry for this model')
            print("-------------------------")
            return 'stop it'

    elif excess<0 or excess%1!=0:
        return 'stop'
    else:
        atoms.excess=excess
    # if singly !=0:
    #     # print('holaaaaa')
    #     # print('singly del none',singly)
    #     # test=[i for i in singly]
    #     # print('test',test)

    #     if len([i for i in singly if atoms[i].symbol==chemicalElements[excesiveIonIndex]])<excess:
    #         print('NP0 does not have enough singly coordinated excess atoms to remove','\n',
    #             'to achive the stoichiometry for this model')
    #         return 'stop'
   

        # print('atoms excess',atoms.excess)
    
    # if excess==0:
    #     return 'stoichiometric'
    # elif excess<0 or excess%1!=0:
    #     return 'stop'

def coordinationv2(symbol,atoms):
    """
    function that calculates the
    coordination based on cutoff
    distances from the crystal,
    the distances was calculated
    by using the MIC
    Args:
        symbol(atoms): atoms object of the crystal
        atoms(atoms): atoms object for the nano
    """
    # get the neigboors for the crystal object by
    # element and keeping the maxima for element
    # as unique len neighbour

    red_nearest_neighbour=[]
    distances=symbol.get_all_distances(mic=True)
    elements=list(set(symbol.get_chemical_symbols()))
    # print(elements)
    for i in elements:
        # print(i)
        nearest_neighbour_by_element=[]
        for atom in symbol:
            if atom.symbol ==i:
                nearest_neighbour_by_element.append(np.min([x for x in distances[atom.index] if x>0]))
        # print(list(set(nearest_neighbour_by_element)))
        red_nearest_neighbour.append(np.max(nearest_neighbour_by_element))
    # print('red_nearest')
    # print(red_nearest_neighbour)

    #construct the nearest_neighbour for the nano
    nearest_neighbour=[]
    for atom in atoms:
        for n,element in enumerate(elements):
            if atom.symbol==element:
                # print('n',n)
                nearest_neighbour.append(red_nearest_neighbour[n])
    # print('nearest')
    # print(nearest_neighbour) 
    C=[]
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append([i,indices])
    return C
    # 
    # print(C)

def wulffDistanceBased(symbol,atoms,surfaces,distance):
    """
    Function that evaluates if equivalent
    faces has the same lenght from the center of the material
    or not. Also calculates the WLI

    Warning: Written for only one surface energy.

    Args:
        symbol(Atoms): bulk atom object
        atoms(Atoms): nanoparticle atoms type
        surface(list): surface miller index
        distance(float): distance from the center to the wall
    Return:
        results(list): List that contains:
                        Symmetric growing(Bool): True if symmetric

    """
    # if len(surfaces)>1:
    #     error = 'distanceBased method only available for one surface'
    #     raise NotImplementedError(error)
    # print('surfaces',surfaces)
    result=[]
    # Get the equivalent surfaces and give them the distance
    # Creating the spacegroup object
    sg=Spacegroup((int(str(symbol.info['spacegroup'])[0:3])))

    positions=np.array([atom.position[:] for atom in atoms])
    centroid=positions.mean(axis=0)

    #Create the ConvexHull  object
    hull=ConvexHull(positions)
    # print(hull.area)

    simplices=[]
    for simplex in hull.simplices:
        simplices.extend(simplex)
    surfaceAtoms=sorted(list(set(simplices)))

    #Save the atoms surface in a new atoms object
    outershell=copy.deepcopy(atoms)
    del outershell[[atom.index for atom in outershell if atom.index not in surfaceAtoms]]

    #Get the equivalent surfaces
    for s in surfaces:
        equivalentSurfaces=sg.equivalent_reflections(s)
        # print('-------------------')
        # print(s,equivalentSurfaces)
        # print('-------------------')
        equivalentSurfacesStrings=[]
        for ss in equivalentSurfaces:
            equivalentSurfacesStrings.append(ss)
        # break
        # Get the direction per each miller index
        #Project the position to the direction of the miller index
        # by calculating the dot produequivalentSurfacesStringsct
        rs=[]
        auxrs=[]
        # print('test distances based')
        for i in equivalentSurfacesStrings:
            rlist=[]
            direction= np.dot(i,symbol.get_reciprocal_cell())
            direction = direction / np.linalg.norm(direction)
            for n,j in enumerate(outershell.get_positions()):
                rlist.append(np.dot(j-centroid,direction))
            # print('surface,distance',i,np.max(rlist))
            # print('position',outershell.get_positions()[np.argmax(rlist)])
            # print('...........................')
            auxrs.append(np.max(rlist))
            rs.append([i,np.max(rlist)])
        maxD=np.max(auxrs)
        #Normalize each distance by the maximum
        totalD=0.0
        for i in rs:
            # print(i[1]/maxD)
            i[1]=i[1]/maxD
            totalD+=i[1]
        # #Calculate the area of each ones
        percentages=[]
        auxPercentage=[]
        for i in rs:
            areaPerPlane=hull.area*i[1]/totalD
            percentages.append([''.join(map(str,i[0])),np.round(areaPerPlane/hull.area,3)])
            auxPercentage.append(np.round(areaPerPlane/hull.area,3))
        # print('auxPercentage',len(auxPercentage))
        ### evaluate if those are equal, limit to  0.1 of difference(10%)
        minArea=np.min(auxPercentage)
        maxArea=np.max(auxPercentage)
        # print('minArea,maxArea',minArea,maxArea)
        avArea=(minArea+maxArea)/2
        # print(avArea)
        symmetric=[]
        for i in percentages:
            # if(np.abs(i[1]-avArea))<0.001:
            if (np.abs(i[1]-maxArea)/maxArea)<0.005:
                symmetric.append(0)
            else:
                symmetric.append(1)
        # print(symmetric)
        if 1 in symmetric:
            result.append(False)
        else:
            result.append(True)
        # print(result)
    ##
    #Calculate the WLI
    ####
    #Ideal surface contribution percentage
    idealAreasPerEquivalent=[]
    if len(surfaces)==1:
        area=1.0/len(equivalentSurfacesStrings)
        for i in equivalentSurfacesStrings:
            indexString=''.join(map(str,tuple(i)))
            idealAreasPerEquivalent.append([indexString,area])

    #Sorting
    idealAreasPerEquivalentSort=sorted(idealAreasPerEquivalent,key=lambda x:x[1],reverse=True)
    realAreasPerEquivalentSort=sorted(percentages,key=lambda x:x[1],reverse=True)

    #Test the order
    # print('heerreee')
    # print('idealAreasPerEquivalentSort',idealAreasPerEquivalentSort)
    # print('realAreasPerEquivalentSort',realAreasPerEquivalentSort)
    # for real,ideal in zip(realAreasPerEquivalentSort,idealAreasPerEquivalentSort):
    #     if str(real[0])==str(ideal[0]):
    #         sameOrder=True
    #     else: 
    #         sameOrder=False
    #     break
    # 
    # print(sameOrder)
    # result.append(sameOrder)
    #Calculate the index
    wlindex=0
    for n,indexArea in enumerate(idealAreasPerEquivalent):
        wlindex+=abs((indexArea[1]-realAreasPerEquivalentSort[n][1]))

    result.append(wlindex)

    return result

def xaviSingulizator(C,singly,father,fatherFull,excess,allowedCoordination):
    """
    Function that returns list of atoms to be removed
    to achieve stoichiometry(partial randoms)
    Args:
        C: coordination list 
        singly: list of singly coordinated atoms
        father: list of fathers of singly coordinated atoms
        fatherFull: list of fathers and their coordination
        excess: number of singly to remove
    Return:
        S: list of list of atmos to remove.
    """
    start=time.time()
    fatherFull_bak=copy.deepcopy(fatherFull)
    singly_bak=copy.deepcopy(singly)

    # allowedCoordination=list(range(maxCord,mid,-1))

    S=[]
    replicas=1000
    for r in range(replicas):
        toRemove=[]
        fatherFull=copy.deepcopy(fatherFull_bak)
        singly=copy.deepcopy(singly_bak)
        for cordLevel in allowedCoordination:
            fathersAtthisLevel=[[n,i] for n,i in enumerate(fatherFull) if i[1]==cordLevel]
            #Structure of each element of fathersAtThisLevel is [position[atomIndex,coordination]]
            startShuffle=time.time()
            shuffle(fathersAtthisLevel)
            # print('cordLevel',cordLevel)
            # print('sample',fathersAtthisLevel)
            end=time.time()
            for j in fathersAtthisLevel:
                # print(j[1][0])
                singlyFather=[k for k in C[j[1][0]][1] if k in singly]
                # print(singlyFather)
                if len(singlyFather)>0:
                    chosen=choice(singlyFather)   
                    toRemove.append(chosen)
                    # print(toRemove)
                    singly.remove(chosen)
                    if len(toRemove)==excess:
                        # print('coordinationAllowed',i)
                        break
                    fatherFull[j[0]][1]=fatherFull[j[0]][1]-1
            if len(toRemove)==excess:
                # print('coordinationAllowed',i)
                break
            # print(fatherFull)
            # print('lalala',fathersAtthisLevel)
            # break

        S.append(sorted(toRemove))
        # print(len(S))


    # at the end we get an array S with 10000 list of atoms
    # to be removed. Previous to the removal and to make the things faster
    # we remove duplicates (I guess that is not duplicates in the list)

    nanoList=[]

    # Generate the list of pairs and select the repeated pairs
    # the aceptance criteria is if the intersection between
    # two s are iqual to the len of the first s. 
    # The repeated list is set and reversed before remove 
    # s elements 

    pairs=[c for c in combinations(range(1000),2)]

    repeatedS=[]
    for c in pairs:
        # print (c[0],S[c[0]])

        if len(list(set(S[c[0]]).intersection(S[c[1]]))) == len(S[c[0]]):
            # print(c,' are duplicates')
            repeatedS.append(c[0])
            # del S[c[0]]
    # print (n)

    uniqueRepeated=list(set(repeatedS))
    uniqueRepeated.sort(reverse=True)

    for i in uniqueRepeated:
        del S[i]
    end=time.time()
    # print('ExecutionTime,finalSamples,excess',end-start,len(S),excess)
    # print(len(S))
    return(S)

def daniloSingulizator(C,singly,father,fatherFull,excess,allowedCoordination,sampleSize):
    """
    Function that returns list of atoms to be removed
    to achieve stoichiometry(random over all fathers)
    Args:
        C: coordination list 
        singly: list of singly coordinated atoms
        father: list of fathers of singly coordinated atoms
        fatherFull: list of fathers and their coordination
        excess: number of singly to remove
        allowedCoordination(list): list of allowed coordinations
        sampleSize(int): Number of replicas
    Return:
        S: list of list of atoms to remove.
    """
    # print('singulizator')
    start=time.time()

    fatherFull_bak=copy.deepcopy(fatherFull)
    singly_bak=copy.deepcopy(singly)

    # The loop basically select the metal
    # atom of higest coordination,aka father, identify the singly coordinated 
    # atoms bounded to it and choose one randomly.
    # Then append the selected and reduce the coordination of father.
    # the process is repeated until the len of remove are equal to 
    # excess.
    S=[]
    replicas=sampleSize

    for r in range(replicas):
        toRemove=[]
        fatherFull=copy.deepcopy(fatherFull_bak)
        singly=copy.deepcopy(singly_bak)
        for i in allowedCoordination:
            shuffle(fatherFull)
            for n,j in enumerate(fatherFull):
                # get the ones that have the allowed coordination
                if fatherFull[n][1]==i:
                    # if debug>0:
                    # print('fatherFull[n][1]',fatherFull[n])
                    #create a list with the single coordinated atoms joined to that atom
                    singlyFather=[k for k in C[j[0]][1] if k in singly]
                    # if that list is larger than 0 choice one
                    if len(singlyFather)>0:
                        # if debug>0:
                        #     print('singlyFather',singlyFather)
                        chosen=choice(singlyFather)
                        # print('chosen',chosen)
                        if chosen not in toRemove:
                            #If that atom was not previously chosen
                            #keep it, remove from the singly coordinated
                            #list and decrease the coordination
                            if len(toRemove)==excess:
                                break
                            toRemove.append(chosen)
                            # print('singly',singly)
                            singly.remove(chosen)
                            fatherFull[n][1]=fatherFull[n][1]-1
                            # print(fatherFull)
                            if fatherFull[n][1] < min(allowedCoordination):
                                break
                # print(len(toRemove),'toRemove',toRemove)
                # print('fatherFull',fatherFull)
            if len(toRemove)==excess:
                break
        # print('len(toRemove)',len(toRemove))
        # print(len(toRemove))
        # exit(1)
        if len(toRemove)< excess:
            print ('Is not possible to achieve coordination with the available coordinaation limits')
            print('allowedCoordination:',allowedCoordination)
            print("-------------------------")
            # exit(1)
            S.append(None)
            break
        else:
            S.append(sorted(toRemove))
        # print(len(S))
    # at the end we get an array S with 10000 list of atoms
    # to be removed. Previous to the removal and to make the things faster
    # we remove duplicates (I guess that is not duplicates in the list)
    # print('Inside singu',S)
    if None in S:
        return None 
    nanoList=[]

    # Generate the list of pairs and select the repeated pairs
    # the aceptance criteria is if the intersection between
    # two s are iqual to the len of the first s. 
    # The repeated list is set and reversed before remove 
    # s elements 

    pairs=[c for c in combinations(range(sampleSize),2)]

    repeatedS=[]
    for c in pairs:
        # print (c[0],S[c[0]])

        if len(list(set(S[c[0]]).intersection(S[c[1]]))) == len(S[c[0]]):
            # print(c,' are duplicates')
            repeatedS.append(c[0])
            # del S[c[0]]
    # print (n)

    uniqueRepeated=list(set(repeatedS))
    uniqueRepeated.sort(reverse=True)

    for i in uniqueRepeated:
        del S[i]
    end=time.time()
    # print('ExecutionTime,finalSamples,excess',end-start,len(S),excess)

    return(S)

def wulfflikeLayerBased(symbol,surfaces,layers,distances,ideal_wulff_fractions):
    '''
    Function that creates the wulff polyhedron from distances
    instead of surface energies, based on the proportionality
    relationship between distances and surface energies.
    and compares with the ideal wulff fractions, computed by using the
    initial surface energies
    Args:
        symbol(Atoms): Atoms type 
        surfaces([srt]):List of surface indexesk
        layers([float]): List of number of layers
        distances([float]): List of interplanar distances
    Return:
      Percentages([float]): Area percentages contribution
    '''
    # Round the layers
    layersRound = [np.round(l).astype(int) for l in layers]

    # print('layers after rounding',layersRound)
    lenghtPerPlane=[]
    # Get the distances and use it to calculate the areas contribution per orientation

    for layer,distanceValue in zip(layersRound,distances):
        # print(distanceValue*(layer[0]+0.5))
        # print(layer) 
        lenghtPerPlane.append(distanceValue*(layer+0.5))
    # print('lenghtPerPlane',lenghtPerPlane)
    # get all the equivalent surface and the proper lenght 
    sg=Spacegroup((int(str(symbol.info['spacegroup'])[0:3])))
    AllDistances=[]
    AllSurfaces=[]
    numberOfEquivalentFaces=[]
    for s in surfaces:
    #     # get the surfaces and save it on a list
    #     equivalentSurfaces=[]
        numberOfEquivalentFaces.append(len([eqSurf.tolist() for eqSurf in sg.equivalent_reflections(s)]))
    #     AllSurfaces.extend([str(s) for s in equivalentSurfaces])
    #     # Get the equivalent layers and save it on a list
    #     equivalentDistances=[lenght for i in range(len(equivalentSurfaces))]

    
    #     AllDistances.extend(equivalentDistances)
    # print('surfaces',surfaces)
    # print('distances',lenghtPerPlane)
    
    realAreas=idealWulffFractions(symbol,surfaces,lenghtPerPlane)
    print('real',realAreas)
    print('ideal',ideal_wulff_fractions,'\n--------------------------------------------')

    ## Sort the ideal and real 
    sortedIdeal=sorted(ideal_wulff_fractions,key=lambda x:x[1],reverse=True)
    sortedReal=sorted(realAreas,key=lambda x:x[1],reverse=True)

    # print('real',sortedReal)
    # print('ideal',sortedIdeal)
    for real,ideal in zip(sortedIdeal,sortedReal):
        if str(real[0])==str(ideal[0]):
        # break
            # print('real,ideal')
            # print(real,ideal)
            sameOrder=True
        else: 
            sameOrder=False
        break
    # print(sameOrder)
    #Calculate the index
    wlindex=0
    for n,indexArea in enumerate(ideal_wulff_fractions):
        wlindex+=abs((indexArea[1]-realAreas[n][1])/numberOfEquivalentFaces[n])

    # print(wlindex)
    return sameOrder,"%.4f"%wlindex
     
    # exit(1)
    # return wulffLike(symbol,ideal_wulff_fractions,realAreas)

def hybridMethod(symbol,atoms,surfaces,layers,distances,interplanarDistances,ideal_wulff_fractions):
    """
    Function that computes the wulff related quality parameters, equivalent planes
    areas, same order and WLI using two methods, wulfflikeLayerBased and Distancesbased methods
    """
    results=[]

    # Execute the Distancesbased method to get the equivalent grow
    # results.append(wulffDistanceBased(symbol,atoms,surfaces,distances)[0])
    #* Testing the new distanceBased
    results.append(wulffDistanceBasedVer2(symbol,atoms,surfaces,distances))
    #* Execute the layerbased to get the order and wulff-like index
    results.extend(wulfflikeLayerBased(symbol,surfaces,layers,interplanarDistances,ideal_wulff_fractions)[:])

    return results

def totalReduce(symbol,atoms):
    """
    Function that removes all dangling atoms
    no element sensible 
    Args:


        symbol(Atoms): Crystal structure
        atoms(Atoms): Nanoparticle 0
    """
    print('running total reduce')
    # Calculate C
    C=coordinationv2(symbol,atoms)
    toRemove=[]
    for c in C:
        if len(c[1])==0 or len(c[1].tolist())==1:
            # print(c)
            toRemove.append(c[0])
    print(toRemove)
    toRemove=sorted(toRemove,reverse=True)
    del atoms[toRemove]
    # print(toRemove)
    # get the total charge
    for iatom in atoms:
        for jatom in symbol:
            if iatom.symbol==jatom.symbol:
                iatom.charge=jatom.charge
    totalCharge=np.sum(atoms.get_initial_charges())
    print('warning: ',atoms.get_chemical_formula(mode='hill'), 'has total charge', totalCharge)
    # save the reduced Np
    name=str(atoms.get_chemical_formula(mode='hill'))+str('_reduced_f.xyz')
    write(name,atoms,format='xyz',columns=['symbols', 'positions'])
    comment='Total charge:'+str(totalCharge)
    command='sed -i \' 2s/.*/'+comment+'/\' '+name
    subprocess.run(command,shell=True)

def removeUnbounded(symbol,atoms):
    """
    Function that removes the unbounded atoms
    Args:
        symbol(Atoms): crystal structure
        atoms(Atoms): Bulk-cut like nanoparticle
    Return:
        atoms(Atoms): nanoparticle without unbounded atoms
    """
    C=coordinationv2(symbol,atoms)
    notBonded=[]
    for c in C:
        if len(c[1])==0:
            notBonded.append(c[0])

    notBonded=sorted(notBonded,reverse=True)
    del atoms[notBonded]

    # return(atoms)

def dipole(slab):
    """
    Function that calculates the dipole moment
    of a slab model per area unit.
    If that value are larger than threshold, the slab is
    polar
    Args:
        slab(Atoms): slab
    Return:
        bool: normalized dipole per area unit
    """
    normal=np.cross(slab.get_cell()[0],slab.get_cell()[1])
    normal/=np.linalg.norm(normal)

    dipole=np.zeros(3)
    # Get the midpoint
    midPoint=np.sum(slab.get_positions(),axis=0)/len(slab.get_atomic_numbers())
    for atom in slab:
        dipole+=atom.charge*np.dot(atom.position - midPoint,normal)*normal
    area=np.linalg.norm(np.cross(slab.get_cell()[0],slab.get_cell()[1]))
    # print(area)
    dipolePerArea=dipole/area
    return np.linalg.norm(dipolePerArea)

def reduceDipole(symbol,surfaces,distances,interplanarDistances,center):
    """
    Function that increase the cut-offdistance to get less 
    polar nanoparticles. The surfaces,distances and
    interPlanarDistances MUST have the same order
    Args:
        symbol(Atoms): crystal structure
        surfaces([list]): surface miller indexes
        distances([float]): cut-off distances
        interplanarDistances([float]): interplanar distances
        center((float)): center
    return:
        finalDistances([float]): Distances that reduce the 
        dipole.

    """
    charges=[]
    ions=[]
    finalDistances=np.zeros(len(surfaces))
    
    # translate the crystal to center
    crystal=copy.deepcopy(symbol)
    crystal.wrap(center)
    crystal.center()
    
    #Rounded number of layers
    roundedNumberOfLayers=np.ceil(np.asarray(distances)/
    np.asarray(interplanarDistances))
    roundedNumberOfLayers=roundedNumberOfLayers+np.array(np.ones(len(roundedNumberOfLayers)))
    # print('roundedNumberOfLayers',roundedNumberOfLayers)
    
    # build slabs for each polar surface and get the dipole
    for element,charge in zip(symbol.get_chemical_symbols(),symbol.get_initial_charges()):
        if element not in ions:
            ions.append(element)
            if charge not in charges:
                charges.append(charge)
    polarity=evaluateSurfPol(symbol,surfaces,ions,charges)
    polarSurfacesIndex=[i for i,pol in enumerate(polarity) if pol=='polar']
    # polarSurfaces=[surfaces[i] for i in polarSurfacesIndex]   
    # Saving the surfaces distances for non-polar ones 
    for index,s in enumerate(surfaces):
        if index not in polarSurfacesIndex:
            finalDistances[index]=distances[index]
    
    for index in polarSurfacesIndex:
        # print(distances[index])
        direction= np.dot(surfaces[index],symbol.get_reciprocal_cell())
        direction = direction / np.linalg.norm(direction)
        l=int(roundedNumberOfLayers[index])
        slab=slabBuild(crystal,tuple(surfaces[index]),l)
        # view(slab)
        limit=np.dot(slab.get_positions(),direction)
        # print(limit)
        beyondLimit=sorted([atom.index for atom in slab if limit[atom.index]>distances[index]],reverse=True)
        del slab[beyondLimit]
        # view(slab)
        initialDipole=dipole(slab)
        distancesAndDipole=[]
        distancesAndDipole.append([distances[index],initialDipole])
        cycle=0
        slabModels=[]
        slabModels.append(slab)
        # print('distances',distances[index])
        while cycle <10:
            cycle+=1
            dprima=distances[index]+((0.1*interplanarDistances[index])*cycle)
            lprima=l+(1*cycle)
            slab_prima=slabBuild(crystal,tuple(surfaces[index]),lprima)
            # view(slab_prima)
            limit=np.dot(slab_prima.get_positions(),direction)
            # print('dprima',dprima)
            # print(limit)
            beyondLimit=sorted([atom.index for atom in slab_prima if limit[atom.index]>dprima],reverse=True)
            del slab_prima[beyondLimit]
            # view(slab_prima)
            slabModels.append(slab_prima)
            distancesAndDipole.append([dprima,dipole(slab_prima)])
            # exit(1)
        # print(s,distancesAndDipole)
        disAndDip=sorted(distancesAndDipole,key=lambda x:x[1]) 
        # print(disAndDip)
        finalDistances[index]=(disAndDip[0][0])
        # break
        # view(slabModels)
    # print('finalDistances',finalDistances)
    # exit(1)
    return finalDistances

def terminations(symbol,atoms,polarSurface):
    """
    Function that evaluates the termination of polar surface
    Equivalent orientation of polar surfaces must have the same termination
    Args:
        symbol(Atoms):crystal structure
        atoms(Atoms):nanoparticle
        polarSurface([list]): polar surface miller indexes
        
    Return:
        finalElements: set of atomic symbols
    """
    # charges=[]
    # ions=[]
    # # Calculate the polarity  and get the polar surfaces
    # for element,charge in zip(symbol.get_chemical_symbols(),symbol.get_initial_charges()):
    #     if element not in ions:
    #         ions.append(element)
    #         if charge not in charges:
    #             charges.append(charge)
    # polarity=evaluateSurfPol(symbol,surfaces,ions,charges)
    # # print('polarity',polarity)
    # polarSurfacesIndex=[n for n,p in enumerate(polarity) if not 'non Polar' in p]
    # # print('polarSurfaceIndex',polarSurfacesIndex)
    # # exit(1)
    # polarSurfaces=[surfaces[i] for i in polarSurfacesIndex] 

    # # Get the equivalent surfaces and give them the distance
    # # Creating the spacegroup object
    # sg=Spacegroup((int(str(symbol.info['spacegroup'])[0:3])))

    positions=[atom.position-atoms.cut_origin for atom in atoms]
    # centroid=positions.mean(axis=0)
    result=[]
    s=polarSurface
    finalElements=[]
    for equSurf in equivalentSurfaces(symbol,s):
        finalElementsPerEquivalent=[]
        rlist=[]
        direction= np.dot(equSurf,symbol.get_reciprocal_cell())
        direction /= np.linalg.norm(direction)
        for pos in positions:
            rlist.append(np.dot(pos,direction))
        testArray=rlist-np.amax(rlist)
        # all the ones that has testArray value equal to 0 
        # belong to that equivalent surface
        for n,val in enumerate(testArray):
            if val==0.0:
                finalElementsPerEquivalent.append(atoms[n].symbol)
        finalElementsPerEquivalent.sort()
        finalElements.append(finalElementsPerEquivalent)
    for p in product(finalElements,repeat=2):
        if p[1]!=p[0]:
            result.append('non equivalents')
            break
    if 'non equivalents' in result:
        return result[0]
    else:
        return finalElements[0]

def addSpecies(symbol,atoms,surfaces,termNature):
    """
    Function that add species on reduced polarity nps
    hydrogens in non metal and OH in metal
    Args:
        symbol(Atoms): crystal structure
        atoms(Atoms): polarity reduced nanoparticle
        surfaces([list]): miller indexes surfaces
        termNature(str): termination nature, metal or non-metal
    Return: 
        adsorbedNP(Atoms)
    """
    sg=Spacegroup((int(str(symbol.info['spacegroup'])[0:3])))

    positions=[atom.position-atoms.cut_origin for atom in atoms]
    
    adsorbedNP=copy.deepcopy(atoms)

    for s in zip(surfaces):
        equivalentSurfaces=sg.equivalent_reflections(s)
        for equSurf in equivalentSurfaces:
            # print(equSurf)
            surfaceAtomsIndexperEq=[]
            surfaceAtomsperEq=[]
            rlist=[]
            direction= np.dot(equSurf,symbol.get_reciprocal_cell())
            direction = direction / np.linalg.norm(direction)
            for pos,num in zip(atoms.get_positions(),atoms.get_atomic_numbers()):
                rlist.append((np.dot(pos,direction)+covalent_radii[num]))
            # exit(1)
            # finalElements.append(atoms[np.argmax(rlist)].symbol)
            testArray=rlist-np.amax(rlist)
            # print(testArray)
            for n,val in enumerate(testArray):
                if val==0.0:
                    # print(val)
                    surfaceAtomsIndexperEq.append(n)
                    surfaceAtomsperEq.append(atoms[n].symbol)
            # print(surfaceAtomsIndexperEq)
            # define the new positions of hydrogen atoms
            # scalling uniformly the position of father atom
            if list(set(surfaceAtomsperEq))[0] in nonMetals and termNature=='non-metal':
                for atomIndex in surfaceAtomsIndexperEq:
                    displacement=0.979
                    newPos=(displacement*direction)+atoms[atomIndex].position
                    # print(atoms[atomIndex].position,newPos)
                    hydrogen=Atom('H',newPos)
                    adsorbedNP.extend(hydrogen)
            if list(set(surfaceAtomsperEq))[0] not in nonMetals and termNature=='metal':
                for atomIndex in surfaceAtomsIndexperEq:
                    # add oxygen and then hydrogen
                    oxygenDisp=2.0
                    oxygenPos=(oxygenDisp*direction)+atoms[atomIndex].position
                    oxygen=Atom('O',oxygenPos)
                    adsorbedNP.extend(oxygen)
                    hydrogenDisp=0.979+oxygenDisp
                    hydrogenPos=(hydrogenDisp*direction)+atoms[atomIndex].position
                    hydrogen=Atom('H',hydrogenPos)
                    adsorbedNP.extend(hydrogen)
    return adsorbedNP 
            # else:
            
def evaluateSurfPol(symbol,surfaces,ions,charges):
    """
    Function that identify if a surface is polar or not
    Args:
        symbol(Atoms): crystal structure
        surfaces([list]): surfaces miller indexes
        ions([]): ions
        charges([]): ionic charges
    """
    # convert atoms into material
    material=AseAtomsAdaptor.get_structure(symbol)
    data=[]
    for element in material.species:
        for i,c in zip(ions,charges):
            if element.name==i:
                data.append(c) 
    material.add_oxidation_state_by_site(data)
    # print(material)
    allSlabs=[]
    polarS=[]
    for s in surfaces:
        slabgen=SlabGenerator(material,s,10,10)
        all_slabs=slabgen.get_slabs()
        slabsPolarity=[]
        # print(s,len(all_slabs))
        for slab in all_slabs:
            allSlabs.append(slab)    
            # slabs.append(AseAtomsAdaptor.get_atoms(slab))
            if slab.is_polar(tol_dipole_per_unit_area=1e-5)==False:
                slabsPolarity.append('non Polar')
            else:
                slabsPolarity.append('polar')
        # print(len(slabsPolarity))
        polarities=list(set(slabsPolarity))
        polarS.append(polarities[:])

    # atomsObjects=[]
    # for slab in allSlabs:
    #     atoms=AseAtomsAdaptor.get_atoms(slab)
    #     atomsObjects.append(atoms)
    # view(atomsObjects)
    # exit(1)
    return(polarS)
    
def forceTermination(symbol,surfaces,distances,interplanarDistances,center,termNature):
    """
    Function that forces polar surfaces nanopartilcle
    terminations:
    Args: 
        symbol(Atoms): crystal structure
        surfaces([list]): miller indexes
        distances([float]): distances
        interplanarDistances([float])
        center([float]): center
        termNature(str): metal or non-metal
    Return:
        finalDistances([float]): final distances.
    """
    # translate the crystal to center
    crystal=copy.deepcopy(symbol)
    crystal.wrap(center)
    crystal.center()
    # round the numer of Layers
    roundedNumberOfLayers=np.ceil(np.asarray(distances)/
    np.asarray(interplanarDistances))
    roundedNumberOfLayers=roundedNumberOfLayers+np.array(np.ones(len(roundedNumberOfLayers)))
    
    # Transforming termNature in ions
    for element in symbol.get_chemical_symbols():
        if termNature=='metal' and element not in nonMetals:
                termIon=element
                break
        elif termNature=='non-metal' and element in nonMetals:
                termIon=element
                break
    
    # build slabs for each polar surface and get the dipole
    ions=[]
    charges=[]
    for element,charge in zip(symbol.get_chemical_symbols(),symbol.get_initial_charges()):
        if element not in ions:
            ions.append(element)
            if charge not in charges:
                charges.append(charge)
    polarity=evaluateSurfPol(symbol,surfaces,ions,charges)
    polarSurfacesIndex=[i for i,pol in enumerate(polarity) if pol=='polar']

    # Saving the surfaces distances for non-polar ones 
    finalDistances=np.zeros(len(distances))
    for index,s in enumerate(surfaces):
        if index not in polarSurfacesIndex:
            finalDistances[index]=distances[index]
    slabs=[]
    for index in polarSurfacesIndex:
        print(surfaces[index])
        # Get the termination 
        topSpecie=[]
        direction= np.dot(surfaces[index],symbol.get_reciprocal_cell())
        direction = direction / np.linalg.norm(direction)
        l=int(roundedNumberOfLayers[index])

        slab=slabBuild(symbol,tuple(surfaces[index]),l)
        limit=np.dot(slab.get_positions(),direction)
        beyondLimit=sorted([atom.index for atom in slab if limit[atom.index]>distances[index]],reverse=True)
        del slab[beyondLimit]
        slabs.append(slab)
        rlist=[] 
        for pos,num in zip(slab.get_positions(),slab.get_atomic_numbers()):
            rlist.append((np.dot(pos,direction)+covalent_radii[num]))
        rlist-=np.amax(rlist)

        for n,val in enumerate(rlist):
            if val==0.0:
                topSpecie.append(slab[n].symbol)
        topIonsType=list(set(topSpecie))

        #cycle to get only one specie 
        # while len(topIonsType)>1:
        distance=distances[index]
        cycleCounter=0
        while topIonsType[0]!=termIon:
            cycleCounter+=1
            dprima=distance+((0.10*interplanarDistances[index]))
            lprima =l+1
            print(termIon,topIonsType,dprima,lprima)
            topSpecie=[]
            slab=slabBuild(symbol,tuple(surfaces[index]),lprima)
            slabs.append(slab)
            limit=np.dot(slab.get_positions(),direction)
            beyondLimit=sorted([atom.index for atom in slab if limit[atom.index]>dprima],reverse=True)
            del slab[beyondLimit]
            slabs.append(slab)
            rlist=[]
            for pos,num in zip(slab.get_positions(),slab.get_atomic_numbers()):
                rlist.append((np.dot(pos,direction)+covalent_radii[num]))
            rlist-=np.amax(rlist)
            for n,val in enumerate(rlist):
                if val==0.0:
                    topSpecie.append(slab[n].symbol)
            # updating the values
            topIonsType=list(set(topSpecie))
            distance=dprima
            l=lprima
            print(termIon,topIonsType,dprima,lprima,'\n--------------------------------')


            if cycleCounter==10:
                break
        
        # view(slabs)
        finalDistances[index]=distance
    # exit(1)

    return(finalDistances)

def forceTermination2(symbol,atoms,surfaces,distances,interplanarDistances,termNature):
    """
    Function that gives a set of distances to
    forces the termination for polar surfaces
    by increasing the distance
    Args:
        atoms(Atoms): NP0
        symbol(Atoms): crystal structure
        surfaces([list]): miller indexes
        distances([float]): distances
        interplanarDistances([float])
        termNature(str): termination element
    Return:
        newDistances([list]):set of cutoff distances 
    """
    cutoffDistancesSets=[]

    # build slabs for each polar surface and get the dipole
    ions=[]
    charges=[]
    for element,charge in zip(symbol.get_chemical_symbols(),symbol.get_initial_charges()):
        if element not in ions:
            ions.append(element)
            if charge not in charges:
                charges.append(charge)
    polarity=evaluateSurfPol(symbol,surfaces,ions,charges)
    # print(polarity)
    # exit(1)
    polarSurfacesIndex=[i for i,pol in enumerate(polarity) if not 'non Polar' in pol]
    # print('polarSurfacesIndex',polarSurfacesIndex)
    # exit(1)
    # print('polar surface')
    # print(surfaces[polarSurfacesIndex[0]])
    # exit(1)

    # Saving the surfaces distances for non-polar ones 
    newDistances=np.zeros(len(distances))
    for index,s in enumerate(surfaces):
        if index not in polarSurfacesIndex:
            newDistances[index]=distances[index]
    terminationsNature=[] 
    # If the orientation has the desired termination preserve the distance
    terminationsNature=[terminationStoichiometry(symbol,atoms,surfaces[index]) for index in polarSurfacesIndex]
    print(terminationsNature)
    # exit(1)
    # first verify if the initial size matches with the desired polarity
    # and remove from list
    toRemove=[]
    for n,index in enumerate(polarSurfacesIndex):            
        if terminationsNature[n]=='nonMetalRich' and termNature=='non-metal':
            newDistances[index]=distances[index]
            toRemove.append(n)
        elif terminationsNature[n]=='metalRich' and termNature=='metal':
            newDistances[index]=distances[index]
            toRemove.append(n)
            # del polarSurfacesIndex[index]
    toRemove.sort(reverse=True)
    for i in toRemove:
        del polarSurfacesIndex[i]

    if polarSurfacesIndex==None:
        return newDistances
    else:
        # geting sets of polar distances         
        polarDistancesSets=[]
        for index in polarSurfacesIndex:
            distancesSet=[]
            # include default distance
            distancesSet.append(distances[index])
            direction= np.dot(surfaces[index],symbol.get_reciprocal_cell())
            direction = direction / np.linalg.norm(direction)

            # unit cell distances 
            distancesUC=np.dot(symbol.get_positions(),direction)
            elements=symbol.get_chemical_symbols()

            # diferences in the unit cell between non equal ionic species 
            diferences=[]
            for a in zip(elements,distancesUC):
                for b in zip(elements,distancesUC):
                    if a[0]!=b[0]:
                        diferences.append(np.round(np.abs(b[1]-a[1]),2))
            # make an exploration in the range between minimum and maximum
            # distance between non equal ionic species 
            for step in np.linspace(np.amin(diferences),np.amax(diferences),5):
                distancesSet.append(distances[index]+step)
            polarDistancesSets.append(distancesSet)

        # Getting the final set of distances
        # as a product of polar distances sets
        for p in product(*polarDistancesSets):
            cutoffD=copy.deepcopy(newDistances)
            i=0
            for n,d in enumerate(cutoffD):
                if cutoffD[n]==0:
                    cutoffD[n]=p[i]
                    i=i+1
            cutoffDistancesSets.append(cutoffD)
        # cutoffDistancesSets.sort(key=lambda x: x[2])
        print(*cutoffDistancesSets,sep='\n')
        # exit(1)  
        return(cutoffDistancesSets)

def wulffDistanceBasedVer2(symbol,atoms,surfaces,distance):
    """
    Function that evaluates if equivalent
    faces has the same lenght from the center of the material
    or not. Also calculates the WLI

    Args:
        symbol(Atoms): bulk atom object
        atoms(Atoms): nanoparticle atoms type
        surface(list): surface miller index
        distance(float): distance from the center to the wall
    Return:
        results(bool): symmetric growing, True if symmetric

    """
    # Get the equivalent surfaces and give them the distance
    # Creating the spacegroup object
    sg=Spacegroup((int(str(symbol.info['spacegroup'])[0:3])))

    positions=np.array([atom.position[:] for atom in atoms])
    centroid=positions.mean(axis=0)

    #Create the ConvexHull  object
    hull=ConvexHull(positions)
    # print(hull.area)

    simplices=[]
    for simplex in hull.simplices:
        simplices.extend(simplex)
    surfaceAtoms=sorted(list(set(simplices)))

    #Save the atoms surface in a new atoms object
    outershell=copy.deepcopy(atoms)
    del outershell[[atom.index for atom in outershell if atom.index not in surfaceAtoms]]
    
    #All distances Maximum
    allMaxDistances=[]

    #Get the equivalent surfaces
    for s in surfaces:
        # Get the direction per each miller index
        # Project the position to the direction of the miller index
        # by calculating the dot produequivalentSurfacesStringsct
        maxDistances=[]
        for i in sg.equivalent_reflections(s):
            rlist=[]
            direction= np.dot(i,symbol.get_reciprocal_cell())
            direction/= np.linalg.norm(direction)
            for j in outershell.get_positions():
                rlist.append(np.dot(j-centroid,direction))
            maxDistances.append(np.amax(rlist))
        allMaxDistances.append(maxDistances)
    # Phase 1: get the contribution per initial surface by relationships
    # aka likePercentage
    likePercentage=[np.sum(i)/np.sum(np.sum(allMaxDistances)) for i in allMaxDistances]
    # print('likePercentage',likePercentage)

    ####################### aka inverse method    ##########################################
    # likePercentage2=[np.sum(1/np.asarray(i))/np.sum(1/np.asarray(np.sum(allMaxDistances))) for i in allMaxDistances]
    # normalizedDistances2=[np.asarray(1/np.asarray(dmax)) /np.amax(1/np.asarray(dmax)) for dmax in allMaxDistances]
    # ratios2=[np.asarray(i)/np.sum(i) for i in normalizedDistances2]
    # areaContributionperEq2=[]
    # for percent,iniPlane in zip(likePercentage2,ratios2):
    #     areaContributionperEq2.append(np.dot(percent,iniPlane))
    # deviations2=[]
    # for iniPlane in areaContributionperEq2:
    #     deviations2.append([np.abs(i-np.max(iniPlane)) for i in iniPlane])
    # print('deviations2')
    # print(*deviations2,sep='\n') 
    #########################################################################################
    
    # Phase 2: get the contribution for each equivalent surface
    normalizedDistances=[np.asarray(dmax) /np.amax(dmax) for dmax in allMaxDistances]
    # print(*allMaxDistances,sep='\n')
    # print('----------------------')
    # print('normalizedDistances',normalizedDistances)
    # print('normalizedDistances2',normalizedDistances2)
    
    ratios=[np.asarray(i)/np.sum(i) for i in normalizedDistances]
    # print('ratios',ratios)
    # print('ratios2',ratios2)

    areaContributionperEq=[]
    for percent,iniPlane in zip(likePercentage,ratios):
        areaContributionperEq.append(np.dot(percent,iniPlane))
    # print('areaContributionperEq',areaContributionperEq) 
    
    # Phase 3: use the absolute deviation respect to max
    deviations=[]
    for iniPlane in areaContributionperEq:
        deviations.append([np.abs(i-np.max(iniPlane)) for i in iniPlane])
    # print('deviations')
    # print(*deviations,sep='\n') 
    # Phase 4: If deviation is smaller than 0.1 (10%) the NP grow is
    # symmetric
    # exit(1)
    symmetric=[]
    for iniPlane in deviations:
        if np.max(iniPlane) >0.1:
            symmetric.append(1)
        else:
            symmetric.append(0)
    # print('symmetric',symmetric)
    if 1 in symmetric:
        return False
    else: 
        return True
    
def intertiaTensorSing(atoms,S,C,nanoList):
    """
    Function that uses intertia tensor to 
    compare equivalent structures
    Args: 
        atoms(Atoms): NP0 Nanoparticle
        S([]): lists of atoms to remove to achieve stoichiometry
        C([]): Coordination 
    Return:
        finalNanos([]): unique list of atoms to remove by rotational
    """
    # Get the fathers index and the eigenvalues of the inertia tensor
    start=time.time()
    # fathers=[coord[1][0] for coord in C if len(coord[1])==1]
    eigenVals=[]
    for s in S:
        # itime=time.time()
        danglingAtom=copy.deepcopy(atoms)
        toRemove=sorted([atom.index for atom in danglingAtom if atom.index not in s],reverse=True)
        del danglingAtom[toRemove]
        # write('tmp.xyz',danglingAtom,format='xyz')
        molecule=Molecule(danglingAtom.get_chemical_symbols(),danglingAtom.get_positions())
        # molecule=Molecule.from_file('tmp.xyz')
        sym=PointGroupAnalyzer(molecule)
        eigenVals.append(np.round(sym.eigvals,decimals=5))
        # print('intertia tensor calcs one',time.time()-itime,' s')    
    dataFrame=pd.DataFrame(np.array(eigenVals).reshape(len(eigenVals),3),columns=['0','1','2'])
    # duplicates=dataFrame[dataFrame.duplicated(keep=False)].sort_values(by='0')
    # print(type(duplicates))
    # duplicatesNames=[nanoList[i] for i in list(duplicates.index)] 
    # print(duplicates)
    # print(*duplicatesNames,sep='\n')
    # exit(1)
    # print(dataFrame)
    sindu=dataFrame.drop_duplicates(keep='first')

    uniqueModelsIndex=list(sindu.index)
    sample=range(len(S))
    deletedNanosIndex=[i for i in sample if i not in uniqueModelsIndex]
    end=time.time() 
    
    deleteNanos=[nanoList[i] for i in deletedNanosIndex]
    # print(*deleteNanos,sep='\n')
    for i in deleteNanos:
        remove(i) 

    print('Removed NPs:',len(deletedNanosIndex))
    # print('uniqueIndex',len(uniqueModelsIndex))
    print('Final models:',len(uniqueModelsIndex))
    
    print('Total time inertia tensor singulizator',np.round((end-start),2), 'sec')
        
def orientedReduction(symbol,atoms,surfaces,distances):
    """
    Function that removes dangling atoms on specific orientation
    Args:
        symbol(Atoms): Crystal structure
        atoms(Atoms): nanoparticle
        surfaces([list]): surfaces miller index
    Return:
        atoms(Atoms): nanoparticle orientedly reducted
    """
    # print('distances',distances)
    # print('surfaces',surfaces)
    sg=Spacegroup((int(str(symbol.info['spacegroup'])[0:3])))
    
    # Get the positions of singly coordinated atoms
    # removeUnbounded(symbol,atoms)
    # write('testLarge.xyz',atoms)
    C=coordinationv2(symbol,atoms)
    # print(*C,sep='\n')
    # exit(1)
    singlyCoordinatedAtomsIndex=[c[0] for c in C if len(c[1])==1]
    fatherSinglyIndex=sorted(list(set([c[1][0] for c in C if len(c[1])==1])))
    print('fatherSinglyIndex')
    # print(len(fatherSinglyIndex))
    print(fatherSinglyIndex)
    # exit(1)
    if len(singlyCoordinatedAtomsIndex)==0:
        return atoms
    else:
        # asuming that only one specie is singlycoordinated
        singlySpecie=atoms[singlyCoordinatedAtomsIndex[0]].symbol
        # Father based algorithm
        positions=[atom.position-atoms.cut_origin for atom in atoms if atom.index in fatherSinglyIndex]
        centerAllPositions=[atom.position - atoms.cut_origin for atom in atoms if atom.symbol!=singlySpecie]
        # print(positions)
        # exit(1)
        # get the polarity per surface

        ions=[]
        charges=[]
        for element,charge in zip(symbol.get_chemical_symbols(),symbol.get_initial_charges()):
            if element not in ions:
                ions.append(element)
                if charge not in charges:
                    charges.append(charge)
        
        polarity=evaluateSurfPol(symbol,surfaces,ions,charges)

        noPolarIndex=[n for n,p in enumerate(polarity) if 'non Polar' in p]
        polarIndex=[n for n,p in enumerate(polarity) if not 'non Polar' in p]
        # print('polarIndex',polarIndex)
        # exit(1)
        #  
        # First screening, remove the positions that belong to atoms in the 
        # polar direction
        # print('enter screening polar loop')

        fathersBelongPolar=[]
        for indexP in polarIndex:
            equivalentSurfaces=sg.equivalent_reflections(surfaces[indexP])
            for eq in equivalentSurfaces:
                direction= np.dot(eq,symbol.get_reciprocal_cell())
                direction/=np.linalg.norm(direction)
                distancesAtom=np.dot(positions,direction)
                # boundary=np.amax(distancesAtom)
                boundary=np.amax(np.dot(centerAllPositions,direction))
                test=[d - boundary for d in distancesAtom]
                # print(test)
                fathersPerEqIndex=[n for n,d in enumerate(test) if d==0.0]
                fathersPerEq=[fatherSinglyIndex[i] for i in fathersPerEqIndex]
                # print(eq,fathersPerEq)
                fathersBelongPolar.extend(fathersPerEq)
        print('belongPolar',sorted(list(set(fathersBelongPolar))))


        #Remove all the ones that does not belong to polar surfaces
        fathersToRemove=list(set(fatherSinglyIndex)-set(fathersBelongPolar))
        print('fathersToRemove',fathersToRemove) 
        # exit(1)
        
        if len(fathersToRemove)==0:
            return(atoms)
        else:
            danglingsToRemove=[]
            for i in singlyCoordinatedAtomsIndex:
                if C[i][1] in fathersToRemove:
                    danglingsToRemove.append(i)
            danglingsToRemove.sort(reverse=True)
            # view(atoms)
            del atoms[danglingsToRemove]
            # view(atoms)
            # exit(1) 
            return atoms

def forceTermination3(symbol,atoms,surfaces,distances,termNature):
    """
    Function that forces termination by adding atoms 
    paralell to the plane directions
    Args:
        symbol(Atoms): crystal structure
        atoms(Atoms):nanoparticle
        surfaces([list]): miller indexes
        distances([float]): distances
        termNature(str): metal or non-metal
    Return:
        decoratedAtoms(Atoms): nanoparticle with forced termination
    """
    sg=Spacegroup((int(str(symbol.info['spacegroup'])[0:3])))
    # Calculate the coodination and remove dangling atoms
    C=coordinationv2(symbol,atoms)
    # singlyCoordinatedAtomsIndex=sorted([coord[0] for coord in C if len(coord[1])==1],reverse=True)
    singly=[c[0] for c in C if len(c[1])==1].sort(reverse=True)
    # print(singly)
    # exit(1)
    if singly != None:
        del atoms[singly]
    # print(len(atoms))

    # Get the positions of the remain atoms respect to the cut center
    positions=[atom.position-atoms.cut_origin for atom in atoms]
    # print(positions) 
    # Get the atoms nature in a specific directions 
    # and only use the polars
    ions=[]
    charges=[]
    for element,charge in zip(symbol.get_chemical_symbols(),symbol.get_initial_charges()):
        if element not in ions:
            ions.append(element)
            if charge not in charges:
                charges.append(charge)
    
    polarity=evaluateSurfPol(symbol,surfaces,ions,charges)
    polarIndex=[n for n,p in enumerate(polarity) if not 'non Polar' in p]

    decoratedAtoms=copy.deepcopy(atoms)

    for index in polarIndex:
        for eq in sg.equivalent_reflections(surfaces[index]):
            
            direction= np.dot(eq,symbol.get_reciprocal_cell())
            direction/=np.linalg.norm(direction)
            #Crystal distances and range
            distancesUC=np.dot(symbol.get_positions(),direction)
            elements=symbol.get_chemical_symbols()

            # surface atoms in that direction
            rlist=[]
            for pos,num in zip(positions,atoms.get_atomic_numbers()):
                rlist.append((np.dot(pos,direction)+covalent_radii[num]))
            testArray=rlist-np.amax(rlist)
            # print(rlist)
            # exit(1)
            # print(testArray)
            surfaceAtomsIndexperEq=[]
            surfaceAtomsperEq=[]
            for n,val in enumerate(testArray):
                if val==0.0:
                    # print(val)
                    surfaceAtomsIndexperEq.append(n)
                    surfaceAtomsperEq.append(atoms[n].symbol)
            # Just work for one termination element
            if len(list(set(surfaceAtomsperEq)))==1:
                metalSymbol=[ele for ele in elements if ele not in nonMetals]
                nonMetalSymbol=[ele for ele in elements if ele in nonMetals]
                
                shortestDistance=[]
                for a in zip(elements,distancesUC):
                    for b in zip(elements,distancesUC):
                        if a[0]!= b[0]:
                            if np.abs(a[1]-b[1])>0.0:
                                shortestDistance.append(np.round(np.abs(a[1]-b[1]),2))
                displacement=np.argmin(shortestDistance)
                # print('specie and plane')
                # print(list(set(surfaceAtomsperEq)),eq) 
                if list(set(surfaceAtomsperEq))[0] in nonMetals and termNature=='metal':
                    #Add metal
                    for atomIndex in surfaceAtomsIndexperEq:
                        newPos=(displacement*direction)+atoms[atomIndex].position
                        metalAdd=Atom(str(metalSymbol[0]),newPos)
                        decoratedAtoms.extend(metalAdd)


                elif list(set(surfaceAtomsperEq))[0] not in nonMetals and termNature=='non-metal':
                    # Add non metal
                    for atomIndex in surfaceAtomsIndexperEq:
                        newPos=(displacement*direction)+atoms[atomIndex].position
                        nonMetalAdd=Atom(str(nonMetalSymbol[0]),newPos)
                        decoratedAtoms.extend(nonMetalAdd)
                else:
                    pass 
    return decoratedAtoms




    # add inverse specie, ie, non metals if its metal
    # terminated or metals if its non metal.

def terminationStoichiometry(symbol,atoms,surfaceIndexes):
    """
    Function that return the nature termination in a face
    
    Args:
        symbol(Atoms):crystal strucuture
        atoms(Atoms):nanoparticle
        surfaceIndexez([]): miller indexes 
    Return:
        termFaceNatur(str): metalRich,nonMetalRich,stoich
    """
    # print(surfaceIndexes)
    # Get the cell stoichiometry
    listOfChemicalSymbols=symbol.get_chemical_symbols()
    
    #Count and divide by the greatest common divisor
    chemicalElements=list(set(listOfChemicalSymbols))

    # put the stuff in order, always metals first
    if chemicalElements[0] in nonMetals:
        chemicalElements.reverse()

    counterCell=[]
    for e in chemicalElements:
        counterCell.append(listOfChemicalSymbols.count(e))

    gcd=np.gcd.reduce(counterCell)

    cellStoichiometry=counterCell/gcd
    crystalRatio=cellStoichiometry[0]/cellStoichiometry[1]

    terminationElements=terminations(symbol,atoms,[surfaceIndexes])
    if terminationElements=='non equivalents':
        return None
        
    # get the stoichiometry 
    orientationProp=[]
    for e in chemicalElements:
        orientationProp.append(terminationElements.count(e))
    # print(orientationProp)
    # if orientation prop just have one index 
    if len(list(set(terminationElements)))==1:
        if list(set(terminationElements))[0] in nonMetals:
            termFaceNatur='nonMetalRich'
        else:
            termFaceNatur='metalRich'
    #if orientation has more than one element
    else:
        gcd=np.gcd.reduce(orientationProp)
        orientationStoichiometry=orientationProp/gcd
        orientRatio=orientationStoichiometry[0]/orientationStoichiometry[1]
        # print(chemicalElements) 
        # print(orientationStoichiometry)
        # print(orientRatio,crystalRatio)
        # # # exit(1)
        # print(chemicalElements[0])
        # exit(1)
        if orientRatio==crystalRatio:
            termFaceNatur='stoich'
        
        elif orientRatio>crystalRatio:
            if chemicalElements[0] in nonMetals:
                termFaceNatur='nonMetalRich'
            else:
                termFaceNatur='metalRich'
        elif orientRatio<crystalRatio:
            if chemicalElements[1] in nonMetals:
                termFaceNatur='nonMetalRich'
            else:
                termFaceNatur='metalRich'
    # print(termFaceNatur)
    return termFaceNatur