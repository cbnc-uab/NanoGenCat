# Input 
As a input, BCNM use YAML formatted file. YAML is a human friendly data
serialization standard for all programing languages. 

The input file has three main parts. The first part has the mandatory data to
build the crystal, the second one the surface related data and the third one
the algorithm setup.  To build the crystal BCNM needs the chemical species,
the space group, the ionic charges, the basis and the cell parameters. The
surfaces are characterized by their three digits Miller indices and their
energies. The algorithm needs the size of the NP in Å, which is defined with a
central value and a range around this size and a step size, and the center of
the nanoparticle. 

## List of parameters
### Crystal construction data
* chemicalSpecies[list]:  Elements symbols, maximum 2.
* spaceGroupNumber[integer]: Number of the material space group
* charges[list]: Ionic charges in the same order as chemicalSpecies
* basis [ĺist]: Fractional coordinates ions position in the same order as chemicalSpecies
* cellDimesion[list] Cell parameters (a,b,c, &alpha;,&beta;,&gamma; )

### Surface data
* surfaceEnergy[list]: Surface energies on the same units.
* surfaces[list]: Three Miller indices for each surface.
### Nanoparticle data
* nanoparticleSize[float]: Nanoparticle selected size in angstroms.
* sizeRange[float]: Range of nanoparticle size.   
* step[float]: Step size to get throught the size range.
* centering[string]: nanoparticle center. Available options:
	* automatic: Explores three different types of nanoparticle centers: i) the positions of the basis ions, ii) the center of mass of the irreducible unit cell and ii) several special positions within those presenting the lower symmetry multiplicity in the unit cell. This is the default value.
	* none: Use the origin of coordinates (0.0, 0.0, 0.0) as nanoparticle center. 
### Advanced options
* wulff-like-method[string]: methods to calculate the Wulff-like index. Available options: 
	* surfaceBased: Robust method to compute the area contribution per plane based on theconvex hull algorithm. This is the default method.
	* distancesBased: Alternative approach based on the ratio of normal lenghts of each surface direction. 

## Input example: Generation of IrO<sub>2</sub> nanoparticles.

```bash
chemicalSpecies: [Ir,O] # Chemical Species
spaceGroupNumber: 136 # Space group number
charges: [4,-2] # Ionic charges 
basis:
 - [0.0, 0.0, 0.0] # Fractional coordinates position of Ir
 - [0.306, 0.306,0.0] # Fractional coordinates position of O
cellDimension: [4.497, 4.497, 3.193, 90, 90, 90] # Conventional cell parameters

surfaceEnergy: [0.15, 0.158, 0.185, 0.206] # Surface energies must to be in the same units 
surfaces: 
  - [1, 1, 0] # Three digits Miller Indices
  - [0, 1, 1]
  - [1, 0, 0]
  - [0, 0, 1]

nanoparticleSize: 14 # Central value
sizeRange: 1 # Range
step: 1 # Step size
centering: none #Type of center
```
To generate stoichiometric nanoparticles of IrO<sub>2</sub>
you have to use the command:
```python
python3 -u bcnm.py input.yaml > output.log
```
All executions create a unique directory in at `./tmp/` location.

The code creates three types .XYZ files:
bulk structure (`crystalShape.xyz`), non-stoichiometric nanoparticles (`*_NP0.xyz`)
and stoichiometric nanoparticles (`*.xyz`).





