# Bcnm package 2018

BCN Models is a tool able to generate stoichiometric Wulff-like NP in a systematic way. The code and website is mantained by the Computational BioNanoCat group at UAB. This code is under the GNU General Public License and uses Atomic Simulation Environment (ASE) library. 

### Installation requirements

* Python 3
* Ase
* Pyyaml
* Pandas
* Pymatgen

### Installation

1 - Install `Miniconda Python 3.7 Distribution <http://conda.pydata.org/miniconda.html>`_ for your platform and install it with:

    bash Miniconda3*.sh
        
2 - Update conda:

    conda update conda

3 - Create a new environment called ``bcnm``:

    conda create -n bcnm

4 - Update conda:

    conda update conda

5 - Activate the new environment as proposed:

    conda activate bcnm

6 - Install dependences:

    conda install pip
    pip install --upgrade pip  
    pip install ase pyyaml pandas pymatgen
  
7 - Clone Bcnm code from git:

    git clone git@github.com:dagonzalezfo/NanoGenCat.git


### General usage notes

    cd NanoGencat
    conda activate bcnm
    python3 bcnm.py examples/ruthenium.input.yaml
    
    
### Example

### Contact

bcnm@qf.uab.cat
  
Ciències · Química Física  
Edifici C · C7-153 · Carrer de l'Albareda  
Campus de la UAB · 08193 Bellaterra  
Cerdanyola del Vallès · Barcelona · Spain

### License 

GNU General Public License v3.0
