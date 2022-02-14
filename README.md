# Bcnm Models [![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/dagonzalezfo/NanoGenCat/blob/master/LICENSE) [![Documentation Status](https://readthedocs.org/projects/bcnm/badge/?version=latest)](https://bcnm.readthedocs.io/en/latest/?badge=latest) [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://bcnm.qf.uab.cat)

BCN Models is a tool capable of generating stoichiometric Wulff-like NP in a systematic way. The code and website are mantained by the Computational BioNanoCat group at UAB. This code is under the GNU General Public License and uses Atomic Simulation Environment (ASE) library. 

### Installation requirements

* Python 3.7.3
* Ase 3.17.0
* Pyyaml
* Pandas
* Pymatgen 2018.11.6

### Installation

1 - Install Miniconda Python 3.7 Distribution `<http://conda.pydata.org/miniconda.html>` for your platform. Install it with:

    bash Miniconda3*.sh
        
2 - Update conda:

    conda update conda

3 - Clone Bcnm code from git:

    git clone git@github.com:cbnc-uab/NanoGenCat.git

4 - Create a new environment called ``bcnm``:

    conda create env -f environment.yml 

5 - Activate the new environment as proposed:

    conda activate bcnm



### General usage notes

    cd NanoGencat
    python3 bcnm.py examples/<filename>.yaml
    

### Contact

bcnm@qf.uab.cat
  
Ciències · Química Física  
Edifici C · C7-153 · Carrer de l'Albareda  
Campus de la UAB · 08193 Bellaterra  
Cerdanyola del Vallès · Barcelona · Spain

### License 

