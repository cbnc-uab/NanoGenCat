## How to install

### Installation requirements

* Python 3
* Ase
* Pyyaml
* Pandas
* Pymatgen

### Instalation

1 - Download [Miniconda Python 3.7 Distribution](http://conda.pydata.org/miniconda.html) for your GNU/Linux and install it:

	Miniconda3*.sh

2 - Update conda:
    
	conda update conda

3 - Create a new environment called `bcnm`:

	conda create -n bcnm

4 - Activate the new environment as proposed:

	conda activate bcnm

5 - Install dependences:
    
	conda install pip
	pip install --upgrade pip  
	pip install ase pyyaml pandas pymatgen

6 - Clone Bcnm code from git:

	git clone git@github.com:dagonzalezfo/NanoGenCat.git

This instalation has been tested on several GNU/Linux distributions, if you have any problem with the instalation, please report them via the [issues page](https://github.com/dagonzalezfo/NanoGenCat/issues) page.


