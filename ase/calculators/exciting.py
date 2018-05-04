from __future__ import print_function
import os

import numpy as np
from lxml import etree as ET

from ase.io.exciting import atoms2etree
from ase.units import Bohr, Hartree
from ase.calculators.calculator import PropertyNotImplementedError
from ase.utils import basestring

class Exciting:
    def __init__(self, dir='calc', paramdict=None,
                 speciespath=None,
                 bin='excitingser', kpts=(1, 1, 1),
                 autormt=False, **kwargs):
        """Exciting calculator object constructor

        dir: string
            directory in which to execute exciting
        paramdict: dict
            Dictionary containing XML parameters. String values are
            translated to attributes, nested dictionaries are translated
            to sub elements. A list of dictionaries is translated to a
            list of sub elements named after the key of which the list
            is the value.  Default: None
        speciespath: string
            Directory or URL to look up species files
        bin: string
            Path or executable name of exciting.  Default: ``excitingser``
        kpts: integer list length 3
            Number of k-points
        autormt: bool
            Bla bla?
        kwargs: dictionary like
            list of key value pairs to be converted into groundstate attributes
        
        """
        self.dir = dir
        self.energy = None
        
        self.paramdict = paramdict
        if speciespath is None:
            speciespath = os.environ['EXCITINGROOT'] + '/species'
        self.speciespath = speciespath
        self.converged = False
        self.excitingbinary = bin
        self.autormt = autormt
        self.groundstate_attributes = kwargs
        if  (not 'ngridk' in kwargs.keys() and (not (self.paramdict))):
            self.groundstate_attributes['ngridk'] = ' '.join(map(str, kpts))
 
    def update(self, atoms):
        if (not self.converged or
            len(self.numbers) != len(atoms) or
            (self.numbers != atoms.get_atomic_numbers()).any()):
            self.initialize(atoms)
            self.calculate(atoms)
        elif ((self.positions != atoms.get_positions()).any() or
              (self.pbc != atoms.get_pbc()).any() or
              (self.cell != atoms.get_cell()).any()):
            self.calculate(atoms)

    def initialize(self, atoms):
        self.numbers = atoms.get_atomic_numbers().copy()
        self.write(atoms)

    def get_potential_energy(self, atoms):
        """
        returns potential Energy
        """
        if self.energy is None:
            self.update(atoms)
        return self.energy

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces.copy()

    def get_stress(self, atoms):
        raise PropertyNotImplementedError

    def calculate(self, atoms):
        self.positions = atoms.get_positions().copy()
        self.cell = atoms.get_cell().copy()
        self.pbc = atoms.get_pbc().copy()

        self.initialize(atoms)
        syscall = ('cd %(dir)s; %(bin)s;' %
                   {'dir': self.dir, 'bin': self.excitingbinary})
        print(syscall)
        assert os.system(syscall) == 0
        self.read()

    def write(self, atoms):
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        root = atoms2etree(atoms)
        root.find('structure').attrib['speciespath'] = self.speciespath
        root.find('structure').attrib['autormt'] = str(self.autormt).lower()

        if(self.paramdict):
            self.dicttoxml(self.paramdict, root)
            fd = open('%s/input.xml' % self.dir, 'w')
            fd.write(ET.tostring(root, method='xml', pretty_print=True,
                                 xml_declaration=True, encoding='UTF-8'))
            fd.close()
        else:
            groundstate = ET.SubElement(root, 'groundstate', tforce='true')
            for key, value in self.groundstate_attributes.items():
                if key == 'title':
                    root.findall('title')[0].text = value
                else:
                    groundstate.attrib[key] = str(value)
            fd = open('%s/input.xml' % self.dir, 'w')
            fd.write(ET.tostring(root, method='xml', pretty_print=True,
                                 xml_declaration=True, encoding='UTF-8'))
            fd.close()

    def dicttoxml(self, pdict, element):
        for key, value in pdict.items():
            if (isinstance(value, basestring) and key == 'text()'):
                element.text = value
            elif (isinstance(value, basestring)):
                element.attrib[key] = value
            elif (isinstance(value, list)):
                for item in value:
                    self.dicttoxml(item, ET.SubElement(element, key))
            elif (isinstance(value, dict)):
                if(element.findall(key) == []):
                    self.dicttoxml(value, ET.SubElement(element, key))
                else:
                    self.dicttoxml(value, element.findall(key)[0])
            else:
                print('cannot deal with', key, '=', value)

    def read(self):
        """
        reads Total energy and forces from info.xml
        """
        INFO_file = '%s/info.xml' % self.dir

        try:
            fd = open(INFO_file)
        except IOError:
            raise RuntimeError("output doesn't exist")
        info = ET.parse(fd)
        self.energy = float(info.xpath('//@totalEnergy')[-1]) * Hartree
        forces = []
        forcesnodes = info.xpath(
            '//structure[last()]/species/atom/forces/totalforce/@*')
        for force in forcesnodes:
            forces.append(np.array(float(force)))
        self.forces = np.reshape(forces, (-1, 3)) * Hartree / Bohr
        
        if str(info.xpath('//groundstate/@status')[0]) == 'finished':
            self.converged = True
        else:
            raise RuntimeError('calculation did not finish correctly')
