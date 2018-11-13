from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
 long_description = fh.read()

setup(
 name='bcnm',
 version='0.1.0',
 author='Bruno Camino, Danilo González Forero',
 author_email='bcnm@qf.uab.cat',
 description='BcnM is a computational tool to obtain realistic models of nanoparticles from bulk systems of any type of material',
 long_description=long_description,
 url='https://github.com/dagonzalezfo/NanoGenCat',
 packages=find_packages(),
 classifiers=[
  'Programming Language :: Python',
  'Development Status :: Pre-Alpha',
  'Natural Language :: English',
  'Intended Audience :: Science/Research',
  'Licences :: GNU General Public License GPLv3',
  'Operating System :: OS Independent',
  'Topic :: Scientific/Engineering :: Chemistry',
 ],
 install_requires=['ase', 'pyyaml'],
) 
