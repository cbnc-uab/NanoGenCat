import setuptools

with open("README.md", "r") as fh:
 long_description = fh.read()

setuptools.setup(
 name="bcnm",
 version="0.1.0",
 author="bruno.camino@uab.cat",
 author="dagonzalez@qf.uab.cat",
 author_email="bcnm@qf.uab.cat",
 description="BcnM is a computational tool to obtain realistic models of nanoparticles from bulk systems of any type of material",
 long_description=long_description,
 url="https://github.com/dagonzalezfo/NanoGenCat",
 packages=setuptools.find_packages(),
 classifiers=[
  "Programming Language :: Python :: 3"
  "Licences :: GNU General Public License GPLv3",
  "Operating System :: OS Independent",
 ],
) 