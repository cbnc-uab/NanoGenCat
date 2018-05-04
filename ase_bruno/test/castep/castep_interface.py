"""Simple shallow test of the CASTEP interface"""
import os
import shutil
import tempfile

import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepParam,
                                    create_castep_keywords,
                                    import_castep_keywords)

tmp_dir = tempfile.mkdtemp()
cwd = os.getcwd()

c = Castep(directory=tmp_dir, label='test_label')
c.xc_functional = 'PBE'

lattice = ase.lattice.cubic.BodyCenteredCubic('Li')

print('For the sake of evaluating this test, warnings')
print('about auto-generating pseudo-potentials are')
print('normal behavior and can be safely ignored')

lattice.set_calculator(c)

create_castep_keywords(
    castep_command=os.environ['CASTEP_COMMAND'],
    path=tmp_dir,
    fetch_only=20)

param_fn = os.path.join(tmp_dir, 'myParam.param')
param = open(param_fn, 'w')
param.write('XC_FUNCTIONAL : PBE #comment\n')
param.write('XC_FUNCTIONAL : PBE #comment\n')
param.write('#comment\n')
param.write('CUT_OFF_ENERGY : 450.\n')
param.close()
c.merge_param(param_fn)

# check if the CastepOpt, CastepCell comparison mechanism works

castep_keywords = import_castep_keywords()
p1 = CastepParam(castep_keywords)
p2 = CastepParam(castep_keywords)

assert p1._options == p2._options

p1._options['xc_functional'].value = 'PBE'
p1.xc_functional = 'PBE'

assert p1._options != p2._options

assert c.calculation_required(lattice)

assert c.dryrun_ok()

c.prepare_input_files(lattice)

os.chdir(cwd)
shutil.rmtree(tmp_dir)
