#! /usr/bin/env python

from argparse import ArgumentParser

def basis(s):
    x, y, z = map(float, s.split(','))
    return x, y, z

def millerIndex(s):
	a,b,c=map(int,s.split(','))
	return a,b,c



parser = ArgumentParser( add_help=True, description = 'Bulk nanoparticles cutter (BNC) developed by B. Camino')
parser.add_argument('-b','--basis',nargs='+',type=basis,required=True,help='Positions of atoms in the unit cell(0.0,0.0,0.0)')
parser.add_argument('-c','--cell',nargs='+',type=float,required=True,help='Cell dimenssions in angstroms and angles(a b c alpha beta gamma)')
parser.add_argument('-s','--space',type=int,required=True,help='spacegroup number')
parser.add_argument('-ch','--chemicalSpecie',type=str,required=True,help='Chemical elements in the unit cell')
parser.add_argument('-m', '--miller',nargs='+',type=millerIndex, required = True, help = 'List of miller indexes 1,1,0 1,0,1 0,0,1 ...')
parser.add_argument('-esurf', '--surfacesEnergy',nargs='+',type=float, required = True, help = 'List of surfaces energy.')
parser.add_argument('-size', '--nanoparticleSize', type=float, required = True, help = 'Nanoparticle size.')
args = parser.parse_args()

print args

# ./parser.py -b 0.0,0.0,0.0 0.306,0.306,0.0 -c 4.54 4.54 3.18 90 90 90 -s 136 -ch IrO -m 1,1,1 1,0,1 0,0,1 -esurf 0.94 1.06 1.23 1.55 -size 16

