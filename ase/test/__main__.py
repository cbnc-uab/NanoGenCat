import optparse
import sys

from ase.test import test

usage = ('Usage: python -m ase.test [-c calc1,calc2,...] '
         '[test1.py test2.py ...]')
parser = optparse.OptionParser(usage=usage, description='Test ASE')

parser.add_option('-c', '--calculators',
                  help='Comma-separated list of calculators to test.')
parser.add_option('-v', '--verbosity', type=int, default=2, metavar='N',
                  help='Use 0, 1 or 2.')

opts, args = parser.parse_args()

if opts.calculators:
    calculators = opts.calculators.split(',')
else:
    calculators = []

results = test(verbosity=opts.verbosity,
               calculators=calculators,
               files=args)
sys.exit(len(results.errors + results.failures))
