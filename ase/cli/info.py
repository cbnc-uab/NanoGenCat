from optparse import OptionParser
from ase.io.formats import filetype
from ase.io.ulm import print_ulm_info
from ase.io.pickletrajectory import print_trajectory_info
from ase.io.bundletrajectory import print_bundletrajectory_info

description = 'Print summary of information from trajectory files.'


def main():
    p = OptionParser(usage='%prog file.traj [file2.traj ...]',
                     description=description)

    opts, args = p.parse_args()

    if len(args) == 0:
        p.error('Incorrect number of arguments')

    for f in args:
        ft = filetype(f)
        print("File type of '{0}' appears to be of type '{1}'".format(f, ft))
        if ft == 'traj':
            print_ulm_info(f)
        elif ft == 'trj':
            print_trajectory_info(f)
        elif ft == 'bundletrajectory':
            print_bundletrajectory_info(f)
        else:
            p.error(
                '%s is of type %s; cannot print info about this type of file' %
                (f, ft))
