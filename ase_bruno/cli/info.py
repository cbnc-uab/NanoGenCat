import platform
import sys

from ase.utils import import_module, FileNotFoundError
from ase.io.formats import filetype, all_formats
from ase.io.ulm import print_ulm_info
from ase.io.bundletrajectory import print_bundletrajectory_info


class CLICommand:
    short_description = 'Print information about files or system'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('filenames', nargs='*')
        parser.add_argument('-v', '--verbose', action='store_true')

    @staticmethod
    def run(args):
        if not args.filenames:
            print_info()
            return

        n = max(len(filename) for filename in args.filenames) + 2
        for filename in args.filenames:
            try:
                format = filetype(filename)
            except FileNotFoundError:
                format = '?'
                description = 'No such file'
            else:
                if format and format in all_formats:
                    description, code = all_formats[format]
                else:
                    format = '?'
                    description = '?'

            print('{:{}}{} ({})'.format(filename + ':', n,
                                        description, format))
            if args.verbose:
                if format == 'traj':
                    print_ulm_info(filename)
                elif format == 'bundletrajectory':
                    print_bundletrajectory_info(filename)


def print_info():
    versions = [('platform', platform.platform()),
                ('python-' + sys.version.split()[0], sys.executable)]
    for name in ['ase', 'numpy', 'scipy']:
        try:
            module = import_module(name)
        except ImportError:
            versions.append((name, 'no'))
        else:
            versions.append((name + '-' + module.__version__,
                            module.__file__.rsplit('/', 1)[0] + '/'))

    for a, b in versions:
        print('{:16}{}'.format(a, b))
