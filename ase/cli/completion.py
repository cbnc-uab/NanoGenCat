from __future__ import print_function
import os

# Path of the complete.py script:
my_dir, _ = os.path.split(os.path.realpath(__file__))
filename = os.path.join(my_dir, 'complete.py')


class CLICommand:
    short_description = 'Add tab-completion for Bash'
    cmd = 'complete -o default -C {} ase'.format(filename)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('filename', nargs='?')
        parser.add_argument('-0', '--dry-run', action='store_true')

    @staticmethod
    def run(args):
        filename = args.filename or os.path.expanduser('~/.bashrc')
        cmd = CLICommand.cmd
        print(cmd)
        if args.dry_run:
            return
        with open(filename) as fd:
            if cmd + '\n' in fd.readlines():
                print('Completion script already installed!')
                return
        with open(filename, 'a') as fd:
            print(cmd, file=fd)


def update(filename, commands):
    """Update commands dict.

    Run this when ever options are changed::

        python3 -m ase.cli.complete

    """

    import collections
    import textwrap
    from ase.utils import import_module

    dct = collections.defaultdict(list)

    class Subparser:
        def __init__(self, command):
            self.command = command

        def add_argument(self, *args, **kwargs):
            dct[command].extend(arg for arg in args
                                if arg.startswith('-'))

    for command, module_name in commands:
        module = import_module(module_name)
        module.CLICommand.add_arguments(Subparser(command))

    txt = 'commands = {'
    for command, opts in sorted(dct.items()):
        txt += "\n    '" + command + "':\n        ["
        txt += '\n'.join(textwrap.wrap("'" + "', '".join(opts) + "'],",
                         width=65,
                         break_on_hyphens=False,
                         subsequent_indent='         '))
    txt = txt[:-1] + '}\n'
    with open(filename) as fd:
        lines = fd.readlines()
        a = lines.index('# Beginning of computer generated data:\n')
        b = lines.index('# End of computer generated data\n')
    lines[a + 1:b] = [txt]
    with open(filename + '.new', 'w') as fd:
        print(''.join(lines), end='', file=fd)
    os.rename(filename + '.new', filename)
    os.chmod(filename, 0o775)


if __name__ == '__main__':
    from ase.cli.main import commands
    update(filename, commands)
