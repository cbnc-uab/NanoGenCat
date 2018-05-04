from __future__ import print_function

import argparse
import sys

from ase import __version__
from ase.utils import import_module


commands = [
    ('info', 'ase.cli.info'),
    ('test', 'ase.test'),
    ('gui', 'ase.gui.ag'),
    ('db', 'ase.db.cli'),
    ('run', 'ase.cli.run'),
    ('band-structure', 'ase.cli.band_structure'),
    ('build', 'ase.cli.build'),
    ('eos', 'ase.eos'),
    ('ulm', 'ase.io.ulm'),
    ('find', 'ase.cli.find'),
    ('nomad-upload', 'ase.cli.nomad'),
    ('completion', 'ase.cli.completion')]


def main(prog='ase', description='ASE command line tool',
         version=__version__, commands=commands, hook=None, args=None):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument('--version', action='version',
                        version='%(prog)s-{}'.format(version))
    parser.add_argument('-T', '--traceback', action='store_true')
    subparsers = parser.add_subparsers(title='Sub-commands',
                                       dest='command')

    subparser = subparsers.add_parser('help',
                                      description='Help',
                                      help='Help for sub-command')
    subparser.add_argument('helpcommand', nargs='?')

    functions = {}
    parsers = {}
    for command, module_name in commands:
        cmd = import_module(module_name).CLICommand
        subparser = subparsers.add_parser(
            command,
            help=cmd.short_description,
            description=getattr(cmd, 'description', cmd.short_description))
        cmd.add_arguments(subparser)
        functions[command] = cmd.run
        parsers[command] = subparser

    if hook:
        args = hook(parser)
    else:
        args = parser.parse_args(args)

    if args.command == 'help':
        if args.helpcommand is None:
            parser.print_help()
        else:
            parsers[args.helpcommand].print_help()
    elif args.command is None:
        parser.print_usage()
    else:
        f = functions[args.command]
        try:
            if f.__code__.co_argcount == 1:
                f(args)
            else:
                f(args, parsers[args.command])
        except KeyboardInterrupt:
            pass
        except Exception as x:
            if args.traceback:
                raise
            else:
                print('{}: {}'.format(x.__class__.__name__, x),
                      file=sys.stderr)
                print('To get a full traceback, use: {} -T {} ...'
                      .format(prog, args.command), file=sys.stderr)


def old():
    cmd = sys.argv[0].split('-')[-1]
    print('Please use "ase {cmd}" instead of "ase-{cmd}"'.format(cmd=cmd))
    sys.argv[:1] = ['ase', cmd]
    main()
