from __future__ import print_function
import os
import os.path as op
import subprocess

from ase.utils import FileNotFoundError

class CLICommand:
    short_description = 'Upload files to NOMAD'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('folders', nargs='+', metavar='folder')
        parser.add_argument('-t', '--token')
        parser.add_argument('-n', '--do-not-save-token', action='store_true')
        parser.add_argument('-0', '--dry-run', action='store_true')

    @staticmethod
    def run(args):
        dotase = op.expanduser('~/.ase')
        tokenfile = op.join(dotase, 'nomad-token')
        if args.token:
            token = args.token
        else:
            try:
                with open(tokenfile) as fd:
                    token = fd.readline().strip()
            except FileNotFoundError:
                print('Could not find token!')
                return

        cmd = ('tar cf - {} | '
               'curl -XPUT -# -HX-Token:{} '
               '-N -F file=@- http://nomad-repository.eu:8000 | '
               'xargs echo').format(' '.join(args.folders), token)

        if args.dry_run:
            print(cmd)
        else:
            subprocess.check_call(cmd, shell=True)

        if args.token and not args.do_not_save_token:
            msg = 'Should I save your token for later use? (yes/No): '
            if input(msg).lower() == 'yes':
                if not op.isdir(dotase):
                    os.mkdir(dotase)
                with open(tokenfile, 'w') as fd:
                    print(token, file=fd)
                os.chmod(tokenfile, 0o600)
                print('Wrote token to', tokenfile)
