from __future__ import print_function
import os
import sys
import shutil
import subprocess
import tempfile
import unittest
from glob import glob

from ase.calculators.calculator import names as calc_names, get_calculator
from ase.parallel import paropen
from ase.utils import devnull
from ase.cli.info import print_info


NotAvailable = unittest.SkipTest


test_calculator_names = []


def require(calcname):
    if calcname not in test_calculator_names:
        raise NotAvailable('use --calculators={0} to enable'.format(calcname))


class CustomTextTestRunner(unittest.TextTestRunner):
    def __init__(self, logname, descriptions=1, verbosity=1):
        self.f = paropen(logname, 'w')
        unittest.TextTestRunner.__init__(self, self.f, descriptions, verbosity)

    def run(self, test):
        stderr_old = sys.stderr
        try:
            sys.stderr = self.f
            testresult = unittest.TextTestRunner.run(self, test)
        finally:
            sys.stderr = stderr_old
        return testresult


class ScriptTestCase(unittest.TestCase):
    def __init__(self, methodname='testfile', filename=None):
        unittest.TestCase.__init__(self, methodname)
        self.filename = filename

    def testfile(self):
        try:
            with open(self.filename) as fd:
                exec(compile(fd.read(), self.filename, 'exec'), {})
        except ImportError as ex:
            module = ex.args[0].split()[-1].replace("'", '').split('.')[0]
            if module in ['scipy', 'matplotlib', 'Scientific', 'lxml',
                          'flask', 'gpaw', 'GPAW']:
                raise unittest.SkipTest('no {} module'.format(module))
            else:
                raise

    def id(self):
        return self.filename

    def __str__(self):
        return self.filename.split('test/')[-1]

    def __repr__(self):
        return "ScriptTestCase(filename='%s')" % self.filename


def get_tests(files=None):
    if files:
        files = [os.path.join(__path__[0], f) for f in files]
    else:
        files = glob(__path__[0] + '/*')

    sdirtests = []  # tests from subdirectories: only one level assumed
    tests = []
    for f in files:
        if os.path.isdir(f):
            # add test subdirectories (like calculators)
            sdirtests.extend(glob(f + '/*.py'))
        else:
            # add py files in testdir
            if f.endswith('.py'):
                tests.append(f)
    tests.sort()
    sdirtests.sort()
    tests.extend(sdirtests)  # run test subdirectories at the end
    tests = [test for test in tests if not test.endswith('__.py')]
    return tests


def test(verbosity=1, calculators=[],
         testdir=None, stream=sys.stdout, files=None):
    test_calculator_names.extend(calculators)
    disable_calculators([name for name in calc_names
                         if name not in calculators])

    tests = get_tests(files)

    ts = unittest.TestSuite()

    for test in tests:
        ts.addTest(ScriptTestCase(filename=os.path.abspath(test)))

    if verbosity > 0:
        print_info()

    sys.stdout = devnull

    if verbosity == 0:
        stream = devnull
    ttr = unittest.TextTestRunner(verbosity=verbosity, stream=stream)

    origcwd = os.getcwd()

    if testdir is None:
        testdir = tempfile.mkdtemp(prefix='ase-test-')
    else:
        if os.path.isdir(testdir):
            shutil.rmtree(testdir)  # clean before running tests!
        os.mkdir(testdir)
    os.chdir(testdir)
    if verbosity:
        print('test-dir       ', testdir, '\n', file=sys.__stdout__)
    try:
        results = ttr.run(ts)
    finally:
        os.chdir(origcwd)
        sys.stdout = sys.__stdout__

    return results


def disable_calculators(names):
    for name in names:
        if name in ['emt', 'lj', 'eam', 'morse', 'tip3p']:
            continue
        try:
            cls = get_calculator(name)
        except ImportError:
            pass
        else:
            def get_mock_init(name):
                def mock_init(obj, *args, **kwargs):
                    raise NotAvailable('use --calculators={0} to enable'
                                       .format(name))
                return mock_init

            def mock_del(obj):
                pass
            cls.__init__ = get_mock_init(name)
            cls.__del__ = mock_del


def cli(command, calculator_name=None):
    if (calculator_name is not None and
        calculator_name not in test_calculator_names):
        return
    proc = subprocess.Popen(' '.join(command.split('\n')),
                            shell=True,
                            stdout=subprocess.PIPE)
    print(proc.stdout.read().decode())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError('Failed running a shell command.  '
                           'Please set you $PATH environment variable!')


class must_raise:
    """Context manager for checking raising of exceptions."""
    def __init__(self, exception):
        self.exception = exception

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            raise RuntimeError('Failed to fail: ' + str(self.exception))
        return issubclass(exc_type, self.exception)


class CLICommand:
    short_description = 'Test ASE'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            '-c', '--calculators',
            help='Comma-separated list of calculators to test.')
        parser.add_argument('-v', '--verbose', help='verbose mode',
                            action='store_true')
        parser.add_argument('-q', '--quiet', action='store_true',
                            help='quiet mode')
        parser.add_argument('--list', action='store_true',
                            help='print all tests and exit')
        parser.add_argument('--list-calculators', action='store_true',
                            help='print all calculator names and exit')
        parser.add_argument('tests', nargs='*')

    @staticmethod
    def run(args):
        if args.calculators:
            calculators = args.calculators.split(',')
        else:
            calculators = []

        if args.list:
            for testfile in get_tests():
                print(testfile)
            sys.exit(0)

        if args.list_calculators:
            for name in calc_names:
                print(name)
            sys.exit(0)

        for calculator in calculators:
            if calculator not in calc_names:
                sys.stderr.write('No calculator named "{}".\n'
                                 'Possible CALCULATORS are: '
                                 '{}.\n'.format(calculator,
                                                ', '.join(calc_names)))
                sys.exit(1)

        results = test(verbosity=1 + args.verbose - args.quiet,
                       calculators=calculators,
                       files=args.tests)
        sys.exit(len(results.errors + results.failures))


if __name__ == '__main__':
    # Run pyflakes3 on all code in ASE:
    try:
        output = subprocess.check_output(['pyflakes3', 'ase', 'doc'],
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as ex:
        output = ex.output.decode()

    lines = []
    for line in output.splitlines():
        # Ignore these:
        for txt in ['jacapo', 'list comprehension redefines']:
            if txt in line:
                break
        else:
            lines.append(line)
    if lines:
        print('\n'.join(lines))
        sys.exit(1)
