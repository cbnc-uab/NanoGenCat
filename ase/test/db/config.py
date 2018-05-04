from ase.db.web import creates

title = 'TEST'

default_columns = ['formula', 'answer', 'kind']

special_keys = [('SELECT', 'kind'),
                ('BOOL', 'foo'),
                ('RANGE', 'ans', 'Answer', [('A1', 'answer'),
                                            ('B2', 'answer')])]

key_descriptions = {
    'kind': ('Type', 'Type of system', ''),
    'answer': ('Answer', 'Answer to question', 'eV')}


@creates('xy.png', 'abc.png')
def xy(row):
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(111)
    ax.plot([0, 1, 2, 3, 0, 1])
    plt.savefig('xy.png')
    if row.natoms > 1:
        ax.plot([2, 2, 2, 3, 3, 3])
        plt.savefig('abc.png')


stuff = ('Stuff', ['energy', 'fmax', 'charge', 'mass', 'magmom', 'volume'])
things = ('Things', ['answer', 'kind'])
calc = ('Calculator Setting', ['calculator'])

layout = [
    ('Basic properties',
     [stuff, 'ATOMS',
      things, 'CELL']),
    ('Calculation details',
     [calc, None,
      'FORCES', None,
      'xy.png', 'abc.png'])]
