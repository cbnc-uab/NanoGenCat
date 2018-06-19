from ase import Atoms
from ase.db import connect
import ase.db.app as app
c = connect('test.db', append=False)
plot = {'title': 'A test',
        'data': [{'label': 't1', 'x': 'x', 'y': 't1'},
                 {'label': 't2', 'style': 'y--',
                  'x': 'x', 'y': 't2'}],
        'xlabel': 'x',
        'ylabel': 'y'}
x = [0, 1, 2]
t1 = [1, 2, 0]
t2 = [[2, 3], [1, 1], [1, 0]]
c.write(Atoms('H2O'),
        foo=42.0,
        bar='abc',
        data={'test': plot,
              'x': x,
              't1': t1,
              't2': t2})
c.metadata = {'title': 'Test title',
              'key_descriptions':
                  {'foo': ('FOO', 'FOO ...', '`m_e`')},
              'default_columns': ['foo', 'formula', 'bar']}
c.python = None
app.databases['default'] = c
app.app.testing = True
c = app.app.test_client()
page = c.get('/').data.decode()
assert 'Test title' in page
assert 'FOO' in page
c.get('/id/1')
c.get('/plot/test-1.png')
c.get('json/1').data
c.get('sqlite/1').data
c.get('sqlite?x=1').data
c.get('json?x=1').data
