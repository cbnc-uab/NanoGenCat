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
        foo='bar',
        data={'test': plot,
              'x': x,
              't1': t1,
              't2': t2})
app.db = c
app.app.testing = True
d = app.app.test_client().get('/')
print(d)
d = app.app.test_client().get('/id/1')
print(d)
d = app.app.test_client().get('/plot/test-1.png')
print(d, app.tmpdir)
