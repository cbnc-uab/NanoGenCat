"""WSGI Flask-app for browsing a database.

You can launch Flask's local webserver like this::

    $ ase-db abc.db -w

For a real webserver, you need to set the $ASE_DB_APP_CONFIG environment
variable to point to a configuration file like this::

    ASE_DB_NAME = '/path/to/db-file/abc.db'
    ASE_DB_HOMEPAGE = '<a href="https://home.page.dk">HOME</a> ::'

Start with something like::

    twistd web --wsgi=ase.db.app.app --port=8000

"""

import collections
import functools
import io
import os
import re
import tempfile

from flask import Flask, render_template, request, send_from_directory

try:
    import matplotlib
    matplotlib.use('Agg', warn=False)
except ImportError:
    pass

import ase.db
from ase.db.plot import atoms2png, dct2plot
from ase.db.summary import Summary
from ase.db.table import Table, all_columns
from ase.visualize import view


# Every client-connetions gets one of these tuples:
Connection = collections.namedtuple(
    'Connection',
    ['query',  # query string
     'nrows',  # number of rows matched
     'page',  # page number
     'columns',  # what columns to show
     'sort',  # what column to sort after
     'limit',  # number of rows per page
     'opened'  # list of id's in the table that are open
     ])

app = Flask(__name__)

db = None
home = ''  # link to homepage
open_ase_gui = True  # click image to open ase-gui

if 'ASE_DB_APP_CONFIG' in os.environ:
    app.config.from_envvar('ASE_DB_APP_CONFIG')
    db = ase.db.connect(app.config['ASE_DB_NAME'])
    home = app.config['ASE_DB_HOMEPAGE']
    open_ase_gui = False

next_con_id = 1
connections = {}
tmpdir = tempfile.mkdtemp()  # used to cache png-files

# Find numbers in formulas so that we can convert H2O to H<sub>2</sub>O:
SUBSCRIPT = re.compile(r'(\d+)')


@app.route('/')
def index():
    global next_con_id
    con_id = int(request.args.get('x', '0'))
    if con_id not in connections:
        con_id = next_con_id
        next_con_id += 1
        query = ''
        columns = list(all_columns)
        sort = 'id'
        limit = 25
        opened = set()
        nrows = None
        page = 0
    else:
        query, nrows, page, columns, sort, limit, opened = connections[con_id]

    if 'sort' in request.args:
        column = request.args['sort']
        if column == sort:
            sort = '-' + column
        elif '-' + column == sort:
            sort = 'id'
        else:
            sort = column
        page = 0
    elif 'query' in request.args:
        query = request.args['query'].encode()
        try:
            limit = max(1, min(int(request.args.get('limit', limit)), 200))
        except ValueError:
            pass
        sort = 'id'
        opened = set()
        page = 0
        nrows = None
    elif 'page' in request.args:
        page = int(request.args['page'])

    if 'toggle' in request.args:
        tcolumns = request.args['toggle'].split(',')
        if tcolumns == ['reset']:
            columns = list(all_columns)
        else:
            for column in tcolumns:
                if column in columns:
                    columns.remove(column)
                    if column == sort.lstrip('-'):
                        sort = 'id'
                        page = 0
                else:
                    columns.append(column)

    if nrows is None:
        nrows = db.count(query)

    table = Table(db)
    table.select(query, columns, sort, limit, offset=page * limit)
    con = Connection(query, nrows, page, columns, sort, limit, opened)
    connections[con_id] = con
    table.format(SUBSCRIPT)
    addcolumns = [column for column in all_columns + table.keys
                  if column not in table.columns]

    return render_template('table.html', t=table, con=con, cid=con_id,
                           home=home, pages=pages(page, nrows, limit),
                           nrows=nrows,
                           addcolumns=addcolumns,
                           row1=page * limit + 1,
                           row2=min((page + 1) * limit, nrows))


@app.route('/open_row/<int:id>')
def open_row(id):
    con_id = int(request.args['x'])
    opened = connections[con_id].opened
    if id in opened:
        opened.remove(id)
        return ''
    opened.add(id)
    return render_template('more.html',
                           dct=db.get(id), id=id, cid=con_id)


@app.route('/image/<name>')
def image(name):
    path = os.path.join(tmpdir, name)
    if not os.path.isfile(path):
        id = int(name[:-4])
        atoms = db.get_atoms(id)
        atoms2png(atoms, path)

    return send_from_directory(tmpdir, name)


@app.route('/cif/<name>')
def cif(name):
    path = os.path.join(tmpdir, name)
    if not os.path.isfile(path):
        id = int(name[:-4])
        atoms = db.get_atoms(id)
        atoms.write(path)
    return send_from_directory(tmpdir, name)


@app.route('/plot/<png>')
def plot(png):
    path = os.path.join(tmpdir, png)
    if not os.path.isfile(path):
        name, id = png[:-4].split('-')
        dct = db[int(id)].data
        dct2plot(dct, name, path, show=False)

    return send_from_directory(tmpdir, png)


@app.route('/gui/<int:id>')
def gui(id):
    if open_ase_gui:
        atoms = db.get_atoms(id)
        view(atoms)
    return '', 204, []


@app.route('/id/<int:id>')
def summary(id):
    s = Summary(db.get(id), SUBSCRIPT)
    return render_template('summary.html', s=s, home=home,
                           open_ase_gui=open_ase_gui)


def tofile(query, type, limit=0):
    fd, name = tempfile.mkstemp(suffix='.' + type)
    con = ase.db.connect(name, use_lock_file=False)
    for dct in db.select(query, limit=limit):
        con.write(dct,
                  data=dct.get('data', {}),
                  **dct.get('key_value_pairs', {}))
    os.close(fd)
    data = open(name).read()
    os.unlink(name)
    return data


def download(f):
    @functools.wraps(f)
    def ff(*args, **kwargs):
        text, name = f(*args, **kwargs)
        headers = [('Content-Disposition',
                    'attachment; filename="{0}"'.format(name)),
                   ]  # ('Content-type', 'application/sqlite3')]
        return text, 200, headers
    return ff


@app.route('/xyz/<int:id>')
@download
def xyz(id):
    fd = io.BytesIO()
    from ase.io.xyz import write_xyz
    write_xyz(fd, db.get_atoms(id))
    data = fd.getvalue()
    return data, '{0}.xyz'.format(id)


@app.route('/json')
@download
def jsonall():
    con_id = int(request.args['x'])
    con = connections[con_id]
    data = tofile(con.query, 'json', con.limit)
    return data, 'selection.json'


@app.route('/json/<int:id>')
@download
def json(id):
    data = tofile(id, 'json')
    return data, '{0}.json'.format(id)


@app.route('/sqlite')
@download
def sqliteall():
    con_id = int(request.args['x'])
    con = connections[con_id]
    data = tofile(con.query, 'db', con.limit)
    return data, 'selection.db'


@app.route('/sqlite/<int:id>')
@download
def sqlite(id):
    data = tofile(id, 'db')
    return data, '{0}.db'.format(id)


@app.route('/robots.txt')
def robots():
    return 'User-agent: *\nDisallow: /\n', 200


def pages(page, nrows, limit):
    npages = (nrows + limit - 1) // limit
    p1 = min(5, npages)
    p2 = max(page - 4, p1)
    p3 = min(page + 5, npages)
    p4 = max(npages - 4, p3)
    pgs = list(range(p1))
    if p1 < p2:
        pgs.append(-1)
    pgs += list(range(p2, p3))
    if p3 < p4:
        pgs.append(-1)
    pgs += list(range(p4, npages))
    pages = [(page - 1, 'previous')]
    for p in pgs:
        if p == -1:
            pages.append((-1, '...'))
        elif p == page:
            pages.append((-1, str(p + 1)))
        else:
            pages.append((p, str(p + 1)))
    nxt = min(page + 1, npages - 1)
    if nxt == page:
        nxt = -1
    pages.append((nxt, 'next'))
    return pages
