import sys
import psycopg2

from ase.db.sqlite import init_statements, index_statements, VERSION
from ase.db.sqlite import all_tables, SQLite3Database


class Connection:
    def __init__(self, con):
        self.con = con

    def cursor(self):
        return Cursor(self.con.cursor())
    
    def commit(self):
        self.con.commit()

    def close(self):
        self.con.close()


class Cursor:
    def __init__(self, cur):
        self.cur = cur

    def fetchone(self):
        return self.cur.fetchone()

    def fetchall(self):
        return self.cur.fetchall()

    def execute(self, statement, *args):
        self.cur.execute(statement.replace('?', '%s'), *args)

    def executemany(self, statement, *args):
        self.cur.executemany(statement.replace('?', '%s'), *args)


def parse_name(name):
    """Parse user:password@host:port string.
    
    Defaults are ase:ase@localhost:5432.  Returns user, password, host, port.
    """
    if '@' in name:
        user, host = name.split('@')
    else:
        user = 'ase'
        host = name
    if ':' in user:
        user, password = user.split(':')
    else:
        password = 'ase'
    if ':' in host:
        host, port = host.split(':')
        port = int(port)
    else:
        port = 5432
    host = host or None
    return user, password, host, port
    
    
class PostgreSQLDatabase(SQLite3Database):
    default = 'DEFAULT'
    
    def _connect(self):
        user, password, host, port = parse_name(self.filename)
        if host == 'localhost':
            host = None
        con = psycopg2.connect(database='postgres',
                               user=user, password=password,
                               host=host, port=port)
        return Connection(con)

    def _initialize(self, con):
        self.version = VERSION
    
    def get_last_id(self, cur):
        cur.execute('SELECT last_value FROM systems_id_seq')
        id = cur.fetchone()[0]
        return int(id)


def reset():
    con = psycopg2.connect(database='postgres', user='postgres')
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) FROM pg_tables WHERE tablename='systems'")
    if cur.fetchone()[0] == 1:
        cur.execute('DROP TABLE %s CASCADE' % ', '.join(all_tables))
        cur.execute('DROP TABLE information CASCADE')
        cur.execute('DROP ROLE ase')
    if len(sys.argv) == 2:
        pw = sys.argv[1]
    else:
        pw = 'ase'
    cur.execute("CREATE ROLE ase LOGIN PASSWORD %s", (pw,))
    con.commit()

    sql = ';\n'.join(init_statements)
    for a, b in [('BLOB', 'BYTEA'),
                 ('REAL', 'DOUBLE PRECISION'),
                 ('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')]:
        sql = sql.replace(a, b)
        
    cur.execute(sql)
    cur.execute(';\n'.join(index_statements))
    cur.execute('GRANT ALL PRIVILEGES ON %s TO ase' %
                ', '.join(all_tables + ['systems_id_seq']))
    con.commit()


if __name__ == '__main__':
    # Debian
    # sudo -u postgres PYTHONPATH=/path/to/ase python -m ase.db.postgresql
    # RHEL:
    # su -c "yum -y remove postgresql-server"
    # su -c "rm -rf /var/lib/pgsql"
    # su -c "yum -y install postgresql-server authd"
    # su -c "postgresql-setup initdb"
    # su -c "systemctl start auth.socket"
    # su -c "systemctl start postgresql.service"
    # su -
    # su - postgres
    # psql -d postgres -U postgres -c "create role ase login password 'ase';"
    # PYTHONPATH=/path/to/ase python -m ase.db.postgresql
    reset()
