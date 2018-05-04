import numpy as np
from ase.io.ulm import open


class A:
    def write(self, writer):
        writer.write(x=np.ones((2, 3)))

    @staticmethod
    def read(reader):
        a = A()
        a.x = reader.x
        return a

w = open('a.ulm', 'w')
w.write(a=A(), y=9)
w.write(s='abc')
w.sync()
w.write(s='abc2')
w.sync()
w.write(s='abc3', z=np.ones(7, int))
w.close()
print(w.data)

r = open('a.ulm')
print(r.y, r.s)
print(A.read(r.a).x)
print(r.a.x)
print(r[1].s)
print(r[2].s)
print(r[2].z)

w = open('a.ulm', 'a')
print(w.nitems, w.offsets)
w.write(d={'h': [1, 'asdf']})
w.add_array('psi', (4, 3))
w.fill(np.ones((1, 3)))
w.fill(np.ones((1, 3)) * 2)
w.fill(np.ones((2, 3)) * 3)
w.close()
print(open('a.ulm', 'r', 3).d)
print(open('a.ulm')[2].z)
print(open('a.ulm', index=3).proxy('psi')[0:3])
for d in open('a.ulm'):
    print(d)
