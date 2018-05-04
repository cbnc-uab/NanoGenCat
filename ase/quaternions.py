import numpy as np
from ase.atoms import Atoms


class Quaternions(Atoms):
    def __init__(self, *args, **kwargs):
        quaternions = None
        if 'quaternions' in kwargs:
            quaternions = np.array(kwargs['quaternions'])
            del kwargs['quaternions']
        Atoms.__init__(self, *args, **kwargs)
        if quaternions is not None:
            self.set_array('quaternions', quaternions, shape=(4,))
            # set default shapes
            self.set_shapes(np.array([[3, 2, 1]] * len(self)))

    def set_shapes(self, shapes):
        self.set_array('shapes', shapes, shape=(3,))

    def set_quaternions(self, quaternions):
        self.set_array('quaternions', quaternions, quaternion=(4,))

    def get_shapes(self):
        return self.get_array('shapes')

    def get_quaternions(self):
        return self.get_array('quaternions').copy()

        
class Quaternion:
    def __init__(self, qin=[1, 0, 0, 0]):
        assert(len(qin) == 4)
        self.q = np.array(qin)

    def __str__(self):
        return self.q.__str__()

    def __mul__(self, other):
        sw, sx, sy, sz = self.q[0], self.q[1], self.q[2], self.q[3]
        ow, ox, oy, oz = other.q[0], other.q[1], other.q[2], other.q[3]
        return Quaternion([sw * ow - sx * ox - sy * oy - sz * oz,
                           sw * ox + sx * ow + sy * oz - sz * oy,
                           sw * oy + sy * ow + sz * ox - sx * oz,
                           sw * oz + sz * ow + sx * oy - sy * ox])

    def conjugate(self):
        return Quaternion(-self.q * np.array([-1., 1., 1., 1.]))

    def rotate(self, vector):
        """Apply the rotation matrix to a vector."""
        qw, qx, qy, qz = self.q[0], self.q[1], self.q[2], self.q[3]
        x, y, z = vector[0], vector[1], vector[2]
        
        ww = qw * qw
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        
        return np.array(
            [(ww + xx - yy - zz) * x + 2 * ((xy - wz) * y + (xz + wy) * z),
             (ww - xx + yy - zz) * y + 2 * ((xy + wz) * x + (yz - wx) * z),
             (ww - xx - yy + zz) * z + 2 * ((xz - wy) * x + (yz + wx) * y)])

    def rotation_matrix(self):

        qw, qx, qy, qz = self.q[0], self.q[1], self.q[2], self.q[3]
        
        ww = qw * qw
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz

        return np.array([[ww + xx - yy - zz, 2 * (xy + wz), 2 * (xz - wy)],
                         [2 * (xy - wz), ww - xx + yy - zz, 2 * (yz + wx)],
                         [2 * (xz + wy), 2 * (yz - wx), ww - xx - yy + zz]])

    def arc_distance(self, other):
        """Gives a metric of the distance between two quaternions,
        expressed as 1-|q1.q2|"""

        return 1.0 - np.abs(np.dot(self.q, other.q))

    @staticmethod
    def rotate_byq(q, vector):
        """Apply the rotation matrix to a vector."""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        x, y, z = vector[0], vector[1], vector[2]
        
        ww = qw * qw
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        
        return np.array(
            [(ww + xx - yy - zz) * x + 2 * ((xy - wz) * y + (xz + wy) * z),
             (ww - xx + yy - zz) * y + 2 * ((xy + wz) * x + (yz - wx) * z),
             (ww - xx - yy + zz) * z + 2 * ((xz - wy) * x + (yz + wx) * y)])
    
    @staticmethod
    def from_matrix(matrix):
        """Build quaternion from rotation matrix."""
        m = np.array(matrix)
        assert m.shape == (3, 3)

        # Now we need to find out the whole quaternion
        # This method takes into account the possibility of qw being nearly
        # zero, so it picks the stablest solution

        if m[2, 2] < 0:
            if (m[0, 0] > m[1, 1]):
                # Use x-form
                qx = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) / 2.0
                fac = 1.0 / (4 * qx)
                qw = (m[2, 1] - m[1, 2]) * fac
                qy = (m[0, 1] + m[1, 0]) * fac
                qz = (m[0, 2] + m[2, 0]) * fac
            else:
                # Use y-form
                qy = np.sqrt(1 - m[0, 0] + m[1, 1] - m[2, 2]) / 2.0
                fac = 1.0 / (4 * qy)
                qw = (m[0, 2] - m[2, 0]) * fac
                qx = (m[0, 1] + m[1, 0]) * fac
                qz = (m[1, 2] + m[2, 1]) * fac
        else:
            if (m[0, 0] < -m[1, 1]):
                # Use z-form
                qz = np.sqrt(1 - m[0, 0] - m[1, 1] + m[2, 2]) / 2.0
                fac = 1.0 / (4 * qz)
                qw = (m[1, 0] - m[0, 1]) * fac
                qx = (m[2, 0] + m[0, 2]) * fac
                qy = (m[1, 2] + m[2, 1]) * fac
            else:
                # Use w-form
                qw = np.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
                fac = 1.0 / (4 * qw)
                qx = (m[2, 1] - m[1, 2]) * fac
                qy = (m[0, 2] - m[2, 0]) * fac
                qz = (m[1, 0] - m[0, 1]) * fac

        return Quaternion(np.array([qw, qx, qy, qz]))
