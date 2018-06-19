import numpy as np
from ase.quaternions import Quaternion


def rand_rotm():
    """Axis & angle rotations."""
    u = np.random.random(3)
    u /= np.linalg.norm(u)
    theta = np.random.random() * np.pi * 2
    
    # Cross product matrix for u
    ucpm = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    
    # Rotation matrix
    rotm = (np.cos(theta) * np.identity(3) + np.sin(theta) * ucpm +
            (1 - np.cos(theta)) * np.kron(u[:, None], u[None, :]))
    
    return rotm
    
# First: test that rotations DO work
for i in range(10):
    # 10 random tests
        
    rotm = rand_rotm()

    q = Quaternion.from_matrix(rotm)

    # Now test this with a vector
    v = np.random.random(3)

    vrotM = np.dot(rotm, v)
    vrotQ = q.rotate(v)

    assert np.allclose(vrotM, vrotQ)

# Second: test the special case of a PI rotation

rotm = np.identity(3)
rotm[:2, :2] *= -1               # Rotate PI around z axis

q = Quaternion.from_matrix(rotm)

assert not np.isnan(q.q).any()

# Third: test compound rotations and operator overload
for i in range(10):

    rotm1 = rand_rotm()
    rotm2 = rand_rotm()
    
    q1 = Quaternion.from_matrix(rotm1)
    q2 = Quaternion.from_matrix(rotm2)
    
    # Now test this with a vector
    v = np.random.random(3)
    
    vrotM = np.dot(rotm2, np.dot(rotm1, v))
    vrotQ = (q2 * q1).rotate(v)
    
    assert np.allclose(vrotM, vrotQ)
