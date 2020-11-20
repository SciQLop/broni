from . import Shape
from .. import Trajectory

import numpy as np
from astropy.units.quantity import Quantity
from astropy.units import km


class Sphere(Shape):
    def __init__(self, x: Quantity, y: Quantity, z: Quantity, r: Quantity):
        if r <= 0:
            raise ValueError("r has to be greater than 0 to define a sphere")

        self.center = np.array((x.to('km').value, y.to('km').value, z.to('km').value)) * km
        self.radius = r.to('km')

    def intersect(self, trajectory: Trajectory):
        dist = np.linalg.norm(self.center - trajectory.cartesian, axis=1)
        return dist <= self.radius


class Cuboid(Shape):
    def __init__(self, x0: Quantity, y0: Quantity, z0: Quantity, x1: Quantity, y1: Quantity, z1: Quantity):
        self.p1 = np.array((x0.to('km').value, y0.to('km').value, z0.to('km').value), dtype=float) * km
        self.p2 = np.array((x1.to('km').value, y0.to('km').value, z0.to('km').value), dtype=float) * km
        self.p3 = np.array((x0.to('km').value, y1.to('km').value, z0.to('km').value), dtype=float) * km
        self.p4 = np.array((x0.to('km').value, y0.to('km').value, z1.to('km').value), dtype=float) * km

        if (x0, y0, z0) == (x1, y1, z1):
            raise ValueError("p0 is equal to p1, a Cuboid of zero volume is not supported.")

    def intersect(self, trajectory: Trajectory):
        def f(b):
            v = self.p1 - b
            vp1, vb = sorted([np.dot(v, p) for p in (self.p1, b)])
            o = np.dot(trajectory.cartesian, v)
            return vp1, vb, o

        u_p1, u_p2, uO = f(self.p2)
        v_p1, v_p3, vO = f(self.p3)
        w_p1, w_p4, wO = f(self.p4)

        return np.logical_and.reduce((u_p1 <= uO, uO <= u_p2,
                                      v_p1 <= vO, vO <= v_p3,
                                      w_p1 <= wO, wO <= w_p4))
