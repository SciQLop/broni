from . import Shape
from .. import Trajectory

import numpy as np


class Sphere(Shape):
    def __init__(self, x: float, y: float, z: float, r: float):
        if r <= 0:
            raise ValueError("r has to be bigger than 0 to define a sphere")

        self.center = np.array((x, y, z), dtype=float)
        self.radius = r

    def intersect(self, trajectory: Trajectory):
        dist = np.linalg.norm(self.center - trajectory.data, axis=1)
        return dist <= self.radius

    def to_mesh(self):
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = self.center[0] + self.radius * np.cos(u) * np.sin(v)
        y = self.center[1] + self.radius * np.sin(u) * np.sin(v)
        z = self.center[2] + self.radius * np.cos(v)

        return np.array((x, y, z), dtype=float)


class Cuboid(Shape):
    def __init__(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float):
        self.p1 = np.array((x0, y0, z0), dtype=float)
        self.p2 = np.array((x1, y0, z0), dtype=float)
        self.p3 = np.array((x0, y1, z0), dtype=float)
        self.p4 = np.array((x0, y0, z1), dtype=float)

        if (x0, y0, z0) == (x1, y1, z1):
            raise ValueError("p0 is equal to p1, a Cuboid of zero volume is not supported.")

    def intersect(self, trajectory: Trajectory):
        if len(trajectory.data) == 0:
            return np.array([])

        def f(b):
            v = self.p1 - b
            vp1, vb = sorted([np.dot(v, p) for p in (self.p1, b)])
            o = np.dot(trajectory.data, v)
            return vp1, vb, o

        u_p1, u_p2, uO = f(self.p2)
        v_p1, v_p3, vO = f(self.p3)
        w_p1, w_p4, wO = f(self.p4)

        return np.logical_and.reduce((u_p1 <= uO, uO <= u_p2,
                                      v_p1 <= vO, vO <= v_p3,
                                      w_p1 <= wO, wO <= w_p4))

    def to_mesh(self):
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        p4 = self.p4

        X = [[p1[0], p2[0], p2[0], p1[0], p1[0]],
             [p1[0], p2[0], p2[0], p1[0], p1[0]],
             [p1[0], p2[0], p2[0], p1[0], p1[0]],
             [p1[0], p2[0], p2[0], p1[0], p1[0]]]

        Y = [[p1[1], p1[1], p3[1], p3[1], p1[1]],
             [p1[1], p1[1], p3[1], p3[1], p1[1]],
             [p1[1], p1[1], p1[1], p1[1], p1[1]],
             [p3[1], p3[1], p3[1], p3[1], p3[1]]]

        Z = [[p1[2], p1[2], p1[2], p1[2], p1[2]],
             [p4[2], p4[2], p4[2], p4[2], p4[2]],
             [p1[2], p1[2], p4[2], p4[2], p1[2]],
             [p1[2], p1[2], p4[2], p4[2], p1[2]]]

        return np.array((X, Y, Z), dtype=float)
