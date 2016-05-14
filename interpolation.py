#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`LST` for generating linear synchronous transit
interpolation paths.
"""

import numpy as np
from scipy.optimize import minimize

import warnings

from node import Node

###############################################################################

class LST(object):
    """
    Contains the main functions required to generate an LST interpolation path.
    The attributes are:

    =============== ======================== ===================================
    Attribute       Type                     Description
    =============== ======================== ===================================
    `node_start`    :class:`node.Node`       A node object representing the start of the LST path
    `node_end`      :class:`node.Node`       A node object representing the end of the LST path
    `node_start_r`  :class:`numpy.ndarray`   An N x N array of the distance matrix coordinates of `node_start`
    `node_end_r`    :class:`numpy.ndarray`   An N x N array of the distance matrix coordinates of `node_end`
    =============== ======================== ===================================

    """

    def __init__(self, node_start, node_end):
        if node_start.number != node_end.number:
            raise Exception('Atom labels at the start and end of LST path do not match')
        self.node_start = node_start
        self.node_end = node_end
        self.node_start_r = self.getDistMat(node_start.coordinates)
        self.node_end_r = self.getDistMat(node_end.coordinates)

    def __str__(self):
        """
        Return a human readable string representation of the object
        """
        return 'LST path object between nodes\n' + self.node_start.__str__() + '\nand\n' + self.node_end.__str__()

    @staticmethod
    def getDistMat(coords):
        """
        Calculate and return distance matrix form given a vector (np.array) of
        Cartesian coordinates. The matrix is N x N. Only the upper diagonal
        elements contain the distances. All other elements are set to 0.
        """
        assert len(coords) % 3 == 0
        coord_array = coords.reshape(len(coords) / 3, 3)
        x = coord_array[:, 0]
        y = coord_array[:, 1]
        z = coord_array[:, 2]
        dx = x[..., np.newaxis] - x[np.newaxis, ...]
        dy = y[..., np.newaxis] - y[np.newaxis, ...]
        dz = z[..., np.newaxis] - z[np.newaxis, ...]
        return np.triu((np.array([dx, dy, dz]) ** 2).sum(axis=0) ** 0.5)

    def LSTobjective(self, w, f):
        """
        Defines the objective function for LST interpolation. Interpolated
        Cartesian coordinates and distance matrix coordinates are calculated
        based on the fractional distance along the LST path, `f`. Cartesian
        coordinates and distance matrix coordinates are denoted by `w` and `r`,
        respectively.
        """
        r_interpolated = self.node_start_r + f * (self.node_end_r - self.node_start_r)
        w_interpolated = self.node_start.coordinates + f * (self.node_end.coordinates - self.node_start.coordinates)
        r = self.getDistMat(w)

        distance_term = 0.0
        for d in range(1, len(self.node_start.number)):
            distance_term += np.array((r_interpolated.diagonal(d) - r.diagonal(d)) ** 2.0 /
                                      r_interpolated.diagonal(d) ** 4.0).sum()
        cartesian_term = 1e-6 * np.array((w_interpolated - w) ** 2.0).sum()
        return distance_term + cartesian_term

    def getLSTnode(self, f):
        """
        Minimize LST objective function to find the coordinates on the LST path
        defined by the fractional distance along the path, `f`. A Node object
        containing the optimized Cartesian coordinates is returned.
        """
        w_guess = self.node_start.coordinates + f * (self.node_end.coordinates - self.node_start.coordinates)
        result = minimize(self.LSTobjective, w_guess, args=(f,), method='BFGS')
        if not result.success:
            message = 'LST minimization terminated with status ' + str(result.status) + ':\n' + result.message
            warnings.warn(message)
        return Node(result.x, self.node_start.number, self.node_start.multiplicity)

    def getLSTpath(self, nnodes=100):
        """
        Generates an LST path between `node_start` and `node_end` represented
        as a list of Node objects containing `nnodes` nodes along the path.
        `node_start` and `node_end` are included as the path endpoints.
        """
        inc = 1.0 / float(nnodes - 1)
        path = [self.getLSTnode(f) for f in np.linspace(inc, 1.0 - inc, nnodes - 2)]
        path.insert(0, self.node_start)
        path.append(self.node_end)
        return path
