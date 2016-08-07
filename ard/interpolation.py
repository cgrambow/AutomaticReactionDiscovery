#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#
#   ARD - Automatic Reaction Discovery
#
#   Copyright (c) 2016 Prof. William H. Green (whgreen@mit.edu) and Colin
#   Grambow (cgrambow@mit.edu)
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
###############################################################################

"""
Contains the :class:`CartesianInterp` for computing linearly interpolated
Cartesian nodes and the :class:`LST` for generating linear synchronous transit
interpolation paths.

Two functions are defined which enable pickling and unpickling of class methods
so that these can be serialized for map functions of parallelization packages.
"""

from __future__ import division

import copy_reg
import multiprocessing
import types
import warnings

import numpy as np
from scipy import optimize

import util
from node import Node

###############################################################################

def _pickle_method(method):
    """
    Pickles method.
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    """
    Unpickles method.
    """
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

# Add functionality for pickling bound methods
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

###############################################################################

class CartesianInterp(object):
    """
    Cartesian interpolation object.
    The attributes are:

    =============== ======================== ==================================
    Attribute       Type                     Description
    =============== ======================== ==================================
    `node_start`    :class:`node.Node`       A node object representing one end for the interpolation
    `node_end`      :class:`node.Node`       A node object representing the other end
    =============== ======================== ==================================

    """

    def __init__(self, node_start, node_end):
        if node_start.atoms != node_end.atoms:
            raise Exception('Atom labels at the start and end of LST path do not match')
        self.node_start = node_start
        self.node_end = node_end

    def getCartNode(self, f):
        """
        Generates a node at fractional distance, `f`, between start and end
        nodes computed using simple Cartesian interpolation.
        """
        return Node(self.node_start.coordinates + f * (self.node_end.coordinates - self.node_start.coordinates),
                    self.node_start.atoms, self.node_start.multiplicity)

    def getCartNodeAtDistance(self, distance):
        """
        Generates a node at a specified distance from `node_start`.
        """
        diff = self.node_end.coordinates.flatten() - self.node_start.coordinates.flatten()
        dist_factor = diff / (diff.dot(diff)) ** 0.5
        return Node(self.node_start.coordinates.flatten() + distance * dist_factor,
                    self.node_start.atoms, self.node_start.multiplicity)

###############################################################################

class LST(CartesianInterp):
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

    def __init__(self, node_start, node_end, nproc=1):
        super(LST, self).__init__(node_start, node_end)
        self.nproc = nproc
        self.node_start_r = util.getDistMat(node_start.coordinates)
        self.node_end_r = util.getDistMat(node_end.coordinates)

    def LSTobjective(self, w, f):
        """
        Defines the objective function for LST interpolation. Interpolated
        Cartesian coordinates and distance matrix coordinates are calculated
        based on the fractional distance along the LST path, `f`. Cartesian
        coordinates and distance matrix coordinates are denoted by `w` and `r`,
        respectively.
        """
        # Compute interpolated coordinates based on fraction along LST path
        r_interpolated = self.node_start_r + f * (self.node_end_r - self.node_start_r)
        w_interpolated = self.node_start.coordinates.flatten() + f * (self.node_end.coordinates.flatten() -
                                                                      self.node_start.coordinates.flatten())
        r = util.getDistMat(w)

        distance_term = 0.0
        for d in range(1, len(self.node_start.atoms)):
            distance_term += np.array((r_interpolated.diagonal(d) - r.diagonal(d)) ** 2.0 /
                                      r_interpolated.diagonal(d) ** 4.0).sum()
        cartesian_term = 1e-6 * (w_interpolated - w).dot(w_interpolated - w)
        return distance_term + cartesian_term

    def getLSTnode(self, f):
        """
        Minimize LST objective function to find the coordinates on the LST path
        defined by the fractional distance along the path, `f`. A Node object
        containing the optimized Cartesian coordinates is returned.
        """
        # Start with Cartesian interpolation guess
        w_guess = self.node_start.coordinates.flatten() + f * (self.node_end.coordinates.flatten() -
                                                               self.node_start.coordinates.flatten())
        # Compute LST node by minimizing objective function
        result = optimize.minimize(self.LSTobjective, w_guess, args=(f,), method='BFGS', options={'gtol': 1e-3})
        if not result.success:
            msg = 'LST minimization terminated with status ' + str(result.status) + ':\n' + result.message + '\n'
            warnings.warn(msg)
        return Node(result.x, self.node_start.atoms, self.node_start.multiplicity)

    def getLSTpath(self, nnodes=100):
        """
        Generates an LST path between `node_start` and `node_end` represented
        as a list of Node objects containing `nnodes` nodes along the path.
        `node_start` and `node_end` are included as the path endpoints.
        Additionally, the integrated arc length along the path is returned as a
        list with each element being the arc length from `node_start` to the
        node corresponding to the element. A large enough number of nodes
        should be used so that the arc length can be computed to sufficient
        accuracy.
        """
        inc = 1.0 / (nnodes - 1)

        # Compute LST path and add start and end node to path
        if self.nproc > 1:
            pool = multiprocessing.Pool(self.nproc)
            path = list(pool.map(self.getLSTnode, np.linspace(inc, 1.0 - inc, nnodes - 2)))
        else:
            path = [self.getLSTnode(f) for f in np.linspace(inc, 1.0 - inc, nnodes - 2)]
        path.insert(0, self.node_start)
        path.append(self.node_end)

        # Compute arc length along path
        arclength = [0]
        for n in range(1, len(path)):
            s = path[n].getDistance(path[n - 1])
            arclength.append(arclength[n - 1] + s)

        return path, arclength

    def getTotalArclength(self, nnodes=100):
        """
        Returns the total arc length between the two end nodes of the LST path.
        """
        path, arclength = self.getLSTpath(nnodes)
        return arclength[-1]
