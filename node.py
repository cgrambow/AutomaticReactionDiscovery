#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`Node` for working with three-dimensional representations
of molecules in Cartesian coordinates and evaluating energies and gradients
using quantum chemical calculations.
"""

import numpy as np

import os

import gaussian

###############################################################################

class Node(object):
    """
    Three-dimensional representation of a molecular configuration.
    The attributes are:

    =============== ======================= ===================================
    Attribute       Type                    Description
    =============== ======================= ===================================
    `coordinates`   :class:`numpy.ndarray`  A 3N x 3 array containing the 3D coordinates of each atom (in Angstrom)
    `number`        :class:`tuple`          A tuple of length N containing the integer atomic number of each atom
    `multiplicity`  ``int``                 The multiplicity of this species, multiplicity = 2*total_spin+1
    =============== ======================= ===================================

    N is the total number of atoms in the molecule. Each row in the coordinate
    array represents one atom.
    """

    # Dictionary of elements corresponding to atomic numbers
    elements = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

    def __init__(self, coordinates, number, multiplicity=1):
        try:
            self.coordinates = np.array(coordinates).reshape(len(number), 3)
        except ValueError:
            print 'One or more atoms are missing a coordinate'
            raise
        self.number = tuple([int(round(num, 0)) for num in number])
        self.multiplicity = multiplicity

    def __str__(self):
        """
        Return a human readable string representation of the object
        """
        return_string = ''
        for atom_num, atom in enumerate(self.coordinates):
            return_string += '{0}  {1[0]: 14.8f}{1[1]: 14.8f}{1[2]: 14.8f}\n'.format(
                self.elements[self.number[atom_num]], atom)
        return return_string[:-1]

    def __repr__(self):
        """
        Return a representation of the object.
        """
        return 'Node({coords}, {self.number}, {self.multiplicity})'.format(coords=self.coordinates.tolist(), self=self)

    def getCentroid(self):
        """
        Compute and return non-mass weighted centroid of molecular
        configuration.
        """
        return self.coordinates.sum(axis=0) / float(len(self.number))

    def translate(self, trans_vec):
        """
        Translate all atoms in the molecular configuration by `trans_vec`,
        which is of type :class:`numpy.ndarray` and of size 3 x 1.
        """
        self.coordinates += trans_vec

    def displaceCoordinates(self, mod_array):
        """
        Displaces the coordinates by adding the 3N x 3 :class:`numpy.ndarray`
        to the current coordinates.
        """
        self.coordinates += mod_array

    def rotate(self, rot_mat):
        """
        Rotate molecular configuration using orthogonal rotation matrix
        `rot_mat` which is of type :class:`numpy.ndarray` and of size 3 x 3.

        The node should first be translated to the origin, since rotation
        matrices can only describe rotations about the origin.
        """
        self.coordinates = (rot_mat.dot(self.coordinates.T)).T

    def getEnergy(self, gaussian_ver='g09', level_of_theory='um062x/cc-pvtz', nproc=32, mem='1500mb'):
        """
        Compute and return energy of node.
        """
        logfile = gaussian.executeGaussianJob(self, 'energy', 'sp', gaussian_ver, level_of_theory, nproc, mem)
        energy = gaussian.getEnergy(logfile)
        os.remove(logfile)
        return energy

    def getDistance(self, other=None):
        """
        Compute and return linear distance between `self` and `other`. Returns
        -1 if other is not specified.
        """
        if other is None:
            return -1
        assert np.size(self.coordinates) == np.size(other.coordinates)
        diff = self.coordinates.flatten() - other.coordinates.flatten()
        return diff.dot(diff) ** 0.5

    def getTangent(self, other):
        """
        Calculate and return tangent direction between two nodes based on
        straight line path between the nodes. The tangent vector points from
        `self` to `other`, which are both of type :class:`node.Node`.
        """
        self_coord_vec = self.coordinates.flatten()
        other_coord_vec = other.coordinates.flatten()
        assert len(self_coord_vec) == len(other_coord_vec)
        diff = other_coord_vec - self_coord_vec
        return diff / np.linalg.norm(diff)
