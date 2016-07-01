#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`Node` for working with three-dimensional representations
of molecules in Cartesian coordinates and evaluating energies and gradients
using quantum chemical calculations.
"""

from __future__ import print_function, division
import sys

import numpy as np

import props

###############################################################################

class Node(object):
    """
    Three-dimensional representation of a molecular configuration.
    The attributes are:

    =============== ======================= ===================================
    Attribute       Type                    Description
    =============== ======================= ===================================
    `coordinates`   :class:`numpy.ndarray`  A 3N x 3 array containing the 3D coordinates of each atom (in Angstrom)
    `atoms`         :class:`list`           A list of length N containing the integer atomic number of each atom
    `multiplicity`  ``int``                 The multiplicity of this species, multiplicity = 2*total_spin+1
    `mass`          :class:`list`           A list of length N containing the masses of each atom
    =============== ======================= ===================================

    N is the total number of atoms in the molecule. Each row in the coordinate
    array represents one atom.
    """

    def __init__(self, coordinates, atoms, multiplicity=1):
        try:
            self.coordinates = np.array(coordinates).reshape(len(atoms), 3)
        except ValueError:
            print('One or more atoms are missing a coordinate', file=sys.stderr)
            raise

        # self.atoms can be generated from atomic numbers or from atom labels
        self.atoms = []
        for num in atoms:
            if num in props.atomnum.values():
                self.atoms.append(props.atomnum_inv[num])
                continue
            if int(num) not in props.atomnum.keys():
                raise ValueError('Invalid atomic number or symbol: {0}'.format(num))
            self.atoms.append(int(num))

        self.multiplicity = int(multiplicity)

        self.mass = [props.atomweights[atom] for atom in self.atoms]

    def __str__(self):
        """
        Return a human readable string representation of the object
        """
        return_string = ''
        for atom_num, atom in enumerate(self.coordinates):
            return_string += '{0}  {1[0]: 14.8f}{1[1]: 14.8f}{1[2]: 14.8f}\n'.format(
                props.atomnum[self.atoms[atom_num]], atom)
        return return_string[:-1]

    def __repr__(self):
        """
        Return a representation of the object.
        """
        return 'Node({coords}, {self.atoms}, {self.multiplicity})'.format(coords=self.coordinates.tolist(), self=self)

    def getTotalMass(self, atoms=None):
        """
        Compute and return total mass in g/mol. If a list of atoms is specified
        in `atoms`, only the corresponding atoms will be used to calculate the
        total mass.
        """
        if atoms is None:
            atoms = range(len(self.mass))
        return sum([self.mass[atom] for atom in atoms])

    def getCentroid(self):
        """
        Compute and return non-mass weighted centroid of molecular
        configuration.
        """
        return self.coordinates.sum(axis=0) / len(self.atoms)

    def getCenterOfMass(self, atoms=None):
        """
        Compute and return the position of the center of mass of the molecular
        configuration. If a list of atoms is specified in `atoms`, only the
        corresponding atoms will be used to calculate the center of mass.
        """
        if atoms is None:
            atoms = range(len(self.mass))
        mass = self.getTotalMass(atoms=atoms)
        center = sum([self.mass[atom] * self.coordinates[atom] for atom in atoms])
        return center / mass

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

    def getEnergy(self, qclass, **kwargs):
        """
        Compute and return energy of node.
        """
        # Create instance of quantum class
        q = qclass()

        # Execute job and retrieve energy
        q.executeJob(self, jobtype='energy', **kwargs)
        energy = q.getEnergy()
        q.clear()
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
