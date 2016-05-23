#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`Node` for working with three-dimensional representations
of molecules in Cartesian coordinates and evaluating energies and gradients
using quantum chemical calculations.
"""

import numpy as np

import os
import logging

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

    def getLSTtangent(self, other):
        """
        Calculate and return tangent direction between two nodes based on LST
        path between the nodes. The tangent vector points from `self` to
        `other`, which are both of type :class:`node.Node`.
        """
        self_coord_vec = self.coordinates.flatten()
        other_coord_vec = other.coordinates.flatten()
        assert len(self_coord_vec) == len(other_coord_vec)
        diff = other_coord_vec - self_coord_vec
        return diff / np.linalg.norm(diff)

    def perpOpt(self, tangent, nsteps=4, min_desired_energy_change=2.5, line_search_factor=0.7,
                gaussian_ver='g09', level_of_theory='um062x/cc-pvtz', nproc=32, mem='1500mb'):
        """
        Optimize node in direction of negative perpendicular gradient using the
        Newton-Raphson method with a BFGS Hessian update scheme. Requires input
        of tangent vector between closest two nodes on the string so that the
        appropriate perpendicular gradient can be calculated.

        Returns the energy of the optimized node in Hartrees.

        Set `min_desired_energy_change` to the energy difference between
        reactant and product if the difference is less than 2.5 kcal/mol.
        """
        if not isinstance(nsteps, int):
            raise TypeError('nsteps has to be an integer value')
        identity_mat = np.eye(3 * len(self.number))
        hessian_inv = identity_mat
        min_desired_energy_change /= 627.5095

        # Calculate gradient and energy
        logfile = gaussian.executeGaussianJob(self, 'step_1', 'force', gaussian_ver, level_of_theory, nproc, mem)
        grad = gaussian.getGradient(logfile).flatten()
        print 'Gradient:'
        print grad.reshape(len(self.number), 3)
        print '\n'
        energy = gaussian.getEnergy(logfile)
        print 'Energy:'
        print energy
        print '\n'
        energy_old = energy
        os.remove(logfile)

        # Calculate perpendicular gradient
        perp_grad = (identity_mat - np.outer(tangent, tangent)).dot(grad)
        print 'Perpendicular gradient:'
        print perp_grad.reshape(len(self.number), 3)
        print '\n'

        # Create empty arrays for storing old values
        coord_vec_old = np.empty_like(perp_grad)
        perp_grad_old = np.empty_like(perp_grad)
        hessian_inv_old = np.empty_like(hessian_inv)

        k = 1
        unstable = False
        while k <= nsteps:
            # Calculate search direction
            direction = hessian_inv.dot(perp_grad)
            direction_norm = np.linalg.norm(direction)
            search_dir = - direction / direction_norm
            print 'Search direction:'
            print search_dir.reshape(len(self.number), 3)
            print '\n'

            # Calculate scaling factor
            scale_factor_max = 0.05 / np.absolute(search_dir).max()
            scale_factor_min = 1.0 - line_search_factor * direction_norm
            desired_energy_change = max(abs(energy - energy_old), min_desired_energy_change)
            line_search_term = perp_grad.dot(search_dir)
            scale_factor = - 2.0 * desired_energy_change / line_search_term
            if scale_factor < scale_factor_min < scale_factor_max:
                scale_factor = scale_factor_min
            elif scale_factor > scale_factor_max:
                scale_factor = scale_factor_max
            if scale_factor < 0.0:
                # Terminate if resetting Hessian did not resolve instability
                if unstable:
                    logging.warning('Optimization terminated due to unstable scaling factor')
                    break
                unstable = True
                hessian_inv = identity_mat
                print 'Skipped remainder of loop.\n'
                continue
            unstable = False
            print 'Scaling factor:'
            print scale_factor
            print '\n'

            # Update
            np.copyto(coord_vec_old, self.coordinates.flatten())
            print 'Old coordinates:'
            print coord_vec_old.reshape(len(self.number), 3)
            print '\n'
            np.copyto(perp_grad_old, perp_grad)
            np.copyto(hessian_inv_old, hessian_inv)
            energy_old = energy

            # Take minimization step
            self.coordinates += (scale_factor * search_dir).reshape(len(self.number), 3)
            print 'Coordinates:'
            print self
            print '\n'

            # Terminate after maximum number of gradient calls
            if k == nsteps:
                logfile = gaussian.executeGaussianJob(self, 'final_energy', 'sp',
                                                      gaussian_ver, level_of_theory, nproc, mem)
                energy = gaussian.getEnergy(logfile)
                break

            # Calculate new gradient and energy
            name = 'step_' + str(k+1)
            logfile = gaussian.executeGaussianJob(self, name, 'force', gaussian_ver, level_of_theory, nproc, mem)
            grad = gaussian.getGradient(logfile).flatten()
            print 'Gradient:'
            print grad.reshape(len(self.number), 3)
            print '\n'
            energy = gaussian.getEnergy(logfile)
            print 'Energy:'
            print energy
            print '\n'
            os.remove(logfile)

            # Calculate perpendicular gradient
            perp_grad = (identity_mat - np.outer(tangent, tangent)).dot(grad)
            print 'Perpendicular gradient:'
            print perp_grad.reshape(len(self.number), 3)
            print '\n'

            # Check remaining termination conditions
            energy_change = abs(energy - energy_old)
            if (energy_change < 0.5 / 627.5095 or
                    abs(perp_grad.dot(search_dir)) <= - line_search_factor * line_search_term):
                break

            # Update inverse Hessian
            perp_grad_change = perp_grad - perp_grad_old
            print 'Coordinates:'
            print self
            print 'Old coordinates:'
            print coord_vec_old.reshape(len(self.number), 3)
            print '\n'
            step = self.coordinates.flatten() - coord_vec_old
            denom = step.dot(perp_grad_change)
            print 'Hessian inverse:'
            print hessian_inv_old
            print 'Perpendicular gradient change:'
            print perp_grad_change.reshape(len(self.number), 3)
            print 'Step'
            print step.reshape(len(self.number), 3)
            print 'Denominator in gradient expression:'
            print denom
            hessian_inv += (1.0 + perp_grad_change.dot((hessian_inv_old.dot(perp_grad_change))) / denom) * \
                np.outer(step, step) / denom - (np.outer(step, perp_grad_change.dot(hessian_inv_old)) +
                                                np.outer(hessian_inv_old.dot(perp_grad_change), step)) / denom

            # Update counter
            k += 1

        return energy
