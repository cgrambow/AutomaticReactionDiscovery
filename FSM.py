#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`FSM` for executing a transition state search using the
freezing string method. The resulting transition state should be further
optimized using another method in order to find the true transition state.
"""

import logging
import time

from bisect import bisect_left
import numpy as np
from scipy.optimize import minimize

from interpolation import LST
from node import Node

###############################################################################

def rotationMatrix(angles):
    """
    Calculates and returns the rotation matrix defined by three angles of
    rotation about the x, y, and z axes.
    """
    Rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(angles[0]), -np.sin(angles[0])],
         [0.0, np.sin(angles[0]), np.cos(angles[0])]]
    )
    Ry = np.array(
        [[np.cos(angles[1]), 0.0, np.sin(angles[1])],
         [0.0, 1.0, 0.0],
         [-np.sin(angles[1]), 0.0, np.cos(angles[1])]]
    )
    Rz = np.array(
        [[np.cos(angles[2]), -np.sin(angles[2]), 0.0],
         [np.sin(angles[2]), np.cos(angles[2]), 0.0],
         [0.0, 0.0, 1.0]]
    )
    return Rx.dot(Ry).dot(Rz)

def findClosest(a, x):
    """
    Returns index of value closest to `x` in sorted sequence `a`.
    """
    idx = bisect_left(a, x)
    if idx == 0:
        return a[0]
    if idx == len(a):
        return a[-1]
    if a[idx] - x < x - a[idx-1]:
        return idx
    else:
        return idx - 1

###############################################################################

class FSM(object):
    """
    Freezing string method.
    The attributes are:

    ================= ====================== ==================================
    Attribute         Type                   Description
    ================= ====================== ==================================
    `reactant`        :class:`node.Node`     A node object containing the coordinates and atoms of the reactant molecule
    `product`         :class:`node.Node`     A node object containing the coordinates and atoms of the product molecule
    `nsteps`          ``int``                The number of gradient evaluations per node optimization
    `nnode`           ``int``                The desired number of nodes, which determines the spacing between them
    `nLSTnodes`       ``int``                The number of nodes on a high-density LST interpolation path
    `gaussian_ver`    ``str``                The version of Gaussian available to the user
    `level_of_theory` ``str``                The level of theory (method/basis) for the quantum calculations
    `nproc`           ``int``                The number of processors available for the FSM calculation
    `mem`             ``str``                The memory requirements
    `node_spacing`    ``float``              The arc length between nodes calculated from `nnode`
    ================= ====================== ==================================

    """

    def __init__(self, reactant, product, nsteps=4, nnode=15, nLSTnodes=100,
                 gaussian_ver='g09', level_of_theory='um062x/cc-pvtz', nproc=32, mem='2000mb'):
        self.reactant = reactant
        self.product = product
        self.nsteps = nsteps
        self.nnode = nnode
        self.nLSTnodes = nLSTnodes
        self.gaussian_ver = gaussian_ver
        self.level_of_theory = level_of_theory
        self.nproc = nproc
        self.mem = mem

        distance = self.align()
        self.node_spacing = distance / float(nnode)
        # self.node_spacing = LST(reactant, product).getDistance(nLSTnodes) / float(nnode)

    def coincidenceObjective(self, angles):
        """
        Defines the objective function for rotating the product structure to
        obtain maximum coincidence in non-mass weighted Cartesian coordinates.
        The rotation matrix is defined by the product of three separate
        rotation matrices which describe rotations about the three principal
        axes and are each defined by an angle in `angles` (i.e., angles[0]
        corresponds to the angle of rotation about the x-axis, etc.). The objective
        function is a measure of the "distance" between reactant and product.
        """
        rotated_product = (rotationMatrix(angles).dot(self.product.coordinates.T)).T.flatten()
        diff = self.reactant.coordinates.flatten() - rotated_product
        return diff.dot(diff)

    # def getDistance(self):
    #     distance = 0.0
    #     for reac_atom, prod_atom in zip(self.reactant.coordinates, self.product.coordinates):
    #         distance += np.linalg.norm(reac_atom - prod_atom)
    #     return distance

    def align(self):
        """
        Align the reactant and product structures to maximum coincidence in
        non-mass weighted Cartesian coordinates. This is done by shifting the
        centroids of both structures to the origin and rotating the molecules
        in order to minimize the distance between them.

        Returns the final linear distance between the structures.
        """
        # Translate reactant and product so that centroids coincide at the origin
        self.reactant.translate(-self.reactant.getCentroid())
        self.product.translate(-self.product.getCentroid())

        # Find optimal rotation matrix iteratively
        angles_guess = np.array([0.0, 0.0, 0.0])
        result = minimize(self.coincidenceObjective, angles_guess, method='BFGS')
        if not result.success:
            message = ('Maximum coincidence alignment terminated with status ' +
                       str(result.status) + ':\n' + result.message + '\n')
            logging.warning(message)

        # Check for positive eigenvalues to ensure that aligned structure is a minimum
        eig_val = np.linalg.eig(result.hess_inv)[0]
        if not all(eig_val > 0.0):
            logging.warning('Not all Hessian eigenvalues were positive for the alignment process.\n' +
                            'The aligned structure may not be optimal.\n')

        # Rotate product to maximum coincidence
        self.product.rotate(rotationMatrix(result.x))
        return result.fun ** 0.5

    def getNodes(self, node1, node2):
        """
        Generates new FSM nodes based on an LST interpolation path between the
        two nodes. If the distance between the nodes is less than the desired
        node spacing, then nothing is returned. Only one node is generated
        halfway between the two previous nodes if the distance between them is
        between one and two times the desired node spacing. Otherwise, two
        nodes are generated at the desired node spacing from the end nodes.
        """
        # Create high density LST path between nodes
        path, arclength = LST(node1, node2).getLSTpath(self.nLSTnodes)

        # Create new nodes depending on spacing between `node1` and `node2`
        if arclength[-1] < self.node_spacing:
            return None
        elif self.node_spacing <= arclength[-1] <= 2.0 * self.node_spacing:
            return path[findClosest(arclength, arclength[-1] / 2.0)]
        new_node1_idx = findClosest(arclength, self.node_spacing)
        new_node2_idx = findClosest(arclength, arclength[-1] - self.node_spacing)
        return path[new_node1_idx], path[new_node2_idx]

    def initialize(self):
        """
        Initialize the FSM job.
        """
        # Log start timestamp
        logging.info('FSM job execution initiated at ' + time.asctime() + '\n')

        # Print FSM header
        self.logHeader()

    def execute(self):
        """
        Run the freezing string method and return a tuple containing all
        optimized nodes along the FSM path and a tuple containing the energies
        of each node.
        """
        self.initialize()

        # Initialize FSM path by adding reactant and product structures and computing their energies
        FSMpath = [self.reactant, self.product]
        energies = [self.reactant.getEnergy(self.gaussian_ver, self.level_of_theory, self.nproc, self.mem),
                    self.product.getEnergy(self.gaussian_ver, self.level_of_theory, self.nproc, self.mem)]

        # The minimum desired energy change in an optimization step should be
        # at least the difference between reactant and product
        energy_diff = abs(energies[1] - energies[0]) * 627.5095
        if energy_diff < 2.5:
            min_en_change = energy_diff
        else:
            min_en_change = 2.5

        # Setting for perpendicular optimization
        settings = {
            'nsteps': self.nsteps,
            'min_desired_energy_change': min_en_change,
            'gaussian_ver': self.gaussian_ver,
            'level_of_theory': self.level_of_theory,
            'nproc': self.nproc,
            'mem': self.mem
        }

        # Impose restriction on maximum number of nodes that can be created in
        # case FSM does not converge
        for i in range(10 * self.nnode):
            innernode_p_idx = len(FSMpath) / 2
            innernode_r_idx = innernode_p_idx - 1
            nodes = self.getNodes(FSMpath[innernode_r_idx], FSMpath[innernode_p_idx])
            if nodes is None:
                logging.info('')
                logging.info('FSM job terminated successfully at ' + time.asctime() + '\n')
                return tuple(FSMpath), tuple(energies)
            elif isinstance(nodes, Node):
                logging.info('Added one node:\n' + str(nodes))
                tangent = FSMpath[innernode_r_idx].getLSTtangent(FSMpath[innernode_p_idx])
                energy = nodes.perpOpt(tangent, **settings)
                logging.info('Optimized node:\n' + str(nodes))
                FSMpath.insert(innernode_p_idx, nodes)
                energies.insert(innernode_p_idx, energy)
            else:
                logging.info('Added two nodes:\n' + str(nodes[0]) + str(nodes[1]))
                tangent = nodes[0].getLSTtangent(nodes[1])
                energy0 = nodes[0].perpOpt(tangent, **settings)
                energy1 = nodes[1].perpOpt(tangent, **settings)
                logging.info('Optimized nodes:\n' + str(nodes[0]) + str(nodes[1]))
                FSMpath.insert(innernode_p_idx, nodes[1])
                FSMpath.insert(innernode_p_idx, nodes[0])
                energies.insert(innernode_p_idx, energy1)
                energies.insert(innernode_p_idx, energy0)
        logging.error('FSM job terminated abnormally at ' + time.asctime() + '\n')
        raise Exception('FSM did not converge')

    def logHeader(self, level=logging.INFO):
        """
        Output a log file header containing identifying information about the
        FSM job.
        """
        logging.log(level, '############################################################')
        logging.log(level, 'Freezing String Method')
        logging.log(level, 'Gradient calls per optimization step: {0:d}'.format(self.nsteps))
        logging.log(level, 'nnode parameter: {0:d}'.format(self.nnode))
        logging.log(level, 'High density LST nodes: {0:d}'.format(self.nLSTnodes))
        logging.log(level, 'Interpolation distance for new nodes: {0:.3f}'.format(self.node_spacing))
        logging.log(level, 'Level of theory for quantum calculations: {0}'.format(self.level_of_theory))
        logging.log(level, '############################################################')
        logging.log(level, 'Reactant structure:\n' + str(self.reactant))
        logging.log(level, 'Product structure:\n' + str(self.product))
