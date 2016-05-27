#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`FSM` for executing a transition state search using the
freezing string method. The resulting transition state should be further
optimized using another method in order to find the true transition state.
"""

import numpy as np
from scipy import optimize
import bisect

import os
import logging
import time

import gaussian
from node import Node
from interpolation import CartesianInterp, LST

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
    idx = bisect.bisect_left(a, x)
    if idx == 0:
        return a[0]
    if idx == len(a):
        return a[-1]
    if a[idx] - x < x - a[idx - 1]:
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
    `node_spacing`    ``float``              The interpolation distance between nodes
    ================= ====================== ==================================

    """

    def __init__(self, reactant, product, nsteps=4, nnode=15, nLSTnodes=100, interpolation='LST',
                 gaussian_ver='g09', level_of_theory='um062x/cc-pvtz', nproc=32, mem='2000mb'):
        if reactant.number != product.number:
            raise Exception('Atom labels of reactant and product do not match')
        if nLSTnodes < 3 * nnode:
            raise ValueError('Increase the number of LST nodes')
        self.reactant = reactant
        self.product = product
        self.nsteps = nsteps
        self.nnode = nnode
        self.nLSTnodes = nLSTnodes
        self.interpolation = interpolation.lower()
        if (self.interpolation != 'cartesian') and (self.interpolation != 'lst'):
            raise Exception('Invalid interpolation method')
        self.gaussian_ver = gaussian_ver
        self.level_of_theory = level_of_theory
        self.nproc = nproc
        self.mem = mem

        self.node_spacing = None  # Set in initialize method

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

    def align(self):
        """
        Align the reactant and product structures to maximum coincidence in
        non-mass weighted Cartesian coordinates. This is done by shifting the
        centroids of both structures to the origin and rotating the molecules
        in order to minimize the distance between them.

        Returns the final linear distance between the structures.
        """
        # Translate reactant and product so that centroids coincide at the origin
        reac_centroid = self.reactant.getCentroid()
        self.reactant.translate(-reac_centroid)
        self.product.translate(-self.product.getCentroid())

        # Find optimal rotation matrix iteratively
        angles_guess = np.array([0.0, 0.0, 0.0])
        result = optimize.minimize(self.coincidenceObjective, angles_guess, method='BFGS')
        if not result.success:
            message = ('Maximum coincidence alignment terminated with status ' +
                       str(result.status) + ':\n' + result.message + '\n')
            logging.warning(message)

        # Check for positive eigenvalues to ensure that aligned structure is a minimum
        eig_val = np.linalg.eig(result.hess_inv)[0]
        if not all(eig_val > 0.0):
            logging.warning('Not all Hessian eigenvalues were positive for the alignment process.\n' +
                            'The aligned structure may not be optimal.\n')

        # Rotate product to maximum coincidence and return distance
        self.product.rotate(rotationMatrix(result.x))
        if self.interpolation == 'cartesian':
            self.reactant.translate(reac_centroid)
            self.product.translate(reac_centroid)
        return result.fun ** 0.5

    def getNodes(self, node1, node2):
        """
        Generates new FSM nodes based on Cartesian interpolation or an LST
        interpolation path between the two nodes. If the distance between the
        nodes is less than the desired node spacing, then nothing is returned.
        Only one node is generated halfway between the two previous nodes if
        the distance between them is between one and two times the desired node
        spacing. Otherwise, two nodes are generated at the desired node spacing
        from the end nodes.

        The tangent vectors at each node are also returned. For the LST path,
        this vector is determined from the two LST nodes that are directly
        adjacent to the interpolated node.
        """
        if self.node_spacing is None:
            raise Exception('Interpolation distance has to be set first')

        # Find new nodes based on simple linear interpolation
        if self.interpolation == 'cartesian':
            total_distance = node1.getDistance(node2)
            if total_distance < self.node_spacing:
                return None, None
            elif self.node_spacing <= total_distance <= 2.0 * self.node_spacing:
                return CartesianInterp(node1, node2).getCartNode(0.5), node1.getTangent(node2)
            else:
                cart = CartesianInterp(node1, node2)
                new_node1 = cart.getCartNodeAtDistance(self.node_spacing)
                new_node2 = cart.getCartNodeAtDistance(total_distance - self.node_spacing)
                tangent = new_node1.getTangent(new_node2)
                return (new_node1, new_node2), (tangent, -tangent)

        # Create high density LST path between nodes
        path, arclength = LST(node1, node2, self.nproc).getLSTpath(self.nLSTnodes)

        # Find new nodes based on nodes that are closest to desired arc length spacing
        if arclength[-1] < self.node_spacing:
            return None, None
        elif self.node_spacing <= arclength[-1] <= 2.0 * self.node_spacing:
            new_node_idx = findClosest(arclength, arclength[-1] / 2.0)
            return path[new_node_idx], path[new_node_idx - 1].getTangent(path[new_node_idx + 1])
        new_node1_idx = findClosest(arclength, self.node_spacing)
        new_node2_idx = findClosest(arclength, arclength[-1] - self.node_spacing)
        tangent1 = path[new_node1_idx - 1].getTangent(path[new_node1_idx + 1])
        tangent2 = path[new_node2_idx + 1].getTangent(path[new_node2_idx - 1])
        return (path[new_node1_idx], path[new_node2_idx]), (tangent1, tangent2)

        # Compute distances from node1 and node2
        # distance_from_node1 = [node1.getDistance(node) for node in path]
        # distance_from_node2 = [node2.getDistance(node) for node in path]
        #
        # # Create new nodes depending on spacing between `node1` and `node2`
        # if distance_from_node1[-1] < self.node_spacing:
        #     return None
        # elif self.node_spacing <= distance_from_node1[-1] <= 2.0 * self.node_spacing:
        #     closest1 = findClosest(distance_from_node1, distance_from_node1[-1] / 2.0)
        #     closest2 = (len(distance_from_node2) - 1 -
        #                 findClosest(distance_from_node2[::-1], distance_from_node2[0] / 2.0))
        #     if distance_from_node1[closest1] < distance_from_node2[closest2]:
        #         return path[closest1]
        #     return path[closest2]
        # new_node1_idx = findClosest(distance_from_node1, self.node_spacing)
        # new_node2_idx = len(distance_from_node2) - 1 - findClosest(distance_from_node2[::-1], self.node_spacing)
        # return path[new_node1_idx], path[new_node2_idx]

    def initialize(self):
        """
        Initialize the FSM job.
        """
        # Log start timestamp
        logging.info('\nFSM job execution initiated on ' + time.asctime() + '\n')

        # Print FSM header
        self.logHeader()

        # Find distance between product and reactant nodes and calculate
        # interpolation distance after aligning product and reactant to maximum
        # coincidence
        logging.info('Aligning product and reactant structure to maximum coincidence')
        distance = self.align()
        if self.interpolation == 'lst':
            distance = LST(self.reactant, self.product, self.nproc).getDistance(self.nLSTnodes)
        self.node_spacing = distance / float(self.nnode)
        logging.info('Aligned reactant structure:\n' + str(self.reactant))
        logging.info('Aligned product structure:\n' + str(self.product))
        if self.interpolation == 'cartesian':
            logging.info('Total reactant to product distance:   {0:>8.4f} Angstrom'.format(distance))
            logging.info('Interpolation distance for new nodes: {0:>8.4f} Angstrom'.format(self.node_spacing))
        else:
            logging.info('Total reactant to product arc length:   {0:>8.4f} Angstrom'.format(distance))
            logging.info('Interpolation arc length for new nodes: {0:>8.4f} Angstrom'.format(self.node_spacing))

    def execute(self):
        """
        Run the freezing string method and return a tuple containing all
        optimized nodes along the FSM path and a tuple containing the energies
        of each node.
        """
        start_time = time.time()
        self.initialize()

        logging.info('Calculating reactant and product energies')
        # Initialize FSM path by adding reactant and product structures and computing their energies
        FSMpath = [self.reactant, self.product]
        energies = [self.reactant.getEnergy(self.gaussian_ver, self.level_of_theory, self.nproc, self.mem),
                    self.product.getEnergy(self.gaussian_ver, self.level_of_theory, self.nproc, self.mem)]
        logging.info('Reactant: {0[0]:.9f} Hartrees; Product: {0[1]:.9f} Hartrees'.format(energies))

        # The minimum desired energy change in an optimization step should be
        # at least the difference between reactant and product
        energy_diff = abs(energies[1] - energies[0]) * 627.5095
        if energy_diff < 2.5:
            min_en_change = energy_diff
        else:
            min_en_change = 2.5

        # Settings for perpendicular optimization
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
            logging.info('\nStarting iteration {0}\n'.format(i + 1))

            # Compute indices for inserting new nodes into FSM path
            innernode_p_idx = len(FSMpath) / 2
            innernode_r_idx = innernode_p_idx - 1

            # Compute distance between innermost nodes
            distance = FSMpath[innernode_r_idx].getDistance(FSMpath[innernode_p_idx])
            logging.info('Linear distance between innermost nodes: {0:.4f} Angstrom'.format(distance))

            # Obtain interpolated nodes
            start_time_interp = time.time()
            nodes, tangents = self.getNodes(FSMpath[innernode_r_idx], FSMpath[innernode_p_idx])
            if self.interpolation == 'lst':
                logging.info('LST interpolation completed in {0:.2f} s'.format(time.time() - start_time_interp))

            # Return if no new nodes were generated
            if nodes is None:
                logging.info('No new nodes were generated')
                logging.info('\nFSM job terminated successfully on ' + time.asctime())
                logging.info('Total run time: {0:.1f} s\n'.format(time.time() - start_time))
                return tuple(FSMpath), tuple(energies)

            # Optimize halfway node if only one was generated and return
            elif isinstance(nodes, Node):
                logging.info('Added one node:\n' + str(nodes))

                # Compute distance from reactant and product side innermost nodes
                logging.info('Linear distance from innermost reactant side node: {0:>8.4f} Angstrom'.
                             format(nodes.getDistance(FSMpath[innernode_r_idx])))
                logging.info('Linear distance from innermost product side node:  {0:>8.4f} Angstrom'.
                             format(nodes.getDistance(FSMpath[innernode_p_idx])))

                # Compute tangent based on previous two nodes
                # tangent = FSMpath[innernode_r_idx].getTangent(FSMpath[innernode_p_idx])

                # Perpendicular optimization
                logging.info('Optimizing final node')
                start_time_opt = time.time()
                energy = self.perpOpt(nodes, tangents, **settings)
                logging.info('Optimization completed in {0:.1f} s'.format(time.time() - start_time_opt))
                logging.info('Optimized node:\n' + str(nodes))
                logging.info('After opt distance from innermost reactant side node: {0:>8.4f} Angstrom'.
                             format(nodes.getDistance(FSMpath[innernode_r_idx])))
                logging.info('After opt distance from innermost product side node:  {0:>8.4f} Angstrom'.
                             format(nodes.getDistance(FSMpath[innernode_p_idx])))
                logging.info('')

                # Insert optimized node and corresponding energy into path
                FSMpath.insert(innernode_p_idx, nodes)
                energies.insert(innernode_p_idx, energy)

                logging.info('\nFSM job terminated successfully on ' + time.asctime())
                logging.info('Total run time: {0:.1f} s\n'.format(time.time() - start_time))
                return tuple(FSMpath), tuple(energies)

            # Optimize new nodes
            else:
                logging.info('Added two nodes:\n' + str(nodes[0]) + '\n****\n' + str(nodes[1]))

                # Compute distance from reactant and product side innermost nodes
                logging.info('Linear distance between previous and current reactant side nodes: {0:>8.4f} Angstrom'.
                             format(nodes[0].getDistance(FSMpath[innernode_r_idx])))
                logging.info('Linear distance between previous and current product side nodes:  {0:>8.4f} Angstrom'.
                             format(nodes[1].getDistance(FSMpath[innernode_p_idx])))

                # Compute tangent based on new nodes
                # tangent = nodes[0].getTangent(nodes[1])

                # Perpendicular optimization
                logging.info('Optimizing new reactant side node')
                start_time_opt = time.time()
                energy0 = self.perpOpt(nodes[0], tangents[0], nodes[1], **settings)
                logging.info('Optimization completed in {0:.1f} s'.format(time.time() - start_time_opt))
                logging.info('Optimizing new product side node')
                start_time_opt = time.time()
                energy1 = self.perpOpt(nodes[1], tangents[1], nodes[0], **settings)
                logging.info('Optimization completed in {0:.1f} s'.format(time.time() - start_time_opt))
                logging.info('Optimized nodes:\n' + str(nodes[0]) + '\n****\n' + str(nodes[1]))
                logging.info('After opt distance between previous and current reactant side nodes: {0:>8.4f} Angstrom'.
                             format(nodes[0].getDistance(FSMpath[innernode_r_idx])))
                logging.info('After opt distance between previous and current product side nodes:  {0:>8.4f} Angstrom'.
                             format(nodes[1].getDistance(FSMpath[innernode_p_idx])))
                logging.info('')

                # Insert optimized nodes and corresponding energies into path
                FSMpath.insert(innernode_p_idx, nodes[1])
                FSMpath.insert(innernode_p_idx, nodes[0])
                energies.insert(innernode_p_idx, energy1)
                energies.insert(innernode_p_idx, energy0)

        logging.error('FSM job terminated abnormally on ' + time.asctime())
        logging.info('Total run time: {0:.1f} s\n'.format(time.time() - start_time))
        raise Exception('FSM did not converge')

    def perpOpt(self, node, tangent, other_node=None, nsteps=4, min_desired_energy_change=2.5, line_search_factor=0.7,
                gaussian_ver='g09', level_of_theory='um062x/cc-pvtz', nproc=32, mem='1500mb'):

        """
        Optimize node in direction of negative perpendicular gradient using the
        Newton-Raphson method with a BFGS Hessian update scheme. Requires input
        of tangent vector between closest two nodes on the string so that the
        appropriate perpendicular gradient can be calculated. Also requires
        that the innermost node on the other side of the string is input so
        that the forward progress towards joining string ends can be assessed.
        If no other node is specified, then the node is optimized without such
        a constraint.

        Returns the energy of the optimized node in Hartrees.

        Set `min_desired_energy_change` to the energy difference between
        reactant and product if the difference is less than 2.5 kcal/mol.
        """
        if not isinstance(nsteps, int):
            raise TypeError('nsteps has to be an integer value')

        # Gaussian settings
        g_settings = {
            'ver': gaussian_ver,
            'level_of_theory': level_of_theory,
            'nproc': nproc,
            'mem': mem
        }

        # Compute maximum allowed distance between nodes based on initial distance
        if other_node is not None:
            max_distance = node.getDistance(other_node) + 0.5 * self.node_spacing
        else:
            max_distance = 0

        # Initialize Hessian inverse to identity matrix
        identity_mat = np.eye(3 * len(node.number))
        hessian_inv = identity_mat

        # Convert units to Hartrees
        min_desired_energy_change /= 627.5095

        # Calculate gradient and energy
        start_time = time.time()
        logfile = gaussian.executeGaussianJob(node, name='step_1', jobtype='force', **g_settings)
        logging.info('Gradient calculation (step_1) completed in {0:.3f} s'.format(time.time() - start_time))
        grad = gaussian.getGradient(logfile).flatten()
        energy = gaussian.getEnergy(logfile)
        logging.debug('Gradient:\n' + str(grad.reshape(len(node.number), 3)))
        logging.debug('Energy:\n' + str(energy))
        energy_old = energy
        os.remove(logfile)

        # Calculate perpendicular gradient
        perp_grad = (identity_mat - np.outer(tangent, tangent)).dot(grad)
        logging.debug('Perpendicular gradient:\n' + str(perp_grad.reshape(len(node.number), 3)))

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
            search_dir = direction / direction_norm  # Gaussian outputs the negative gradient

            # Calculate maximum and minimum scaling factors
            scale_factor_max = 0.05 / np.absolute(search_dir).max()
            scale_factor_min = (1.0 - line_search_factor) * direction_norm

            # Calculate desired change in energy
            desired_energy_change = max(abs(energy_old - energy), min_desired_energy_change)

            # Calculate scaling factor
            line_search_term = perp_grad.dot(search_dir)
            scale_factor = - 2.0 * desired_energy_change / line_search_term

            # Refine scaling factor based on limits
            if scale_factor < scale_factor_min:
                scale_factor = scale_factor_min
            if scale_factor > scale_factor_max:
                scale_factor = scale_factor_max

            # Handle unstable searches by reinitializing the Hessian
            if scale_factor < 0.0:
                # Terminate if resetting Hessian did not resolve instability
                if unstable:
                    logging.warning('Optimization terminated prematurely due to unstable scaling factor')
                    break
                unstable = True
                hessian_inv = identity_mat
                continue
            unstable = False

            # Save old values
            np.copyto(coord_vec_old, node.coordinates.flatten())
            np.copyto(perp_grad_old, perp_grad)
            np.copyto(hessian_inv_old, hessian_inv)
            energy_old = energy

            # Take minimization step
            step = scale_factor * search_dir
            node.displaceCoordinates(step.reshape(len(node.number), 3))
            logging.debug('Updated coordinates:\n' + str(node))

            # Terminate after maximum number of gradient calls or if maximum distance between nodes is exceeded
            if k == nsteps or node.getDistance(other_node) > max_distance:
                start_time = time.time()
                logfile = gaussian.executeGaussianJob(node, name='final_energy', jobtype='sp', **g_settings)
                logging.info('Final energy calculation completed in {0:.3f} s'.format(time.time() - start_time))
                logging.info('Energy = {0:.9f}'.format(energy))
                energy = gaussian.getEnergy(logfile)
                os.remove(logfile)
                break

            # Calculate new gradient and energy
            name = 'step_' + str(k + 1)
            start_time = time.time()
            logfile = gaussian.executeGaussianJob(node, name=name, jobtype='force', **g_settings)
            logging.info('Gradient calculation ({0}) completed in {1:.3f} s'.format(name, time.time() - start_time))
            grad = gaussian.getGradient(logfile).flatten()
            energy = gaussian.getEnergy(logfile)
            logging.debug('Gradient:\n' + str(grad.reshape(len(node.number), 3)))
            logging.debug('Energy:\n' + str(energy))
            os.remove(logfile)

            # Calculate perpendicular gradient
            perp_grad = (identity_mat - np.outer(tangent, tangent)).dot(grad)
            logging.debug('Perpendicular gradient:\n' + str(perp_grad.reshape(len(node.number), 3)))

            # Check remaining termination conditions
            energy_change = abs(energy - energy_old)
            if energy_change < 0.5 / 627.5095:
                logging.info('Optimization terminated due to small energy change')
                logging.info('Energy = {0:.9f}'.format(energy))
                break
            if abs(perp_grad.dot(search_dir)) <= - line_search_factor * line_search_term:
                logging.info('Optimization terminated due to stable line search condition')
                logging.info('Energy = {0:.9f}'.format(energy))
                break

            # Update inverse Hessian
            perp_grad_diff = perp_grad - perp_grad_old
            denom = step.dot(perp_grad_diff)
            hessian_inv += (1.0 + perp_grad_diff.dot((hessian_inv_old.dot(perp_grad_diff))) / denom) * \
                np.outer(step, step) / denom - (np.outer(step, perp_grad_diff.dot(hessian_inv_old)) +
                                                np.outer(hessian_inv_old.dot(perp_grad_diff), step)) / denom

            # Update counter
            k += 1

        # Return optimized node energy
        return energy

    def logHeader(self, level=logging.INFO):
        """
        Output a log file header containing identifying information about the
        FSM job.
        """
        logging.log(level, '############################################################################')
        logging.log(level, '########################## FREEZING STRING METHOD ##########################')
        logging.log(level, '############################################################################')
        logging.log(level, '# Number of gradient calculations per optimization step:     {0:>5}         #'.
                    format(self.nsteps))
        logging.log(level, '# Number of nodes for calculation of interpolation distance: {0:>5}         #'.
                    format(self.nnode))
        if self.interpolation == 'lst':
            logging.log(level, '# Number of high density LST nodes:                          {0:>5}         #'.
                        format(self.nLSTnodes))
        logging.log(level, '# Level of theory for quantum calculations: {0:<31}#'.format(self.level_of_theory))
        logging.log(level, '############################################################################')
        logging.log(level, 'Reactant structure:\n' + str(self.reactant))
        logging.log(level, 'Product structure:\n' + str(self.product))
        logging.log(level, '############################################################################\n')
