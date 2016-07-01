#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains relevant classes for executing transition state searches using the
freezing string or growing string methods. The resulting transition state
should be further optimized using another method in order to find the true
transition state.
"""

from __future__ import division

import numpy as np
from scipy import optimize
import bisect

import os
import logging
import time

from quantum import Gaussian, NWChem, QChem
from node import Node
from interpolation import LST

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

class String(object):
    """
    Base class from which freezing and growing string methods can inherit.
    The attributes are:

    ================== ===================== ==================================
    Attribute          Type                  Description
    ================== ===================== ==================================
    `reactant`         :class:`node.Node`    A node object containing the coordinates and atoms of the reactant molecule
    `product`          :class:`node.Node`    A node object containing the coordinates and atoms of the product molecule
    `nsteps`           ``int``               The number of gradient evaluations per node optimization
    `nnode`            ``int``               The desired number of nodes, which determines the spacing between them
    `tol`              ``float``             The gradient convergence tolerance (Hartree/Angstrom)
    `nLSTnodes`        ``int``               The number of nodes on a high-density LST interpolation path
    `qprog`            ``str``               A software for quantum calculations
    `level_of_theory`  ``str``               The level of theory (method/basis) for the quantum calculations
    `nproc`            ``int``               The number of processors available for the FSM calculation
    `mem`              ``str``               The memory requirements
    `qclass`           ``class``             A class representing the quantum software
    `qsettings`        :class:`dict`         A dictionary containing settings for the quantum calculations
    `output_file`      ``str``               The name of the output file for path nodes and energies
    `output_dir`       ``str``               The path to the output directory
    `node_spacing`     ``float``             The interpolation distance between nodes
    `ngrad`            ``int``               The total number of gradient evaluations in one FSM run
    ================== ===================== ==================================

    """

    def __init__(self, reactant, product, nsteps=4, nnode=15, tol=0.1, nlstnodes=100,
                 qprog='g09', level_of_theory='m062x/cc-pvtz', nproc=32, mem='2000mb',
                 output_file='stringfile.txt', output_dir=''):
        if reactant.atoms != product.atoms:
            raise Exception('Atom labels of reactant and product do not match')
        self.reactant = reactant
        self.product = product
        self.nsteps = int(nsteps)
        self.nnode = int(nnode)
        self.tol = float(tol)

        self.nLSTnodes = int(nlstnodes)
        if self.nLSTnodes < 4 * self.nnode:
            raise ValueError('Increase the number of LST nodes to at least {0}'.format(4 * self.nnode))

        self.qprog = qprog.lower()
        self.level_of_theory = level_of_theory.lower()
        self.nproc = int(nproc)
        self.mem = mem

        if self.qprog == 'g03' or self.qprog == 'g09':
            self.qclass = Gaussian
        elif 'nwchem' in self.qprog:
            self.qclass = NWChem
        elif self.qprog == 'qchem':
            self.qclass = QChem
        else:
            raise NameError('Invalid quantum software')

        self.qsettings = {
            'cmd': self.qprog,
            'level_of_theory': self.level_of_theory,
            'nproc': self.nproc,
            'mem': self.mem,
            'output_dir': output_dir
        }

        self.output_file = output_file
        self.output_dir = output_dir

        # Set in initialize method
        self.node_spacing = None
        self.ngrad = None

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
        """
        # Translate reactant and product so that centroids coincide at the origin
        self.reactant.translate(-self.reactant.getCentroid())
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

        # Rotate product to maximum coincidence
        self.product.rotate(rotationMatrix(result.x))

    def initialize(self, logHeader):
        """
        Initialize the FSM/GSM job. Prints the header specified in the function
        `logHeader`, aligns the product and reactant structure to maximum
        coincidence in non-mass weighted Cartesian coordinates, and computes
        the product and reactant energies.

        A tuple of two lists is returned. The first one contains the reactant
        and product nodes, and the second one contains their energies.
        """
        # Log start timestamp
        logging.info('\nJob execution initiated on ' + time.asctime() + '\n')

        # Print FSM header
        logHeader()

        # Find distance between product and reactant nodes and calculate
        # interpolation distance after aligning product and reactant to maximum
        # coincidence
        logging.info('Aligning product and reactant structure to maximum coincidence')
        self.align()
        arclength = LST(self.reactant, self.product, self.nproc).getTotalArclength(self.nLSTnodes)
        self.node_spacing = arclength / self.nnode
        logging.info('Aligned reactant structure:\n' + str(self.reactant))
        logging.info('Aligned product structure:\n' + str(self.product))
        logging.info('Total reactant to product arc length:   {0:>8.4f} Angstrom'.format(arclength))
        logging.info('Interpolation arc length for new nodes: {0:>8.4f} Angstrom'.format(self.node_spacing))

        # Initialize path by adding reactant and product structures and computing their energies
        logging.info('Calculating reactant and product energies')
        path = [self.reactant, self.product]
        energies = [self.reactant.getEnergy(self.qclass, **self.qsettings),
                    self.product.getEnergy(self.qclass, **self.qsettings)]
        logging.info('Reactant: {0[0]:.9f} Hartree; Product: {0[1]:.9f} Hartree'.format(energies))

        # Initialize gradient counter
        self.ngrad = 0

        return path, energies

    def finalize(self, start_time, success=True):
        """
        Finalize the job.
        """
        logging.info('Total number of gradient evaluations: {0}'.format(self.ngrad))
        if success:
            logging.info('\nJob terminated successfully on ' + time.asctime())
        else:
            logging.error('\nJob terminated abnormally on ' + time.asctime())
        logging.info('Total run time: {0:.1f} s\n'.format(time.time() - start_time))

    def writeStringfile(self, path, energies):
        """
        Writes the nodes along the path and their corresponding energies to
        the output file.
        """
        with open(os.path.join(self.output_dir, self.output_file), 'w') as f:
            for node_num, (node, energy) in enumerate(zip(path, energies)):
                f.write('Node ' + str(node_num + 1) + ':\n')
                f.write('Energy = ' + str(energy) + '\n')
                f.write(str(node) + '\n')

    def getPerpGrad(self, node, tangent, name='grad'):
        """
        Calculate and return a tuple of the energy, the perpendicular gradient,
        and its magnitude given a node and the string tangent. `name` is the
        name of the quantum job to be executed.
        """
        # Calculate gradient and energy
        start_time = time.time()
        q = self.qclass()
        q.executeJob(node, name=name, jobtype='grad', **self.qsettings)
        logging.info('Gradient calculation ({0}) completed in {1:.3f} s'.format(name, time.time() - start_time))
        grad = q.getGradient().flatten()
        energy = q.getEnergy()
        logging.debug('Gradient:\n' + str(grad.reshape(len(node.atoms), 3)))
        logging.debug('Energy: ' + str(energy))
        q.clear()

        # Calculate perpendicular gradient and its magnitude
        perp_grad = (np.eye(3 * len(node.atoms)) - np.outer(tangent, tangent)).dot(grad)
        perp_grad_mag = np.linalg.norm(perp_grad)
        logging.debug('Perpendicular gradient:\n' + str(perp_grad.reshape(len(node.atoms), 3)))
        logging.debug('Magnitude: ' + str(perp_grad_mag))

        return energy, perp_grad, perp_grad_mag

    def getEnergy(self, node, name='energy'):
        """
        Calculate and return energy given a node. `name` is the name of the
        quantum job to be executed. This method is typically used if only the
        energy and not the perpendicular gradient is required
        """
        start_time = time.time()
        q = self.qclass()
        q.executeJob(node, name=name, jobtype='energy', **self.qsettings)
        logging.info('Energy calculation completed in {0:.3f} s'.format(time.time() - start_time))
        energy = q.getEnergy()
        logging.info('Energy = {0:.9f}'.format(energy))
        q.clear()

        return energy

    @staticmethod
    def getSearchDir(hess_inv, perp_grad, line_search_factor, desired_energy_change):
        """
        Calculate and return Newton-Raphson search direction, scaling factor,
        and exact line search condition given the perpendicular gradient, the
        inverse Hessian, the line search factor, and the minimum desired energy
        change.
        """
        # Calculate search direction
        direction = hess_inv.dot(perp_grad)
        direction_norm = np.linalg.norm(direction)
        search_dir = - direction / direction_norm

        # Calculate maximum and minimum scaling factors
        scale_factor_max = 0.05 / np.absolute(search_dir).max()
        scale_factor_min = (1.0 - line_search_factor) * direction_norm

        # Calculate scaling factor
        scale_factor = - 2.0 * desired_energy_change / (perp_grad.dot(search_dir))

        # Refine scaling factor based on limits
        if scale_factor < scale_factor_min:
            scale_factor = scale_factor_min
        if scale_factor > scale_factor_max:
            scale_factor = scale_factor_max

        return search_dir, scale_factor

    @staticmethod
    def updateHess(hess_inv, step, grad_diff):
        """
        Update and return the inverse Hessian according to the BFGS update
        scheme given the step and the difference in the gradients.
        """
        denom = step.dot(grad_diff)
        return hess_inv + (1.0 + grad_diff.dot((hess_inv.dot(grad_diff))) / denom) * np.outer(step, step) / denom - \
            (np.outer(step, grad_diff.dot(hess_inv)) + np.outer(hess_inv.dot(grad_diff), step)) / denom

###############################################################################

class FSM(String):
    """
    Freezing string method.
    The attributes are:

    ================== ===================== ==================================
    Attribute          Type                  Description
    ================== ===================== ==================================
    `reactant`         :class:`node.Node`    A node object containing the coordinates and atoms of the reactant molecule
    `product`          :class:`node.Node`    A node object containing the coordinates and atoms of the product molecule
    `nsteps`           ``int``               The number of gradient evaluations per node optimization
    `nnode`            ``int``               The desired number of nodes, which determines the spacing between them
    `lsf`              ``float``             A line search factor determining how strong the line search is
    `tol`              ``float``             The gradient convergence tolerance (Hartree/Angstrom)
    `nLSTnodes`        ``int``               The number of nodes on a high-density LST interpolation path
    `qprog`            ``str``               A software for quantum calculations
    `level_of_theory`  ``str``               The level of theory (method/basis) for the quantum calculations
    `nproc`            ``int``               The number of processors available for the FSM calculation
    `mem`              ``str``               The memory requirements
    `output_file`      ``str``               The name of the output file for FSM path nodes and energies
    `output_dir`       ``str``               The path to the output directory
    `node_spacing`     ``float``             The interpolation distance between nodes (set in parent)
    `ngrad`            ``int``               The total number of gradient evaluations in one FSM run (set in parent)
    ================== ===================== ==================================

    """

    def __init__(self, reactant, product, nsteps=4, nnode=15, lsf=0.7, tol=0.5, nlstnodes=100,
                 qprog='g09', level_of_theory='m062x/cc-pvtz', nproc=32, mem='2000mb',
                 output_file='stringfile.txt', output_dir=''):
        super(FSM, self).__init__(
            reactant,
            product,
            nsteps=nsteps,
            nnode=nnode,
            tol=tol,
            nlstnodes=nlstnodes,
            qprog=qprog,
            level_of_theory=level_of_theory,
            nproc=nproc,
            mem=mem,
            output_file=output_file,
            output_dir=output_dir
        )
        self.lsf = float(lsf)
        if not (0.0 < self.lsf < 1.0):
            raise ValueError('Line search factor must be between 0 and 1')

    def getNodes(self, node1, node2):
        """
        Generates new FSM nodes based on an LST interpolation path between the
        two nodes. If the distance between the nodes is less than the desired
        node spacing, then nothing is returned. Only one node is generated
        halfway between the two previous nodes if the distance between them is
        between one and two times the desired node spacing. Otherwise, two
        nodes are generated at the desired node spacing from the end nodes.

        The tangent vectors at each node are also returned. For the LST path,
        this vector is determined from the two LST nodes that are directly
        adjacent to the interpolated node.
        """
        if self.node_spacing is None:
            raise Exception('Interpolation distance has to be set first')

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

    def execute(self):
        """
        Run the freezing string method and return a tuple containing all
        optimized nodes along the FSM path and a tuple containing the energies
        of each node. The output file is updated each time nodes have been
        optimized.
        """
        start_time = time.time()
        FSMpath, energies = self.initialize(self.logHeader)

        # The minimum desired energy change in an optimization step should be
        # at least the difference between reactant and product
        energy_diff = abs(energies[1] - energies[0]) * 627.5095
        if energy_diff < 2.5:
            min_en_change = energy_diff
        else:
            min_en_change = 2.5

        # Settings for perpendicular optimization
        settings = {'min_desired_energy_change': min_en_change}

        # Impose restriction on maximum number of nodes that can be created in
        # case FSM does not converge
        for i in range(10 * self.nnode):
            logging.info('\nStarting iteration {0}\n'.format(i + 1))

            # Compute indices for inserting new nodes into FSM path
            innernode_p_idx = len(FSMpath) // 2
            innernode_r_idx = innernode_p_idx - 1

            # Compute distance between innermost nodes
            distance = FSMpath[innernode_r_idx].getDistance(FSMpath[innernode_p_idx])
            logging.info('Linear distance between innermost nodes: {0:.4f} Angstrom'.format(distance))

            # Obtain interpolated nodes
            start_time_interp = time.time()
            nodes, tangents = self.getNodes(FSMpath[innernode_r_idx], FSMpath[innernode_p_idx])
            logging.info('New nodes generated in {0:.2f} s'.format(time.time() - start_time_interp))

            # Return if no new nodes were generated
            if nodes is None:
                logging.info('No new nodes were generated')
                self.finalize(start_time)
                self.writeStringfile(FSMpath, energies)
                return tuple(FSMpath), tuple(energies)

            # Optimize halfway node if only one was generated and return
            elif isinstance(nodes, Node):
                logging.info('Added one node:\n' + str(nodes))

                # Compute distance from reactant and product side innermost nodes
                logging.info('Linear distance from innermost reactant side node: {0:>8.4f} Angstrom'.
                             format(nodes.getDistance(FSMpath[innernode_r_idx])))
                logging.info('Linear distance from innermost product side node:  {0:>8.4f} Angstrom'.
                             format(nodes.getDistance(FSMpath[innernode_p_idx])))

                # Perpendicular optimization
                logging.info('Optimizing final node')
                start_time_opt = time.time()
                logging.debug('Tangent:\n' + str(tangents.reshape(len(nodes.atoms), 3)))
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

                self.finalize(start_time)
                self.writeStringfile(FSMpath, energies)
                return tuple(FSMpath), tuple(energies)

            # Optimize new nodes
            else:
                logging.info('Added two nodes:\n' + str(nodes[0]) + '\n****\n' + str(nodes[1]))

                # Compute distance from reactant and product side innermost nodes
                logging.info('Linear distance between previous and current reactant side nodes: {0:>8.4f} Angstrom'.
                             format(nodes[0].getDistance(FSMpath[innernode_r_idx])))
                logging.info('Linear distance between previous and current product side nodes:  {0:>8.4f} Angstrom'.
                             format(nodes[1].getDistance(FSMpath[innernode_p_idx])))

                # Perpendicular optimization
                logging.info('Optimizing new reactant side node')
                start_time_opt = time.time()
                logging.debug('Tangent:\n' + str(tangents[0].reshape(len(nodes[0].atoms), 3)))
                energy0 = self.perpOpt(nodes[0], tangents[0], nodes[1], **settings)
                logging.info('Optimization completed in {0:.1f} s'.format(time.time() - start_time_opt))
                logging.info('Optimizing new product side node')
                start_time_opt = time.time()
                logging.debug('Tangent:\n' + str(tangents[1].reshape(len(nodes[1].atoms), 3)))
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
                self.writeStringfile(FSMpath, energies)

        self.finalize(start_time, success=False)
        raise Exception('FSM did not converge')

    def perpOpt(self, node, tangent, other_node=None, min_desired_energy_change=2.5):
        """
        Optimize node in direction of negative perpendicular gradient using the
        Newton-Raphson method with a BFGS Hessian update scheme. Requires input
        of tangent vector between closest two nodes on the string so that the
        appropriate perpendicular gradient can be calculated. Also requires
        that the innermost node on the other side of the string is input so
        that the forward progress towards joining string ends can be assessed.
        If no other node is specified, then the node is optimized without such
        a constraint.

        Returns the energy of the optimized node in Hartree.

        Set `min_desired_energy_change` to the energy difference between
        reactant and product if the difference is less than 2.5 kcal/mol.
        """

        # Compute maximum allowed distance between nodes based on initial distance
        if other_node is not None:
            max_distance = node.getDistance(other_node) + 0.5 * self.node_spacing
        else:
            max_distance = 0

        # Initialize Hessian inverse to identity matrix
        identity_mat = np.eye(3 * len(node.atoms))
        hess_inv = np.copy(identity_mat)

        # Convert units to Hartree
        min_desired_energy_change /= 627.5095

        # Calculate perpendicular gradient and energy
        energy, perp_grad, perp_grad_mag = self.getPerpGrad(node, tangent, name='grad0')
        self.ngrad += 1
        energy_old = energy

        # Create empty array for storing old perpendicular gradient
        perp_grad_old = np.empty_like(perp_grad)

        k = 1
        unstable = False
        while k <= self.nsteps:
            # Calculate desired change in energy
            desired_energy_change = max(energy_old - energy, min_desired_energy_change)

            # Calculate search direction, scaling factor, and exact line search condition
            search_dir, scale_factor = self.getSearchDir(hess_inv, perp_grad, self.lsf, desired_energy_change)

            # Handle unstable searches by reinitializing the Hessian
            if scale_factor < 0.0:
                # Terminate if resetting Hessian did not resolve instability
                if unstable:
                    logging.warning('Optimization terminated prematurely due to unstable scaling factor')
                    break
                unstable = True
                np.copyto(hess_inv, identity_mat)
                continue
            unstable = False

            # Take minimization step
            step = scale_factor * search_dir
            node.displaceCoordinates(step.reshape(len(node.atoms), 3))
            logging.debug('Updated coordinates:\n' + str(node))

            # Save old values
            np.copyto(perp_grad_old, perp_grad)
            energy_old = energy

            # Calculate new perpendicular gradient and energy
            energy, perp_grad, perp_grad_mag = self.getPerpGrad(node, tangent, name='grad' + str(k))
            self.ngrad += 1

            # Check termination conditions:
            #     - Energy increase from previous step
            #     - Small energy change
            #     - Maximum number of steps
            #     - Stable line search condition
            #     - Perpendicular gradient tolerance reached
            #     - Exceeding of maximum distance
            if k > 1:
                energy_change = energy - energy_old
                if energy_change > 0.0:
                    logging.info('Optimization terminated due to energy increase')
                    return energy_old
                if abs(energy_change) < 0.5 / 627.5095:
                    logging.info('Optimization terminated due to small energy change')
                    return energy
            if k == self.nsteps:
                return energy
            if perp_grad_mag < self.tol:
                logging.info('Perpendicular gradient convergence criterion satisfied')
                return energy
            if abs(perp_grad.dot(search_dir)) <= - self.lsf * perp_grad_old.dot(search_dir):
                logging.info('Optimization terminated due to stable line search condition')
                return energy
            if node.getDistance(other_node) > max_distance:
                logging.warning('Optimization terminated because maximum distance between nodes was exceeded')
                return energy

            # Update inverse Hessian
            perp_grad_diff = perp_grad - perp_grad_old
            hess_inv = self.updateHess(hess_inv, step, perp_grad_diff)
            logging.debug('Hessian inverse:\n' + str(hess_inv))

            # Update counter
            k += 1

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
        logging.log(level, '# Gradient convergence tolerance (Hartree/Angstrom):         {0:>5.2f}         #'.
                    format(self.tol))
        logging.log(level, '# Number of high density LST nodes:                          {0:>5}         #'.
                    format(self.nLSTnodes))
        logging.log(level, '# Level of theory for quantum calculations: {0:<31}#'.format(self.level_of_theory))
        logging.log(level, '############################################################################')
        logging.log(level, 'Reactant structure:\n' + str(self.reactant))
        logging.log(level, 'Product structure:\n' + str(self.product))
        logging.log(level, '############################################################################\n')

###############################################################################

class GSM(String):
    """
    Growing string method.
    The attributes are:

    ================== ===================== ==================================
    Attribute          Type                  Description
    ================== ===================== ==================================
    `reactant`         :class:`node.Node`    A node object containing the coordinates and atoms of the reactant molecule
    `product`          :class:`node.Node`    A node object containing the coordinates and atoms of the product molecule
    `nsteps`           ``int``               The number of gradient evaluations per node optimization
    `nnode`            ``int``               The desired number of nodes, which determines the spacing between them
    `tol`              ``float``             The tolerance for adding frontier nodes (Hartree/Angstrom)
    `gtol`             ``float``             The overall string tolerance based on the sum of gradient magnitudes
    `nLSTnodes`        ``int``               The number of nodes on a high-density LST interpolation path
    `qprog`            ``str``               A software for quantum calculations
    `level_of_theory`  ``str``               The level of theory (method/basis) for the quantum calculations
    `nproc`            ``int``               The number of processors available for the calculation
    `mem`              ``str``               The memory requirements
    `output_file`      ``str``               The name of the output file for GSM path nodes and energies
    `output_dir`       ``str``               The path to the output directory
    `node_spacing`     ``float``             The interpolation distance between nodes (set in parent)
    `ngrad`            ``int``               The total number of gradient evaluations in one GSM run (set in parent)
    ================== ===================== ==================================

    """

    def __init__(self, reactant, product, nsteps=3, nnode=15, tol=0.1, gtol=0.3, nlstnodes=100,
                 qprog='g09', level_of_theory='m062x/cc-pvtz', nproc=32, mem='2000mb',
                 output_file='stringfile.txt', output_dir=''):
        super(GSM, self).__init__(
            reactant,
            product,
            nsteps=nsteps,
            nnode=nnode,
            tol=tol,
            nlstnodes=nlstnodes,
            qprog=qprog,
            level_of_theory=level_of_theory,
            nproc=nproc,
            mem=mem,
            output_file=output_file,
            output_dir=output_dir
        )
        self.gtol = float(gtol)

    def reparameterize(self, nodes_r, nodes_p, add_r=False, add_p=False):
        """
        Reparameterizes a GSM path based on the desired node spacing and adds
        new nodes if desired based on an LST interpolation path containing all
        nodes. Requires a list of reactant and product side nodes and an
        indicator specifying whether new nodes should be added. No nodes are
        returned if the string is fully grown, otherwise the reparameterized
        lists of nodes are returned. At most one node is generated halfway
        between the two innermost nodes if the distance between them is between
        one and two times the desired node spacing.

        The tangent vectors at each node are also returned. For the LST path,
        this vector is determined from the two LST nodes that are directly
        adjacent to the interpolated node.
        """
        if self.node_spacing is None:
            raise Exception('Interpolation distance has to be set first')

        # Compute distances between nodes and assign LST nodes
        distances = np.array([nodes[i].getDistance(nodes[i + 1]) for i in range(0, len(nodes) - 1)])
        nnodes_list = [int(round(d)) for d in distances / distances.sum() * nnodes]

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

    def execute(self):
        """
        Run the growing string method and return a tuple containing all
        optimized nodes along the FSM path and a tuple containing the energies
        of each node. The output file is updated each time nodes have been
        optimized.
        """
        start_time = time.time()
        GSMpath, energies = self.initialize(self.logHeader)

    def finalize(self, start_time, success=True):
        """
        Finalize the GSM job.
        """
        logging.info('Total number of gradient evaluations: {0}'.format(self.ngrad))
        if success:
            logging.info('\nGSM job terminated successfully on ' + time.asctime())
        else:
            logging.error('\nGSM job terminated abnormally on ' + time.asctime())
        logging.info('Total run time: {0:.1f} s\n'.format(time.time() - start_time))

    def perpOpt(self, node, tangent, nsteps=3, other_node=None, min_desired_energy_change=2.5, line_search_factor=0.7,
                frontier_node=False):
        """
        Optimize node in direction of negative perpendicular gradient using the
        Newton-Raphson method with a BFGS Hessian update scheme. Requires input
        of tangent vector between closest two nodes on the string so that the
        appropriate perpendicular gradient can be calculated. Also requires
        that the innermost node on the other side of the string is input so
        that the forward progress towards joining string ends can be assessed.
        If no other node is specified, then the node is optimized without such
        a constraint.

        Returns the energy of the optimized node in Hartree and the
        perpendicular gradient magnitude in Hartree/Angstrom.

        Set `min_desired_energy_change` to the energy difference between
        reactant and product if the difference is less than 2.5 kcal/mol.
        """
        # If the node is a frontier node, only optimize until the tolerance is reached
        if frontier_node:
            tol = self.tol
        else:
            tol = 0.0

        # Compute maximum allowed distance between nodes based on initial distance
        if other_node is not None:
            max_distance = node.getDistance(other_node) + 0.5 * self.node_spacing
        else:
            max_distance = 0

        # Initialize Hessian inverse to identity matrix
        identity_mat = np.eye(3 * len(node.atoms))
        hess_inv = np.copy(identity_mat)

        # Convert units to Hartree
        min_desired_energy_change /= 627.5095

        # Calculate perpendicular gradient and energy
        energy, perp_grad, perp_grad_mag = self.getPerpGrad(node, tangent, name='grad0')
        self.ngrad += 1
        energy_old = energy

        # Create empty array for storing old perpendicular gradient
        perp_grad_old = np.empty_like(perp_grad)

        k = 1
        unstable = False
        while k <= nsteps and perp_grad_mag >= tol:
            # Calculate desired change in energy
            desired_energy_change = max(energy_old - energy, min_desired_energy_change)

            # Calculate search direction, scaling factor, and exact line search condition
            search_dir, scale_factor = self.getSearchDir(hess_inv, perp_grad, self.lsf, desired_energy_change)

            # Handle unstable searches by reinitializing the Hessian
            if scale_factor < 0.0:
                # Terminate if resetting Hessian did not resolve instability
                if unstable:
                    logging.warning('Optimization terminated prematurely due to unstable scaling factor')
                    break
                unstable = True
                np.copyto(hess_inv, identity_mat)
                continue
            unstable = False

            # Take minimization step
            step = scale_factor * search_dir
            node.displaceCoordinates(step.reshape(len(node.atoms), 3))
            logging.debug('Updated coordinates:\n' + str(node))

            # Save old values
            np.copyto(perp_grad_old, perp_grad)
            energy_old = energy

            # Calculate new perpendicular gradient and energy
            energy, perp_grad, perp_grad_mag = self.getPerpGrad(node, tangent, name='grad' + str(k))
            self.ngrad += 1

            # Check termination conditions:
            #     - Energy increase from previous step
            #     - Small energy change
            #     - Maximum number of steps
            #     - Stable line search condition
            #     - Perpendicular gradient tolerance reached
            #     - Exceeding of maximum distance
            if k > 1:
                energy_change = energy - energy_old
                if energy_change > 0.0:
                    logging.info('Optimization terminated due to energy increase')
                    return energy_old
                if abs(energy_change) < 0.5 / 627.5095:
                    logging.info('Optimization terminated due to small energy change')
                    return energy
            if k == self.nsteps:
                return energy
            if perp_grad_mag < self.tol:
                logging.info('Perpendicular gradient convergence criterion satisfied')
                return energy
            if abs(perp_grad.dot(search_dir)) <= - self.lsf * perp_grad_old.dot(search_dir):
                logging.info('Optimization terminated due to stable line search condition')
                return energy
            if node.getDistance(other_node) > max_distance:
                logging.warning('Optimization terminated because maximum distance between nodes was exceeded')
                return energy

            # Update inverse Hessian
            perp_grad_diff = perp_grad - perp_grad_old
            hess_inv = self.updateHess(hess_inv, step, perp_grad_diff)
            logging.debug('Hessian inverse:\n' + str(hess_inv))

            # Update counter
            k += 1

        # Return optimized node energy, perpendicular gradient magnitude, and number of steps
        return energy, perp_grad_mag, k - 1

    def logHeader(self, level=logging.INFO):
        """
        Output a log file header containing identifying information about the
        FSM job.
        """
        logging.log(level, '###########################################################################')
        logging.log(level, '########################## GROWING STRING METHOD ##########################')
        logging.log(level, '###########################################################################')
        logging.log(level, '# Frontier node addition threshold (Hartree/Angstrom):       {0:>5.2f}        #'.
                    format(self.tol))
        logging.log(level, '# Overall convergence threshold (Hartree/Angstrom):          {0:>5.2f}        #'.
                    format(self.gtol))
        logging.log(level, '# Number of nodes for calculation of interpolation distance: {0:>5}        #'.
                    format(self.nnode))
        logging.log(level, '# Number of high density LST nodes:                          {0:>5}        #'.
                    format(self.nLSTnodes))
        logging.log(level, '# Level of theory for quantum calculations: {0:<31}#'.format(self.level_of_theory))
        logging.log(level, '###########################################################################')
        logging.log(level, 'Reactant structure:\n' + str(self.reactant))
        logging.log(level, 'Product structure:\n' + str(self.product))
        logging.log(level, '###########################################################################\n')