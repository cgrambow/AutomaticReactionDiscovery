#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains relevant classes for executing transition state searches using the
freezing string method. The resulting transition state should be further
optimized using another method in order to find the true transition state.
"""

from __future__ import division

import logging
import os
import time

import numpy as np
from scipy import optimize

import util
import constants
import props
from quantum import QuantumError
from node import Node
from interpolation import LST

###############################################################################

def removeDuplicateBondChanges(bc):
    """
    Given a list of bond changes as a numpy vector, remove duplicates.
    """
    if bc.size:
        out = []
        for b in bc:
            if b[1] > b[0]:
                out.append(b)

        return np.vstack(out)
    else:
        return bc

###############################################################################

class String(object):
    """
    Base class from which the freezing string method can inherit.
    The attributes are:

    =============== ======================== ===================================
    Attribute       Type                     Description
    =============== ======================== ===================================
    `name`          ``str``                  The name of the object
    `reactant`      :class:`node.Node`       A node object containing the coordinates and atoms of the reactant molecule
    `product`       :class:`node.Node`       A node object containing the coordinates and atoms of the product molecule
    `reac_cmat`     :class:`numpy.ndarray`   The connectivity matrix of the reactant
    `bc`            ``list``                 A list of bond changes between reactant and product
    `nsteps`        ``int``                  The number of gradient evaluations per node optimization
    `nnode`         ``int``                  The desired number of nodes, which determines the spacing between them
    `tol`           ``float``                The gradient convergence tolerance (Hartree/Angstrom)
    `nLSTnodes`     ``int``                  The number of nodes on a high-density LST interpolation path
    `Qclass`        ``class``                A class representing the quantum software
    `nproc`         ``int``                  The number of processors available for the string method
    `output_dir`    ``str``                  The path to the output directory
    `kwargs`        ``dict``                 Additional arguments for quantum calculations
    `node_spacing`  ``float``                The interpolation distance between nodes
    `ngrad`         ``int``                  The total number of gradient evaluations
    `logger`        :class:`logging.Logger`  The logger
    =============== ======================== ===================================

    """

    def __init__(self, reactant, product, name='0000', logger=None,
                 nsteps=4, nnode=15, tol=0.1, nlstnodes=100, qprog='gau', **kwargs):
        if reactant.atoms != product.atoms:
            raise Exception('Atom labels of reactant and product do not match')
        self.reactant = reactant
        self.product = product
        self.name = name

        self.reac_cmat = self.reactant.toConnectivityMat()
        self.bc = None
        self.findBondChanges()

        self.nsteps = int(nsteps)
        self.nnode = int(nnode)
        self.tol = float(tol)
        self.nLSTnodes = int(nlstnodes)

        self.Qclass = util.assignQclass(qprog)
        self.nproc = int(kwargs.get('nproc', 1))
        self.output_dir = kwargs.get('output_dir', '')
        self.kwargs = kwargs

        self.node_spacing = None
        self.ngrad = None

        # Set up logger
        if logger is None:
            log_level = logging.INFO
            logfile = 'output.' + self.name + '.log'
            self.logger = util.initializeLog(log_level, os.path.join(self.output_dir, logfile))
        else:
            self.logger = logger

    def findBondChanges(self):
        """
        Save the list of bond changes between reactant and product.
        """
        prod_cmat = self.product.toConnectivityMat()
        Rmat = prod_cmat - self.reac_cmat
        bc = np.transpose(np.nonzero(Rmat))
        bc = removeDuplicateBondChanges(bc)
        self.bc = bc.tolist()

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
        rotated_product = (util.rotationMatrix(angles).dot(self.product.coords.T)).T.flatten()
        diff = self.reactant.coords.flatten() - rotated_product
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
            self.logger.warning(message)

        # Check for positive eigenvalues to ensure that aligned structure is a minimum
        eig_val = np.linalg.eig(result.hess_inv)[0]
        if not all(eig_val > 0.0):
            self.logger.warning('Not all Hessian eigenvalues were positive for the alignment process.\n' +
                                'The aligned structure may not be optimal.\n')

        # Rotate product to maximum coincidence
        self.product.rotate(util.rotationMatrix(result.x))

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
        self.logger.info('\n----------------------------------------------------------------------')
        self.logger.info('String method initiated on ' + time.asctime() + '\n')

        # Print FSM header
        logHeader()

        # Find distance between product and reactant nodes and calculate
        # interpolation distance after aligning product and reactant to maximum
        # coincidence
        self.logger.info('Aligning product and reactant structure to maximum coincidence')
        self.align()
        arclength = LST(self.reactant, self.product, self.nproc).getTotalArclength(self.nLSTnodes)
        self.node_spacing = arclength / self.nnode
        self.logger.info('Aligned reactant structure:\n' + str(self.reactant))
        self.logger.info('Aligned product structure:\n' + str(self.product))
        self.logger.info('Total reactant to product arc length:   {0:>8.4f} Angstrom'.format(arclength))
        self.logger.info('Interpolation arc length for new nodes: {0:>8.4f} Angstrom'.format(self.node_spacing))

        # Initialize gradient counter
        self.ngrad = 0

        # Initialize path by adding reactant and product structures and computing their energies
        self.logger.info('Calculating reactant and product energies')
        path = [self.reactant, self.product]
        self.reactant.computeEnergy(self.Qclass, name='reac_energy.' + self.name, **self.kwargs)
        self.product.computeEnergy(self.Qclass, name='prod_energy.' + self.name, **self.kwargs)
        self.logger.info(
            'Reactant: {0:.9f} Hartree; Product: {1:.9f} Hartree'.format(self.reactant.energy, self.product.energy)
        )

        return path

    def finalize(self, start_time, success=True):
        """
        Finalize the job.
        """
        self.logger.info('Number of gradient evaluations during string method: {0}'.format(self.ngrad))
        if success:
            self.logger.info('\nString method terminated successfully on ' + time.asctime())
        else:
            self.logger.warning('String method terminated abnormally on ' + time.asctime())
        self.logger.info('Total string method run time: {0:.1f} s'.format(time.time() - start_time))
        self.logger.info('----------------------------------------------------------------------\n')

    def writeStringfile(self, path):
        """
        Write the nodes along the path and their corresponding energies
        relative to the reactant energy (in kcal/mol) to the output file.
        """
        with open(os.path.join(self.output_dir, 'string.{}.out'.format(self.name)), 'w') as f:
            for node_num, node in enumerate(path):
                f.write(str(len(node.atoms)) + '\n')

                energy = (node.energy - self.reactant.energy) * constants.hartree_to_kcal_per_mol
                f.write('Energy = ' + str(energy) + '\n')
                f.write(str(node) + '\n')

    def writeDistMat(self, node, msg=None):
        """
        Write the distance matrix at a node and check for undesired bond
        changes.
        """
        with open(os.path.join(self.output_dir, 'bond_changes.{}.out'.format(self.name)), 'a') as f:
            if msg is not None:
                f.write(msg + '\n')

            dist_mat = util.getDistMat(node.coords)

            dmat_string = ''
            for anum, row in enumerate(dist_mat):
                line = ' '.join(['{:7.4f}'.format(d) for d in row])
                dmat_string += '{}  {}\n'.format(props.atomnum[node.atoms[anum]], line)

            f.write(dmat_string)

            if self.detectUndesiredBondChange(node):
                f.write('Above distance matrix contains undesired bond change.\n')

    def detectUndesiredBondChange(self, node):
        """
        Detect undesired bond changes that do not correspond to the desired
        reaction between the given reactant and product. Returns a boolean.
        """
        cmat = node.toConnectivityMat()
        Rmat = cmat - self.reac_cmat
        bc = np.transpose(np.nonzero(Rmat))
        bc = removeDuplicateBondChanges(bc)
        bc = bc.tolist()

        for b in bc:
            if b not in self.bc:
                return True
        else:
            return False

    @util.timeFn
    def getPerpGrad(self, node, tangent, name='grad.0000'):
        """
        Calculate and return a tuple of the perpendicular gradient and its
        magnitude given a node and the string tangent. `name` is the name of
        the quantum job to be executed.
        """
        # Calculate gradient and energy
        node.computeGradient(self.Qclass, name=name, **self.kwargs)
        self.logger.debug('Gradient:\n' + str(node.gradient.reshape(len(node.atoms), 3)))
        self.logger.debug('Energy: ' + str(node.energy))

        # Calculate perpendicular gradient and its magnitude
        perp_grad = (np.eye(3 * len(node.atoms)) - np.outer(tangent, tangent)).dot(node.gradient.flatten())
        perp_grad_mag = np.linalg.norm(perp_grad)
        self.logger.debug('Perpendicular gradient:\n' + str(perp_grad.reshape(len(node.atoms), 3)))
        self.logger.debug('Magnitude: ' + str(perp_grad_mag))

        return perp_grad, perp_grad_mag

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
    `lsf`              ``float``             A line search factor determining how strong the line search is
    ================== ===================== ==================================

    """

    def __init__(self, *args, **kwargs):
        self.lsf = float(kwargs.pop('lsf', 0.7))
        if not (0.0 < self.lsf < 1.0):
            raise ValueError('Line search factor must be between 0 and 1')

        super(FSM, self).__init__(*args, **kwargs)

    @util.timeFn
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
            new_node_idx = util.findClosest(arclength, arclength[-1] / 2.0)
            return path[new_node_idx], path[new_node_idx - 1].getTangent(path[new_node_idx + 1])
        new_node1_idx = util.findClosest(arclength, self.node_spacing)
        new_node2_idx = util.findClosest(arclength, arclength[-1] - self.node_spacing)
        tangent1 = path[new_node1_idx - 1].getTangent(path[new_node1_idx + 1])
        tangent2 = path[new_node2_idx + 1].getTangent(path[new_node2_idx - 1])
        return (path[new_node1_idx], path[new_node2_idx]), (tangent1, tangent2)

    def execute(self):
        """
        Run the freezing string method and return a tuple containing all
        optimized nodes along the FSM path. The output file is updated each
        time nodes have been optimized.
        """
        start_time = time.time()
        FSMpath = self.initialize(self.logHeader)

        # The minimum desired energy change in an optimization step should be
        # at least the difference between reactant and product
        energy_diff = abs(FSMpath[1].energy - FSMpath[0].energy) * 627.5095
        if energy_diff < 2.5:
            min_en_change = energy_diff
        else:
            min_en_change = 2.5

        # Impose restriction on maximum number of nodes that can be created in
        # case FSM does not converge
        for i in range(2 * self.nnode):
            self.logger.info('\nStarting iteration {0}\n'.format(i + 1))

            # Compute indices for inserting new nodes into FSM path
            innernode_p_idx = len(FSMpath) // 2
            innernode_r_idx = innernode_p_idx - 1

            # Compute distance between innermost nodes
            distance = FSMpath[innernode_r_idx].getDistance(FSMpath[innernode_p_idx])
            self.logger.info('Linear distance between innermost nodes: {0:.4f} Angstrom'.format(distance))

            # Obtain interpolated nodes
            nodes, tangents = self.getNodes(FSMpath[innernode_r_idx], FSMpath[innernode_p_idx])

            # Return if no new nodes were generated
            if nodes is None:
                self.logger.info('No new nodes were generated')
                self.finalize(start_time)
                self.writeStringfile(FSMpath)
                return FSMpath

            # Optimize halfway node if only one was generated and return
            elif isinstance(nodes, Node):
                self.logger.info('Added one node:\n' + str(nodes))

                # Compute distance from reactant and product side innermost nodes
                self.logger.info('Linear distance from innermost reactant side node: {0:>8.4f} Angstrom'.
                                 format(nodes.getDistance(FSMpath[innernode_r_idx])))
                self.logger.info('Linear distance from innermost product side node:  {0:>8.4f} Angstrom'.
                                 format(nodes.getDistance(FSMpath[innernode_p_idx])))

                # Perpendicular optimization
                self.logger.info('Optimizing final node')
                self.writeDistMat(nodes, msg='Distance matrix before perpendicular optimization (final node):')
                self.logger.debug('Tangent:\n' + str(tangents.reshape(len(nodes.atoms), 3)))
                self.perpOpt(nodes, tangents, min_desired_energy_change=min_en_change)
                self.logger.info('Energy = {0:.9f} Hartree'.format(nodes.energy))

                self.logger.info('Optimized node:\n' + str(nodes))
                self.logger.info('After opt distance from innermost reactant side node: {0:>8.4f} Angstrom'.
                                 format(nodes.getDistance(FSMpath[innernode_r_idx])))
                self.logger.info('After opt distance from innermost product side node:  {0:>8.4f} Angstrom'.
                                 format(nodes.getDistance(FSMpath[innernode_p_idx])))
                self.logger.info('')

                # Insert optimized node and corresponding energy into path
                FSMpath.insert(innernode_p_idx, nodes)

                self.finalize(start_time)
                self.writeStringfile(FSMpath)
                return FSMpath

            # Optimize new nodes
            else:
                self.logger.info('Added two nodes:\n' + str(nodes[0]) + '\n****\n' + str(nodes[1]))

                # Compute distance from reactant and product side innermost nodes
                self.logger.info('Linear distance between previous and current reactant side nodes: {0:>8.4f} Angstrom'.
                                 format(nodes[0].getDistance(FSMpath[innernode_r_idx])))
                self.logger.info('Linear distance between previous and current product side nodes:  {0:>8.4f} Angstrom'.
                                 format(nodes[1].getDistance(FSMpath[innernode_p_idx])))

                # Perpendicular optimization
                self.logger.info('Optimizing new reactant side node')
                self.writeDistMat(nodes[0], msg='Distance matrix before perpendicular optimization (reactant side):')
                self.logger.debug('Tangent:\n' + str(tangents[0].reshape(len(nodes[0].atoms), 3)))
                self.perpOpt(nodes[0], tangents[0], nodes[1], min_en_change)
                self.logger.info('Energy = {0:.9f} Hartree'.format(nodes[0].energy))

                self.logger.info('Optimizing new product side node')
                self.writeDistMat(nodes[1], msg='Distance matrix before perpendicular optimization (product side):')
                self.logger.debug('Tangent:\n' + str(tangents[1].reshape(len(nodes[1].atoms), 3)))
                self.perpOpt(nodes[1], tangents[1], nodes[0], min_en_change)
                self.logger.info('Energy = {0:.9f} Hartree'.format(nodes[1].energy))

                self.logger.info('Optimized nodes:\n' + str(nodes[0]) + '\n****\n' + str(nodes[1]))
                self.logger.info(
                    'After opt distance between previous and current reactant side nodes: {0:>8.4f} Angstrom'.
                    format(nodes[0].getDistance(FSMpath[innernode_r_idx]))
                )
                self.logger.info(
                    'After opt distance between previous and current product side nodes:  {0:>8.4f} Angstrom'.
                    format(nodes[1].getDistance(FSMpath[innernode_p_idx]))
                )
                self.logger.info('')

                # Insert optimized nodes and corresponding energies into path
                FSMpath.insert(innernode_p_idx, nodes[1])
                FSMpath.insert(innernode_p_idx, nodes[0])
                self.writeStringfile(FSMpath)

        self.finalize(start_time, success=False)
        self.writeStringfile(FSMpath)
        return FSMpath

    @util.timeFn
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
        min_desired_energy_change /= constants.hartree_to_kcal_per_mol

        # Calculate perpendicular gradient and set node energy
        perp_grad, perp_grad_mag = self.getPerpGrad(node, tangent, name='grad.{}.0'.format(self.name))
        self.ngrad += 1
        energy_old = node.energy

        # Create empty array for storing old perpendicular gradient
        perp_grad_old = np.empty_like(perp_grad)

        k = 1
        unstable = False
        while k <= self.nsteps:
            # Calculate desired change in energy
            desired_energy_change = max(energy_old - node.energy, min_desired_energy_change)

            # Calculate search direction, scaling factor, and exact line search condition
            search_dir, scale_factor = self.getSearchDir(hess_inv, perp_grad, self.lsf, desired_energy_change)

            # Handle unstable searches by reinitializing the Hessian
            if scale_factor < 0.0:
                # Terminate if resetting Hessian did not resolve instability
                if unstable:
                    self.logger.warning('Optimization terminated prematurely due to unstable scaling factor')
                    break
                unstable = True
                np.copyto(hess_inv, identity_mat)
                continue
            unstable = False

            # Take minimization step
            step = scale_factor * search_dir
            node.displaceCoordinates(step.reshape(len(node.atoms), 3))
            self.logger.debug('Updated coordinates:\n' + str(node))
            self.writeDistMat(node, msg='After step {}:'.format(k))

            # Save old values
            np.copyto(perp_grad_old, perp_grad)
            energy_old = node.energy

            # Calculate new perpendicular gradient and set energy
            try:
                perp_grad, perp_grad_mag = self.getPerpGrad(node, tangent, name='grad.{}.{}'.format(self.name, k))
            except QuantumError:
                self.logger.info('SCF error ignored. Previous gradient used.')
                grad_success = False
                pass  # Ignore error and use previous gradient
            else:
                grad_success = True

            self.ngrad += 1

            # Check termination conditions:
            #     - Maximum number of steps
            #     - Energy increase from previous step
            #     - Small energy change
            #     - Stable line search condition
            #     - Perpendicular gradient tolerance reached
            #     - Exceeding of maximum distance
            if k == self.nsteps:
                self.logger.info('Optimization terminated because maximum number of steps was reached')
                break
            if grad_success:
                if k > 1:
                    energy_change = node.energy - energy_old
                    if energy_change > 0.0:
                        self.logger.info('Optimization terminated due to energy increase')
                        node.displaceCoordinates(-step.reshape(len(node.atoms), 3))
                        node.energy = energy_old
                        break
                    if abs(energy_change) < 0.5 / constants.hartree_to_kcal_per_mol:
                        self.logger.info('Optimization terminated due to small energy change')
                        break
                if perp_grad_mag < self.tol:
                    self.logger.info('Perpendicular gradient convergence criterion satisfied')
                    break
                if abs(perp_grad.dot(search_dir)) <= - self.lsf * perp_grad_old.dot(search_dir):
                    self.logger.info('Optimization terminated due to stable line search condition')
                    break
                if node.getDistance(other_node) > max_distance:
                    self.logger.warning('Optimization terminated because maximum distance between nodes was exceeded')
                    break

                # Update inverse Hessian
                perp_grad_diff = perp_grad - perp_grad_old
                hess_inv = self.updateHess(hess_inv, step, perp_grad_diff)
                self.logger.debug('Hessian inverse:\n' + str(hess_inv))

            # Update counter
            k += 1

    def logHeader(self):
        """
        Output a log file header containing identifying information about the
        FSM job.
        """
        self.logger.info('######################################################################')
        self.logger.info('####################### FREEZING STRING METHOD #######################')
        self.logger.info('######################################################################')
        self.logger.info('# Number of gradient calculations per optimization step:     {0:>5}   #'.format(self.nsteps))
        self.logger.info('# Number of nodes for calculation of interpolation distance: {0:>5}   #'.format(self.nnode))
        self.logger.info('# Line search factor during Newton-Raphson optimization:     {0:>5.2f}   #'.format(self.lsf))
        self.logger.info('# Gradient convergence tolerance (Hartree/Angstrom):         {0:>5.2f}   #'.format(self.tol))
        self.logger.info('# Number of high density LST nodes:                          {0:>5}   #'.
                         format(self.nLSTnodes))
        self.logger.info('######################################################################')
        self.logger.info('Reactant structure:\n' + str(self.reactant))
        self.logger.info('Product structure:\n' + str(self.product))
        self.logger.info('######################################################################\n')

###############################################################################

if __name__ == '__main__':
    import argparse

    from main import readInput

    # Set up parser for reading the input filename from the command line
    parser = argparse.ArgumentParser(description='A freezing string method transition state search')
    parser.add_argument('-n', '--nproc', default=1, type=int, metavar='N', help='number of processors')
    parser.add_argument('-m', '--mem', default=2000, type=int, metavar='M', help='memory requirement')
    parser.add_argument('file', type=str, metavar='infile', help='an input file describing the FSM job options')
    args = parser.parse_args()

    # Read input file
    input_file = os.path.abspath(args.file)
    options = readInput(input_file)

    # Set output directory
    output_dir = os.path.abspath(os.path.dirname(input_file))
    options['output_dir'] = output_dir

    # Set number of processors
    options['nproc'] = args.nproc
    options['mem'] = str(args.mem) + 'mb'

    # Execute job
    fsm = FSM(**options)
    fsm.execute()
