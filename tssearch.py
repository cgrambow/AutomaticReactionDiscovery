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
Contains the :class:`TSSearch` for finding transition states and reaction paths
using FSM.
"""

import glob
import logging
import os
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ard.constants as constants
import ard.util as util
from ard.quantum import QuantumError
from ard.node import Node
from ard.sm import FSM

###############################################################################

class TSError(Exception):
    """
    An exception class for errors that occur during a TS search.
    """
    pass

###############################################################################

class TSSearch(object):
    """
    Transition state finding using FSM with subsequent exact TS search and
    verification of the reaction path by intrinsic reaction coordinate
    calculation.
    The attributes are:

    ============== ======================== ===================================
    Attribute      Type                     Description
    ============== ======================== ===================================
    `reactant`     :class:`node.Node`       A node object containing the coordinates and atoms of the reactant molecule
    `product`      :class:`node.Node`       A node object containing the coordinates and atoms of the product molecule
    `ts`           :class:`node.Node`       The exact transition state
    `irc`          ``list``                 The IRC path corresponding to the transition state
    `fsm`          ``list``                 The FSM path
    `barrier`      ``float``                The reaction barrier in kcal/mol
    `dH`           ``float``                The reaction energy in kcal/mol
    `output_dir`   ``str``                  The path to the output directory
    `Qclass`       ``class``                A class representing the quantum software
    `kwargs`       ``dict``                 Options for FSM and quantum calculations
    `ngrad`        ``int``                  The total number of gradient evaluations
    `logger`       :class:`logging.Logger`  The logger
    ============== ======================== ===================================

    """

    def __init__(self, reactant, product, logname=None, **kwargs):
        if reactant.atoms != product.atoms:
            raise Exception('Atom labels of reactant and product do not match')
        self.reactant = reactant
        self.product = product

        self.output_dir = kwargs.get('output_dir', '')
        qprog = kwargs.get('qprog', 'gau')
        self.Qclass = util.assignQclass(qprog)
        self.kwargs = kwargs

        self.ts = None
        self.irc = None
        self.fsm = None
        self.barrier = None
        self.dH = None
        self.ngrad = None

        # Set up log file
        log_level = logging.INFO
        if logname is None:
            filename = 'TSSearch.log'
        else:
            filename = logname + '.log'
            self.__name__ = logname
        self.logger = util.initializeLog(log_level, os.path.join(self.output_dir, filename), logname=logname)

    def initialize(self):
        """
        Initialize the TS search job.
        """
        self.logger.info('\nTS search initiated on ' + time.asctime() + '\n')
        self.logHeader()
        self.ngrad = 0

    def execute(self):
        """
        Run the string method, exact transition state search, IRC calculation,
        and check the results. The highest energy node is selected for the
        exact transition state search.
        """
        start_time = time.time()
        self.initialize()

        reac_opt_success = self.optimizeReactant()
        if reac_opt_success:
            reac_energy = self.reactant.energy
        prod_opt_success = self.optimizeProduct()
        if prod_opt_success:
            prod_energy = self.product.energy

        self.executeStringMethod()

        energy_max = self.fsm[0].energy
        for node in self.fsm[1:-1]:
            if node.energy > energy_max:
                self.ts = node
                energy_max = node.energy

        self.executeExactTSSearch()
        chkfile = os.path.join(self.output_dir, 'chkf.chk')
        self.computeFrequencies(chkfile)
        correct_reac, correct_prod = self.executeIRC(chkfile)

        if correct_reac:
            if not reac_opt_success:
                reac = self.irc[0]
                reac_opt, reac_opt_success = self.optimizeNode('irc_reac', reac)

                if reac_opt_success:
                    self.reactant = reac_opt
                    reac_energy = self.reactant.energy
                    writeNode(self.reactant, 'reac', self.output_dir)

            if not correct_prod or not prod_opt_success:
                prod = self.irc[-1]
                prod_opt, prod_opt_success = self.optimizeNode('irc_prod', prod)

                if prod_opt_success:
                    self.product = prod_opt
                    prod_energy = self.product.energy
                    writeNode(self.product, 'prod', self.output_dir)

            if reac_opt_success:
                self.barrier = (self.ts.energy - reac_energy) * constants.hartree_to_kcal_per_mol

                if prod_opt_success:
                    self.dH = (prod_energy - reac_energy) * constants.hartree_to_kcal_per_mol

        self.finalize(start_time, correct_reac, correct_prod)

    def finalize(self, start_time, correct_reac, correct_prod):
        """
        Finalize the job.
        """
        fsm_energies = ['{:.1f}'.format((node.energy - self.reactant.energy) * constants.hartree_to_kcal_per_mol)
                        for node in self.fsm]
        irc_energies = ['{:.1f}'.format((node.energy - self.reactant.energy) * constants.hartree_to_kcal_per_mol)
                        for node in self.irc]
        self.logger.info('\nFSM path energies: ' + ' '.join(fsm_energies))
        self.logger.info('IRC path energies: ' + ' '.join(irc_energies))

        if self.barrier is not None:
            self.logger.info('Barrier height = {:.2f} kcal/mol'.format(self.barrier))
        elif correct_reac:
            self.barrier = (self.ts.energy - self.reactant.energy) * constants.hartree_to_kcal_per_mol

            if self.irc is not None:
                barrier_irc = (self.ts.energy - self.irc[0].energy) * constants.hartree_to_kcal_per_mol
                self.barrier = max(self.barrier, barrier_irc)

            self.logger.info('Barrier height (estimate) = {:.2f} kcal/mol'.format(self.barrier))

        if self.dH is not None:
            if self.dH < self.barrier:
                self.logger.info('Reaction energy = {:.2f} kcal/mol'.format(self.dH))

                if not correct_prod:
                    self.logger.info('Note: Reaction energy is based on unintended product')

        self.logger.info('\nTS search terminated on ' + time.asctime())
        self.logger.info('Total TS search run time: {:.1f} s'.format(time.time() - start_time))
        self.logger.info('Total number of gradient evaluations: {}'.format(self.ngrad))

    @util.logStartAndFinish
    @util.timeFn
    def optimizeReactant(self):
        """
        Optimize reactant geometry and set reactant energy. The optimization is
        done separately for each molecule in the reactant structure. Return
        `True` if successful, `False` otherwise.
        """
        success = True
        name = 'reac_opt'
        reac_mol = self.reactant.toMolecule()
        reac_cmat = self.reactant.toConnectivityMat()
        reac_node = self.reactant.copy()

        try:
            ngrad = reac_mol.optimizeGeometry(self.Qclass, name=name, **self.kwargs)
        except QuantumError as e:
            success = False
            self.logger.warning('Optimization of reactant structure was unsuccessful')
            self.logger.info('Error message: {}'.format(e))

            # Read number of gradients even if the optimization failed
            ngrad = 0
            for logname in glob.glob('{}*.log'.format(name)):
                q = self.Qclass(logfile=os.path.join(self.output_dir, logname))
                ngrad += q.getNumGrad()

            self.logger.info('\nNumber of gradient evaluations during failed reactant optimization: {}'.format(ngrad))
            self.logger.info('Proceeding with force field or partially optimized geometry\n')
            self.reactant = reac_mol.toNode()
        else:
            self.reactant = reac_mol.toNode()
            self.logger.info('Optimized reactant structure:\n' + str(self.reactant))
            self.logger.info('Energy ({}) = {}'.format(self.kwargs['theory'].upper(), self.reactant.energy))
            self.logger.info('\nNumber of gradient evaluations during reactant optimization: {}\n'.format(ngrad))

        reac_cmat_new = self.reactant.toConnectivityMat()
        if not np.array_equal(reac_cmat, reac_cmat_new):
            success = False
            self.logger.warning('Optimized geometry has wrong connectivity and will not be used\n')
            self.reactant = reac_node

        if success:
            writeNode(self.reactant, 'reac', self.output_dir)

        self.ngrad += ngrad
        return success

    @util.logStartAndFinish
    @util.timeFn
    def optimizeProduct(self):
        """
        Optimize product geometry and set product energy. The optimization is
        done separately for each molecule in the product structure. Return
        `True` if successful, `False` otherwise.
        """
        success = True
        name = 'prod_opt'
        prod_mol = self.product.toMolecule()
        prod_cmat = self.product.toConnectivityMat()
        prod_node = self.product.copy()

        try:
            ngrad = prod_mol.optimizeGeometry(self.Qclass, name=name, **self.kwargs)
        except QuantumError as e:
            success = False
            self.logger.warning('Optimization of product structure was unsuccessful')
            self.logger.info('Error message: {}'.format(e))

            # Read number of gradients even if the optimization failed
            ngrad = 0
            for logname in glob.glob('{}*.log'.format(name)):
                q = self.Qclass(logfile=os.path.join(self.output_dir, logname))
                ngrad += q.getNumGrad()

            self.logger.info('\nNumber of gradient evaluations during failed product optimization: {}'.format(ngrad))
            self.logger.info('Proceeding with force field or partially optimized geometry\n')
            self.product = prod_mol.toNode()
        else:
            self.product = prod_mol.toNode()
            self.logger.info('Optimized product structure:\n' + str(self.product))
            self.logger.info('Energy ({}) = {}'.format(self.kwargs['theory'].upper(), self.product.energy))
            self.logger.info('\nNumber of gradient evaluations during product optimization: {}\n'.format(ngrad))

        prod_cmat_new = self.product.toConnectivityMat()
        if not np.array_equal(prod_cmat, prod_cmat_new):
            success = False
            self.logger.warning('Optimized geometry has wrong connectivity and will not be used\n')
            self.product = prod_node

        if success:
            writeNode(self.product, 'prod', self.output_dir)

        self.ngrad += ngrad
        return success

    @util.logStartAndFinish
    @util.timeFn
    def optimizeNode(self, name, node):
        """
        Optimize copy of node and set energy. The optimization is done
        separately for each molecule in the structure. Return node copy and
        return `True` if successful, `False` otherwise.
        """
        success = True
        mol = node.toMolecule()
        cmat_old = node.toConnectivityMat()

        try:
            ngrad = mol.optimizeGeometry(self.Qclass, name=name, **self.kwargs)
        except QuantumError as e:
            success = False
            self.logger.warning('Optimization of structure was unsuccessful')
            self.logger.info('Error message: {}'.format(e))

            # Read number of gradients even if the optimization failed
            ngrad = 0
            for logname in glob.glob('{}*.log'.format(name)):
                q = self.Qclass(logfile=os.path.join(self.output_dir, logname))
                ngrad += q.getNumGrad()

            self.logger.info('\nNumber of gradient evaluations during failed optimization: {}'.format(ngrad))
            node_new = mol.toNode()
        else:
            node_new = mol.toNode()
            self.logger.info('Optimized structure:\n' + str(node_new))
            self.logger.info('Energy ({}) = {}'.format(self.kwargs['theory'].upper(), node_new.energy))
            self.logger.info('\nNumber of gradient evaluations during optimization: {}\n'.format(ngrad))

            cmat_new = node_new.toConnectivityMat()
            if not np.array_equal(cmat_old, cmat_new):
                success = False
                self.logger.warning('Optimized geometry has wrong connectivity\n')

        self.ngrad += ngrad
        return node_new, success

    def executeStringMethod(self):
        """
        Run the string method with the options specified in `kwargs`.
        """
        fsm = FSM(self.reactant, self.product, logger=self.logger, **self.kwargs)
        try:
            self.fsm = fsm.execute()
        except QuantumError as e:
            self.ngrad += fsm.ngrad
            self.logger.error('String method failed and terminated with the message: {}'.format(e))
            self.logger.info('Number of gradient evaluations during failed string method: {}'.format(fsm.ngrad))
            self.logger.info('Total number of gradient evaluations: {}'.format(self.ngrad))
            raise TSError('TS search failed during string method')

        self.ngrad += fsm.ngrad

        filepath = os.path.join(self.output_dir, 'FSMpath.png')
        drawPath(self.fsm, filepath)

    @util.logStartAndFinish
    @util.timeFn
    def executeExactTSSearch(self):
        """
        Run the exact transition state search and update `self.ts`.
        """
        name = 'TSopt'
        self.logger.info('Initial TS structure:\n' + str(self.ts) + '\nEnergy = ' + str(self.ts.energy))

        try:
            ngrad = self.ts.optimizeGeometry(self.Qclass, ts=True, name=name, **self.kwargs)
        except QuantumError as e:
            self.logger.error('Exact TS search did not succeed and terminated with the message: {}'.format(e))

            # Read number of gradients even if the optimization failed
            q = self.Qclass(logfile=os.path.join(self.output_dir, name + '.log'))
            ngrad = q.getNumGrad()
            self.ngrad += ngrad
            self.logger.info('\nNumber of gradient evaluations during failed TS search: {}\n'.format(ngrad))
            self.logger.info('Total number of gradient evaluations: {}'.format(self.ngrad))

            raise TSError('TS search failed during exact TS search')

        with open(os.path.join(self.output_dir, 'ts.out'), 'w') as f:
            f.write('Transition state:\n')
            f.write('Energy ({}) = {}\n'.format(self.kwargs['theory'].upper(), self.ts.energy))
            f.write(str(self.ts) + '\n')

        self.logger.info('Optimized TS structure:\n{}\nEnergy ({}) = {}'.format(self.ts,
                                                                                self.kwargs['theory'].upper(),
                                                                                self.ts.energy))
        self.logger.info('\nNumber of gradient evaluations during exact TS search: {}\n'.format(ngrad))
        self.ngrad += ngrad

    @util.logStartAndFinish
    @util.timeFn
    def computeFrequencies(self, chkfile):
        """
        Run a frequency calculation using the exact TS geometry.
        """
        try:
            nimag, ngrad = self.ts.computeFrequencies(self.Qclass, name='freq', chkfile=chkfile, **self.kwargs)
        except QuantumError as e:
            self.logger.error('Frequency calculation did not succeed and terminated with the message: {}'.format(e))
            self.logger.info('Total number of gradient evaluations: {}'.format(self.ngrad))
            raise TSError('TS search failed during frequency calculation')

        if nimag != 1:
            self.logger.error('Number of imaginary frequencies is different than 1. Geometry is not a TS!')
            self.logger.info('Total number of gradient evaluations: {}'.format(self.ngrad))
            raise TSError('TS search failed due to wrong number of imaginary frequencies')

        self.logger.info('Number of gradient evaluations during frequency calculation: {}\n'.format(ngrad))
        self.ngrad += ngrad

    @util.logStartAndFinish
    @util.timeFn
    def executeIRC(self, chkfile):
        """
        Run an IRC calculation using the exact TS geometry and save the path to
        `self.irc`. Return two booleans indicating whether or not the correct
        reactant and product were found.
        """
        chkf_name, chkf_ext = os.path.splitext(chkfile)
        chkfile_copy = chkf_name + '_copy' + chkf_ext
        shutil.copyfile(chkfile, chkfile_copy)
        forward_path, forward_ngrad = self._runOneDirectionalIRC('IRC_forward', 'forward', chkfile)
        reverse_path, reverse_ngrad = self._runOneDirectionalIRC('IRC_reverse', 'reverse', chkfile_copy)
        ngrad = forward_ngrad + reverse_ngrad

        # Check if endpoints correspond to reactant and product based on connectivity matrices
        # and try to orient IRC path so that it runs from reactant to product
        self.logger.info('Begin IRC endpoint check...')
        reac_cmat = self.reactant.toConnectivityMat()
        prod_cmat = self.product.toConnectivityMat()
        irc_end_1_cmat = forward_path[-1].toConnectivityMat()
        irc_end_2_cmat = reverse_path[-1].toConnectivityMat()

        correct_reac, correct_prod = False, False
        if np.array_equal(reac_cmat, irc_end_1_cmat):
            self.irc = forward_path[::-1] + [self.ts] + reverse_path
            correct_reac = True
        elif np.array_equal(reac_cmat, irc_end_2_cmat):
            self.irc = reverse_path[::-1] + [self.ts] + forward_path
            correct_reac = True
        else:
            self.irc = forward_path[::-1] + [self.ts] + reverse_path

        if np.array_equal(prod_cmat, irc_end_1_cmat):
            correct_prod = True
            if not correct_reac:
                self.irc = reverse_path[::-1] + [self.ts] + forward_path
        elif np.array_equal(prod_cmat, irc_end_2_cmat):
            correct_prod = True

        if correct_reac and correct_prod:
            self.logger.info('IRC check was successful. The IRC path endpoints correspond to the reactant and product.')
        elif correct_reac:
            self.logger.warning('IRC check was unsuccessful. Wrong product!')
        elif correct_prod:
            self.logger.warning('IRC check was unsuccessful. Wrong reactant!')
        else:
            self.logger.warning('IRC check was unsuccessful. Wrong reactant and product!')

        if np.array_equal(reac_cmat, prod_cmat):
            self.logger.warning('Reactant and product are the same! Conformational saddle point was found.')

        reactant_smi = self.reactant.toSMILES()
        product_smi = self.product.toSMILES()
        irc_end_1_smi = self.irc[0].toSMILES()
        irc_end_2_smi = self.irc[-1].toSMILES()
        self.logger.info('Coordinates converted to SMILES:')
        self.logger.info('Reactant: {}\nProduct: {}\nIRC endpoint 1: {}\nIRC endpoint 2: {}'.
                         format(reactant_smi, product_smi, irc_end_1_smi, irc_end_2_smi))

        with open(os.path.join(self.output_dir, 'irc.out'), 'w') as f:
            for node_num, node in enumerate(self.irc):
                f.write('Node ' + str(node_num + 1) + ':\n')
                f.write('Energy = ' + str(node.energy) + '\n')
                f.write(str(node) + '\n')

        self.logger.info('IRC path endpoints:\n' + str(self.irc[0]) + '\n****\n' + str(self.irc[-1]) + '\n')
        self.logger.info('\nNumber of gradient evaluations during IRC calculation: {}\n'.format(ngrad))
        self.ngrad += ngrad

        filepath = os.path.join(self.output_dir, 'IRCpath.png')
        drawPath(self.irc, filepath)

        return correct_reac, correct_prod

    @util.timeFn
    def _runOneDirectionalIRC(self, name, direction, chkfile):
        """
        Run an IRC calculation in the forward or reverse direction and return
        the path together with the number of gradient evaluations. The results
        are returned even if an error was raised during the calculation.
        """
        try:
            ircpath, ngrad = self.ts.getIRCpath(
                self.Qclass, name=name, direction=direction, freq=False, chkfile=chkfile, **self.kwargs
            )
        except QuantumError as e:
            self.logger.warning(
                '{} IRC calculation did not run to completion and terminated with the message: {}'.format(direction, e)
            )

            q = self.Qclass(logfile=os.path.join(self.output_dir, name + '.log'))
            try:
                path = q.getIRCpath()
                ngrad = q.getNumGrad()
            except QuantumError as e:
                self.logger.error('TS search failed reading IRC logfile: {}\n'.format(e))
                self.barrier = (self.ts.energy - self.reactant.energy) * constants.hartree_to_kcal_per_mol
                self.logger.info('Barrier height (estimate) = {:.2f} kcal/mol\n'.format(self.barrier))
                self.logger.info('Total number of gradient evaluations: {}'.format(self.ngrad))
                raise TSError('TS search failed reading IRC logfile: {}'.format(e))

            q.clearChkfile()
            ircpath = []
            for element in path:
                node = Node(element[0], self.reactant.atoms, self.reactant.multiplicity)
                node.energy = element[1]
                ircpath.append(node)

        return ircpath, ngrad

    def logHeader(self):
        """
        Output a log file header containing identifying information about the
        TS search.
        """
        self.logger.info('#######################################################################')
        self.logger.info('############################## TS SEARCH ##############################')
        self.logger.info('#######################################################################')
        self.logger.info('Reactant SMILES: ' + str(self.reactant.toSMILES()))
        self.logger.info('Reactant structure:\n' + str(self.reactant))
        self.logger.info('Product SMILES: ' + str(self.product.toSMILES()))
        self.logger.info('Product structure:\n' + str(self.product))
        self.logger.info('#######################################################################\n')

###############################################################################

def drawPath(nodepath, filepath):
    """
    Make a plot of the path energies, where `nodepath` is a list of
    :class:`node.Node` objects.
    """
    reac_energy = nodepath[0].energy
    energies = [(node.energy - reac_energy) * constants.hartree_to_kcal_per_mol for node in nodepath]
    n = range(1, len(energies) + 1)

    plt.figure()
    line = plt.plot(n, energies)
    plt.setp(line, c='b', ls='-', lw=2.0, marker='.', mec='k', mew=1.0, mfc='w', ms=17.0)
    plt.xlabel('Node')
    plt.ylabel('Energy (kcal/mol)')
    plt.grid(True)
    plt.savefig(filepath)

def writeNode(node, name, out_dir):
    """
    Write node geometry to file.
    """
    with open(os.path.join(out_dir, name + '.out'), 'w') as f:
        f.write(name + '\n')
        f.write('Energy = {}\n'.format(node.energy))
        f.write(str(node) + '\n')

###############################################################################

if __name__ == '__main__':
    import argparse

    from ard.main import readInput

    # Set up parser for reading the input filename from the command line
    parser = argparse.ArgumentParser(description='A transition state search')
    parser.add_argument('-n', '--nproc', default=1, type=int, metavar='N', help='number of processors')
    parser.add_argument('-m', '--mem', default=2000, type=int, metavar='M', help='memory requirement')
    parser.add_argument('file', type=str, metavar='infile', help='an input file describing the job options')
    args = parser.parse_args()

    # Read input file
    input_file = os.path.abspath(args.file)
    options = readInput(input_file)

    # Set output directory
    output_dir = os.path.abspath(os.path.dirname(input_file))
    options['output_dir'] = output_dir

    # Set number of processors and memory
    options['nproc'] = args.nproc
    options['mem'] = str(args.mem) + 'mb'

    # Execute job
    tssearch = TSSearch(**options)
    tssearch.execute()
