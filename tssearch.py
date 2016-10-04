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
using FSM/GSM.
"""

import logging
import os
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ard.constants as constants
import ard.util as util
from ard.node import Node
from ard.quantum import QuantumError
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
    Transition state finding using FSM/GSM with subsequent exact TS search and
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
    `output_dir`   ``str``                  The path to the output directory
    `Qclass`       ``class``                A class representing the quantum software
    `kwargs`       ``dict``                 Options for FSM/GSM and quantum calculations
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
        if 'theory_preopt' in self.kwargs:
            self.preoptimizeProduct()
        self.executeStringMethod()

        energy_max = self.fsm[0].energy
        for node in self.fsm[1:-1]:
            if node.energy > energy_max:
                self.ts = node
                energy_max = node.energy

        self.executeExactTSSearch()
        chkfile = os.path.join(self.output_dir, 'chkf.chk')
        self.computeFrequencies(chkfile)
        self.executeIRC(chkfile)
        self.finalize(start_time)

    def finalize(self, start_time):
        """
        Finalize the job.
        """
        barrier = (self.ts.energy - self.reactant.energy) * constants.hartree_to_kcal_per_mol
        self.logger.info('\nBarrier height = {:.2f} kcal/mol'.format(barrier))

        self.logger.info('\nTS search terminated on ' + time.asctime())
        self.logger.info('Total TS search run time: {:.1f} s'.format(time.time() - start_time))
        self.logger.info(
            'Total number of gradient evaluations (excluding pre-optimization): {}'.format(self.ngrad)
        )

    @util.logStartAndFinish
    @util.timeFn
    def preoptimizeProduct(self):
        """
        Optimize the product geometry.
        """
        kwargs = self.kwargs.copy()
        kwargs['theory'] = kwargs['theory_preopt']
        try:
            self.product.optimizeGeometry(self.Qclass, name='preopt_prod', **kwargs)
        except QuantumError as e:
            self.logger.warning('Pre-optimization of product structure was unsuccessful')
            self.logger.info('Error message: {}'.format(e))
        else:
            self.logger.info('Optimized product structure:\n' + str(self.product))
            self.logger.info('Energy ({}) = {}'.format(kwargs['theory'], self.product.energy))

    def executeStringMethod(self):
        """
        Run the string method with the options specified in `kwargs`.
        """
        fsm = FSM(self.reactant, self.product, logger=self.logger, **self.kwargs)
        try:
            self.fsm = fsm.execute()
        except QuantumError as e:
            self.logger.error('String method failed and terminated with the message: {}'.format(e))
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
        self.logger.info('Initial TS structure:\n' + str(self.ts) + '\nEnergy = ' + str(self.ts.energy))

        try:
            ngrad = self.ts.optimizeGeometry(self.Qclass, ts=True, name='TSopt', **self.kwargs)
        except QuantumError as e:
            self.logger.error('Exact TS search did not succeed and terminated with the message: {}'.format(e))
            raise TSError('TS search failed during exact TS search')

        with open(os.path.join(self.output_dir, 'ts.out'), 'w') as f:
            f.write('Transition state:\n')
            f.write('Energy = ' + str(self.ts.energy) + '\n')
            f.write(str(self.ts) + '\n')

        self.logger.info('Optimized TS structure:\n' + str(self.ts) + '\nEnergy = ' + str(self.ts.energy))
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
            raise TSError('TS search failed during frequency calculation')

        if nimag != 1:
            self.logger.error('Number of imaginary frequencies is different than 1. Geometry is not a TS!')
            raise TSError('TS search failed due to wrong number of imaginary frequencies')

        self.logger.info('Number of gradient evaluations during frequency calculation: {}\n'.format(ngrad))
        self.ngrad += ngrad

    @util.logStartAndFinish
    @util.timeFn
    def executeIRC(self, chkfile):
        """
        Run an IRC calculation using the exact TS geometry and save the path to
        `self.irc`.
        """
        chkf_name, chkf_ext = os.path.splitext(chkfile)
        chkfile_copy = chkf_name + '_copy' + chkf_ext
        shutil.copyfile(chkfile, chkfile_copy)
        forward_path, forward_ngrad = self._runOneDirectionalIRC('IRC_forward', 'forward', chkfile)
        reverse_path, reverse_ngrad = self._runOneDirectionalIRC('IRC_reverse', 'reverse', chkfile_copy)
        ngrad = forward_ngrad + reverse_ngrad

        # Check if endpoints correspond to reactant and product
        # and try to orient IRC path so that it runs from reactant to product
        self.logger.info('Begin IRC endpoint check...')
        reactant_smi = self.reactant.toSMILES()
        product_smi = self.product.toSMILES()
        irc_end_1_smi = forward_path[-1].toSMILES()
        irc_end_2_smi = reverse_path[-1].toSMILES()

        check1, check2 = False, False
        if reactant_smi == irc_end_1_smi:
            self.irc = forward_path[::-1] + [self.ts] + reverse_path
            check1 = 1
        elif reactant_smi == irc_end_2_smi:
            self.irc = reverse_path[::-1] + [self.ts] + forward_path
            check1 = 2
        else:
            self.irc = forward_path[::-1] + [self.ts] + reverse_path
        if product_smi == irc_end_1_smi or product_smi == irc_end_2_smi:
            check2 = True

        if check1 and check2:
            self.logger.info('IRC check was successful. The IRC path endpoints correspond to the reactant and product.')
        else:
            self.logger.warning('IRC check was unsuccessful. IRC path may correspond to a different reaction.')
            self.logger.warning('Check the results manually.')

        self.logger.info('Coordinates converted to SMILES:')
        endpoint_order = [irc_end_1_smi, irc_end_2_smi]
        if check1 == 2:
            endpoint_order = [irc_end_2_smi, irc_end_1_smi]
        self.logger.info('Reactant: {0}\nProduct: {1}\nIRC endpoint 1: {2[0]}\nIRC endpoint 2: {2[1]}'.
                         format(reactant_smi, product_smi, endpoint_order))

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
        self.logger.info('Reactant structure:\n' + str(self.reactant))
        self.logger.info('Product structure:\n' + str(self.product))
        self.logger.info('#######################################################################\n')

###############################################################################

def drawPath(nodepath, filepath):
    """
    Make a plot of the path energies, where `nodepath` is a list of
    :class:`node.Node` objects.
    """
    reac_energy = nodepath[0].energy
    energies = [(node.energy - reac_energy) * 627.5095 for node in nodepath]
    n = range(1, len(energies) + 1)

    plt.figure()
    line = plt.plot(n, energies)
    plt.setp(line, c='b', ls='-', lw=2.0, marker='.', mec='k', mew=1.0, mfc='w', ms=17.0)
    plt.xlabel('Node')
    plt.ylabel('Energy (kcal/mol)')
    plt.grid(True)
    plt.savefig(filepath)

###############################################################################

if __name__ == '__main__':
    import argparse

    from ard.main import readInput

    # Set up parser for reading the input filename from the command line
    parser = argparse.ArgumentParser(description='A transition state search')
    parser.add_argument('file', type=str, metavar='FILE', help='An input file describing the job options')
    args = parser.parse_args()

    # Read input file
    input_file = os.path.abspath(args.file)
    options = readInput(input_file)

    # Set output directory
    output_dir = os.path.abspath(os.path.dirname(input_file))
    options['output_dir'] = output_dir

    # Execute job
    tssearch = TSSearch(**options)
    tssearch.execute()
