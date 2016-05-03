#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`FSM` for executing a transition state search using the
freezing string method.
"""

from node import Node

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
    `nnode`           ``int``                The number of nodes
    `gaussian_ver`    ``str``                The version of Gaussian available to the user
    `level_of_theory` ``str``                The level of theory (method/basis) for the quantum calculations
    `nproc`           ``int``                The number of processors available for the FSM calculation
    `mem`             ``str``                The memory requirements
    =============== ======================== ==================================

    """

    def __init__(self, reactant, product, nsteps=4, nnode=15,
                 gaussian_ver='g09', level_of_theory='um062x/cc-pvtz', nproc=32, mem='1500mb'):
        self.reactant = reactant
        self.product = product
        self.nsteps = nsteps
        self.nnode = nnode
        self.gaussian_ver = gaussian_ver
        self.level_of_theory = level_of_theory
        self.nproc = nproc
        self.mem = mem

    def execute(self):
        pass
