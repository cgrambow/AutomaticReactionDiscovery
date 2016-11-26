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
Contains the :class:`ARD` for running an automatic reaction discovery. This
includes filtering reactions, generating 3D geometries, and running transition
state searches.
"""

from __future__ import print_function

import logging
import os
import time

import pybel
from rmgpy import settings
from rmgpy.data.thermo import ThermoDatabase

import constants
import gen3D
import util
from quantum import QuantumError
from node import Node
from pgen import Generate

###############################################################################

class ARD(object):
    """
    Automatic reaction discovery class. Filters reactions based on estimated
    thermo of reactant and products, generates force field 3D geometries, and
    runs transition state searches.
    The attributes are:

    =============== ======================== ==================================
    Attribute       Type                     Description
    =============== ======================== ==================================
    `reac_smi`      ``str``                  A valid SMILES string describing the reactant structure
    `nbreak`        ``int``                  The maximum number of bonds that may be broken
    `nform`         ``int``                  The maximum number of bonds that may be formed
    `dh_cutoff`     ``float``                Heat of reaction cutoff (kcal/mol) for reactions that are too endothermic
    `theory_low`    ``str``                  Low level of theory for pre-optimizations
    `forcefield`    ``str``                  The force field for 3D geometry generation
    `distance`      ``float``                The initial distance between molecules
    `Qclass`        ``class``                A class representing the quantum software
    `output_dir`    ``str``                  The path to the output directory
    `logger`        :class:`logging.Logger`  The main logger
    =============== ======================== ==================================

    """

    def __init__(self, reac_smi, nbreak=3, nform=3, dh_cutoff=20.0, theory_low=None,
                 forcefield='mmff94', distance=3.5, output_dir='', **kwargs):
        self.reac_smi = reac_smi
        self.nbreak = int(nbreak)
        self.nform = int(nform)
        self.dh_cutoff = float(dh_cutoff)
        self.theory_low = theory_low
        self.forcefield = forcefield
        self.distance = float(distance)
        qprog = kwargs.get('qprog', 'gau')
        self.Qclass = util.assignQclass(qprog)
        self.output_dir = output_dir
        log_level = logging.INFO
        self.logger = util.initializeLog(log_level, os.path.join(self.output_dir, 'ARD.log'), logname='main')

    def initialize(self):
        """
        Initialize the ARD job. Return the :class:`gen3D.Molecule` object for
        the reactant.
        """
        self.logger.info('\nARD initiated on ' + time.asctime() + '\n')
        reac_mol = self.generateReactant3D()
        self.reac_smi = reac_mol.write('can').strip()
        self.logHeader()
        return reac_mol

    def generateReactant3D(self):
        """
        Convert the reactant SMILES to a :class:`gen3D.Molecule` object and
        generate a 3D geometry. Return the object and store the corresponding
        :class:`node.Node` object in `self.reactant`.
        """
        reac_mol = gen3D.readstring('smi', self.reac_smi)
        reac_mol.addh()
        reac_mol.gen3D(forcefield=self.forcefield)
        return reac_mol

    def preopt(self, mol, **kwargs):
        """
        Optimize `mol` at the low level of theory and return its energy in
        kcal/mol. The optimization is done separately for each molecule in the
        structure. If the optimization was unsuccessful or if no low level of
        theory was specified, `None` is returned.
        """
        if self.theory_low is None:
            return None

        kwargs_copy = kwargs.copy()
        kwargs_copy['theory'] = self.theory_low

        try:
            mol.optimizeGeometry(self.Qclass, name='preopt', **kwargs_copy)
        except QuantumError:
            return None

        return mol.energy * constants.hartree_to_kcal_per_mol

    def execute(self, **kwargs):
        """
        Execute the automatic reaction discovery procedure.
        """
        start_time = time.time()
        reac_mol = self.initialize()
        # self.optimizeReactant(reac_mol, **kwargs)

        gen = Generate(reac_mol)
        self.logger.info('Generating all possible products...')
        gen.generateProducts(nbreak=self.nbreak, nform=self.nform)
        prod_mols = gen.prod_mols
        self.logger.info('{} possible products generated\n'.format(len(prod_mols)))

        # Load thermo database and choose which libraries to search
        thermo_db = ThermoDatabase()
        thermo_db.load(os.path.join(settings['database.directory'], 'thermo'))
        thermo_db.libraryOrder = ['primaryThermoLibrary', 'NISTThermoLibrary', 'thermo_DFT_CCSDTF12_BAC',
                                  'CBS_QB3_1dHR', 'DFT_QCI_thermo', 'KlippensteinH2O2', 'GRI-Mech3.0-N', ]

        # Filter reactions based on standard heat of reaction
        H298_reac = reac_mol.getH298(thermo_db)
        self.logger.info('Filtering reactions...')
        prod_mols_filtered = [mol for mol in prod_mols if self.filterThreshold(H298_reac, mol, thermo_db, **kwargs)]
        self.logger.info('{} products remaining\n'.format(len(prod_mols_filtered)))

        # Generate 3D geometries
        if prod_mols_filtered:
            self.logger.info('Feasible products:\n')
            rxn_dir = util.makeOutputSubdirectory(self.output_dir, 'reactions')

            # These two lines are required so that new coordinates are
            # generated for each new product. Otherwise, Open Babel tries to
            # use the coordinates of the previous molecule if it is isomorphic
            # to the current one, even if it has different atom indices
            # participating in the bonds. a hydrogen atom is chosen
            # arbitrarily, since it will never be the same as any of the
            # product structures.
            Hatom = gen3D.readstring('smi', '[H]')
            ff = pybel.ob.OBForceField.FindForceField(self.forcefield)

            reac_mol_copy = reac_mol.copy()
            for rxn, mol in enumerate(prod_mols_filtered):
                mol.gen3D(forcefield=self.forcefield, make3D=False)
                arrange3D = gen3D.Arrange3D(reac_mol, mol)
                msg = arrange3D.arrangeIn3D()
                if msg != '':
                    self.logger.info(msg)

                ff.Setup(Hatom.OBMol)  # Ensures that new coordinates are generated for next molecule (see above)
                reac_mol.gen3D(make3D=False)
                ff.Setup(Hatom.OBMol)
                mol.gen3D(make3D=False)
                ff.Setup(Hatom.OBMol)

                reactant = reac_mol.toNode()
                product = mol.toNode()

                rxn_num = '{:04d}'.format(rxn)
                rxn_name = 'rxn' + rxn_num
                output_dir = util.makeOutputSubdirectory(rxn_dir, rxn_num)
                kwargs['output_dir'] = output_dir
                kwargs['logname'] = rxn_name

                self.logger.info('Product {}: {}\n{}\n****\n{}\n'.format(rxn, product.toSMILES(), reactant, product))
                self.makeInputFile(reactant, product, **kwargs)

                reac_mol.setCoordsFromMol(reac_mol_copy)
        else:
            self.logger.info('No feasible products found')

        # Finalize
        self.finalize(start_time)

    def finalize(self, start_time):
        """
        Finalize the job.
        """
        self.logger.info('\nARD terminated on ' + time.asctime())
        self.logger.info('Total ARD run time: {:.1f} s'.format(time.time() - start_time))

    def filterThreshold(self, H298_reac, prod_mol, thermo_db, **kwargs):
        """
        Filter threshold based on standard enthalpies of formation of reactants
        and products. Returns `True` if the heat of reaction is less than
        `self.dh_cutoff`, `False` otherwise.
        """
        H298_prod = prod_mol.getH298(thermo_db)
        dH = H298_prod - H298_reac

        if dH < self.dh_cutoff:
            return True
        return False

    @staticmethod
    def makeInputFile(reactant, product, **kwargs):
        """
        Create input file for TS search and return path to file.
        """
        path = os.path.join(kwargs['output_dir'], 'input.txt')

        with open(path, 'w') as f:
            for key, val in kwargs.iteritems():
                if key not in ('reac_smi', 'nbreak', 'nform', 'dh_cutoff', 'forcefield', 'distance', 'theory_low',
                               'output_dir'):
                    f.write('{0}  {1}\n'.format(key, val))
            f.write('\n')
            f.write('geometry (\n0 {0}\n{1}\n****\n{2}\n)\n'.format(reactant.multiplicity, reactant, product))

        return path

    def logHeader(self):
        """
        Output a log file header.
        """
        self.logger.info('######################################################################')
        self.logger.info('#################### AUTOMATIC REACTION DISCOVERY ####################')
        self.logger.info('######################################################################')
        self.logger.info('Reactant SMILES: ' + self.reac_smi)
        self.logger.info('Maximum number of bonds to be broken: ' + str(self.nbreak))
        self.logger.info('Maximum number of bonds to be formed: ' + str(self.nform))
        self.logger.info('Heat of reaction cutoff: {:.1f} kcal/mol'.format(self.dh_cutoff))
        self.logger.info('Force field for 3D structure generation: ' + self.forcefield)
        self.logger.info('######################################################################\n')

###############################################################################

def readInput(input_file):
    """
    Read input parameters from a file. It is assumed that the input file
    contains key-value pairs in the form "key value" on separate lines. If a
    keyword containing the string 'geometry' is encountered, the corresponding
    geometries are read in the form (example for methane dissociation):
        geometry (
        0 1
        C                 -0.03144385    0.03144654    0.00041162
        H                  0.32521058   -0.97736346    0.00041162
        H                  0.32522899    0.53584473    0.87406313
        H                  0.32522899    0.53584473   -0.87323988
        H                 -1.10144385    0.03145972    0.00041162
        ****
        C                 -0.36061854   -0.43406458    0.80670792
        H                  0.14377652   -1.32573293    0.49781771
        H                  0.14379613    0.27926689    1.42446520
        H                  0.56523315    0.87525286   -1.46111753
        H                 -1.36941886   -0.25571437    0.49781777
        )
    If '#' is found in a line, the rest of the line will be ignored.

    A dictionary containing all input parameters and their values is returned.
    """
    # Allowed keywords
    keys = ('reac_smi', 'nbreak', 'nform', 'dh_cutoff', 'forcefield', 'logname',
            'nsteps', 'nnode', 'lsf', 'tol', 'gtol', 'nlstnodes',
            'qprog', 'theory', 'theory_low')

    # Read all data from file
    with open(input_file, 'r') as f:
        input_data = f.read().splitlines()

    # Create dictionary
    input_dict = {}

    # Read geometry block
    read = False
    geometry = []
    sep_loc = -1
    for line in input_data:
        if line.strip().startswith(')'):
            break
        if read and not line.strip().startswith('#') and line != '':
            geometry.append(line)
            if line.strip().startswith('*'):
                sep_loc = len(geometry) - 1
        elif 'geometry' in line:
            read = True

    if geometry:
        if sep_loc == -1:
            raise Exception('Incorrect geometry specification')

        # Extract multiplicity, atoms, and geometries
        multiplicity = geometry[0].split()[1]
        reactant = geometry[1:sep_loc]
        reac_atoms = [line.split()[0] for line in reactant]
        reac_geo = [[float(coord) for coord in line.split()[1:4]] for line in reactant]
        product = geometry[sep_loc + 1:]
        prod_atoms = [line.split()[0] for line in product]
        prod_geo = [[float(coord) for coord in line.split()[1:4]] for line in product]

        # Create nodes
        reac_node = Node(reac_geo, reac_atoms, multiplicity)
        prod_node = Node(prod_geo, prod_atoms, multiplicity)

        # Add to dictionary
        input_dict['reactant'] = reac_node
        input_dict['product'] = prod_node

    # Extract remaining keywords and values
    for line in input_data:
        if line != '' and not line.strip().startswith('#'):
            key = line.split()[0].lower()
            if key not in keys:
                continue
            if line.split()[1] == '=':
                input_dict[key] = line.split()[2]
            else:
                input_dict[key] = line.split()[1]

    # Check if valid method was specified and default to FSM
    try:
        method = input_dict['method'].lower()
    except KeyError:
        input_dict['method'] = 'fsm'
    except AttributeError:
        raise Exception('Invalid method')
    else:
        if method != 'gsm' and method != 'fsm':
            raise Exception('Invalid method: {}'.format(method))

    return input_dict
