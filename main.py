#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filtering reactions, generating 3D geometries, running transition state searches.

Workflow:
- Estimate thermo of reactant and products
- Filter reactions
- Generate FF 3D geometries
- Run TS searches
- Optimize product (and successful reactions??, frequency calcs??)
"""

import sys
import os
import logging
import functools
import subprocess
import time

import gen3D
from pgen import Generate
from rmgpy import settings
from rmgpy.data.thermo import ThermoDatabase

try:
    from scoop import futures
except ImportError:
    logging.info('Could not import SCOOP')

###############################################################################

class ARD(object):
    """
    Automatic reaction discovery class. Filters reactions based on estimated
    thermo of reactant and products, generates force field 3D geometries, and
    runs transition state searches.
    The attributes are:

    =============== ====================== ====================================
    Attribute       Type                   Description
    =============== ====================== ====================================
    `reac_smi`      ``string``             A valid SMILES string describing the reactant structure
    `reactant`      :class:`node.Node`     A node object describing the reactant structure
    `nbreak`        ``int``                The maximum number of bonds that may be broken
    `nform`         ``int``                The maximum number of bonds that may be formed
    `dH_cutoff`     ``float``              Heat of reaction cutoff (kcal/mol) for reactions that are too endothermic
    `forcefield`    ``str``                The force field for 3D geometry generation
    `output_dir`    ``str``                The path to the output directory
    =============== ====================== ====================================

    """

    def __init__(self, reac_smi, nbreak=3, nform=3, dH_cutoff=20.0, forcefield='mmff94', output_dir='', **kwargs):
        self.reac_smi = reac_smi
        self.nbreak = int(nbreak)
        self.nform = int(nform)
        self.dH_cutoff = float(dH_cutoff)
        self.forcefield = forcefield
        self.output_dir = output_dir
        self.reactant = None

    def initialize(self):
        """
        Initialize the ARD job. Return the :class:`gen3D.Molecule` object for
        the reactant.
        """
        logging.info('\nARD initiated on ' + time.asctime() + '\n')
        reac_mol = self.generateReactant3D()
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
        reac_mol.gen3D()
        self.reactant = reac_mol.toNode()
        return reac_mol

    def execute(self, **kwargs):
        """
        Execute the automatic reaction discovery procedure.
        """
        # Initialize job
        start_time = time.time()
        reac_mol = self.initialize()

        # Generate products
        gen = Generate(reac_mol)
        gen.generateProducts(nbreak=self.nbreak, nform=self.nform)
        prod_mols = gen.prod_mols

        # Load thermo database
        thermo_db = ThermoDatabase()
        thermo_db.load(os.path.join(settings['database.directory'], 'thermo'))
        thermo_db.libraryOrder = ['primaryThermoLibrary', 'NISTThermoLibrary', 'DFT_QCI_thermo', 'CBS_QB3_1dHR',
                                  'KlippensteinH2O2', 'GRI-Mech3.0-N', 'thermo_DFT_CCSDTF12_BAC']

        # Filter reactions based on standard heat of reaction
        H298_reac = reac_mol.getH298(thermo_db)
        prod_mols = [mol for mol in prod_mols if self.filterThreshold(H298_reac, mol, thermo_db)]

        # Generate 3D geometries in parallel
        if 'scoop.futures' in sys.modules:
            futures.map(functools.partial(gen3D.gen3D, forcefield=self.forcefield), prod_mols)
        else:
            for i, mol in enumerate(prod_mols):
                mol.gen3D(self.forcefield)

        # Run TS searches
        procs = []
        tssearch_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tssearch.py')
        logging.info('\nFeasible products:')

        for rxn, mol in enumerate(prod_mols):
            output_dir = os.path.join(self.output_dir, '{0:03d}'.format(rxn))
            os.mkdir(output_dir)
            kwargs['output_dir'] = output_dir

            product = mol.toNode()
            logging.info('Reaction {0}:\n{1}\n{2}\n'.format(rxn, mol.write().strip(), product))
            input_path = self.makeInputFile(product, **kwargs)

            procs.append(subprocess.Popen(['python', tssearch_path, input_path]))

        logging.info('Running TS searches...')
        for proc in procs:
            proc.wait()

        # Finalize
        self.finalize(start_time)

    @staticmethod
    def finalize(start_time):
        """
        Finalize the job.
        """
        logging.info('\nARD terminated on ' + time.asctime())
        logging.info('Total ARD run time: {0:.1f} s'.format(time.time() - start_time))

    def filterThreshold(self, H298_reac, prod_mol, thermo_db=None):
        """
        Filter threshold based on standard enthalpies of formation of reactants
        and products. Returns `True` if the heat of reaction is less than
        `self.dH_cutoff`, `False` otherwise.
        """
        # Calculate enthalpy of formation and heat of reaction
        H298_prod = prod_mol.getH298(thermo_db)
        dH = H298_prod - H298_reac

        if dH < self.dH_cutoff:
            return True
        return False

    def makeInputFile(self, product, **kwargs):
        """
        Create input file for TS search and return path to file.
        """
        path = os.path.join(kwargs['output_dir'], 'input.txt')

        with open(path, 'w') as f:
            for key, val in kwargs.iteritems():
                if key not in ('reac_smi', 'nbreak', 'nform', 'output_dir'):
                    f.write('{0}  {1}\n'.format(key, val))
            f.write('\n')
            f.write('geometry (\n0 {0}\n{1}\n****\n{2}\n)\n'.format(self.reactant.multiplicity, self.reactant, product))
            f.write('\n')

        return path

    def logHeader(self):
        """
        Output a log file header.
        """
        logging.info('######################################################################')
        logging.info('#################### AUTOMATIC REACTION DISCOVERY ####################')
        logging.info('######################################################################')
        logging.info('Reactant SMILES: ' + self.reac_smi)
        logging.info('Reactant coordinates:\n' + str(self.reactant))
        logging.info('Maximum number of bonds to be broken: ' + str(self.nbreak))
        logging.info('Maximum number of bonds to be formed: ' + str(self.nform))
        logging.info('Heat of reaction cutoff: {0:.1f} kcal/mol'.format(self.dH_cutoff))
        logging.info('Force field for 3D structure generation: ' + self.forcefield)
        logging.info('######################################################################\n')
