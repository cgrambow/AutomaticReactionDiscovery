#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`ARD` for running an automatic reaction discovery. This
includes filtering reactions, generating 3D geometries, and running transition
state searches.
"""

import logging
import os
import subprocess
import sys
import time

from rmgpy import settings
from rmgpy.data.thermo import ThermoDatabase

import util
import gen3D
from pgen import Generate
from node import Node

try:
    from scoop.futures import map
except ImportError:
    logging.warning('Could not import SCOOP')

###############################################################################

def submitProcessAndWait(cmd, *args):
    """
    Submit process and wait for completion. Return exit code of process.
    """
    full_cmd = [cmd] + list(args)
    proc = subprocess.Popen(full_cmd)
    exit_code = proc.wait()
    return exit_code

class Copier(object):
    """
    Function object that creates picklable function, `fn`, with a constant
    value for some arguments at the beginning of the function, set as
    `self.args`. This enables using `fn` in conjunction with `map` if the
    sequence being mapped to the function does not correspond to the first
    function argument and if the function has multiple arguments.
    """
    def __init__(self, fn, *begin_args):
        self.fn = fn
        self.args = begin_args

    def __call__(self, *end_args):
        all_args = self.args + end_args  # Required in Python 2.7
        return self.fn(*all_args)

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

        # Generate 3D geometries
        for i, mol in enumerate(prod_mols):
            mol.gen3D(self.forcefield)

        # Run TS searches
        input_paths = []
        tssearch_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tssearch.py')
        logging.info('\nFeasible products:')

        # Create input files
        for rxn, mol in enumerate(prod_mols):
            output_dir = util.makeOutputSubdirectory(self.output_dir, '{0:03d}'.format(rxn))
            kwargs['output_dir'] = output_dir

            # Convert to :class:`node.Node` and create input file
            product = mol.toNode()
            logging.info('Reaction {0}:\n{1}\n{2}\n'.format(rxn, mol.write().strip(), product))
            input_path = self.makeInputFile(product, **kwargs)
            input_paths.append(input_path)

        # Run TS searches
        if input_paths:
            logging.info('Running TS searches...')
            exit_codes = list(map(Copier(submitProcessAndWait, sys.executable, tssearch_path), input_paths))

            successful, unsuccessful = [], []
            for rxn, exit_code in enumerate(exit_codes):
                if exit_code == 0:
                    successful.append(rxn)
                else:
                    unsuccessful.append(rxn)
            logging.info('Successful reactions: {}'.format(successful))
            logging.info('Unsuccessful reactions: {}'.format(unsuccessful))
        else:
            logging.info('No feasible products found')

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

###############################################################################

def initializeLog(level, logfile):
    """
    Configure a logger. `level` is an integer parameter specifying how much
    information is displayed in the log file. The levels correspond to those of
    the :data:`logging` module.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    logging.addLevelName(logging.CRITICAL, 'CRITICAL: ')
    logging.addLevelName(logging.ERROR, 'ERROR: ')
    logging.addLevelName(logging.WARNING, 'WARNING: ')
    logging.addLevelName(logging.INFO, '')
    logging.addLevelName(logging.DEBUG, '')

    # Create formatter
    formatter = logging.Formatter('%(levelname)s%(message)s')

    # Create file handler
    if os.path.exists(logfile):
        os.remove(logfile)
    fh = logging.FileHandler(filename=logfile)
    fh.setLevel(min(logging.DEBUG, level))
    fh.setFormatter(formatter)

    # Remove old handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Add file handler
    logger.addHandler(fh)

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
    keys = ('reac_smi', 'nbreak', 'nform', 'dH_cutoff', 'forcefield', 'method', 'nsteps', 'nnode', 'lsf',
            'tol', 'gtol', 'nlstnodes', 'qprog', 'theory', 'theory_preopt', 'reac_preopt', 'nproc', 'mem')

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
            raise Exception('Invalid method: {0}'.format(method))

    return input_dict
