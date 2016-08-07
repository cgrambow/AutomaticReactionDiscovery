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

import glob
import logging
import os
import stat
import time

from rmgpy import settings
from rmgpy.data.thermo import ThermoDatabase

import gen3D
import util
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
    `reac_smi`      ``string``               A valid SMILES string describing the reactant structure
    `reactant`      :class:`node.Node`       A node object describing the reactant structure
    `nbreak`        ``int``                  The maximum number of bonds that may be broken
    `nform`         ``int``                  The maximum number of bonds that may be formed
    `dh_cutoff`     ``float``                Heat of reaction cutoff (kcal/mol) for reactions that are too endothermic
    `forcefield`    ``str``                  The force field for 3D geometry generation
    `distance`      ``float``                The initial distance between molecules
    `output_dir`    ``str``                  The path to the output directory
    `logger`        :class:`logging.Logger`  The main logger
    =============== ======================== ==================================

    """

    def __init__(self, reac_smi, nbreak=3, nform=3, dh_cutoff=20.0,
                 forcefield='mmff94', distance=3.5, output_dir='', **kwargs):
        self.reac_smi = reac_smi
        self.nbreak = int(nbreak)
        self.nform = int(nform)
        self.dh_cutoff = float(dh_cutoff)
        self.forcefield = forcefield
        self.distance = float(distance)
        self.output_dir = output_dir
        self.reactant = None
        log_level = logging.INFO
        self.logger = util.initializeLog(log_level, os.path.join(self.output_dir, 'ARD.log'), logname='main')

    def initialize(self):
        """
        Initialize the ARD job. Return the :class:`gen3D.Molecule` object for
        the reactant.
        """
        self.logger.info('\nARD initiated on ' + time.asctime() + '\n')
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
        reac_mol.gen3D(forcefield=self.forcefield, d=self.distance)
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
        prod_mols = [mol for mol in prod_mols if self.filterThreshold(H298_reac, mol, thermo_db=thermo_db)]

        # Generate 3D geometries (make3D is False because coordinates already exist from the reactant)
        # and make job files
        if prod_mols:
            self.logger.info('Feasible products:\n')
            rxn_dir = util.makeOutputSubdirectory(self.output_dir, 'reactions')

            try:
                example_script = glob.glob('submit.*')[0]
            except IndexError:
                raise Exception('Example submission script cannot be found')
            else:
                example_script_path = os.path.abspath(example_script)

            for rxn, mol in enumerate(prod_mols):
                mol.gen3D(forcefield=self.forcefield, d=self.distance, make3D=False)

                rxn_num = '{:03d}'.format(rxn)
                rxn_name = 'rxn' + rxn_num
                output_dir = util.makeOutputSubdirectory(rxn_dir, rxn_num)
                kwargs['output_dir'] = output_dir
                kwargs['logname'] = 'rxn' + rxn_num

                product = mol.toNode()
                self.logger.info('Reaction {}:\n{}\n{}\n'.format(rxn, mol.write().strip(), product))

                self.makeInputFile(product, **kwargs)
                job_script = makeBatchSubmissionScript(rxn_name, example_script_path, output_dir)

            job_cmd = kwargs.get('job_cmd', 'qsub')
            makeOverallSubmissionScript(job_cmd, job_script, self.output_dir)
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

    def filterThreshold(self, H298_reac, prod_mol, thermo_db=None):
        """
        Filter threshold based on standard enthalpies of formation of reactants
        and products. Returns `True` if the heat of reaction is less than
        `self.dh_cutoff`, `False` otherwise.
        """
        # Calculate enthalpy of formation and heat of reaction
        H298_prod = prod_mol.getH298(thermo_db)
        dH = H298_prod - H298_reac

        if dH < self.dh_cutoff:
            return True
        return False

    def makeInputFile(self, product, **kwargs):
        """
        Create input file for TS search and return path to file.
        """
        path = os.path.join(kwargs['output_dir'], 'input.txt')

        with open(path, 'w') as f:
            for key, val in kwargs.iteritems():
                if key not in ('job_cmd', 'reac_smi', 'nbreak', 'nform', 'dh_cutoff', 'forcefield', 'distance',
                               'output_dir'):
                    f.write('{0}  {1}\n'.format(key, val))
            f.write('\n')
            f.write('geometry (\n0 {0}\n{1}\n****\n{2}\n)\n'.format(self.reactant.multiplicity, self.reactant, product))

        return path

    def logHeader(self):
        """
        Output a log file header.
        """
        self.logger.info('######################################################################')
        self.logger.info('#################### AUTOMATIC REACTION DISCOVERY ####################')
        self.logger.info('######################################################################')
        self.logger.info('Reactant SMILES: ' + self.reac_smi)
        self.logger.info('Reactant coordinates:\n' + str(self.reactant))
        self.logger.info('Maximum number of bonds to be broken: ' + str(self.nbreak))
        self.logger.info('Maximum number of bonds to be formed: ' + str(self.nform))
        self.logger.info('Heat of reaction cutoff: {:.1f} kcal/mol'.format(self.dh_cutoff))
        self.logger.info('Force field for 3D structure generation: ' + self.forcefield)
        self.logger.info('######################################################################\n')

###############################################################################

def makeBatchSubmissionScript(name, example_script, outdir):
    """
    Create a batch submission script for a TS search job given a path to the
    example script and a path to the target directory. Everywhere the string
    "NAME" appears in the model script, it will be replaced by `name`. Return
    the path to the submission script.
    """
    outpath = os.path.join(outdir, 'submit.sh')

    with open(example_script, 'r') as infile, open(outpath, 'wb') as outfile:
        for line in infile:
            line = line.replace('NAME', name)
            outfile.write(line)

    return outpath

def makeOverallSubmissionScript(cmd, job_script, outdir):
    """
    Create a bash shell script that when run will submit all TS search jobs to
    the job scheduler. The name of all job scripts has be given in `job_script`
    (or the path) and the command for submitting a job has to be specified in
    `cmd`.

    Note: The script can be run as often as desired. It will only submit jobs
    that have not yet been submitted, which is useful if there is a limit to
    the number of jobs that can be submitted. Once all jobs in the queue have
    executed, the "submitted_jobs" file can be deleted and the script can be
    re-run to submit jobs that may have failed for an abnormal reason before
    creating a log file.
    """
    script = os.path.join(outdir, 'submitTSjobs.sh')

    with open(script, 'wb') as f:
        f.write('#!/bin/bash\n\n')
        f.write('for d in reactions/[0-9]*/\n')
        f.write('do\n')
        f.write('   cd $d\n')
        f.write("   j=`echo $d | sed 's/[^0-9]*//g'`\n\n")
        f.write('   if [ -s rxn*.log ]\n')
        f.write('   then\n')
        f.write('      cd ../..\n')
        f.write('      continue\n')
        f.write('   fi\n\n')
        f.write('   grep -Fxqs "$j" ../../submitted_jobs\n')
        f.write('   if [ $? -ne 0 ]\n')
        f.write('   then\n')
        f.write('      {} {}\n'.format(cmd, os.path.basename(job_script)))
        f.write('      if [ $? -eq 0 ]\n')
        f.write('      then\n')
        f.write('         echo $j >> ../../submitted_jobs\n')
        f.write('      fi\n')
        f.write('   fi\n\n')
        f.write('   cd ../..\n')
        f.write('done\n')

    os.chmod(script, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH)

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
    keys = ('job_cmd', 'reac_smi', 'nbreak', 'nform', 'dh_cutoff', 'forcefield', 'distance', 'logname'
            'method', 'nsteps', 'nnode', 'lsf', 'tol', 'gtol', 'nlstnodes',
            'qprog', 'theory', 'theory_preopt', 'nproc', 'mem')

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
