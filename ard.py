#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#
#   Automatic Reaction Discovery
#
###############################################################################

"""
Discovers chemical reactions automatically.
"""

import os
import logging

from node import Node

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

###############################################################################

if __name__ == '__main__':
    import argparse

    from main import ARD

    # Set up parser for reading the input filename from the command line
    parser = argparse.ArgumentParser(description='Automatic Reaction Discovery')
    parser.add_argument('file', type=str, metavar='FILE', help='An input file describing the job options')
    args = parser.parse_args()

    # Read input file
    input_file = os.path.abspath(args.file)
    kwargs = readInput(input_file)

    # Set output directory
    output_dir = os.path.abspath(os.path.dirname(input_file))
    kwargs['output_dir'] = output_dir

    # Initialize the logging system
    log_level = logging.INFO
    initializeLog(log_level, os.path.join(output_dir, 'ARD.log'))

    # Execute job
    ard = ARD(**kwargs)
    ard.execute(**kwargs)
