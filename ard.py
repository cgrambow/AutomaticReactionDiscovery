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
import argparse
import logging

from node import Node
from sm import FSM, GSM

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

    logging.addLevelName(logging.CRITICAL, 'Critical: ')
    logging.addLevelName(logging.ERROR, 'Error: ')
    logging.addLevelName(logging.WARNING, 'Warning: ')
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
    keyword containing the strings 'reactant' or 'product' is encountered, the
    corresponding geometries are read in the form (example for methane):
        reactant (
            0 1
            C                 -0.03144385    0.03144654    0.00041162
            H                  0.32521058   -0.97736346    0.00041162
            H                  0.32522899    0.53584473    0.87406313
            H                  0.32522899    0.53584473   -0.87323988
            H                 -1.10144385    0.03145972    0.00041162
        )
    If '#' is found in a line, the rest of the line will be ignored.

    A dictionary containing all input parameters and their values is returned.
    """
    if not os.path.exists(input_file):
        raise IOError('Input file "{0}" does not exist'.format(input_file))

    # Allowed keywords
    keys = ('method', 'reactant', 'product', 'nsteps', 'nnode', 'lsf', 'tol', 'gtol', 'nlstnodes',
            'qprog', 'level_of_theory', 'nproc', 'mem', 'output_file')

    # Read all data from file
    with open(input_file, 'r') as f:
        input_data = f.read().splitlines()

    # Read geometries
    read = False
    first_line = False
    geometry = -1  # 0 for reactant, 1 for product
    reactant_geo, reactant_atoms, product_geo, product_atoms = [], [], [], []
    reactant_multiplicity, product_multiplicity = 1, 1
    lines = []
    for line_num, line in enumerate(input_data):
        if line != '':
            if line.split()[0] != '#':  # Ignore comments
                if ')' in line and read:
                    read = False
                    geometry = -1
                    lines.append(line_num)
                if read and geometry == 0 and not first_line:
                    reactant_atoms.append(line.split()[0])
                    reactant_geo.append([float(coord) for coord in line.split()[1:4]])
                    lines.append(line_num)
                elif read and geometry == 0 and first_line:
                    reactant_multiplicity = line.split()[1]
                    first_line = False
                    lines.append(line_num)
                if read and geometry == 1 and not first_line:
                    product_atoms.append(line.split()[0])
                    product_geo.append([float(coord) for coord in line.split()[1:4]])
                    lines.append(line_num)
                elif read and geometry == 1 and first_line:
                    product_multiplicity = line.split()[1]
                    first_line = False
                    lines.append(line_num)

                if 'reactant' in line.lower():
                    read = True
                    first_line = True
                    geometry = 0
                    lines.append(line_num)
                if 'product' in line.lower():
                    read = True
                    first_line = True
                    geometry = 1
                    lines.append(line_num)

    # Check if reactant and product geometries were found
    if not reactant_geo:
        raise IOError('Missing reactant geometry')
    if not product_geo:
        raise IOError('Missing product geometry')

    # Create nodes
    reactant_node = Node(reactant_geo, reactant_atoms, reactant_multiplicity)
    product_node = Node(product_geo, product_atoms, product_multiplicity)

    # Delete geometries from input data
    for idx in sorted(lines, reverse=True):
        del input_data[idx]

    # Create and initialize dictionary
    input_dict = {'reactant': reactant_node, 'product': product_node}

    # Extract remaining keywords and values
    for line in input_data:
        # Ignore lines containing only whitespace or comments
        if line != '':
            if line.split()[0] != '#':
                key = line.lower().split()[0]
                if key not in keys:
                    raise KeyError('Keyword {0} not recognized'.format(key))
                if line.split()[1] == '=':
                    input_dict[key] = line.split()[2]
                else:
                    input_dict[key] = line.split()[1]

    # Check if valid method was specified and default to FSM
    try:
        method = input_dict['method'].lower()
        if method != 'gsm' and method != 'fsm':
            raise ValueError('Invalid method: {0}'.format(method))
    except KeyError:
        input_dict['method'] = 'fsm'
    except AttributeError:
        raise ValueError('Invalid method')

    return input_dict

###############################################################################

if __name__ == '__main__':

    # Set up parser for reading the input filename from the command line
    parser = argparse.ArgumentParser(description='A freezing/growing string method transition state search')
    parser.add_argument('file', type=str, metavar='FILE', help='An input file describing the FSM/GSM job to execute')
    args = parser.parse_args()

    # Set output directory
    input_file = args.file
    output_dir = os.path.abspath(os.path.dirname(input_file))

    # Read input file
    options = readInput(input_file)
    jobtype = options['method'].upper()
    del options['method']

    # Initialize the logging system
    log_level = logging.INFO
    initializeLog(log_level, os.path.join(output_dir, jobtype + '.log'))

    # Execute job
    if jobtype == 'FSM':
        fsm = FSM(**options)
        fsm.execute()
    else:
        gsm = GSM(**options)
        gsm.execute()
