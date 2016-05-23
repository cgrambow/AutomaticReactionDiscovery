#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#
#   Automatic Reaction Discovery
#
###############################################################################

"""
Discovers chemical reactions automatically.
Currently, it just runs a freezing string method transition state search.
"""

import os
import sys
import logging

from node import Node
from fsm import FSM

###############################################################################

def initializeLog(level, logfile):
    """
    Configure a logger. `level` is an integer parameter specifying how much
    information is displayed in the log file. The levels correspond to those of
    the :data:`logging` module.
    """
    # Create logger, console handler, formatter
    logger = logging.getLogger()
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    logging.addLevelName(logging.CRITICAL, 'Critical: ')
    logging.addLevelName(logging.ERROR, 'Error: ')
    logging.addLevelName(logging.WARNING, 'Warning: ')
    logging.addLevelName(logging.INFO, '')
    logging.addLevelName(logging.DEBUG, '')

    # Create formatter
    formatter = logging.Formatter('%(levelname)s%(message)s')
    ch.setFormatter(formatter)

    # Create file handler
    if os.path.exists(logfile):
        os.remove(logfile)
    fh = logging.FileHandler(filename=logfile)
    fh.setLevel(min(logging.DEBUG, level))
    fh.setFormatter(formatter)

    # Remove old handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Add console and file handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

###############################################################################

if __name__ == '__main__':

    # Initialize the logging system
    level = logging.INFO
    initializeLog(level, 'FSM.log')

    # Create FSM input nodes
    # Korcek reaction
    atoms = (6, 6, 6, 1, 1, 1, 1, 1, 8, 8, 1, 8)
    reactant = [[4.43125529,   0.07919777,   0.23143382],
                [2.89342009,   0.11975358,   0.16059524],
                [2.30386003,  -0.60715121,   1.38355409],
                [4.96628850,  -0.73161823,  -0.21713823],
                [2.56324649,   1.13753262,   0.15724650],
                [2.56472719,  -0.36478984,  -0.73499320],
                [2.63255292,  -0.12260779,   2.27914253],
                [2.63403363,  -1.62493026,   1.38690283],
                [0.87587019,  -0.56949224,   1.31777541],
                [0.37053300,  -1.19255350,   2.36602586],
                [-0.58811752, -1.16726934,   2.32186829],
                [5.05864729,   0.99963811,   0.81687333]]

    product = [[1.67823370,  -1.48014525,   1.42090357],
               [1.50203321,  -0.95735353,   0.01705726],
               [1.30824135,  -2.24160181,  -0.68086569],
               [2.68969585,  -1.72829100,   1.66639900],
               [0.58688205,  -0.40902620,  -0.06501275],
               [2.29612445,  -0.32545165,  -0.32208744],
               [2.19418642,  -2.84159727,  -0.67826161],
               [0.99487246,  -2.12525301,  -1.69731190],
               [0.28615035,  -2.79519060,   0.19386353],
               [0.91921584,  -2.73204886,   1.39906438],
               [0.29867020,  -0.27448633,   2.14539727],
               [1.21294235,  -0.49081137,   2.34265323]]
    reactant_node = Node(reactant, atoms)
    product_node = Node(product, atoms)

    fsm = FSM(reactant_node, product_node, nsteps=4, nnode=15, nLSTnodes=100,
              gaussian_ver='g03', level_of_theory='hf/sto-3g', nproc=1)

    FSMpath, energies = fsm.execute()

    # Create file containing FSM path geometries and energies
    node_num = 1
    with open('my_stringfile.txt', 'w') as f:
        for node, energy in zip(FSMpath, energies):
            f.write('Node ' + str(node_num) + ':\n')
            f.write('Energy = ' + str(energy) + '\n')
            f.write(str(node) + '\n')
            node_num += 1
