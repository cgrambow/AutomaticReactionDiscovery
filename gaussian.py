#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains functions for reading data from Gaussian logfiles.
"""

import numpy as np

###############################################################################

class GaussianError(Exception):
    """
    An exception class for errors that occur while using Gaussian.
    """
    pass

###############################################################################

def getEnergy(logfile='forceJob.log'):
    """
    Extract and return energy (in Hartrees) from Gaussian job.
    """
    with open(logfile, 'r') as f:
        gaussian_output = f.read().splitlines()
    for line in reversed(gaussian_output):
        if 'SCF Done' in line:
            return float(line.split()[4])
    else:
        raise GaussianError('Energy could not be found in Gaussian logfile')

###############################################################################

def getGradient(logfile='forceJob.log'):
    """
    Extract and return gradient (forces) from Gaussian job. Results are
    returned as a 3N x 1 array in units of Hartrees/Angstrom.
    """
    with open(logfile, 'r') as f:
        gaussian_output = f.read().splitlines()
    for line in gaussian_output:
        if 'NAtoms' in line:
            natoms = int(line.split()[1])
            break
    else:
        raise GaussianError('Number of atoms could not be found in Gaussian logfile')
    for line_num, line in enumerate(reversed(gaussian_output)):
        if 'Forces (Hartrees/Bohr)' in line:
            force_mat_str = gaussian_output[-(line_num-2):-(line_num-2-natoms)]
            break
    else:
        raise GaussianError('Forces could not be found in Gaussian logfile')

    force_mat = np.array([])
    for row in force_mat_str:
        force_mat = np.append(force_mat, [float(force_comp) for force_comp in row.split()[-3:]])
    return 1.88972613 * force_mat.flatten()

###############################################################################

def getGeometry(logfile='optJob.log'):
    """
    Extract and return final geometry from Gaussian job. Results are returned
    as a 3N x 1 array in units of Angstrom.
    """
    with open(logfile, 'r') as f:
        gaussian_output = f.read().splitlines()
    for line in gaussian_output:
        if 'NAtoms' in line:
            natoms = int(line.split()[1])
            break
    else:
        raise GaussianError('Number of atoms could not be found in Gaussian logfile')
    for line_num, line in enumerate(reversed(gaussian_output)):
        if 'Input orientation' in line:
            coord_mat_str = gaussian_output[-(line_num-4):-(line_num-4-natoms)]
            break
    else:
        raise GaussianError('Geometry could not be found in Gaussian logfile')

    coord_mat = np.array([])
    for row in coord_mat_str:
        coord_mat = np.append(coord_mat, [float(coord) for coord in row.split()[-3:]])
    return coord_mat.flatten()
