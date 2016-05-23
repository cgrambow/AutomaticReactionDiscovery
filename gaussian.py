#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains functions for reading data from Gaussian logfiles and executing
quantum jobs using Gaussian.
"""

import numpy as np

from sys import platform as _platform
import os
import subprocess

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
    # Read entire file
    with open(logfile, 'r') as f:
        gaussian_output = f.read().splitlines()

    # Return final energy
    for line in reversed(gaussian_output):
        if 'SCF Done' in line:
            return float(line.split()[4])
    else:
        raise GaussianError('Energy could not be found in Gaussian logfile')

###############################################################################

def getGradient(logfile='forceJob.log'):
    """
    Extract and return gradient (forces) from Gaussian job. Results are
    returned as a 3N x 3 array in units of Hartrees/Angstrom.
    """
    # Read entire file
    with open(logfile, 'r') as f:
        gaussian_output = f.read().splitlines()
    for line in gaussian_output:
        if 'NAtoms' in line:
            natoms = int(line.split()[1])
            break
    else:
        raise GaussianError('Number of atoms could not be found in Gaussian logfile')

    # Read last occurrence of forces
    for line_num, line in enumerate(reversed(gaussian_output)):
        if 'Forces (Hartrees/Bohr)' in line:
            force_mat_str = gaussian_output[-(line_num-2):-(line_num-2-natoms)]
            break
    else:
        raise GaussianError('Forces could not be found in Gaussian logfile')

    # Create force array and convert units
    force_mat = np.array([])
    for row in force_mat_str:
        force_mat = np.append(force_mat, [float(force_comp) for force_comp in row.split()[-3:]])
    return 1.88972613 * force_mat

###############################################################################

def getGeometry(logfile='optJob.log'):
    """
    Extract and return final geometry from Gaussian job. Results are returned
    as a 3N x 3 array in units of Angstrom.
    """
    # Read entire file
    with open(logfile, 'r') as f:
        gaussian_output = f.read().splitlines()
    for line in gaussian_output:
        if 'NAtoms' in line:
            natoms = int(line.split()[1])
            break
    else:
        raise GaussianError('Number of atoms could not be found in Gaussian logfile')

    # Read last occurrence of geometry
    for line_num, line in enumerate(reversed(gaussian_output)):
        if 'Input orientation' in line:
            coord_mat_str = gaussian_output[-(line_num-4):-(line_num-4-natoms)]
            break
    else:
        raise GaussianError('Geometry could not be found in Gaussian logfile')

    # Create and return array containing geometry
    coord_mat = np.array([])
    for row in coord_mat_str:
        coord_mat = np.append(coord_mat, [float(coord) for coord in row.split()[-3:]])
    return coord_mat

###############################################################################

def executeGaussianJob(node, name='forceJob', jobtype='force', ver='g09', level_of_theory='um062x/cc-pvtz',
                           nproc=32, mem='1500mb'):
    """
    Execute quantum job type using the Gaussian software package. This method
    can only be run on a UNIX system where Gaussian is installed. Requires that
    the geometry is input in the form of a :class:`node.Node` object.

    Return filename of Gaussian logfile.
    """
    # Create Gaussian input file
    input_file = name + '.com'
    with open(input_file, 'w') as f:
        f.write('%chk=' + name + '.chk\n')
        f.write('%mem=' + mem + '\n')
        f.write('%nprocshared=' + str(int(nproc)) + '\n')
        f.write('# ' + jobtype + ' ' + level_of_theory + '\n\n')
        f.write(name + '\n\n')
        f.write('0 ' + str(node.multiplicity) + '\n')

        for atom_num, atom in enumerate(node.coordinates):
            f.write(' {0}              {1[0]: 14.8f}{1[1]: 14.8f}{1[2]: 14.8f}\n'.format(
                node.elements[node.number[atom_num]], atom))
        f.write('\n')

    # Run job and wait until termination
    if _platform == 'linux' or _platform == 'linux2':
        output_file = name + '.log'
        subprocess.Popen([ver, input_file, output_file]).wait()
        os.remove(input_file)
        os.remove(name + '.chk')
    else:
        os.remove(input_file)
        raise OSError('Invalid operating system')

    # Check if job completed or if it terminated with an error
    if os.path.isfile(output_file):
        completed = False
        with open(output_file, 'r') as f:
            gaussian_output = f.readlines()
        for line in gaussian_output:
            if 'Error termination' in line:
                raise GaussianError('Force job terminated with an error')
            elif 'Normal termination' in line:
                completed = True
        if not completed:
            raise GaussianError('Force job did not terminate')
        else:
            return output_file
    else:
        raise IOError('Gaussian output file could not be found')
