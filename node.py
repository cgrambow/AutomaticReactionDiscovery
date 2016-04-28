#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`Node` for working with three-dimensional representations of
molecules in Cartesian coordinates and evaluating energies and gradients using
quantum chemical calculations.
"""

import numpy as np

from sys import platform as _platform
import subprocess
import os

################################################################################

class GaussianError(Exception):
    """
    An exception class for errors that occur while using Gaussian.
    """
    pass

################################################################################

class Node(object):
    """
    Three-dimensional representation of a molecular configuration.
    The attributes are:

    =============== ======================= ====================================
    Attribute       Type                    Description
    =============== ======================= ====================================
    `coordinates`   :class:`numpy.ndarray`  A 3N x 1 array containing the 3D coordinates of each atom in a vector
    `number`        :class:`list`           A list of length N containing the integer atomic number of each atom
    `multiplicity`  ``int``                 The multiplicity of this species, multiplicity = 2*total_spin+1
    =============== ======================= ====================================

    N is the total number of atoms in the molecule. The integer index of each
    atom corresponds to three subsequent entries in the coordinates vector,
    which represent the set of x, y, and z coordinates of the atom.
    """

    # Dictionary of elements corresponding to atomic numbers
    elements = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

    def __init__(self, coordinates, number, multiplicity=1):
        self.coordinates = np.array(coordinates).reshape(len(coordinates), 1)
        self.number = [int(num) for num in number]
        assert len(coordinates) == 3 * len(number)
        self.multiplicity = multiplicity

    def __str__(self):
        """
        Return a human readable string representation of the object
        """
        return_string = '0 ' + str(self.multiplicity) + '\n'
        coord_array = self.coordinates.reshape(len(self.number), 3)
        for atom_num, atom in enumerate(coord_array[:]):
            return_string += self.elements[self.number[atom_num]] + '    ' + np.array_str(atom) + '\n'
        return return_string

    def getTangent(self, other):
        """
        Calculate and return tangent direction between two nodes based on LST
        path between the nodes. The tangent vector points from self to other.
        """
        return (other.coordinates - self.coordinates) / np.linalg.norm(other.coordinates - self.coordinates)

    def executeGaussianForce(self, name='forceJob', ver='g09', level_of_theory='um062x/cc-pvtz', nproc=32, mem='1500mb'):
        """
        Execute 'force' job type using the Gaussian software package. This
        method can only be run on a UNIX system where Gaussian is installed.
        Return filename of Gaussian logfile.
        """
        coord_array = self.coordinates.reshape(len(self.number), 3)

        # Create Gaussian input file
        input_file = name + '.com'
        with open(input_file, 'w') as f:
            f.write('%chk=' + name + '.chk\n')
            f.write('%mem=' + mem + '\n')
            f.write('%nprocshared=' + str(nproc) + '\n')
            f.write('# force ' + level_of_theory + '\n\n')
            f.write(name + '\n\n')
            f.write('0 ' + str(self.multiplicity) + '\n')

            for atom_num, atom in enumerate(coord_array[:]):
                f.write(self.elements[self.number[atom_num]] + '                 ')
                for coord in atom:
                    f.write(str(coord) + '    ')
                f.write('\n')

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

    def getEnergy(self, logfile='forceJob.log'):
        """
        Extract and return energy (in Hartrees) from Gaussian force job.
        """
        with open(logfile, 'r') as f:
            for line in f:
                if 'SCF Done' in line:
                    return float(line.split()[4])
            else:
                raise GaussianError('Energy could not be found in Gaussian logfile')

    def getGradient(self, logfile='forceJob.log'):
        """
        Extract and return gradient (forces) from Gaussian force job. Results
        are returned as a 3N x 1 array in units of Hartrees/Bohr.
        """
        with open(logfile, 'r') as f:
            gaussian_output = f.read().splitlines()
        for line_num, line in enumerate(gaussian_output):
            if 'Forces (Hartrees/Bohr)' in line:
                force_mat_str = gaussian_output[line_num+3:line_num+3+len(self.number)]
                break
        else:
            raise GaussianError('Forces could not be found in Gaussian logfile')

        force_mat = np.array([])
        for row in force_mat_str:
            force_mat = np.append(force_mat, [float(force_comp) for force_comp in row.split()[-3:]])
        return force_mat.reshape(3*len(self.number), 1)
