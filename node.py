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


class Node(object):
    """
    Three-dimensional representation of a molecular configuration.
    The attributes are:

    =============== ======================= ====================================
    Attribute       Type                    Description
    =============== ======================= ====================================
    `coordinates`   :class:`numpy.ndarray`  A 3N x 1 array containing the 3D coordinates of each atom in a vector
    `number`        :class:`numpy.ndarray`  A N x 1 array containing the integer atomic number of each atom
    `multiplicity`  ``int``                 The multiplicity of this species, multiplicity = 2*total_spin+1
    =============== ======================= ====================================

    N is the total number of atoms in the molecule. The integer index of each
    atom corresponds to three subsequent entries in the coordinates vector,
    which represent the set of x, y, and z coordinates of the atom.
    """

    # Dictionary of elements corresponding to atomic numbers
    elements = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

    def __init__(self, coordinates, number, multiplicity=1):
        self.coordinates = np.array(coordinates)
        self.number = np.array(number)
        self.multiplicity = multiplicity

    def __str__(self):
        """
        Return a human readable string representation of the object
        """
        return_string = '0 ' + str(self.multiplicity) + '\n'
        coord_array = self.coordinates.reshape(len(self.number), 3)
        atom_num = 0
        for atom in coord_array[:]:
            return_string += self.elements[self.number[atom_num]] + '    ' + np.array_str(atom) + '\n'
            atom_num += 1
        return return_string

    def getTangent(self, other):
        """
        Calculate and return tangent direction between two nodes based on LST
        path between the nodes
        """
        return (other.coordinates - self.coordinates) / np.linalg.norm(other.coordinates - self.coordinates)

    def executeGaussianForce(self, name='forceJob', ver='g09', level_of_theory='um062x/cc-pvtz', nproc=32, mem='1500mb'):
        """
        Execute 'force' job type using the Gaussian software package. This
        method can only be run on a UNIX system where Gaussian is installed.
        Return filename of Gaussian logfile.
        """
        coord_array = self.coordinates.reshape(len(self.number), 3)
        atom_num = 0

        # Create Gaussian input file
        input_file = name + '.com'
        with open(input_file, 'w') as f:
            f.write('%chk=' + name + '.chk\n')
            f.write('%mem=' + mem + '\n')
            f.write('%nprocshared=' + str(nproc) + '\n')
            f.write('# force ' + level_of_theory + '\n\n')
            f.write(name + '\n\n')
            f.write('0 ' + str(self.multiplicity) + '\n')

            for atom in coord_array[:]:
                f.write(self.elements[self.number[atom_num]] + '                 ')
                atom_num += 1
                for coord in atom:
                    f.write(str(coord) + '    ')
                f.write('\n')

            f.write('\n')

        # Run job and wait until termination
        if _platform == 'linux' or _platform == 'linux2':
            output_file = name + '.log'
            subprocess.Popen(ver + ' < ' + input_file + ' > ' + output_file).wait()
        else:
            raise OSError('Invalid operating system')
        os.remove(input_file)

        # Check if job completed or if it terminated with an error
        if os.path.isfile(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    if 'Error termination' in line:
                        pass
                    elif 'Normal termination' in line:
                        pass
                    else:
                        pass
