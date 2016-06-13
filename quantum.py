#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains classes for reading data from quantum software log files and for
executing quantum jobs.
"""

import numpy as np

from sys import platform as _platform
import os
import re
import subprocess

###############################################################################

class GaussianError(Exception):
    """
    An exception class for errors that occur while using Gaussian.
    """
    pass

class NWChemError(Exception):
    """
    An exception class for errors that occur while using NWChem.
    """
    pass

class QChemError(Exception):
    """
    An exception class for errors that occur while using Q-Chem.
    """
    pass

###############################################################################

class Quantum(object):
    """
    Base class from which other quantum software classes can inherit.

    The attribute `logfile` represents the path where the log file of interest
    is located and the attribute `lf_contents` contains the contents of the log
    file in a list.
    """

    def __init__(self, logfile=None):
        self.logfile = logfile
        if logfile is not None:
            self.lf_contents = self.read()
        else:
            self.lf_contents = None

    def read(self):
        """
        Reads and returns the contents of the log file.
        """
        with open(self.logfile, 'r') as f:
            output = f.read().splitlines()
        return output

    def clear(self):
        """
        Deletes the log file and clears the associated attributes.
        """
        if self.logfile is not None:
            os.remove(self.logfile)
            self.logfile = None
            self.lf_contents = None

    @staticmethod
    def formatArray(a):
        """
        Converts raw geometry or gradient array of strings, `a`, to a formatted
        :class:`numpy.ndarray` of size N x 3. Only the rightmost 3 values of
        each row in `a` are retained.
        """
        vec = np.array([])
        for row in a:
            vec = np.append(vec, [float(e) for e in row.split()[-3:]])
        return vec.reshape(len(a), 3)

###############################################################################

class Gaussian(Quantum):
    """
    Class for reading data from Gaussian log files and for executing Gaussian
    jobs.

    The attribute `logfile` represents the path where the log file of interest
    is located and the attribute `lf_contents` contains the contents of the log
    file in a list.
    """

    def __init__(self, logfile=None):
        super(Gaussian, self).__init__(logfile)

    def getNumAtoms(self):
        """
        Extract and return number of atoms from Gaussian job.
        """
        for line in self.lf_contents:
            if 'NAtoms' in line:
                return int(line.split()[1])
        raise GaussianError('Number of atoms could not be found in Gaussian log file')

    def getEnergy(self):
        """
        Extract and return energy (in Hartree) from Gaussian job.
        """
        # Return final energy
        for line in reversed(self.lf_contents):
            if 'SCF Done' in line:
                return float(line.split()[4])
        raise GaussianError('Energy could not be found in Gaussian log file')

    def getGradient(self):
        """
        Extract and return gradient (forces) from Gaussian job. Results are
        returned as an N x 3 array in units of Hartree/Angstrom.
        """
        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read last occurrence of forces
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'Forces (Hartrees/Bohr)' in line:
                force_mat_str = self.lf_contents[-(line_num - 2):-(line_num - 2 - natoms)]
                break
        else:
            raise GaussianError('Forces could not be found in Gaussian log file')

        # Create force array and convert units
        return - 1.88972613 * self.formatArray(force_mat_str)  # Return negative because gradient is desired

    def getGeometry(self):
        """
        Extract and return final geometry from Gaussian job. Results are
        returned as an N x 3 array in units of Angstrom.
        """
        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read last occurrence of geometry
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'Input orientation' in line:
                coord_mat_str = self.lf_contents[-(line_num - 4):-(line_num - 4 - natoms)]
                break
        else:
            raise GaussianError('Geometry could not be found in Gaussian log file')

        # Create and return array containing geometry
        return self.formatArray(coord_mat_str)

    def executeJob(self, node, name='gau', jobtype='force', args=None, cmd='g09',
                   level_of_theory='m062x/cc-pvtz', nproc=32, mem='2000mb', output_dir=''):
        """
        Execute quantum job type using the Gaussian software package. This
        method can only be run on a UNIX system where Gaussian is installed.
        Requires that the geometry is input in the form of a :class:`node.Node`
        object.
        """
        jobtype = jobtype.lower()
        if jobtype == 'energy':
            jobtype = 'sp'
        elif jobtype == 'optimize' or jobtype == 'optimization':
            jobtype = 'opt'
        elif jobtype == 'gradient' or jobtype == 'grad':
            jobtype = 'force'
        elif jobtype == 'rpath':
            jobtype = 'irc'

        # Use correct string for memory
        match = re.match(r"([0-9]+) ([a-z]+)", mem, re.I)
        if match:
            mem = ''.join(match.groups())

        # Create Gaussian input file
        try:
            input_file = os.path.join(output_dir, name + '.com')
            with open(input_file, 'w') as f:
                f.write('%chk=' + name + '.chk\n')
                f.write('%mem=' + mem + '\n')
                f.write('%nprocshared=' + str(int(nproc)) + '\n')
                if args is None:
                    f.write('# ' + jobtype + ' ' + level_of_theory + '\n\n')
                else:
                    f.write('# ' + jobtype + '=(' + args + ') ' + level_of_theory + '\n\n')
                f.write(name + '\n\n')
                f.write('0 ' + str(node.multiplicity) + '\n')

                for atom_num, atom in enumerate(node.coordinates):
                    f.write(' {0}              {1[0]: 14.8f}{1[1]: 14.8f}{1[2]: 14.8f}\n'.format(
                        node.elements[node.number[atom_num]], atom))
                f.write('\n')
        except:
            print 'An error occurred while creating the Gaussian input file'
            raise

        # Run job and wait until termination
        if _platform == 'linux' or _platform == 'linux2':
            output_file = os.path.join(output_dir, name + '.log')
            subprocess.Popen([cmd, input_file, output_file]).wait()
            os.remove(input_file)
            os.remove(name + '.chk')
        else:
            os.remove(input_file)
            raise OSError('Invalid operating system')

        # Check if job completed or if it terminated with an error
        if os.path.isfile(output_file):
            completed = False
            self.logfile = output_file
            gaussian_output = self.read()
            for line in reversed(gaussian_output):
                if 'Error termination' in line:
                    raise GaussianError('Quantum job terminated with an error')
                elif 'Normal termination' in line:
                    completed = True
                    break
            if not completed:
                raise GaussianError('Quantum job did not terminate')
            self.lf_contents = gaussian_output
        else:
            raise IOError('Gaussian output file could not be found')

###############################################################################

class NWChem(Quantum):
    """
    Class for reading data from NWChem log files and for executing NWChem jobs.

    The attribute `logfile` represents the path where the log file of interest
    is located and the attribute `lf_contents` contains the contents of the log
    file in a list.
    """

    def __init__(self, logfile=None):
        super(NWChem, self).__init__(logfile)

    def getNumAtoms(self):
        """
        Extract and return number of atoms from NWChem job.
        """
        for line_num, line in enumerate(self.lf_contents):
            if 'XYZ format geometry' in line:
                return int(self.lf_contents[line_num + 2].split()[0])
        raise NWChemError('Number of atoms could not be found in NWChem log file')

    def getEnergy(self):
        """
        Extract and return energy (in Hartree) from NWChem job.
        """
        # Return final energy
        for line in reversed(self.lf_contents):
            if any(s in line for s in ('Total SCF energy', 'Total DFT energy')):
                return float(line.split()[4])
            if any(s in line for s in ('Total CCSD energy:', 'Total CCSD(T) energy:')):
                return float(line.split()[3])
        else:
            raise NWChemError('Energy could not be found in NWChem log file')

    def getGradient(self):
        """
        Extract and return gradient from NWChem job. Results are returned as an
        N x 3 array in units of Hartree/Angstrom.
        """
        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read last occurrence of gradient
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'ENERGY GRADIENTS' in line:
                grad_mat_str = self.lf_contents[-(line_num - 3):-(line_num - 3 - natoms)]
                break
        else:
            raise NWChemError('Gradient could not be found in NWChem log file')

        # Create gradient array and convert units
        return 1.88972613 * self.formatArray(grad_mat_str)

    def getGeometry(self):
        """
        Extract and return final geometry from NWChem job. Results are returned
        as an N x 3 array in units of Angstrom.
        """
        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read last occurrence of geometry
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'Geometry' in line:
                coord_mat_str = self.lf_contents[-(line_num - 6):-(line_num - 6 - natoms)]
                break
        else:
            raise NWChemError('Geometry could not be found in NWChem log file')

        # Create array containing geometry and convert units
        return 0.529177211 * self.formatArray(coord_mat_str)

    def executeJob(self, node, name='nwc', jobtype='gradient', cmd='nwchem',
                   level_of_theory='m062x/cc-pvtz', nproc=32, mem='2000mb', output_dir=''):
        """
        Execute quantum job type using the NWChem software package. This method
        can only be run on a UNIX system where NWChem is installed. Requires
        that the geometry is input in the form of a :class:`node.Node` object.
        """
        jobtype = jobtype.lower()
        if jobtype == 'sp':
            jobtype = 'energy'
        elif jobtype == 'opt' or jobtype == 'optimization':
            jobtype = 'optimize'
        elif jobtype == 'force' or jobtype == 'grad':
            jobtype = 'gradient'

        # Split theory and basis
        theory, basis = level_of_theory.lower().split('/')

        # Reformat m06 and hf theories
        match = re.match(r"(m06)(\w+)", theory, re.I)
        if match:
            theory = '-'.join(match.groups())
        if theory == 'hf' and node.multiplicity > 1:
            theory = 'uhf'
        elif theory == 'hf':
            theory = 'rhf'

        # Reformat memory string
        match = re.match(r"([0-9]+)([a-z]+)", mem, re.I)
        if match:
            mem = ' '.join(match.groups())

        # Create NWChem input file
        try:
            input_file = os.path.join(output_dir, name + '.nw')
            with open(input_file, 'w') as f:
                f.write('title "' + name + '"\n')
                f.write('memory ' + mem + '\n')
                f.write('geometry nocenter noautosym\n')  # Have to disable symmetry to obtain input orientation
                f.write(str(node) + '\n')
                f.write('end\n')
                f.write('basis\n')
                f.write('* library ' + basis + '\n')
                f.write('end\n')

                if theory in ('rhf', 'rohf', 'uhf'):
                    f.write('scf\n')
                    f.write(theory + '; ')
                    f.write('nopen ' + str(node.multiplicity - 1) + '\n')
                    f.write('end\n')
                    f.write('task scf ' + jobtype + '\n')
                elif theory in ('ccsd', 'ccsd+t(ccsd)', 'ccsd(t)'):
                    f.write('task ' + theory + ' ' + jobtype + '\n')
                else:
                    f.write('dft\n')
                    f.write('xc ' + theory + '\n')
                    f.write('mult ' + str(node.multiplicity) + '\n')
                    f.write('end\n')
                    f.write('task dft ' + jobtype + '\n')
        except:
            print 'An error occurred while creating the NWChem input file'
            raise

        # Run job and wait until termination
        if _platform == 'linux' or _platform == 'linux2':
            output_file = os.path.join(output_dir, name + '.log')
            subprocess.Popen(
                'srun -n {0} {1} {2} >& {3}'.format(nproc, cmd, input_file, output_file), shell=True
            ).wait()
            os.remove(input_file)
            file_endings = ('.b', '.b^-1', '.c', '.cfock', '.db', '.movecs', '.p', '.zmat', '.drv.hess', '.t2')
            for e in file_endings:
                try:
                    os.remove(name + e)
                except OSError:
                    pass
        else:
            os.remove(input_file)
            raise OSError('Invalid operating system')

        # Check if job completed or if it terminated with an error
        if os.path.isfile(output_file):
            completed = False
            self.logfile = output_file
            nwchem_output = self.read()
            for line in reversed(nwchem_output):
                if 'error' in line:
                    raise NWChemError('Quantum job terminated with an error')
                elif 'Total times' in line:
                    completed = True
                    break
            if not completed:
                raise NWChemError('Quantum job did not terminate')
            self.lf_contents = nwchem_output
        else:
            raise IOError('NWChem output file could not be found')

###############################################################################

class QChem(Quantum):
    """
    Class for reading data from Q-Chem log files and for executing Q-Chem jobs.

    The attribute `logfile` represents the path where the log file of interest
    is located and the attribute `lf_contents` contains the contents of the log
    file in a list.
    """

    def __init__(self, logfile=None):
        super(QChem, self).__init__(logfile)

    def getNumAtoms(self):
        """
        Extract and return number of atoms from Q-Chem job.
        """
        read = False
        natoms = 0
        i = 0
        for line in self.lf_contents:
            if read:
                i += 1
                try:
                    natoms = int(line.split()[0])
                except ValueError:
                    if i > 2:
                        return natoms
                    continue
            if 'Standard Nuclear Orientation' in line:
                read = True
        raise NWChemError('Number of atoms could not be found in Q-Chem log file')

    def getEnergy(self):
        """
        Extract and return energy (in Hartree) from Q-Chem job.
        """
        # Return final energy
        for line in reversed(self.lf_contents):
            if 'Final energy is' in line:
                return float(line.split()[3])
            if 'Total energy' in line:
                return float(line.split()[8])
        raise GaussianError('Energy could not be found in Gaussian log file')

    def getGradient(self):
        """
        Extract and return gradient (forces) from Q-Chem job. Results are
        returned as an N x 3 array in units of Hartree/Angstrom.
        """
        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read last occurrence of gradient
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'Gradient of SCF Energy' in line:
                num_lines = int(np.ceil(natoms / 6.0)) * 4
                grad_mat_str = self.lf_contents[-line_num:-(line_num - num_lines)]
                break
        else:
            raise NWChemError('Gradient could not be found in NWChem log file')

        # Remove every 4th line and 1st column, and extract and flatten gradient components
        grad_mat_str = [row.split()[1:] for (i, row) in enumerate(grad_mat_str) if i % 4]
        xgrad = [row for (i, row) in enumerate(grad_mat_str) if not i % 3]
        ygrad = [row for (i, row) in enumerate(grad_mat_str[1:]) if not i % 3]
        zgrad = [row for (i, row) in enumerate(grad_mat_str[2:]) if not i % 3]
        xgrad = [x for row in xgrad for x in row]
        ygrad = [y for row in ygrad for y in row]
        zgrad = [z for row in zgrad for z in row]

        # Create gradient array and convert units
        return 1.88972613 * np.array([xgrad, ygrad, zgrad]).astype(float).reshape(natoms, 3)

    def getGeometry(self):
        """
        Extract and return final geometry from Q-Chem job. Results are returned
        as an N x 3 array in units of Angstrom.
        """
        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read last occurrence of geometry
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'Coordinates' in line:
                coord_mat_str = self.lf_contents[-(line_num - 1):-(line_num - 1 - natoms)]
            if 'Standard Nuclear Orientation' in line:
                coord_mat_str = self.lf_contents[-(line_num - 2):-(line_num - 2 - natoms)]
                break
        else:
            raise QChemError('Geometry could not be found in Q-Chem log file')

        # Create and return array containing geometry
        return self.formatArray(coord_mat_str)

    def executeJob(self, node, name='qc', jobtype='force', cmd='qchem',
                   level_of_theory='m062x/cc-pvtz', nproc=32, mem='2000mb', output_dir=''):
        """
        Execute quantum job type using the Q-Chem software package. This method
        can only be run on a UNIX system where Q-Chem is installed. Requires
        that the geometry is input in the form of a :class:`node.Node` object.
        """
        jobtype = jobtype.lower()
        if jobtype == 'energy':
            jobtype = 'sp'
        elif jobtype == 'optimize':
            jobtype = 'opt'
        elif jobtype == 'gradient' or jobtype == 'grad':
            jobtype = 'force'
        elif jobtype == 'irc':
            jobtype = 'rpath'

        # Split theory and basis
        theory, basis = level_of_theory.lower().split('/')

        # Create Q-Chem input file
        try:
            input_file = os.path.join(output_dir, name + '.in')
            with open(input_file, 'w') as f:
                f.write('$comment\n')
                f.write(name + '\n')
                f.write('$end\n')

                f.write('$molecule\n')
                f.write('0 ' + str(node.multiplicity) + '\n')
                f.write(str(node) + '\n')
                f.write('$end\n')

                f.write('$rem\n')
                f.write('jobtype ' + str(jobtype) + '\n')
                f.write('basis ' + basis + '\n')
                f.write('method ' + theory + '\n')
                f.write('sym_ignore true\n')  # Have to disable symmetry to obtain input orientation
                f.write('$end')
        except:
            print 'An error occurred while creating the Q-Chem input file'
            raise

        # Run job and wait until termination
        if _platform == 'linux' or _platform == 'linux2':
            output_file = os.path.join(output_dir, name + '.log')
            subprocess.Popen('{0} -np {1} {2} {3}'.format(cmd, nproc, input_file, output_file), shell=True).wait()
            os.remove(input_file)
            os.remove('pathtable')
        else:
            # os.remove(input_file)
            raise OSError('Invalid operating system')

        # Check if job completed or if it terminated with an error
        if os.path.isfile(output_file):
            completed = False
            self.logfile = output_file
            qchem_output = self.read()
            for line in reversed(qchem_output):
                if 'errno' in line:
                    raise QChemError('Quantum job terminated with an error')
                elif 'Total job time:' in line:
                    completed = True
                    break
            if not completed:
                raise QChemError('Quantum job did not terminate')
            self.lf_contents = qchem_output
        else:
            raise IOError('Q-Chem output file could not be found')
