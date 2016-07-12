#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains classes for reading data from quantum software log files and for
executing quantum jobs.
"""

from __future__ import print_function, division

import numpy as np

import sys
import os
import re
import subprocess

import props

###############################################################################

class QuantumError(Exception):
    """
    An exception class for errors that occur during quantum calculations.
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
            self.read()
        else:
            self.lf_contents = None

    def read(self):
        """
        Reads and returns the contents of the log file.
        """
        with open(self.logfile, 'r') as f:
            self.lf_contents = f.read().splitlines()

    def clear(self):
        """
        Deletes the log file and clears the associated attributes.
        """
        if self.logfile is not None:
            os.remove(self.logfile)
            self.logfile = None
            self.lf_contents = None

    @staticmethod
    def _formatArray(a):
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

    # def getNumAtoms(self):
    #     """
    #     Extract and return number of atoms from Gaussian job.
    #     """
    #     for line in self.lf_contents:
    #         if 'NAtoms' in line:
    #             return int(line.split()[1])
    #     raise QuantumError('Number of atoms could not be found in Gaussian log file')

    def getNumAtoms(self):
        """
        Extract and return number of atoms from Gaussian job.
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
                    if i > 5:
                        return natoms
                    continue
            elif 'Input orientation' in line:
                read = True
        raise QuantumError('Number of atoms could not be found in Gaussian log file')

    def getEnergy(self):
        """
        Extract and return energy (in Hartree) from Gaussian job.
        """
        # Return final energy
        for line in reversed(self.lf_contents):
            if 'SCF Done' in line:
                return float(line.split()[4])
        raise QuantumError('Energy could not be found in Gaussian log file')

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
            raise QuantumError('Forces could not be found in Gaussian log file')

        # Create force array and convert units
        return - 1.88972613 * self._formatArray(force_mat_str)  # Return negative because gradient is desired

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
            raise QuantumError('Geometry could not be found in Gaussian log file')

        # Create and return array containing geometry
        return self._formatArray(coord_mat_str)

    def getIRCpath(self):
        """
        Extract and return IRC path from Gaussian job. Results are returned as
        a list of tuples of N x 3 coordinate arrays in units of Angstrom and
        corresponding energies.
        """
        # Ensure that logfile contains IRC calculation
        for line in self.lf_contents:
            if 'IRC-IRC' in line:
                break
        else:
            raise QuantumError('Gaussian log file does not contain IRC calculation')

        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read forward IRC path
        forwardpath = []
        for line_num, line in enumerate(self.lf_contents):
            if 'Input orientation' in line:
                coord_mat = self._formatArray(self.lf_contents[line_num + 5:line_num + 5 + natoms])
            elif 'SCF Done' in line:
                energy = float(line.split()[4])
                forwardpath.append((coord_mat, energy))
            elif 'FORWARD' in line:
                break
        else:
            raise QuantumError('Forward IRC path could not be found in Gaussian log file')

        # Read reverse IRC path
        reversepath = []
        energy = None
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'SCF Done' in line:
                energy = float(line.split()[4])
            if 'Input orientation' in line and energy is not None:
                coord_mat = self._formatArray(self.lf_contents[-(line_num - 4):-(line_num - 4 - natoms)])
                reversepath.append((coord_mat, energy))
            elif 'FORWARD' in line:
                break
        else:
            raise QuantumError('Reverse IRC path could not be found in Gaussian log file')

        return reversepath + forwardpath

    def getNumGrad(self):
        """
        Extract and return the total number of gradient evaluations from a
        Gaussian job.
        """
        ngrad = 0
        for line in self.lf_contents:
            if 'Forces (Hartrees/Bohr)' in line:
                ngrad += 1
        return ngrad

    def executeJob(self, node, name='gau', jobtype='force',
                   theory='m062x/cc-pvtz', nproc=32, mem='2000mb',
                   output_dir='', **kwargs):
        """
        Execute quantum job type using the Gaussian software package. This
        method can only be run on a UNIX system where Gaussian is installed.
        Requires that the geometry is input in the form of a :class:`node.Node`
        object.
        """
        if not (sys.platform == 'linux' or sys.platform == 'linux2'):
            raise OSError('Invalid operating system')

        jobtype = jobtype.lower()
        if jobtype == 'energy':
            jobtype = 'sp'
        elif jobtype == 'optimize' or jobtype == 'optimization':
            jobtype = 'opt'
        elif jobtype == 'gradient' or jobtype == 'grad':
            jobtype = 'force'
        elif jobtype == 'saddle':
            jobtype = 'ts'
        elif jobtype == 'rpath' or jobtype == 'mepgs':
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
                if jobtype == 'opt':
                    f.write('# opt=(maxcycles=100) ' + theory + '\n\n')
                elif jobtype == 'ts':
                    f.write('# opt=(ts,noeigen,calcfc,maxcycles=100) ' + theory + '\n\n')
                elif jobtype == 'irc':
                    f.write('# irc=(calcfc,maxpoints=50,stepsize=5,maxcycle=50) ' + theory +
                            ' iop(1/7=300) \n\n')
                else:
                    f.write('# ' + jobtype + ' ' + theory + '\n\n')
                f.write(name + '\n\n')
                f.write('0 ' + str(node.multiplicity) + '\n')

                for atom_num, atom in enumerate(node.coordinates):
                    f.write(' {0}              {1[0]: 14.8f}{1[1]: 14.8f}{1[2]: 14.8f}\n'.format(
                        props.atomnum[node.atoms[atom_num]], atom))
                f.write('\n')
        except:
            print('An error occurred while creating the Gaussian input file', file=sys.stderr)
            raise

        # Run job and wait until termination
        output_file = os.path.join(output_dir, name + '.log')
        try:
            subprocess.check_call(['g09', input_file, output_file])
        except subprocess.CalledProcessError:
            if os.path.isfile(output_file):
                raise QuantumError('Gaussian job terminated with an error')
            try:
                subprocess.check_call(['g03', input_file, output_file])
            except subprocess.CalledProcessError:
                if os.path.isfile(output_file):
                    raise QuantumError('Gaussian job terminated with an error')
                raise Exception('Gaussian is not available')

        # Remove unnecessary files and save log file
        os.remove(input_file)
        os.remove(name + '.chk')
        self.logfile = output_file
        self.read()

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
        raise QuantumError('Number of atoms could not be found in NWChem log file')

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
            raise QuantumError('Energy could not be found in NWChem log file')

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
            raise QuantumError('Gradient could not be found in NWChem log file')

        # Create gradient array and convert units
        return 1.88972613 * self._formatArray(grad_mat_str)

    def getGeometry(self):
        """
        Extract and return final geometry from NWChem job. Results are returned
        as an N x 3 array in units of Angstrom.
        """
        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read last occurrence of geometry
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'Geometry "geometry"' in line:
                coord_mat_str = self.lf_contents[-(line_num - 6):-(line_num - 6 - natoms)]
                break
        else:
            raise QuantumError('Geometry could not be found in NWChem log file')

        # Create and return array containing geometry
        return self._formatArray(coord_mat_str)

    def getIRCpath(self):
        """
        Extract and return IRC path from Gaussian job. Results are returned as
        a list of tuples of N x 3 coordinate arrays in units of Angstrom and
        corresponding energies.
        """
        # Ensure that logfile contains IRC calculation
        for line in self.lf_contents:
            if 'IRC optimization' in line:
                break
        else:
            raise QuantumError('Log file does not contain IRC calculation')

        # Find number of atoms
        natoms = self.getNumAtoms()

        # Read TS geometry
        for line_num, line in enumerate(self.lf_contents):
            if 'Geometry "geometry"' in line:
                ts_mat = self._formatArray(self.lf_contents[line_num + 7:line_num + 7 + natoms])
            if any(s in line for s in ('Total SCF energy', 'Total DFT energy')):
                ts_energy = float(line.split()[4])
                break
        else:
            raise QuantumError('TS geometry/energy could not be found in NWChem log file')

        # Read forward IRC path
        forwardpath = []
        energy = None
        for line_num, line in enumerate(self.lf_contents):
            if 'Optimization converged' in line:
                energy = float(self.lf_contents[line_num + 6].split()[2])
            if 'Geometry "geometry"' in line and energy is not None:
                coord_mat = self._formatArray(self.lf_contents[line_num + 7:line_num + 7 + natoms])
                forwardpath.append((coord_mat, energy))
            elif 'Backward IRC' in line:
                break
        else:
            raise QuantumError('Forward IRC path could not be found in NWChem log file')
        forwardpath.insert(0, (ts_mat, ts_energy))

        # Read reverse IRC path
        reversepath = []
        for line_num, line in enumerate(reversed(self.lf_contents)):
            if 'Geometry "geometry"' in line:
                coord_mat = self._formatArray(self.lf_contents[-(line_num - 6):-(line_num - 6 - natoms)])
            if 'Optimization converged' in line:
                energy = float(self.lf_contents[-(line_num - 5)].split()[2])
                reversepath.append((coord_mat, energy))
            elif 'Backward IRC' in line:
                break
        else:
            raise QuantumError('Reverse IRC path could not be found in NWChem log file')

        return reversepath + forwardpath

    def executeJob(self, node, name='nwc', jobtype='gradient',
                   theory='m062x/cc-pvtz', nproc=32, mem='2000mb',
                   output_dir='', **kwargs):
        """
        Execute quantum job type using the NWChem software package. This method
        can only be run on a UNIX system where NWChem is installed. Requires
        that the geometry is input in the form of a :class:`node.Node` object.
        """
        if not (sys.platform == 'linux' or sys.platform == 'linux2'):
            raise OSError('Invalid operating system')

        jobtype = jobtype.lower()
        if jobtype == 'sp':
            jobtype = 'energy'
        elif jobtype == 'opt' or jobtype == 'optimization':
            jobtype = 'optimize'
        elif jobtype == 'force' or jobtype == 'grad':
            jobtype = 'gradient'
        elif jobtype == 'ts':
            jobtype = 'saddle'
        elif jobtype == 'irc' or jobtype == 'rpath':
            jobtype = 'mepgs'

        # Split method and basis
        method, basis = theory.lower().split('/')

        # Reformat m06 and hf theories
        match = re.match(r"(m06)(\w+)", method, re.I)
        if match:
            method = '-'.join(match.groups())
        if method == 'hf' and node.multiplicity > 1:
            method = 'uhf'
        elif method == 'hf':
            method = 'rhf'

        # Reformat memory string
        match = re.match(r"([0-9]+)([a-z]+)", mem, re.I)
        if match:
            mem = ' '.join(match.groups())

        # Create NWChem input file
        try:
            input_file = os.path.join(output_dir, name + '.nw')
            with open(input_file, 'w') as f:
                f.write('title "' + name + '"\n')
                if jobtype == 'mepgs':
                    f.write('print low "total time"')
                f.write('memory ' + mem + '\n')
                f.write('geometry nocenter noautosym\n')  # Have to disable symmetry to obtain input orientation
                f.write(str(node) + '\n')
                f.write('end\n')
                f.write('basis\n')
                f.write('* library ' + basis + '\n')
                f.write('end\n')

                if jobtype == 'mepgs':
                    s = ('freq\n' + 'reuse ' + name + '.hess\n' + 'end\n' + 'mepgs\n' +
                         'opttol 0.0003\n' +
                         'stride 0.05\n' +
                         'maxmep 50\n' +
                         'maxiter 50\n' +
                         'inhess 2\n' +
                         'mswg\n' +
                         'end\n')
                else:
                    s = ''

                if method in ('rhf', 'rohf', 'uhf'):
                    f.write('scf\n')
                    f.write(method + '; ')
                    f.write('nopen ' + str(node.multiplicity - 1) + '\n')
                    f.write('end\n')
                    if jobtype == 'mepgs':
                        f.write('task scf freq\n')
                        f.write(s)
                    f.write('task scf ' + jobtype + '\n')
                elif method in ('ccsd', 'ccsd+t(ccsd)', 'ccsd(t)'):
                    f.write('task ' + method + ' ' + jobtype + '\n')
                else:
                    f.write('dft\n')
                    f.write('xc ' + method + '\n')
                    f.write('mult ' + str(node.multiplicity) + '\n')
                    f.write('end\n')
                    if jobtype == 'mepgs':
                        f.write('task dft freq\n')
                        f.write(s)
                    f.write('task dft ' + jobtype + '\n')
        except:
            print('An error occurred while creating the NWChem input file', file=sys.stderr)
            raise

        # Run job and wait until termination
        output_file = os.path.join(output_dir, name + '.log')
        try:
            subprocess.check_call(
                'srun -n {0} {1} {2} >& {3}'.format(nproc, 'nwchem', input_file, output_file), shell=True
            )
        except subprocess.CalledProcessError:
            if os.path.isfile(output_file):
                raise QuantumError('NWChem job terminated with an error')
            try:
                subprocess.check_call(
                    'mpiexec -n {0} {1} {2} > {3}'.format(nproc, 'nwchem', input_file, output_file), shell=True
                )
            except subprocess.CalledProcessError:
                if os.path.isfile(output_file):
                    raise QuantumError('NWChem job terminated with an error')
                raise Exception('NWChem is not available')

        # Remove unnecessary files and save log file
        os.remove(input_file)
        file_endings = ('.b', '.b^-1', '.c', '.cfock', '.cphf_rhs.00', '.cphf_sol.00', '.db', '.fd_ddipole',
                        '.fdrst', '.gsopt.hess', '.hess', '.irc.hess', '.movecs', '.movecs.hess', '.nmode', '.p',
                        '.zmat', '.drv.hess', '.t2')
        for e in file_endings:
            try:
                os.remove(name + e)
            except OSError:
                pass
        self.logfile = output_file
        self.read()

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
            elif 'Standard Nuclear Orientation' in line:
                read = True
        raise QuantumError('Number of atoms could not be found in Q-Chem log file')

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
        raise QuantumError('Energy could not be found in Gaussian log file')

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
                num_lines = int(np.ceil(natoms / 6)) * 4
                grad_mat_str = self.lf_contents[-line_num:-(line_num - num_lines)]
                break
        else:
            raise QuantumError('Gradient could not be found in NWChem log file')

        # Remove every 4th line and 1st column, and extract and flatten gradient components
        grad_mat_str = [row.split()[1:] for (i, row) in enumerate(grad_mat_str) if i % 4]
        xgrad = [row for (i, row) in enumerate(grad_mat_str) if not i % 3]
        ygrad = [row for (i, row) in enumerate(grad_mat_str[1:]) if not i % 3]
        zgrad = [row for (i, row) in enumerate(grad_mat_str[2:]) if not i % 3]
        xgrad = np.array([x for row in xgrad for x in row]).astype(float)
        ygrad = np.array([y for row in ygrad for y in row]).astype(float)
        zgrad = np.array([z for row in zgrad for z in row]).astype(float)

        # Create gradient array and convert units
        return 1.88972613 * np.column_stack((xgrad, ygrad, zgrad))

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
                break
            if 'Standard Nuclear Orientation' in line:
                coord_mat_str = self.lf_contents[-(line_num - 2):-(line_num - 2 - natoms)]
                break
        else:
            raise QuantumError('Geometry could not be found in Q-Chem log file')

        # Create and return array containing geometry
        return self._formatArray(coord_mat_str)

    def executeJob(self, node, name='qc', jobtype='force',
                   theory='m062x/cc-pvtz', nproc=32, mem='2000mb',
                   output_dir='', **kwargs):
        """
        Execute quantum job type using the Q-Chem software package. This method
        can only be run on a UNIX system where Q-Chem is installed. Requires
        that the geometry is input in the form of a :class:`node.Node` object.
        """
        if not (sys.platform == 'linux' or sys.platform == 'linux2'):
            raise OSError('Invalid operating system')

        jobtype = jobtype.lower()
        if jobtype == 'energy':
            jobtype = 'sp'
        elif jobtype == 'optimize':
            jobtype = 'opt'
        elif jobtype == 'gradient' or jobtype == 'grad':
            jobtype = 'force'
        elif jobtype == 'saddle':
            jobtype = 'ts'
        elif jobtype == 'irc' or jobtype == 'mepgs':
            jobtype = 'rpath'

        # Split theory and basis
        method, basis = theory.lower().split('/')

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
                f.write('method ' + method + '\n')
                f.write('sym_ignore true\n')  # Have to disable symmetry to obtain input orientation
                f.write('$end')
        except:
            print('An error occurred while creating the Q-Chem input file', file=sys.stderr)
            raise

        # Run job and wait until termination
        output_file = os.path.join(output_dir, name + '.log')
        try:
            subprocess.check_call('{0} -np {1} {2} {3}'.format('qchem', nproc, input_file, output_file), shell=True)
        except subprocess.CalledProcessError:
            if os.path.isfile(output_file):
                raise QuantumError('Q-Chem job terminated with an error')
            raise Exception('Q-Chem is not available')

        # Remove unnecessary files and save log file
        os.remove(input_file)
        os.remove('pathtable')
        self.logfile = output_file
        self.read()
