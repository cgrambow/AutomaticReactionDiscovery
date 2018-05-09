#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains classes for reading data from quantum software log files and for
executing quantum jobs.
"""

from __future__ import print_function, division

import os
import re
import subprocess
import sys

import numpy as np

import constants
import props

###############################################################################

class QuantumError(Exception):
    """
    An exception class for errors that occur during quantum calculations.
    """
    pass

###############################################################################

def submitProcess(cmd, *args):
    """
    Submit a process with the command, `cmd`, and arguments, `args`.
    """
    args = [str(arg) for arg in args]
    full_cmd = [cmd] + args
    subprocess.check_call(full_cmd)

def which(program):
    """
    Check if an executable file exists and return the path to it or `None` if
    it does not exist.
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

###############################################################################

class Quantum(object):
    """
    Base class from which other quantum software classes can inherit.

    The attribute `input_file` represents the path where the input file for the
    quantum job is located, the attribute `logfile` represents the path where
    the log file containing the results is located, the attribute `chkfile`
    represents the path where a checkpoint file for reading from a previous job
    is located, and the attribute `output` contains the output of the
    calculation in a list.
    """

    def __init__(self, input_file=None, logfile=None, chkfile=None):
        self.input_file = input_file
        self.logfile = logfile
        self.chkfile = chkfile
        if logfile is not None:
            self.read()
        else:
            self.output = None

    def read(self):
        """
        Reads the contents of the log file.
        """
        with open(self.logfile, 'r') as f:
            self.output = f.read().splitlines()

    def clear(self):
        """
        Deletes the input and log file and clears the associated attributes.
        """
        try:
            os.remove(self.input_file)
        except (OSError, TypeError):
            pass

        try:
            os.remove(self.logfile)
        except (OSError, TypeError):
            pass

        self.input_file = None
        self.logfile = None
        self.output = None

    def clearChkfile(self):
        """
        Deletes the checkpoint file and clears the associated attribute.
        """
        try:
            os.remove(self.chkfile)
        except (OSError, TypeError):
            pass

        self.chkfile = None

    def submitProcessAndCheck(self, cmd, *args):
        """
        Submits a quantum calculation and checks if errors occurred.
        """
        try:
            submitProcess(cmd, *args)
        except subprocess.CalledProcessError:
            if os.path.isfile(self.logfile):
                with open(self.logfile, 'r') as f:
                    f.seek(0, 2)
                    fsize = f.tell()
                    f.seek(max(fsize - 1024, 0), 0)  # Read last 1 kB of file
                    lines = f.readlines()

                lines = lines[-4:]  # Return last 4 lines

                msg = 'Quantum job terminated with an error:\n' + ''.join(lines)
                raise QuantumError(msg)
            raise

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
    """

    def __init__(self, input_file=None, logfile=None, chkfile=None):
        super(Gaussian, self).__init__(input_file=input_file, logfile=logfile, chkfile=chkfile)

    def getNumAtoms(self):
        """
        Extract and return number of atoms from Gaussian job.
        """
        read = False
        natoms = 0
        i = 0
        for line in self.output:
            if read:
                i += 1
                try:
                    natoms = int(line.split()[0])
                except ValueError:
                    if i > 5:
                        return natoms
                    continue
            elif 'Input orientation' in line or 'Z-Matrix orientation' in line:
                read = True
        raise QuantumError('Number of atoms could not be found in Gaussian output')

    def getEnergy(self):
        """
        Extract and return energy (in Hartree) from Gaussian job.
        """
        # Read last occurrence of energy
        for line in reversed(self.output):
            if 'SCF Done' in line:
                energy = float(line.split()[4])
                return energy
        raise QuantumError('Energy could not be found in Gaussian output')

    def getGradient(self):
        """
        Extract and return gradient (forces) from Gaussian job. Results are
        returned as an N x 3 array in units of Hartree/Angstrom.
        """
        natoms = self.getNumAtoms()

        # Read last occurrence of forces
        for line_num, line in enumerate(reversed(self.output)):
            if 'Forces (Hartrees/Bohr)' in line:
                force_mat_str = self.output[-(line_num - 2):-(line_num - 2 - natoms)]
                break
        else:
            raise QuantumError('Forces could not be found in Gaussian output')

        gradient = - self._formatArray(force_mat_str) / constants.bohr_to_ang  # Make negative to get gradient
        return gradient

    def getGeometry(self):
        """
        Extract and return final geometry from Gaussian job. Results are
        returned as an N x 3 array in units of Angstrom.
        """
        natoms = self.getNumAtoms()

        # Read last occurrence of geometry
        for line_num, line in enumerate(reversed(self.output)):
            if 'Input orientation' in line or 'Z-Matrix orientation' in line:
                coord_mat_str = self.output[-(line_num - 4):-(line_num - 4 - natoms)]
                break
        else:
            raise QuantumError('Geometry could not be found in Gaussian output')

        geometry = self._formatArray(coord_mat_str)
        return geometry

    def getIRCpath(self):
        """
        Extract and return IRC path from Gaussian job. Results are returned as
        a list of tuples of N x 3 coordinate arrays in units of Angstrom and
        corresponding energies in Hartrees. Path does not include TS geometry.
        """
        for line in self.output:
            if 'IRC-IRC' in line:
                break
        else:
            raise QuantumError('Gaussian output does not contain IRC calculation')

        natoms = self.getNumAtoms()

        # Read IRC path (does not include corrector steps of last point if there was an error termination)
        path = []
        for line_num, line in enumerate(self.output):
            if 'Input orientation' in line or 'Z-Matrix orientation' in line:
                coord_mat = self._formatArray(self.output[line_num + 5:line_num + 5 + natoms])
            elif 'SCF Done' in line:
                energy = float(line.split()[4])
            elif 'Forces (Hartrees/Bohr)' in line:
                force_mat_str = self.output[line_num + 3:line_num + 3 + natoms]
                gradient = - self._formatArray(force_mat_str) / constants.bohr_to_ang
            elif 'NET REACTION COORDINATE UP TO THIS POINT' in line:
                path.append((coord_mat, energy, gradient))

        if not path:
            raise QuantumError('IRC path is too short')
        return path

    def getNumImaginaryFrequencies(self):
        """
        Extract and return the number of imaginary frequencies from a Gaussian
        job.
        """
        for line in self.output:
            if 'imaginary frequencies' in line:
                nimag = int(line.split()[1])
                return nimag
        raise QuantumError('Frequencies could not be found in Gaussian output')

    def getNumGrad(self):
        """
        Extract and return the total number of gradient evaluations from a
        Gaussian job.
        """
        ngrad = 0
        for line in self.output:
            if 'Forces (Hartrees/Bohr)' in line:
                ngrad += 1
        return ngrad

    def makeInputFile(self, node, name='gau', jobtype='force', direction='forward', output_dir='',
                      theory='m062x/cc-pvtz', nproc=1, mem='2000mb', **kwargs):
        """
        Create Gaussian input file.
        """
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
        if jobtype not in ('sp', 'opt', 'force', 'freq', 'ts', 'irc'):
            raise Exception('Invalid job type')

        # Join memory string if there is a space between the number and unit
        match = re.match(r"([0-9]+) ([a-z]+)", mem, re.I)
        if match:
            mem = ''.join(match.groups())

        # Add dispersion for PM6 if Gaussian 09 is used
        g09path = which('g09')
        if g09path is not None and theory == 'pm6':
            dispersion = 'EmpiricalDispersion=gd3'
        else:
            dispersion = ''

        # Create Gaussian input file
        try:
            input_file = os.path.join(output_dir, name + '.com')
            with open(input_file, 'w') as f:
                fc = 'calcfc'
                if self.chkfile is not None:
                    f.write('%chk=' + self.chkfile + '\n')
                    if os.path.exists(self.chkfile):
                        fc = 'rcfc'
                f.write('%mem=' + mem + '\n')
                f.write('%nprocshared=' + str(int(nproc)) + '\n')
                if jobtype == 'opt':
                    f.write('# opt=(maxcycles=200) {} {} nosymm test\n\n'.format(theory, dispersion))
                elif jobtype == 'ts':
                    f.write('# opt=(ts,noeigen,{},maxcycles=100) {} {} nosymm test\n\n'.format(fc, theory, dispersion))
                elif jobtype == 'irc':
                    if direction == 1:
                        direction = 'forward'
                    elif direction == -1:
                        direction = 'reverse'
                    if direction not in ('forward', 'reverse'):
                        raise Exception('Invalid IRC direction')
                    if g09path is not None:
                        f.write(
                            '# irc=({},{},recalcfc=(predictor=10,corrector=5),maxpoints=60,'.format(direction, fc) +
                            'stepsize=8,maxcycle=25) {} {} iop(1/7=300) nosymm test\n\n'.format(theory, dispersion)
                        )
                    else:
                        f.write('# irc=({},{},maxpoints=60,stepsize=8,maxcycle=25) {} {} iop(1/7=300) nosymm test\n\n'.
                                format(direction, fc, theory, dispersion))

                else:
                    f.write('# {} {} {} nosymm test\n\n'.format(jobtype, theory, dispersion))
                f.write(name + '\n\n')
                f.write('0 ' + str(node.multiplicity) + '\n')

                for atom_num, atom in enumerate(node.coords):
                    f.write(' {0}              {1[0]: 14.8f}{1[1]: 14.8f}{1[2]: 14.8f}\n'.format(
                        props.atomnum[node.atoms[atom_num]], atom))
                f.write('\n')
        except:
            print('An error occurred while creating the Gaussian input file', file=sys.stderr)
            raise

        self.input_file = input_file

    def executeJob(self, node, name='gau', output_dir='', **kwargs):
        """
        Execute quantum job type using the Gaussian software package. This
        method can only be run on a UNIX system where Gaussian is installed.
        Requires that the geometry is input in the form of a :class:`node.Node`
        object.
        """
        if not (sys.platform == 'linux' or sys.platform == 'linux2'):
            raise OSError('Invalid operating system')

        if self.input_file is None:
            self.makeInputFile(node, name=name, output_dir=output_dir, **kwargs)

        # Run job (try Gaussian 09 first and then resort to Gaussian 03)
        self.logfile = os.path.join(output_dir, name + '.log')
        try:
            self.submitProcessAndCheck('g09', self.input_file, self.logfile)
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                self.submitProcessAndCheck('g03', self.input_file, self.logfile)
            else:
                raise

        # Read output
        self.read()

###############################################################################

class QChem(Quantum):
    """
    Class for reading data from Q-Chem log files and for executing Q-Chem jobs.
    """

    def __init__(self, input_file=None, logfile=None, chkfile=None):
        super(QChem, self).__init__(input_file=input_file, logfile=logfile, chkfile=chkfile)

    def getNumAtoms(self):
        """
        Extract and return number of atoms from Q-Chem job.
        """
        read = False
        natoms = 0
        i = 0
        for line in self.output:
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
        # Read last occurrence of energy
        for line in reversed(self.output):
            if 'Final energy is' in line:
                energy = float(line.split()[3])
                return energy
            if 'Total energy' in line:
                energy = float(line.split()[8])
                return energy
        raise QuantumError('Energy could not be found in Q-Chem log file')

    def getGradient(self):
        """
        Extract and return gradient (forces) from Q-Chem job. Results are
        returned as an N x 3 array in units of Hartree/Angstrom.
        """
        natoms = self.getNumAtoms()

        # Read last occurrence of gradient
        for line_num, line in enumerate(reversed(self.output)):
            if 'Gradient of SCF Energy' in line:
                num_lines = int(np.ceil(natoms / 6)) * 4
                grad_mat_str = self.output[-line_num:-(line_num - num_lines)]
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

        gradient = np.column_stack((xgrad, ygrad, zgrad)) / constants.bohr_to_ang
        return gradient

    def getGeometry(self):
        """
        Extract and return final geometry from Q-Chem job. Results are returned
        as an N x 3 array in units of Angstrom.
        """
        natoms = self.getNumAtoms()

        # Read last occurrence of geometry
        for line_num, line in enumerate(reversed(self.output)):
            if 'Coordinates' in line:
                coord_mat_str = self.output[-(line_num - 1):-(line_num - 1 - natoms)]
                break
            if 'Standard Nuclear Orientation' in line:
                coord_mat_str = self.output[-(line_num - 2):-(line_num - 2 - natoms)]
                break
        else:
            raise QuantumError('Geometry could not be found in Q-Chem log file')

        geometry = self._formatArray(coord_mat_str)
        return geometry

    def getIRCpath(self):
        """
        Extract and return IRC path from Q-Chem job. Results are returned as a
        list of tuples of N x 3 coordinate arrays in units of Angstrom and
        corresponding energies in Hartrees. Path does not include TS geometry
        """
        for line in self.output:
            if 'starting direction =' in line:
                break
        else:
            raise QuantumError('Q-Chem output does not contain IRC calculation')

        natoms = self.getNumAtoms()

        # Read IRC path
        path = []
        for line_num, line in enumerate(self.output):
            if 'Standard Nuclear Orientation' in line:
                coord_mat = self._formatArray(self.output[line_num + 3:line_num + 3 + natoms])
            elif 'Total energy' in line:
                energy = float(line.split()[8])
            elif 'Reaction path following' in line:
                path.append((coord_mat, energy))

        if len(path) == 1:
            raise QuantumError('IRC path is too short')
        return path[1:]

    def getNumGrad(self):
        """
        Extract and return the total number of gradient evaluations from a
        Q-Chem job.
        """
        ngrad = 0
        for line in self.output:
            if 'Gradient of SCF Energy' in line:
                ngrad += 1
        return ngrad

    def makeInputFile(self, node, name='qc', jobtype='force', direction=1, output_dir='',
                      theory='m062x/cc-pvtz', **kwargs):
        """
        Create Q-Chem input file.
        """
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
        if jobtype not in ('sp', 'opt', 'force', 'freq', 'ts', 'rpath'):
            raise Exception('Invalid job type')

        if theory == 'pm6':
            raise QuantumError('PM6 level of theory is not available in Q-Chem')

        # Split theory and basis into separate strings
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

                if jobtype == 'ts':
                    f.write('$rem\n')
                    f.write('jobtype freq\n')
                    f.write('basis ' + basis + '\n')
                    f.write('method ' + method + '\n')
                    f.write('$end\n')
                    f.write('\n@@@\n\n')
                    f.write('$molecule\n')
                    f.write('read\n')
                    f.write('$end\n')

                f.write('$rem\n')
                f.write('jobtype ' + str(jobtype) + '\n')
                f.write('basis ' + basis + '\n')
                f.write('method ' + method + '\n')
                f.write('sym_ignore true\n')  # Have to disable symmetry to obtain input orientation

                if jobtype == 'opt':
                    f.write('geom_opt_max_cycles 100\n')
                elif jobtype == 'ts':
                    f.write('geom_opt_max_cycles 100\n')
                    f.write('scf_guess read\n')
                    f.write('geom_opt_hessian read\n')
                elif jobtype == 'rpath':
                    if direction == 'forward':
                        direction = 1
                    elif direction == 'reverse':
                        direction = -1
                    if direction not in (1, -1):
                        raise Exception('Invalid IRC direction')
                    f.write('rpath_direction ' + str(direction) + '\n')
                    f.write('scf_guess read\n')
                    f.write('rpath_max_cycles 60\n')
                    f.write('rpath_max_stepsize 80\n')

                f.write('$end\n')
        except:
            print('An error occurred while creating the Q-Chem input file', file=sys.stderr)
            raise

        self.input_file = input_file

    def executeJob(self, node, name='qc', output_dir='', nproc=1, **kwargs):
        """
        Execute quantum job type using the Q-Chem software package. This method
        can only be run on a UNIX system where Q-Chem is installed. Requires
        that the geometry is input in the form of a :class:`node.Node` object.
        """
        if not (sys.platform == 'linux' or sys.platform == 'linux2'):
            raise OSError('Invalid operating system')

        if self.input_file is None:
            self.makeInputFile(node, name=name, output_dir=output_dir, **kwargs)

        # Run job
        self.logfile = os.path.join(output_dir, name + '.log')
        self.submitProcessAndCheck('qchem', '-np', 1, '-nt', nproc, self.input_file, self.logfile, 'save')
        os.remove(os.path.join(output_dir, 'pathtable'))

        # Read output
        self.read()

###############################################################################

class NWChem(Quantum):
    """
    Class for reading data from NWChem log files and for executing NWChem jobs.
    """

    def __init__(self, input_file=None, logfile=None, chkfile=None):
        super(NWChem, self).__init__(input_file=input_file, logfile=logfile, chkfile=chkfile)

    def getNumAtoms(self):
        """
        Extract and return number of atoms from NWChem job.
        """
        for line_num, line in enumerate(self.output):
            if 'XYZ format geometry' in line:
                natoms = int(self.output[line_num + 2].split()[0])
                return natoms
        raise QuantumError('Number of atoms could not be found in NWChem log file')

    def getEnergy(self):
        """
        Extract and return energy (in Hartree) from NWChem job.
        """
        # Read last occurrence of energy
        for line in reversed(self.output):
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
        natoms = self.getNumAtoms()

        # Read last occurrence of gradient
        for line_num, line in enumerate(reversed(self.output)):
            if 'ENERGY GRADIENTS' in line:
                grad_mat_str = self.output[-(line_num - 3):-(line_num - 3 - natoms)]
                break
        else:
            raise QuantumError('Gradient could not be found in NWChem log file')

        gradient = self._formatArray(grad_mat_str) / constants.bohr_to_ang
        return gradient

    def getGeometry(self):
        """
        Extract and return final geometry from NWChem job. Results are returned
        as an N x 3 array in units of Angstrom.
        """
        natoms = self.getNumAtoms()

        # Read last occurrence of geometry
        for line_num, line in enumerate(reversed(self.output)):
            if 'Geometry "geometry"' in line:
                coord_mat_str = self.output[-(line_num - 6):-(line_num - 6 - natoms)]
                break
        else:
            raise QuantumError('Geometry could not be found in NWChem log file')

        geometry = self._formatArray(coord_mat_str)
        return geometry

    def getIRCpath(self):
        """
        Extract and return IRC path from NWChem job. Results are returned as a
        list of tuples of N x 3 coordinate arrays in units of Angstrom and
        corresponding energies in Hartrees.
        """
        for line in self.output:
            if 'IRC optimization' in line:
                break
        else:
            raise QuantumError('Log file does not contain IRC calculation')

        natoms = self.getNumAtoms()

        # Read TS geometry
        for line_num, line in enumerate(self.output):
            if 'Geometry "geometry"' in line:
                ts_mat = self._formatArray(self.output[line_num + 7:line_num + 7 + natoms])
            if any(s in line for s in ('Total SCF energy', 'Total DFT energy')):
                ts_energy = float(line.split()[4])
                break
        else:
            raise QuantumError('TS geometry/energy could not be found in NWChem log file')

        # Read forward IRC path
        forwardpath = []
        energy = None
        for line_num, line in enumerate(self.output):
            if 'Optimization converged' in line:
                energy = float(self.output[line_num + 6].split()[2])
            if 'Geometry "geometry"' in line and energy is not None:
                coord_mat = self._formatArray(self.output[line_num + 7:line_num + 7 + natoms])
                forwardpath.append((coord_mat, energy))
            elif 'Backward IRC' in line:
                break
        else:
            raise QuantumError('Forward IRC path could not be found in NWChem log file')
        forwardpath.insert(0, (ts_mat, ts_energy))

        # Read reverse IRC path
        reversepath = []
        for line_num, line in enumerate(reversed(self.output)):
            if 'Geometry "geometry"' in line:
                coord_mat = self._formatArray(self.output[-(line_num - 6):-(line_num - 6 - natoms)])
            if 'Optimization converged' in line:
                energy = float(self.output[-(line_num - 5)].split()[2])
                reversepath.append((coord_mat, energy))
            elif 'Backward IRC' in line:
                break
        else:
            raise QuantumError('Reverse IRC path could not be found in NWChem log file')

        return reversepath + forwardpath

    def getNumGrad(self):
        """
        Extract and return the total number of gradient evaluations from an
        NWChem job.
        """
        # Implement this later on
        raise NotImplementedError

    def makeInputFile(self, node, name='nwc', jobtype='gradient', output_dir='',
                      theory='m062x/cc-pvtz', mem='2000mb', **kwargs):
        """
        Create NWChem input file.
        """
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

        # Split method and basis into separate strings
        method, basis = theory.lower().split('/')

        # Reformat m06 and hf theories (m06 has to be written as m-06)
        match = re.match(r"(m06)(\w+)", method, re.I)
        if match:
            method = '-'.join(match.groups())
        if method == 'hf' and node.multiplicity > 1:
            method = 'uhf'
        elif method == 'hf':
            method = 'rhf'

        # Separate number and unit in memory string if there is no space in between
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

        self.input_file = input_file

    def executeJob(self, node, name='nwc', output_dir='', nproc=1, **kwargs):
        """
        Execute quantum job type using the NWChem software package. This method
        can only be run on a UNIX system where NWChem is installed. Requires
        that the geometry is input in the form of a :class:`node.Node` object.
        """
        if not (sys.platform == 'linux' or sys.platform == 'linux2'):
            raise OSError('Invalid operating system')

        if self.input_file is None:
            self.makeInputFile(node, name=name, output_dir=output_dir, **kwargs)

        # Run job (try `srun` for SLURM and `mpiexec` for other systems)
        self.logfile = os.path.join(output_dir, name + '.log')
        try:
            self.submitProcessAndCheck('srun', '-n', nproc, 'nwchem', self.input_file, self.logfile)
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                self.submitProcessAndCheck('mpiexec', '-n', nproc, 'nwchem', self.input_file, self.logfile)
            else:
                raise

        # Remove unnecessary files
        file_endings = ('.b', '.b^-1', '.c', '.cfock', '.cphf_rhs.00', '.cphf_sol.00', '.db', '.fd_ddipole',
                        '.fdrst', '.gsopt.hess', '.hess', '.irc.hess', '.movecs', '.movecs.hess', '.nmode', '.p',
                        '.zmat', '.drv.hess', '.t2')
        for e in file_endings:
            try:
                os.remove(os.path.join(output_dir, name + e))
            except OSError:
                pass

        # Read output
        self.read()
