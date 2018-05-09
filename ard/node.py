#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`Node` for working with three-dimensional representations
of molecules in Cartesian coordinates and evaluating energies and gradients
using quantum chemical calculations.
"""

from __future__ import print_function, division

import sys

import numpy as np
import pybel

import props
import gen3D

###############################################################################

class Atom(object):
    """
    An atom with coordinates.
    The attributes are:

    =============== ======================= ===================================
    Attribute       Type                    Description
    =============== ======================= ===================================
    `atomicnum`     ``int``                 The atomic number of the atom
    `coords`        :class:`numpy.ndarray`  A 1 x 3 array containing the coordinates of the atom (in Angstrom)
    `idx`           ``int``                 The index of the atom
    =============== ======================= ===================================
    """

    def __init__(self, atomicnum, coords, idx):
        if isinstance(atomicnum, basestring):
            if atomicnum.upper() in props.atomnum.values():
                self.atomicnum = props.atomnum_inv[atomicnum]
        elif int(atomicnum) not in props.atomnum.keys():
            raise ValueError('Invalid atomic number or symbol: {0}'.format(atomicnum))
        else:
            self.atomicnum = int(atomicnum)

        self.coords = np.array(coords)
        if len(self.coords) > 3:
            raise Exception('Atom has more than three coordinates')

        self.idx = int(idx)

    @property
    def covrad(self): return props.covrad[self.atomicnum]

###############################################################################

class Node(object):
    """
    Three-dimensional representation of a molecular configuration.
    The attributes are:

    =============== ======================= ===================================
    Attribute       Type                    Description
    =============== ======================= ===================================
    `coords`        :class:`numpy.ndarray`  An N x 3 array containing the 3D coordinates of each atom (in Angstrom)
    `atoms`         ``list``                A list of length N containing the integer atomic number of each atom
    `multiplicity`  ``int``                 The multiplicity of this species, multiplicity = 2*total_spin+1
    `masses`        ``list``                A list of length N containing the masses of each atom
    `energy`        ``float``               The energy of the node (in Hartree)
    `gradient`      :class:`numpy.ndarray`  The gradient for the current node geometry (in Hartree/Angstrom)
    =============== ======================= ===================================

    N is the total number of atoms in the molecule. Each row in the coordinate
    array represents one atom.
    """

    def __init__(self, coords, atoms, multiplicity=1):
        try:
            self.coords = np.array(coords).reshape(len(atoms), 3)
        except ValueError:
            print('One or more atoms are missing a coordinate', file=sys.stderr)
            raise

        # self.atoms can be generated from atomic numbers or from atom labels
        self.atoms = []
        for num in atoms:
            if isinstance(num, basestring):
                if num.upper() in props.atomnum.values():
                    self.atoms.append(props.atomnum_inv[num])
                    continue
            if int(num) not in props.atomnum.keys():
                raise ValueError('Invalid atomic number or symbol: {0}'.format(num))
            self.atoms.append(int(num))

        self.multiplicity = int(multiplicity)

        self.masses = [props.atomweights[atom] for atom in self.atoms]
        self.energy = None
        self.gradient = None

    def __str__(self):
        """
        Return a human readable string representation of the object.
        """
        return_string = ''
        for anum, atom in enumerate(self.coords):
            return_string += '{0}  {1[0]: 14.8f}{1[1]: 14.8f}{1[2]: 14.8f}\n'.format(
                props.atomnum[self.atoms[anum]], atom)
        return return_string[:-1]

    def __repr__(self):
        """
        Return a representation of the object.
        """
        return 'Node({coords}, {self.atoms}, {self.multiplicity})'.format(coords=self.coords.tolist(), self=self)

    def copy(self):
        """
        Create copy of `self`.
        """
        new = Node(self.coords, self.atoms, self.multiplicity)
        new.masses = self.masses
        new.energy = self.energy
        new.gradient = self.gradient
        return new

    def hasSameConnectivity(self, other):
        """
        Check if `self` has the same connectivity as `other`. Return `True` if
        this is the case.
        """
        cmat = self.toConnectivityMat()
        cmat_other = other.toConnectivityMat()

        if np.array_equal(cmat, cmat_other):
            return True
        else:
            return False

    def getXYZ(self):
        """
        Return a string of the node in the XYZ file format.
        """
        return str(len(self.atoms)) + '\n\n' + str(self)

    def getListOfAtoms(self):
        """
        Make and return a list of `Atom` objects corresponding to the atoms in
        the node.
        """
        atoms = []
        for idx, (anum, coord) in enumerate(zip(self.atoms, self.coords)):
            atom = Atom(anum, coord, idx)
            atoms.append(atom)
        return atoms

    def getBonds(self):
        """
        Return a list of bonds of the structure. Each bond is a tuple of the
        two atom indices and the bond order (here only single bonds).
        """
        atoms = self.getListOfAtoms()
        bonds = []

        # Ensure there are coordinates to work with
        for atom in atoms:
            assert len(atom.coords) != 0

        # Sort atoms by distance on the z-axis
        sortedAtoms = sorted(atoms, key=lambda x: x.coords[2])

        for i, atom1 in enumerate(sortedAtoms):
            for atom2 in sortedAtoms[i + 1:]:
                # Set upper limit for bond distance
                criticalDistance = (atom1.covrad + atom2.covrad + 0.45) ** 2

                # First atom that is more than 4.0 Angstroms away in the z-axis, break the loop
                # Atoms are sorted along the z-axis, so all following atoms should be even further
                zBoundary = (atom1.coords[2] - atom2.coords[2]) ** 2
                if zBoundary > 16.0:
                    break

                distanceSquared = sum((atom1.coords - atom2.coords) ** 2)

                if distanceSquared > criticalDistance or distanceSquared < 0.40:
                    continue
                else:
                    bond = (atom1.idx, atom2.idx, 1)
                    bonds.append(bond)

        return bonds

    def toPybelMol(self):
        """
        Convert node to a :class:`pybel.Molecule` object.
        """
        mol = pybel.readstring('xyz', self.getXYZ())

        # Detect hydrogen molecules separately, since Open Babel often does not create a bond for these
        Hatoms = []
        for atom in mol:
            if atom.atomicnum == 1 and atom.OBAtom.BOSum() == 0:
                Hatoms.append(atom)
            elif atom.spin == 3 and self.multiplicity == 1:
                atom.OBAtom.SetSpinMultiplicity(1)

        if len(Hatoms) > 1:
            potential_Hmols = [[Hatom1, Hatom2]
                               for (idx1, Hatom1) in enumerate(Hatoms[:-1])
                               for Hatom2 in Hatoms[idx1 + 1:]]

            for potential_Hmol in potential_Hmols:
                coords1 = np.array(potential_Hmol[0].coords)
                coords2 = np.array(potential_Hmol[1].coords)

                distance = np.linalg.norm(coords1 - coords2)

                if distance <= 1.05:
                    mol.OBMol.AddBond(potential_Hmol[0].idx, potential_Hmol[1].idx, 1)
                    potential_Hmol[0].OBAtom.SetSpinMultiplicity(0)
                    potential_Hmol[1].OBAtom.SetSpinMultiplicity(0)

        mol.OBMol.SetTotalSpinMultiplicity(self.multiplicity)
        if self.energy is not None:
            mol.OBMol.SetEnergy(self.energy)
        return mol

    def toMolecule(self):
        """
        Convert node to a :class:`gen3D.Molecule` object.
        """
        mol = self.toPybelMol()
        mol = gen3D.Molecule(mol.OBMol)
        return mol

    def toRMGMolecule(self):
        """
        Convert node to a :class:`rmgpy.molecule.Molecule` object.
        """
        mol = self.toMolecule()
        return mol.toRMGMolecule()

    def toSMILES(self):
        """
        Return a SMILES representation of the node.
        """
        mol = self.toPybelMol()
        smiles = mol.write('can').strip()
        return smiles

    def toBEMat(self):
        """
        Return a bond and electron matrix (as a :class:`numpy.ndarray`)
        containing bonding information about the node.

        Note: The diagonal values are not populated.
        """
        natoms = len(self.atoms)
        BEmat = np.zeros((natoms, natoms), dtype=np.int)

        mol = self.toPybelMol()
        for bond in pybel.ob.OBMolBondIter(mol.OBMol):
            bond_tup = (bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBondOrder())

            BEmat[bond_tup[0], bond_tup[1]] = bond_tup[2]
            BEmat[bond_tup[1], bond_tup[0]] = bond_tup[2]

        return BEmat

    def toConnectivityMat(self):
        """
        Return the connectivity matrix of the node as :class:`numpy.ndarray`.
        """
        natoms = len(self.atoms)
        cmat = np.zeros((natoms, natoms), dtype=np.int)

        mol = self.toPybelMol()
        for bond in pybel.ob.OBMolBondIter(mol.OBMol):
            bond_tup = (bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1)

            cmat[bond_tup[0], bond_tup[1]] = 1
            cmat[bond_tup[1], bond_tup[0]] = 1

        return cmat

    def getTotalMass(self, atoms=None):
        """
        Compute and return total mass in g/mol. If a list of atoms is specified
        in `atoms`, only the corresponding atoms will be used to calculate the
        total mass.
        """
        if atoms is None:
            atoms = range(len(self.masses))
        return sum([self.masses[atom] for atom in atoms])

    def getCentroid(self):
        """
        Compute and return non-mass weighted centroid of molecular
        configuration.
        """
        return self.coords.sum(axis=0) / len(self.atoms)

    def getCenterOfMass(self, atoms=None):
        """
        Compute and return the position of the center of mass of the molecular
        configuration. If a list of atoms is specified in `atoms`, only the
        corresponding atoms will be used to calculate the center of mass.
        """
        if atoms is None:
            atoms = range(len(self.masses))
        mass = self.getTotalMass(atoms=atoms)
        center = sum([self.masses[atom] * self.coords[atom] for atom in atoms])
        return center / mass

    def translate(self, trans_vec):
        """
        Translate all atoms in the molecular configuration by `trans_vec`,
        which is of type :class:`numpy.ndarray` and of size 3 x 1.
        """
        self.coords += trans_vec

    def displaceCoordinates(self, mod_array):
        """
        Displaces the coordinates by adding the N x 3 :class:`numpy.ndarray`
        to the current coordinates.
        """
        self.coords += mod_array

    def rotate(self, rot_mat):
        """
        Rotate molecular configuration using orthogonal rotation matrix
        `rot_mat` which is of type :class:`numpy.ndarray` and of size 3 x 3.

        The node should first be translated to the origin, since rotation
        matrices can only describe rotations about the origin.
        """
        self.coords = (rot_mat.dot(self.coords.T)).T

    def computeEnergy(self, Qclass, **kwargs):
        """
        Compute and set energy of the node using the quantum program specified
        in `Qclass` with the options in `kwargs`.
        """
        q = Qclass()
        q.executeJob(self, jobtype='energy', **kwargs)
        self.energy = q.getEnergy()
        q.clear()

    def computeGradient(self, Qclass, **kwargs):
        """
        Compute and set gradient and energy of the node using the quantum
        program specified in `Qclass` with the options in `kwargs`.
        """
        q = Qclass()
        q.executeJob(self, jobtype='grad', **kwargs)
        self.energy = q.getEnergy()
        self.gradient = q.getGradient()
        q.clear()

    def computeFrequencies(self, Qclass, chkfile=None, **kwargs):
        """
        Compute force constants and frequencies of the node using the quantum
        program specified in `Qclass` with the options in `kwargs`. The number
        of imaginary frequencies and of gradient evaluations is returned.
        """
        q = Qclass(chkfile=chkfile)
        q.executeJob(self, jobtype='freq', **kwargs)
        self.energy = q.getEnergy()
        self.gradient = q.getGradient()
        nimag = q.getNumImaginaryFrequencies()
        ngrad = q.getNumGrad()

        return nimag, ngrad

    def optimizeGeometry(self, Qclass, ts=False, chkfile=None, **kwargs):
        """
        Perform a geometry optimization of the node using the quantum program
        specified in `Qclass` with the options in `kwargs`. If `ts` is True, a
        transition state search will be run. The node coordinates, gradient,
        and energy are updated. The number of gradient evaluations is returned.
        """
        q = Qclass(chkfile=chkfile)
        if ts:
            q.executeJob(self, jobtype='ts', **kwargs)
        else:
            q.executeJob(self, jobtype='opt', **kwargs)
        self.energy = q.getEnergy()
        self.gradient = q.getGradient()
        self.coords = q.getGeometry()
        ngrad = q.getNumGrad()
        q.clearChkfile()

        return ngrad

    def getIRCpath(self, Qclass, direction='forward', chkfile=None, **kwargs):
        """
        Execute an IRC path calculation in the given direction assuming that
        the current node geometry corresponds to a transition state. A list of
        :class:`node.Node` objects (with coordinates and energies) representing
        the nodes along the IRC path, and the number of gradient evaluations
        are returned. The transition state node is included in the IRC path.
        """
        q = Qclass(chkfile=chkfile)
        q.executeJob(self, jobtype='irc', direction=direction, **kwargs)
        path = q.getIRCpath()
        ngrad = q.getNumGrad()
        q.clearChkfile()

        nodepath = []
        for element in path:
            node = Node(element[0], self.atoms, self.multiplicity)
            node.energy = element[1]
            nodepath.append(node)

        return nodepath, ngrad

    def getDistance(self, other=None):
        """
        Compute and return linear distance between `self` and `other`. Returns
        -1 if other is not specified.
        """
        if other is None:
            return -1
        assert np.size(self.coords) == np.size(other.coords)
        diff = self.coords.flatten() - other.coords.flatten()
        return diff.dot(diff) ** 0.5

    def getTangent(self, other):
        """
        Calculate and return tangent direction between two nodes based on
        straight line path between the nodes. The tangent vector points from
        `self` to `other`, which are both of type :class:`node.Node`.
        """
        self_coord_vec = self.coords.flatten()
        other_coord_vec = other.coords.flatten()
        assert len(self_coord_vec) == len(other_coord_vec)
        diff = other_coord_vec - self_coord_vec
        return diff / np.linalg.norm(diff)
