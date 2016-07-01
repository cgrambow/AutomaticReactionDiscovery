#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains functions and classes for generating 3D geometries using Openbabel.
"""

from __future__ import division

import pybel

from node import Node

###############################################################################

def readstring(format, string):
    """
    Read in a molecule from a string and convert to a :class:`OBGen` object.
    """
    mol = pybel.readstring(format, string)
    return OBGen(mol.OBMol)

def make3DandOpt(mol, forcefield='mmff94'):
    """
    Generate 3D coordinates and optimize them using a force field.
    """
    mol.make3D(forcefield=forcefield)
    mol.localopt(forcefield=forcefield)

###############################################################################

class OBGen(pybel.Molecule):
    """
    Extension of :class:`pybel.Molecule` for the generation of 3D geometries
    for structures containing more than one molecule.
    The attributes are:

    =============== ======================== ==================================
    Attribute       Type                     Description
    =============== ======================== ==================================
    `OBMol`         :class:`pybel.ob.OBMol`  An Openbabel molecule object
    `forcefield`    ``string``               Force field for coordinate generation
    `mols_indices`  :class:`tuple`           Tuple of lists containing indices of atoms in the molecules
    =============== ======================== ==================================

    """

    def __init__(self, OBMol, forcefield='mmff94'):
        super(OBGen, self).__init__(OBMol)
        self.forcefield = forcefield
        self.mols_indices = None

        # Delete stereochemistry information to prevent segmentation faults
        self.OBMol.DeleteData('StereoData')

    def copy(self):
        """
        Create copy of `self`. The copy is somewhat reduced in that it only
        contains atoms and bonds.
        """
        # Create new empty instance
        m = OBGen(pybel.ob.OBMol(), self.forcefield)

        # Add atoms and bonds
        for atom in self:
            m.OBMol.AddAtom(atom.OBAtom)
        for bond in pybel.ob.OBMolBondIter(self.OBMol):
            m.OBMol.AddBond(bond)

        # Return copy
        return m

    def toNode(self):
        """
        Convert to :class:`node.Node` object and return the object.
        """
        atoms = []
        coords = []
        for atom in self:
            atoms.append(atom.atomicnum)
            coords.append(atom.coords)
        return Node(coords, atoms, self.spin)

    def gen3D(self):
        """
        Generate 3D coordinates. If more than one molecule is present `self`,
        as indicated by the atom indices in `mols_indices`, then the geometries
        are generated separately for each molecule and are merged before being
        returned.
        """
        # Perform connectivity analysis to determine which molecules are present
        self.connectivityAnalysis()

        # Generate 3D geometry directly if there is only one molecule
        if len(self.mols_indices) == 1:
            make3DandOpt(self, forcefield=self.forcefield)

        # Generate 3D geometries separately for each molecule
        else:
            # Separate molecules
            mols = self.separateMol()

            # Arrange molecules in space
            arrange3D = Arrange3D(mols, forcefield=self.forcefield)
            arrange3D.arrangeIn3D()

            # Merge molecules
            self.mergeMols(mols)

    def separateMol(self):
        """
        Separate molecule and return a list of separated :class:`OBGen` objects
        based on the indices in `self.mols_indices`.
        """
        mols = []
        for mol_idx in range(len(self.mols_indices)):
            # Create copy
            mol = self.copy()

            # Obtain indices of all atoms to be deleted for current molecule and obtain corresponding atoms
            del_indices = [atom_idx for mol_idx_2, mol_indices in enumerate(self.mols_indices) if mol_idx_2 != mol_idx
                           for atom_idx in mol_indices]
            del_atoms = [mol.atoms[idx].OBAtom for idx in del_indices]

            # Delete atoms not in current molecule
            for atom in del_atoms:
                mol.OBMol.DeleteAtom(atom)

            # Append to list of molecules
            mols.append(mol)

        return mols

    def mergeMols(self, mols):
        """
        Merge molecules by clearing the current molecule and rewriting all
        atoms and bonds. The atoms are reordered according to the indices in
        `self.mols_indices`.
        """
        # Clear current molecule
        self.OBMol.Clear()

        # Loop through molecules and append atoms and bonds in order
        natoms = 0
        for mol in mols:
            for atom in mol:
                self.OBMol.AddAtom(atom.OBAtom)
            for bond in pybel.ob.OBMolBondIter(mol.OBMol):
                self.OBMol.AddBond(
                    bond.GetBeginAtomIdx() + natoms, bond.GetEndAtomIdx() + natoms, bond.GetBondOrder()
                )
            natoms += len(mol.atoms)

        # Reorder atoms
        mols_indices_new = [atom_idx for mol_indices in self.mols_indices for atom_idx in mol_indices]

        neworder = natoms * [0]
        for i, atom_idx in enumerate(mols_indices_new):
            neworder[atom_idx] = i + 1
        self.OBMol.RenumberAtoms(neworder)

    def connectivityAnalysis(self):
        """
        Analyze bonds to determine which atoms are connected and form a
        molecule.
        """
        # Extract bonds
        bonds = [[bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1] for bond in pybel.ob.OBMolBondIter(self.OBMol)]

        # Create first molecular fragment from first bond and start keeping track of atoms
        molecules = [bonds[0][:]]
        atoms_used = bonds[0][:]

        # Loop over remaining bonds
        for bond in bonds[1:]:
            ind1, ind2 = -1, -2

            for idx, molecule in enumerate(molecules):
                if bond[0] in molecule:
                    ind1 = idx
                if bond[1] in molecule:
                    ind2 = idx

            # Skip bond if both atoms are already contained in the same molecule
            if ind1 == ind2:
                continue
            # Combine fragments if they are connected through bond
            if ind1 != -1 and ind2 != -2:
                molecules[ind1].extend(molecules[ind2])
                del molecules[ind2]
            # Add new atom to fragment if it is connected
            elif ind1 != -1:
                molecules[ind1].append(bond[1])
                atoms_used.append(bond[1])
            elif ind2 != -2:
                molecules[ind2].append(bond[0])
                atoms_used.append(bond[0])
            # Add new fragment if it does not connect to any other ones
            else:
                molecules.append(bond)
                atoms_used.extend(bond)

        # Add atoms that are not involved in bonds
        for atom in range(len(self.atoms)):
            if atom not in atoms_used:
                molecules.append([atom])

        # Sort molecules before returning
        self.mols_indices = tuple(sorted(molecule) for molecule in molecules)

###############################################################################

class Arrange3D(object):
    """
    Arranging of :class:`OBGen` molecules in 3D space.
    The attributes are:

    =============== ===================== =====================================
    Attribute       Type                  Description
    =============== ===================== =====================================
    `mols`          :class:`list`         A list of :class:`OBGen` objects
    `forcefield`    ``string``            Force field for coordinate generation
    `d`             ``float``             Separation distance between molecules in Angstrom (excluding molecular radii)
    =============== ===================== =====================================

    """

    def __init__(self, mols, forcefield='mmff94', d=10.0):
        if not 1 < len(mols) <= 4:
            raise Exception('More than 4 molecules are not supported')
        self.mols = mols
        self.forcefield = forcefield
        self.d = d

    def gen3D(self):
        """
        Generate 3D geometries for each molecule.
        """
        for mol in self.mols:
            if len(mol.atoms) == 1:
                mol.atoms[0].OBAtom.SetVector(0.0, 0.0, 0.0)
            else:
                make3DandOpt(mol, forcefield=self.forcefield)

    def arrangeIn3D(self):
        """
        Arrange the molecules in 3D-space by modifying their coordinates. Two
        molecules are arranged in a line, three molecules in a triangle, and
        four molecules in a square.
        """
        # Generate geometries
        self.gen3D()

        # Center molecules and find their approximate radii
        sizes = self.centerAndFindDistances()

        nmols = len(self.mols)
        # Separate molecules by `d` if there are two molecules
        if nmols == 2:
            t = pybel.ob.vector3(self.d + sizes[0] + sizes[1], 0.0, 0.0)
            self.mols[1].OBMol.Translate(t)
        # Arrange molecules in triangle if there are three molecules
        elif nmols == 3:
            t1 = pybel.ob.vector3(self.d + sizes[0] + sizes[1], 0.0, 0.0)
            d1 = self.d + sizes[0] + sizes[1]
            d2 = self.d + sizes[0] + sizes[2]
            d3 = self.d + sizes[1] + sizes[2]
            y = (-d1 ** 4.0 + 2.0 * d1 ** 2.0 * d2 ** 2.0 + 2.0 * d1 ** 2.0 * d3 ** 2.0 -
                 d2 ** 4.0 + 2.0 * d2 ** 2.0 * d3 ** 2.0 - d3 ** 4.0) ** 0.5 / (2.0 * d1)
            x = (d2 ** 2.0 - y ** 2.0) ** 0.5
            t2 = pybel.ob.vector3(x, -y, 0.0)
            self.mols[1].OBMol.Translate(t1)
            self.mols[2].OBMol.Translate(t2)
        # Arrange molecules in square if there are four molecules
        elif nmols == 4:
            x = max(self.d + sizes[0] + sizes[1], self.d + sizes[2] + sizes[3])
            y = max(self.d + sizes[0] + sizes[2], self.d + sizes[1] + sizes[3])
            t1 = pybel.ob.vector3(x, 0.0, 0.0)
            t2 = pybel.ob.vector3(0.0, -y, 0.0)
            t3 = pybel.ob.vector3(x, -y, 0.0)
            self.mols[1].OBMol.Translate(t1)
            self.mols[2].OBMol.Translate(t2)
            self.mols[3].OBMol.Translate(t3)

    def centerAndFindDistances(self):
        """
        Center the molecules about the origin and return the distances between
        the origin and the atom farthest from the origin, which can be used as
        size estimates for the molecules.
        """
        max_distances = []
        for mol in self.mols:
            mol.OBMol.Center()
            max_distance, distance_prev = 0.0, 0.0
            for atom in mol:
                distance = atom.vector.length()
                if distance > distance_prev:
                    max_distance = distance
                distance_prev = distance
            max_distances.append(max_distance)
        return max_distances
