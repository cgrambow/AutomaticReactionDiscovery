#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains the :class:`Generate` for generating product structures.
"""

import pybel

import props
import gen3D

###############################################################################

class StructureError(Exception):
    """
    An exception class for errors that occur while generating structures.
    """
    pass

###############################################################################

class Generate(object):
    """
    Generation of product structures.
    The attributes are:

    =============== ========================== ================================
    Attribute       Type                       Description
    =============== ========================== ================================
    `reac_mol`      :class:`gen3D.Molecule`    A molecule object for the reactant structure
    `reac_smi`      ``str``                    Canonical SMILES of the reactant structure
    `atoms`         ``tuple``                  A tuple containing the atomic numbers of reactant/product structures
    `prod_mols`     ``list``                   A list of :class:`gen3D.Molecule` product structures
    =============== ========================== ================================

    Note: Bonds are represented as
          (beginAtomIdx, endAtomIdx, bondOrder)
    """

    def __init__(self, reac_mol):
        self.reac_mol = reac_mol
        self.reac_smi = None
        self.atoms = None
        self.prod_mols = []

        self.initialize()

    def initialize(self):
        """
        Set the canonical SMILES for the reactant and extract the atomic
        numbers.
        """
        self.reac_smi = self.reac_mol.write('can').strip()
        self.atoms = tuple(atom.atomicnum for atom in self.reac_mol)

    def generateProducts(self, nbreak=3, nform=3):
        """
        Generate all possible products from the reactant under the constraints
        of breaking a maximum of `nbreak` and forming a maximum of `nform`
        bonds.
        """
        if nbreak > 3 or nform > 3:
            raise Exception('Breaking/forming bonds is limited to a maximum of 3')

        # Extract bonds as an unmutable sequence (indices are made compatible with atom list)
        reactant_bonds = tuple(sorted(
            [(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond.GetBondOrder())
             for bond in pybel.ob.OBMolBondIter(self.reac_mol.OBMol)]
        ))

        # Extract valences as a mutable sequence
        reactant_valences = [atom.OBAtom.BOSum() for atom in self.reac_mol]

        # Initialize set for storing bonds of products
        # A set is used to ensure that no duplicate products are added
        products_bonds = set()

        # Generate all possibilities for forming bonds
        natoms = len(self.atoms)
        bonds_form_all = [(atom1_idx, atom2_idx, 1)
                          for atom1_idx in range(natoms - 1)
                          for atom2_idx in range(atom1_idx + 1, natoms)]

        # Generate products
        bf_combinations = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3))
        for bf in bf_combinations:
            if bf[0] <= nbreak and bf[1] <= nform:
                self._generateProductsHelper(
                    bf[0],
                    bf[1],
                    products_bonds,
                    reactant_bonds,
                    reactant_valences,
                    bonds_form_all
                )

        # Convert all products to Molecule objects and append to list of product molecules
        if products_bonds:
            reac_rmg_mol = self.reac_mol.toRMGMolecule()
            for bonds in products_bonds:
                mol = gen3D.makeMolFromAtomsAndBonds(self.atoms, bonds, spin=self.reac_mol.spin)
                mol.setCoordsFromMol(self.reac_mol)

                prod_rmg_mol = mol.toRMGMolecule()
                if not prod_rmg_mol.isIsomorphic(reac_rmg_mol):
                    self.prod_mols.append(mol)

    def _generateProductsHelper(self, nbreak, nform, products, bonds, valences, bonds_form_all, bonds_broken=None):
        """
        Generate products recursively given the number of bonds that should be
        broken and formed, a set for storing the products, a sequence of atoms,
        of bonds, and of valences. `bonds_form_all` should contain a tuple of
        tuples of bonds that contains all possibilities for forming bonds.

        Nothing is returned, but formed products are added to `products`.
        """
        if bonds_broken is None:
            bonds_broken = []
        if nbreak == 0 and nform == 0:
            # If no more bonds are to be changed, then add product (base case)
            products.add((tuple(sorted(bonds))))
        if nbreak > 0:
            # Break bond
            for bond_break_idx, bond_break in enumerate(bonds):
                valences_break = self.changeValences(valences, bond_break, -1)
                bonds_break = self.breakBond(bonds, bond_break_idx)

                # Keep track of bonds that have been broken
                if bond_break_idx == 0:
                    bonds_broken.append(bond_break)
                else:
                    bonds_broken[-1] = bond_break

                # Call function recursively to break next bond
                self._generateProductsHelper(
                    nbreak - 1,
                    nform,
                    products,
                    bonds_break,
                    valences_break,
                    bonds_form_all,
                    bonds_broken
                )

            # Remove last bond that has been broken after loop terminates
            del bonds_broken[-1]
        elif nform > 0:
            # Form bond
            for bond_form in bonds_form_all:
                # Do not add bond if it has previously been broken
                if bond_form[:2] in [bond[:2] for bond in bonds_broken]:
                    continue

                # Form new bond and catch exception if it violates constraints
                try:
                    valences_form = self.changeValences(valences, bond_form, 1)
                    bonds_form = self.formBond(bonds, bond_form)
                except StructureError:
                    continue

                # Call function recursively to form next bond
                self._generateProductsHelper(
                    nbreak,
                    nform - 1,
                    products,
                    bonds_form,
                    valences_form,
                    bonds_form_all,
                    bonds_broken
                )

    @staticmethod
    def breakBond(bonds, break_idx):
        """
        Break a bond given a tuple of tuples of bonds in the form:
            (atom index 1, atom index 2, bond type)
        The bond at index `break_idx` is broken. A tuple of tuples of updated
        bonds is returned.
        """
        # Break double or triple bond
        if bonds[break_idx][2] > 1:
            return bonds[:break_idx] + (bonds[break_idx][:2] + (bonds[break_idx][2] - 1,),) + bonds[(break_idx + 1):]

        # Break single bond
        return bonds[:break_idx] + bonds[(break_idx + 1):]

    @staticmethod
    def formBond(bonds, new_bond):
        """
        Form a bond given a tuple of tuples of bonds in the form:
            (atom index 1, atom index 2, bond type)
        If a bond already exists in `bonds`, the bond order is incremented. If
        a bond between atoms that are not yet connected is to be formed, then a
        new bond is added to the tuple of bonds. The bond to be added is
        specified in `new_bond`. A tuple of tuples of updated bonds is
        returned.
        """
        # Ensure that only one bond is added at a time
        assert new_bond[2] == 1

        try:  # Check if bond exists as single bond
            idx = bonds.index(new_bond)
        except ValueError:
            try:  # Check if bond exists as double bond
                idx = bonds.index(new_bond[:2] + (2,))
            except ValueError:
                try:  # Check if bond exists as triple bond
                    idx = bonds.index(new_bond[:2] + (3,))
                except ValueError:  # Add new bond if it does not exist yet
                    return bonds + (new_bond,)
                else:  # Raise exception if trying to exceed triple bond
                    raise StructureError('Bond type cannot be higher than triple bond for bond {}'.format(bonds[idx]))
            else:  # Return bonds with double bond increased to triple bond
                return bonds[:idx] + (bonds[idx][:2] + (3,),) + bonds[(idx + 1):]
        else:  # Return bonds with single bond increased to double bond
            return bonds[:idx] + (bonds[idx][:2] + (2,),) + bonds[(idx + 1):]

    def changeValences(self, valences, bond, inc):
        """
        Update the valences corresponding to each atom in `self.atoms` given
        the current valences in `valences`, the bond that is being affected,
        and the increment (typically 1 or -1). The maximum valences for each
        atom type are respected and an error is raised if they would be
        violated. A sequence of updated valences is returned.
        """
        # Create copy of current valences
        valences_temp = valences[:]

        # Check for invalid operation
        if inc < 0 and (valences_temp[bond[0]] < inc or valences_temp[bond[1]] < inc):
            raise Exception('Cannot decrease valence below zero-valence')

        # Change valences of both atoms participating in bond
        valences_temp[bond[0]] += inc
        valences_temp[bond[1]] += inc

        # Check if maximum valences are exceeded
        if valences_temp[bond[0]] > props.maxvalences[self.atoms[bond[0]]]:
            raise StructureError('Maximum valence on atom {} exceeded'.format(bond[0]))
        if valences_temp[bond[1]] > props.maxvalences[self.atoms[bond[1]]]:
            raise StructureError('Maximum valence on atom {} exceeded'.format(bond[1]))

        # Return valid valences
        return valences_temp

    def writeMolblock(self, bonds, valences):
        """
        Convert a sequence of bonds and valences corresponding to the atoms in
        `self.atoms` to an MDL molfile representation and return it as a
        string.

        Note: Atom indices in the MDL representation start at 1.
        """
        # Create counts line of MDL format
        counts_line = '{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n'.format(len(self.atoms), len(bonds))

        # Create atom block of MDL format
        atom_block = ''
        for idx, atom in enumerate(self.atoms):
            atom_block += ('    0.0000    0.0000    0.0000 {:<3} 0  0  0  0  0{:>3}  0  0  0  0  0  0\n'.
                           format(props.atomnum[atom], valences[idx]))

        # Create bond block of MDL format
        bond_block = ''
        for bond in bonds:
            bond_block += '{:>3}{:>3}{:>3}  0  0  0\n'.format(bond[0] + 1, bond[1] + 1, bond[2])

        return '\n\n\n' + counts_line + atom_block + bond_block + 'M  END\n'
