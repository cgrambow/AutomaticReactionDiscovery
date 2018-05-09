#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains dictionaries of atomic properties, such as atomic numbers and masses.
"""

###############################################################################

# Atomic numbers
atomnum = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
atomnum_inv = dict((v, k) for k, v in atomnum.iteritems())

# Atomic weights in g/mol (from http://www.ciaaw.org/atomic-weights.htm#m)
atomweights = {1: 1.007975, 6: 12.0106, 7: 14.006855, 8: 15.9994, 9: 18.9984031636, 14: 28.085, 15: 30.9737619985,
               16: 32.0675, 17: 35.4515, 35: 79.904}

# Valence electrons of neutral atoms
valenceelec = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7, 14: 4, 15: 5, 16: 6, 17: 7, 35: 7}

# Maximum valences of atoms
maxvalences = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 7, 35: 7}

# Covalent radii in Angstrom (from the Cambridge Structural Database)
covrad = {1: 0.31, 6: 0.73, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.2}
