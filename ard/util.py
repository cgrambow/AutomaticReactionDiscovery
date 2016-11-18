#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#
#   ARD - Automatic Reaction Discovery
#
#   Copyright (c) 2016 Prof. William H. Green (whgreen@mit.edu) and Colin
#   Grambow (cgrambow@mit.edu)
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
###############################################################################

"""
Provides utility functions and classes.
"""

from __future__ import division

import bisect
from functools import wraps
import logging
import os
import shutil
import time

import numpy as np

from quantum import Gaussian, QChem, NWChem

###############################################################################

def initializeLog(level, logfile, logname=None):
    """
    Configure a logger. `level` is an integer parameter specifying how much
    information is displayed in the log file. The levels correspond to those of
    the :data:`logging` module.
    """
    # Create logger
    logger = logging.getLogger(logname)
    logger.propagate = False
    logger.setLevel(level)

    logging.addLevelName(logging.CRITICAL, 'CRITICAL: ')
    logging.addLevelName(logging.ERROR, 'ERROR: ')
    logging.addLevelName(logging.WARNING, 'WARNING: ')
    logging.addLevelName(logging.INFO, '')
    logging.addLevelName(logging.DEBUG, '')

    # Create formatter
    formatter = logging.Formatter('%(levelname)s%(message)s')

    # Create file handler
    if os.path.exists(logfile):
        os.remove(logfile)
    fh = logging.FileHandler(filename=logfile)
    fh.setLevel(min(logging.DEBUG, level))
    fh.setFormatter(formatter)

    # Remove old handlers
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Add file handler
    logger.addHandler(fh)

    return logger

class Copier(object):
    """
    Function object that creates picklable function, `fn`, with a constant
    value for some arguments of the function, set as`self.args` and
    `self.kwargs`. This enables using `fn` in conjunction with `map` if the
    sequence being mapped to the function does not correspond to the first
    function argument and if the function has multiple arguments. `var_kw` has
    to specify the names of variable keyword arguments in a list in the order
    corresponding to any keyword arguments in `var_args`.
    """
    def __init__(self, fn, *const_args, **const_kwargs):
        self.fn = fn
        self.args = const_args
        self.kwargs = const_kwargs
        self.kw = self.kwargs.pop('var_kw', None)

    def __call__(self, *var_args):
        if self.kw is not None:
            num_var_kw = len(self.kw)
            args = self.args + var_args[:-num_var_kw]
            var_kwargs = {key: val for (key, val) in zip(self.kw, var_args[-num_var_kw:])}
            kwargs = dict(self.kwargs, **var_kwargs)
            return self.fn(*args, **kwargs)

        args = self.args + var_args
        return self.fn(*args, **self.kwargs)

def makeOutputSubdirectory(output_dir, folder):
    """
    Create a subdirectory `folder` in the output directory. If the folder
    already exists, its contents are deleted. Returns the path to the
    subdirectory.
    """
    subdir = os.path.join(output_dir, folder)
    if os.path.exists(subdir):
        shutil.rmtree(subdir)
    os.mkdir(subdir)
    return subdir

def timeFn(fn):
    @wraps(fn)
    def fnWithTime(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        final_time = time.time()
        # Has to be used for a method of a class that has a `logger` attribute
        args[0].logger.info('{} completed in {:.2f} s'.format(fn.__name__, final_time - start_time))
        return result
    return fnWithTime

def logStartAndFinish(fn):
    @wraps(fn)
    def fnWrappedWithLog(*args, **kwargs):
        # Has to be used for a method of a class that has a `logger` attribute
        args[0].logger.info('\n----------------------------------------------------------------------')
        args[0].logger.info('{} initiated on {}\n'.format(fn.__name__, time.asctime()))
        result = fn(*args, **kwargs)
        args[0].logger.info('{} terminated on {}'.format(fn.__name__, time.asctime()))
        args[0].logger.info('----------------------------------------------------------------------\n')
        return result
    return fnWrappedWithLog

def assignQclass(qprog):
    """
    Choose the appropriate quantum class based on the `qprog` string.
    """
    if qprog == 'gau':
        Qclass = Gaussian
    elif qprog == 'qchem':
        Qclass = QChem
    elif qprog == 'nwchem':
        Qclass = NWChem
    else:
        raise Exception('Invalid quantum software')

    return Qclass

def findClosest(a, x):
    """
    Returns index of value closest to `x` in sorted sequence `a`.
    """
    idx = bisect.bisect_left(a, x)
    if idx == 0:
        return a[0]
    if idx == len(a):
        return a[-1]
    if a[idx] - x < x - a[idx - 1]:
        return idx
    else:
        return idx - 1

def getDistMat(coords):
    """
    Calculate and return distance matrix form given an N x 3 array of
    Cartesian coordinates. The matrix is N x N. Only the upper diagonal
    elements contain the distances. All other elements are set to 0.
    """
    coords = coords.reshape(np.size(coords) // 3, 3)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    dx = x[..., np.newaxis] - x[np.newaxis, ...]
    dy = y[..., np.newaxis] - y[np.newaxis, ...]
    dz = z[..., np.newaxis] - z[np.newaxis, ...]
    return np.triu((np.array([dx, dy, dz]) ** 2).sum(axis=0) ** 0.5)

def rotationMatrix(angles):
    """
    Calculates and returns the rotation matrix defined by three angles of
    rotation about the x, y, and z axes.
    """
    Rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(angles[0]), -np.sin(angles[0])],
         [0.0, np.sin(angles[0]), np.cos(angles[0])]]
    )
    Ry = np.array(
        [[np.cos(angles[1]), 0.0, np.sin(angles[1])],
         [0.0, 1.0, 0.0],
         [-np.sin(angles[1]), 0.0, np.cos(angles[1])]]
    )
    Rz = np.array(
        [[np.cos(angles[2]), -np.sin(angles[2]), 0.0],
         [np.sin(angles[2]), np.cos(angles[2]), 0.0],
         [0.0, 0.0, 1.0]]
    )
    return Rx.dot(Ry).dot(Rz)
