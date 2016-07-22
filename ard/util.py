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

import bisect
from functools import wraps
import logging
import os
import shutil
import subprocess
import time

###############################################################################

def submitProcess(cmd, *args):
    """
    Submit a process with the command, `cmd`, and arguments, `args`.
    """
    full_cmd = [cmd] + list(args)
    subprocess.check_call(full_cmd)

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
        logging.info('{} completed in {:.2f} s'.format(fn.__name__, final_time - start_time))
        return result
    return fnWithTime

def logStartAndFinish(fn):
    @wraps(fn)
    def fnWrappedWithLog(*args, **kwargs):
        logging.info('\n----------------------------------------------------------------------')
        logging.info('{} initiated on {}\n'.format(fn.__name__, time.asctime()))
        result = fn(*args, **kwargs)
        logging.info('\n{} terminated on {}'.format(fn.__name__, time.asctime()))
        logging.info('----------------------------------------------------------------------\n')
        return result
    return fnWrappedWithLog

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
