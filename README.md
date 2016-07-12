# AutomaticReactionDiscovery
Using quantum chemical computation to find important reactions without
requiring human intuition.

## Description
Currently, this repository contains Python code for executing a freezing string
method search to find transition states. It is also capable of running an exact
transition state search using the highest energy node from the freezing string
method and it can verify the results with an intrinsic reaction coordinate
calculation.

It can be run from the command line by providing the input filename as a second
argument. The full program can be run by executing
`python ard.py input.txt`
from the command line, where input.txt can be replaced by any desired filename.
A freezing string method (without exact TS search and IRC calculation) can be
run by executing
`python sm.py input.txt`

Several arguments can be specified in the input file in the format _arg value_.
The possible arguments are:

* `method`          - FSM or GSM (currently only FSM is supported)
* `reactant`        - Reactant geometry (specified like in Gaussian)
* `product`         - Product geometry
* `nsteps`          - Number or gradient calculations per optimization step
* `nnode`           - Number of nodes for calculation of interpolation distance
* `lsf`             - Line search factor for Newton-Raphson optimization
* `tol`             - Perpendicular gradient tolerance
* `nLSTnodes`       - Number of high density LST nodes
* `qprog`           - Program for quantum calculations (currently only 'gau')
* `theory`          - Level of theory (e.g., m062x/cc-pvtz)
* `theory_preopt`   - Level of theory for pre-optimization
* `nproc`           - Number of available processors
* `mem`             - Memory requirements for quantum software

Only reactant and product have to be specified, all other arguments have
default values. The input file arguments can be specified in any order and
comments can be added. An example is given in _input.txt_.

## Dependencies
Required Python modules in addition to the standard library:
**Numpy**
**Scipy**
**Open Babel**
