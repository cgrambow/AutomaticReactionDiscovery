# AutomaticReactionDiscovery
Using quantum chemical computation to find important reactions without
requiring human intuition.

## Description
Currently, this repository contains Python code for executing a freezing string
method search to find transition states.

It can be run from the command line by providing the input filename as a second
argument. The program can be run by executing
	`python ard.py input.txt`
from the command line, where input.txt can be replaced by any desired filename.

Several arguments can be specified in the input file in the format
"arg value". The possible arguments are:

* `method`          - FSM or GSM (currently only FSM is supported)
* `reactant`        - Reactant geometry (specified like in Gaussian)
* `product`         - Product geometry
* `nsteps`          - Number or gradient calculations per optimization step
* `nnode`           - Number of nodes for calculation of interpolation distance
* `lsf`             - Line search factor for Newton-Raphson optimization
* `tol`             - Perpendicular gradient tolerance
* `nLSTnodes`       - Number of high density LST nodes
* `qprog`           - Program for quantum calculations ('g03', 'g09', 'nwchem', or 'qchem')
* `level_of_theory` - Level of theory for quantum calculations (e.g., m062x/cc-pvtz)
* `nproc`           - Number of available processors
* `mem`             - Memory requirements (only for Gaussian and NWChem)
* `output_file`     - Name of output file to which the FSM nodes and energies are written

Only reactant and product have to be specified, all other arguments have
default values. The input file arguments can be specified in any order and
comments can be added. An example is given in 'input.txt'.
