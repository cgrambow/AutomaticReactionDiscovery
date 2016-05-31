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
"arg = value". The possible arguments are:

* `reactant`        - Reactant geometry (specified like in Gaussian)
* `product`         - Product geometry
* `nsteps`          - Number or gradient calculations per optimization step
* `nnode`           - Number of nodes for calculation of interpolation distance
* `nLSTnodes`       - Number of high density LST nodes
* `interpolation`   - Interpolation method (cartesian or LST)
* `gaussian_ver`    - Gaussian version (typically g03 or g09)
* `level_of_theory` - Level of theory for quantum calculations (e.g., m062x/cc-pvtz)
* `nproc`           - Number of available processors
* `mem`             - Memory requirements
* `output_file`     - Name of output file to which the FSM nodes and energies are written

Only reactant and product have to be specified, all other arguments have
default values. The input file arguments can be specified in any order and
comments can be added. An example is given in 'input.txt'.
