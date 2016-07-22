# AutomaticReactionDiscovery
Using quantum chemical computation to find important reactions without
requiring human intuition.

## Description
This repository contains Python code for automatically discovering chemical
reactions using a freezing string method with a subsequent exact transition
state search and reaction path verification by intrinsic reaction coordinate
calculation.

It can be run from the command line by providing the input filename as a second
argument. The full program can be run by executing
`python ard.py input.txt` or `python -m scoop ard.py input.txt`
from the command line, where input.txt can be replaced by any desired filename.
The latter option enables 3D product generation in parallel if SCOOP is
installed. (See the example SLURM script for extra arguments that may be
required by SCOOP)
A freezing string method (without exact TS search and IRC calculation) can be
run by executing
`python sm.py input.txt`
A full transition state search (for only one reaction) can be run by executing
`python tssearch.py input.txt`

Several arguments can be specified in the input file in the format _arg value_.
The possible arguments are:

* `reac_smi`       - A valid SMILES string describing the reactant structure
* `nbreak`         - The maximum number of bonds that may be broken
* `nform`          - The maximum number of bonds that may be formed
* `dH_cutoff`      - Heat of reaction cutoff (kcal/mol)
* `forcefield`     - The force field for 3D geometry generation
* `method`         - FSM or GSM (currently only FSM is supported)
* `geometry`       - Reactant and product geometry (see ard.py for details)
* `nsteps`         - Number or gradient calculations per optimization step
* `nnode`          - Number of nodes for calculation of interpolation distance
* `lsf`            - Line search factor for Newton-Raphson optimization
* `tol`            - Perpendicular gradient tolerance
* `nLSTnodes`      - Number of high density LST nodes
* `qprog`          - Program for quantum calculations (currently only 'gau')
* `theory`         - Level of theory (e.g., m062x/cc-pvtz)
* `theory_preopt`  - Level of theory for pre-optimization
* `reac_preopt`    - Boolean determining if reactant is pre-optimized
* `nproc`          - Number of processors per quantum calculation
* `mem`            - Memory requirements for quantum software

Only reactant and product have to be specified, all other arguments have
default values. The input file arguments can be specified in any order and
comments can be added. An example is given in _input.txt_.

## Dependencies
Required Python modules in addition to the standard library:

* **Numpy**
* **Scipy**
* **Open Babel**
* Some RMG functionality is also required

