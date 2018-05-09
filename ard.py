#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == '__main__':
    import argparse
    import os

    from ard.main import ARD, readInput

    # Set up parser for reading the input filename from the command line
    parser = argparse.ArgumentParser(description='Automatic Reaction Discovery')
    parser.add_argument('file', type=str, metavar='infile', help='An input file describing the job options')
    args = parser.parse_args()

    # Read input file
    input_file = os.path.abspath(args.file)
    kwargs = readInput(input_file)

    # Set output directory
    output_dir = os.path.abspath(os.path.dirname(input_file))
    kwargs['output_dir'] = output_dir

    # Execute job
    ard = ARD(**kwargs)
    ard.execute(**kwargs)
