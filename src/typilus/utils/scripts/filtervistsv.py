#!/usr/bin/env python
"""
Usage:
    exportreps.py [options] TYPE_FILTER INPUT_VECTORS INPUT_METADATA TARGET_OUTPUT_FOLDER

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""

import os
import sys

from docopt import docopt
from dpu_utils.utils import run_and_debug

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def run_export(type_filter: str, input_vectors_path: str, metadata_vectors_path: str, output_folder: str):
    with open(input_vectors_path) as f:
        vectors = f.readlines()

    with open(metadata_vectors_path) as f:
        metadata = f.readlines()[1:]

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'vectors.tsv'), 'w') as vectors_file,\
            open(os.path.join(output_folder, 'metadata.tsv'), 'w') as metadata_file:

        metadata_file.write('varname\ttype\tkind\tprovenance\n')
        for metadata, vectors in zip(metadata, vectors):
            name, annotated_type, kind, provenance = metadata.strip().split('\t')
            if annotated_type != type_filter:
                continue
            metadata_file.write(metadata)
            vectors_file.write(vectors)
            vectors_file.write('\n')


def run(arguments):
    run_export(arguments['TYPE_FILTER'], arguments['INPUT_VECTORS'], arguments['INPUT_METADATA'], arguments['TARGET_OUTPUT_FOLDER'])


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
