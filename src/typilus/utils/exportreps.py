#!/usr/bin/env python
"""
Usage:
    exportreps.py [options] MODEL_PATH DATA_PATH TARGET_OUTPUT_FOLDER

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""

import os
import sys

from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug

from typilus.model import model_restore_helper

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def assert_valid_str(s: str):
    s = s.strip()
    assert '\n' not in s
    assert '\t' not in s
    return s

def run_export(model_path: RichPath, test_data_path: RichPath, output_folder: str):
    test_hyper_overrides = {
        'run_id': 'exporting',
        "dropout_keep_rate": 1.0,
    }

    data_chunks = test_data_path.get_filtered_files_in_dir('*gz')

    # Restore model
    model = model_restore_helper.restore(
        model_path, is_train=False, hyper_overrides=test_hyper_overrides)

    exporting = model.export_representations(data_chunks)

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'vectors.tsv'), 'w') as vectors_file,\
            open(os.path.join(output_folder, 'metadata.tsv'), 'w') as metadata_file:

        metadata_file.write('varname\ttype\tkind\tprovenance\n')
        for annot in exporting:
            metadata_file.write(f'{assert_valid_str(annot.name)}\t{assert_valid_str(annot.type_annotation)}\t{assert_valid_str(annot.kind)}\t{assert_valid_str(annot.provenance)}\n')
            vectors_file.write('\t'.join(str(e) for e in annot.representation))
            vectors_file.write('\n')


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    data_folder = RichPath.create(arguments['DATA_PATH'], azure_info_path)
    model_path = RichPath.create(arguments['MODEL_PATH'], azure_info_path)

    run_export(model_path, data_folder, arguments['TARGET_OUTPUT_FOLDER'])


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
