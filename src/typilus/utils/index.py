#!/usr/bin/env python
"""
Usage:
    index.py [options] MODEL_PATH DATA_PATH

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


def run_indexing(model_path: RichPath, index_data_path: RichPath):
    test_hyper_overrides = {
        'run_id': 'indexing',
        "dropout_keep_rate": 1.0,
    }

    data_chunks = index_data_path.get_filtered_files_in_dir('*.jsonl.gz')

    # Restore model
    model = model_restore_helper.restore(
        model_path, is_train=False, hyper_overrides=test_hyper_overrides)

    model.create_index(data_chunks)
    model.save(model_path)


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    data_folder = RichPath.create(arguments['DATA_PATH'], azure_info_path)
    model_path = RichPath.create(arguments['MODEL_PATH'])
    run_indexing(model_path, data_folder)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
