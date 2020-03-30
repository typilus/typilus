#!/usr/bin/env python
"""
Usage:
    predict.py [options] MODEL_PATH TEST_DATA_PATH OUTPUT_JSON_PATH

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""
import os
import sys
import time
from typing import Optional

from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug

from typilus.model import model_restore_helper
from typilus.model.utils import ignore_type_annotation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def ignore_annotation(annotation_str: Optional[str]) -> bool:
    if annotation_str is None:
        return False
    return ignore_type_annotation(annotation_str)


def run_predict(model_path: RichPath, test_data_path: RichPath, output_file: RichPath):
    test_run_id = "_".join(
        [time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])

    test_hyper_overrides = {
        'run_id': test_run_id,
        "dropout_keep_rate": 1.0,
    }

    test_data_chunks = test_data_path.get_filtered_files_in_dir('*.jsonl.gz')

    # Restore model
    model = model_restore_helper.restore(model_path, is_train=False, hyper_overrides=test_hyper_overrides)

    def predictions():
        for annotation in model.annotate(test_data_chunks):
            if ignore_annotation(annotation.original_annotation):
                continue
            ordered_annotation_predictions = sorted(annotation.predicted_annotation_logprob_dist,
                                    key=lambda x: -annotation.predicted_annotation_logprob_dist[x])[:10]

            annotation_dict = annotation._asdict()
            logprobs = annotation_dict['predicted_annotation_logprob_dist']
            filtered_logprobs = []
            for annot in ordered_annotation_predictions:
                logprob = float(logprobs[annot])
                if annot == '%UNK%' or annot == '%UNKNOWN%':
                    annot = 'typing.Any'
                filtered_logprobs.append((annot, logprob))
            annotation_dict['predicted_annotation_logprob_dist'] = filtered_logprobs

            yield annotation_dict


    output_file.save_as_compressed_file(predictions())


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    test_folder = RichPath.create(arguments['TEST_DATA_PATH'], azure_info_path)
    model_path = RichPath.create(arguments['MODEL_PATH'])
    output_file = RichPath.create(arguments['OUTPUT_JSON_PATH'])
    run_predict(model_path, test_folder, output_file)

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
