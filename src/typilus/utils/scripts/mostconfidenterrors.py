#!/usr/bin/env python
"""
Usage:
    mostconfidenterrors.py [options] PREDICTIONS_JSONL

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""
import heapq

from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug

from typilus.model.utils import ignore_type_annotation


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    predictions_jsonl = RichPath.create(arguments['PREDICTIONS_JSONL'], azure_info_path)

    def errorenous_predictions():
        for prediction in predictions_jsonl.read_by_file_suffix():
            if ignore_type_annotation(prediction['original_annotation']):
                continue
            top_prediction = prediction["predicted_annotation_logprob_dist"][0][0]
            if top_prediction != prediction['original_annotation']:
                yield prediction

    most_confident_errors = heapq.nlargest(5000, errorenous_predictions(), key=lambda x: x["predicted_annotation_logprob_dist"][0][1])
    RichPath.create('most-confident-errors.jsonl.gz').save_as_compressed_file(most_confident_errors)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
