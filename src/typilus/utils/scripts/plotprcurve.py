#!/usr/bin/env python
"""
Usage:
    plotprcurve.py [options] PREDICTIONS_PATH TYPE_LATTICE_PATH ALIAS_METADATA

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""
import os
import sys
from typing import Tuple, List

import matplotlib
matplotlib.use('agg')

font = dict(family='normal', weight='bold', size=23)

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['ps.useafm'] = True

import matplotlib.pyplot as plt

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath
from sklearn.metrics import precision_recall_curve

from typilus.model.typelattice import TypeLattice
from typilus.model.utils import ignore_type_annotation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class MetricForPrecRec:
    def __init__(self, name: str) -> None:
        self.__name = name
        self.__collected_is_correct = []  # type: List[bool]
        self.__collected_confidences = []  # type: List[float]

    def add(self, is_correct: bool, confidence: float) -> None:
        self.__collected_is_correct.append(is_correct)
        self.__collected_confidences.append(confidence)

    def get_pr_curve(self, num_buckets: int=100) -> Tuple[np.ndarray, np.ndarray]:
        assert len(self.__collected_confidences) == len(self.__collected_is_correct)
        pr_curve = precision_recall_curve(self.__collected_is_correct, np.exp(self.__collected_confidences))
        return (self.__name, ) + pr_curve

    @property
    def num_elements(self) -> int:
        return len(self.__collected_is_correct)



def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    predictions_jsonl = RichPath.create(arguments['PREDICTIONS_PATH'], azure_info_path)
    type_lattice_path = RichPath.create(arguments['TYPE_LATTICE_PATH'], azure_info_path)
    alias_metadata_path = RichPath.create(arguments['ALIAS_METADATA'], azure_info_path)

    type_lattice = TypeLattice(type_lattice_path, 'typing.Any', alias_metadata_path)

    exact_match_metric = MetricForPrecRec('Exact Match')
    up_to_parametric_match_metric = MetricForPrecRec('Match Up to Parametric Type')
    type_correct_metric = MetricForPrecRec('Type Neutral')

    for prediction in predictions_jsonl.read_as_jsonl():
        ground_truth = prediction['original_annotation']
        if ignore_type_annotation(ground_truth):
            continue
        top_prediction, prediction_logprob = prediction["predicted_annotation_logprob_dist"][0]

        is_exact_match = type_lattice.are_same_type(ground_truth, top_prediction)
        exact_match_metric.add(is_exact_match, prediction_logprob)

        correct_up_to_parametric = is_exact_match or type_lattice.are_same_type(ground_truth.split("[")[0], top_prediction.split("[")[0])
        up_to_parametric_match_metric.add(correct_up_to_parametric, prediction_logprob)

        if is_exact_match:
            type_correct_metric.add(True, prediction_logprob)
        elif ground_truth in type_lattice and top_prediction in type_lattice:
            ground_truth_node_idx = type_lattice.id_of(ground_truth)
            predicted_node_idx = type_lattice.id_of(top_prediction)

            intersection_nodes_idx = type_lattice.intersect(ground_truth_node_idx, predicted_node_idx)
            is_ground_subtype_of_predicted = ground_truth_node_idx in intersection_nodes_idx
            type_correct_metric.add(is_ground_subtype_of_predicted, prediction_logprob)

    pr_curves = [
        exact_match_metric.get_pr_curve() + ('r--', exact_match_metric),
        up_to_parametric_match_metric.get_pr_curve() + ('b:', up_to_parametric_match_metric),
        type_correct_metric.get_pr_curve() + ('k-', type_correct_metric)
    ]

    fig = plt.figure(figsize=(5.5,2.5))
    ax = fig.add_subplot(111)
    for name, precision, recall, _, style, metric in pr_curves:
        print(name, metric.num_elements, recall, precision)
        print(f'{name}: {precision[0]:%}')
        ax.plot(recall, precision, style, label=name, linewidth=2)
    plt.grid()
    plt.xlim([0,1.005])
    plt.ylim([0,1.005])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig('test.pdf', dpi=300)



if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
