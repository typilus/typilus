#!/usr/bin/env python
"""
Usage:
    test.py [options] MODEL_PATH TEST_DATA_PATH TYPE_LATTICE_PATH ALIAS_METADATA

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
    --print-predictions        Print raw predictions
"""
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Any, Optional

from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug

from typilus.model import model_restore_helper
from typilus.model.typelattice import TypeLattice
from typilus.model.utils import ignore_type_annotation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class TypePredictionEvaluator:
    def __init__(self, type_lattice_path: RichPath, alias_metadata_path: RichPath, bottom_symbol: str='typing.Any'):
        self.__num_samples = 0
        self.__num_samples_on_lattice = 0
        self.__exact_match = 0
        self.__accuracy_up_to_parametric_type = 0
        self.__type_consistency = 0
        self.__type_consistency_without_any = 0
        self.__reverse_type_consistency = 0
        self.__reciprocal_distance_sum = 0
        self.__sum_normalized_least_upper_bound_depth = 0
        self.__sum_reciprocal_least_upper_bound_depth = 0
        self.__per_type = defaultdict(lambda: defaultdict(int))
        self.__per_type_metrics = defaultdict(lambda: defaultdict(float))

        self.__type_lattice = TypeLattice(type_lattice_path, bottom_symbol, alias_metadata_path)

    def add_sample(self, ground_truth: str, predicted_dist: Dict[str, float]) -> None:
        self.__num_samples += 1

        predicted = max(predicted_dist, key=lambda x: predicted_dist[x])

        # Exact Match
        is_correct = self.__type_lattice.are_same_type(ground_truth, predicted)

        self.__exact_match += 1 if is_correct else 0
        self.__per_type[ground_truth][predicted] += 1

        # Accurancy up to parametric type
        is_accurate_utpt = self.__type_lattice.are_same_type(ground_truth.split("[")[0], predicted.split("[")[0])
        self.__accuracy_up_to_parametric_type += 1 if is_accurate_utpt else 0
        self.__per_type_metrics[ground_truth]["accuracy_up_to_parametric_type"] += 1 if is_accurate_utpt else 0

        if is_correct:
            # Avoid the hassle of computing the things below
            self.__num_samples_on_lattice += 1
            self.__per_type_metrics[ground_truth]["num_samples_on_lattice"] += 1

            self.__type_consistency += 1
            self.__type_consistency_without_any += 1
            self.__per_type_metrics[ground_truth]["type_consistency"] += 1
            self.__per_type_metrics[ground_truth]["type_consistency_without_any"] += 1

            self.__reverse_type_consistency += 1
            self.__per_type_metrics[ground_truth]["reverse_type_consistency"] += 1

            self.__reciprocal_distance_sum += 1
            self.__per_type_metrics[ground_truth]["reciprocal_distance"] += 1

            self.__sum_normalized_least_upper_bound_depth += 1
            self.__per_type_metrics[ground_truth]["sum_normalized_least_upper_bound_depth"] +=  1

            if ground_truth in self.__type_lattice:
                ground_truth_node_idx = self.__type_lattice.id_of(ground_truth)
                depth = self.__type_lattice.get_depth(ground_truth_node_idx)
                self.__sum_reciprocal_least_upper_bound_depth += (1./ depth) if depth > 0 else 0
                self.__per_type_metrics[ground_truth]["sum_reciprocal_least_upper_bound_depth"] += (1./ depth) if depth > 0 else 0

        elif self.__type_lattice is not None and ground_truth in self.__type_lattice and predicted in self.__type_lattice:
            self.__num_samples_on_lattice += 1
            self.__per_type_metrics[ground_truth]["num_samples_on_lattice"] += 1

            # Type Consistency and Directed Distance
            ground_truth_node_idx = self.__type_lattice.id_of(ground_truth)
            predicted_node_idx = self.__type_lattice.id_of(predicted)

            intersection_nodes_idx = self.__type_lattice.intersect(ground_truth_node_idx, predicted_node_idx)
            is_ground_subtype_of_predicted = ground_truth_node_idx in intersection_nodes_idx

            predicted_is_any = self.__type_lattice.are_same_type(predicted, 'typing.Any')
            self.__type_consistency += 1 if is_ground_subtype_of_predicted else 0
            self.__per_type_metrics[ground_truth]["type_consistency"] += 1 if is_ground_subtype_of_predicted else 0

            self.__type_consistency_without_any += 1 if is_ground_subtype_of_predicted and not predicted_is_any else 0
            self.__per_type_metrics[ground_truth]["type_consistency_without_any"] += 1 if is_ground_subtype_of_predicted and not predicted_is_any else 0


            is_predicted_subtype_of_ground = predicted_node_idx in intersection_nodes_idx
            self.__reverse_type_consistency += 1 if is_predicted_subtype_of_ground else 0
            self.__per_type_metrics[ground_truth]["reverse_type_consistency"] += 1 if is_predicted_subtype_of_ground else 0

            # Opportunistically, pick the first one. TODO: Reconsider later
            intersection_node_idx = next(iter(intersection_nodes_idx))

            intersection_depth = self.__type_lattice.get_depth(intersection_node_idx)
            max_depth = max(
                self.__type_lattice.get_depth(ground_truth_node_idx),
                self.__type_lattice.get_depth(intersection_node_idx)
            )

            if max_depth == 0:
                self.__sum_normalized_least_upper_bound_depth += 1
                self.__per_type_metrics[ground_truth]["sum_normalized_least_upper_bound_depth"] +=  1
            else:
                self.__sum_normalized_least_upper_bound_depth += intersection_depth / max_depth
                self.__per_type_metrics[ground_truth]["sum_normalized_least_upper_bound_depth"] +=  intersection_depth / max_depth

            self.__sum_reciprocal_least_upper_bound_depth += (1./ intersection_depth) if intersection_depth > 0 else 0
            self.__per_type_metrics[ground_truth]["sum_reciprocal_least_upper_bound_depth"] += (1./ intersection_depth) if intersection_depth > 0 else 0

            distance = self.__type_lattice.find_distance_to_intersection(ground_truth_node_idx, intersection_node_idx) +\
                            self.__type_lattice.find_distance_to_intersection(predicted_node_idx, intersection_node_idx)
            self.__reciprocal_distance_sum += 1 / (distance + 1.)
            self.__per_type_metrics[ground_truth]["reciprocal_distance"] += 1 / (distance + 1.)


    def metrics(self) -> Dict[str, Any]:
        metrics = {
            'accuracy': self.__exact_match / self.__num_samples,
            'accuracy_up_to_parametric_type': self.__accuracy_up_to_parametric_type / self.__num_samples,
            'count': self.__num_samples,
            'count_on_lattice': self.__num_samples_on_lattice
        }

        if self.__type_lattice is not None:
            metrics.update({
                'type_consistency': self.__type_consistency / self.__num_samples_on_lattice if self.__num_samples_on_lattice != 0 else 0.0,
                'type_consistency_without_any': self.__type_consistency_without_any / self.__num_samples_on_lattice if self.__num_samples_on_lattice != 0 else 0.0,
                'reverse_type_consistency': self.__reverse_type_consistency / self.__num_samples_on_lattice if self.__num_samples_on_lattice != 0 else 0.0,
                'reciprocal_distance': self.__reciprocal_distance_sum / self.__num_samples_on_lattice if self.__num_samples_on_lattice != 0 else 0.0,
                'normalized_least_upper_bound_depth': self.__sum_normalized_least_upper_bound_depth / self.__num_samples_on_lattice if self.__num_samples_on_lattice != 0 else 0.0,
                'reciprocal_least_upper_bound_depth': self.__sum_reciprocal_least_upper_bound_depth / self.__num_samples_on_lattice if self.__num_samples_on_lattice != 0 else 0.0,
            })

        per_type_stats = {}
        for type_annot, classifications in self.__per_type.items():
            num_samples = sum(classifications.values())
            num_samples_on_lattice = self.__per_type_metrics[type_annot]["num_samples_on_lattice"]

            per_type_stats[type_annot] = {
                'accuracy': classifications[type_annot] / num_samples,
                'accuracy_up_to_parametric_type': self.__per_type_metrics[type_annot]['accuracy_up_to_parametric_type'] / num_samples,
                'confusions': {c: classifications[c] / num_samples for c in classifications if c != type_annot},
                'count': num_samples
            }

            if self.__type_lattice is not None:
                per_type_stats[type_annot].update({
                    'type_consistency': self.__per_type_metrics[type_annot]['type_consistency'] / num_samples_on_lattice if num_samples_on_lattice != 0 else 0.0,
                    'type_consistency_without_any': self.__per_type_metrics[type_annot]['type_consistency_without_any'] / num_samples_on_lattice if num_samples_on_lattice != 0 else 0.0,
                    'reverse_type_consistency': self.__per_type_metrics[type_annot]['reverse_type_consistency'] / num_samples_on_lattice if num_samples_on_lattice != 0 else 0.0,
                    'reciprocal_distance': self.__per_type_metrics[type_annot]['reciprocal_distance'] / num_samples_on_lattice if num_samples_on_lattice != 0 else 0.0,
                    'normalized_least_upper_bound_depth': self.__per_type_metrics[type_annot]['sum_normalized_least_upper_bound_depth'] / num_samples_on_lattice if num_samples_on_lattice != 0 else 0.0,
                    'least_upper_bound_depth': self.__per_type_metrics[type_annot]['sum_reciprocal_least_upper_bound_depth'] / num_samples_on_lattice if num_samples_on_lattice != 0 else 0.0,
                })
        metrics['per_type_stats'] = per_type_stats

        return metrics


def run_test(model_path: RichPath, test_data_path: RichPath, type_lattice_path: RichPath, alias_metadata_path: RichPath, print_predictions: bool = False):
    test_run_id = "_".join(
        [time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])

    test_hyper_overrides = {
        'run_id': test_run_id,
        "dropout_keep_rate": 1.0,
    }

    test_data_chunks = test_data_path.get_filtered_files_in_dir('*gz')

    # Restore model
    model = model_restore_helper.restore(
        model_path, is_train=False, hyper_overrides=test_hyper_overrides)

    evaluator = TypePredictionEvaluator(type_lattice_path, alias_metadata_path)

    all_annotations = model.annotate(test_data_chunks)
    for annotation in all_annotations:
        if ignore_type_annotation(annotation.original_annotation):
            continue
        predicted_annotation = max(annotation.predicted_annotation_logprob_dist,
                                   key=lambda x: annotation.predicted_annotation_logprob_dist[x])
        if print_predictions:
            print(
                f'{annotation.provenance} -- {annotation.name}: {annotation.original_annotation} -> {predicted_annotation} ({math.exp(annotation.predicted_annotation_logprob_dist[predicted_annotation])*100:.1f}%)')
        evaluator.add_sample(ground_truth=annotation.original_annotation,
                             predicted_dist=annotation.predicted_annotation_logprob_dist)

    print(json.dumps(evaluator.metrics(), indent=2, sort_keys=True))


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    test_folder = RichPath.create(arguments['TEST_DATA_PATH'], azure_info_path)
    model_path = RichPath.create(arguments['MODEL_PATH'])
    type_lattice_path = RichPath.create(arguments['TYPE_LATTICE_PATH'], azure_info_path)
    alias_metadata_path = RichPath.create(arguments['ALIAS_METADATA'], azure_info_path)
    run_test(model_path, test_folder, type_lattice_path, alias_metadata_path,
             arguments['--print-predictions'])


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
