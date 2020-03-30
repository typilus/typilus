#!/usr/bin/env python
"""
Usage:
    test.py [options] PREDICTIONS_JSONL_GZ TYPE_LATTICE_PATH ALIAS_METADATA_PATH

Options:
    -h --help                  Show this screen.
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""

from collections import defaultdict

from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug

from typilus.model.typelattice import TypeLattice
from typilus.model.utils import ignore_type_annotation


def compute(predictions_path: RichPath, type_lattice_path: RichPath, alias_metadata_path: RichPath):
    type_lattice = TypeLattice(type_lattice_path, 'typing.Any', alias_metadata_path)
    data = predictions_path.read_as_jsonl()

    total_per_kind = defaultdict(int)
    correct_per_kind = defaultdict(int)
    up_to_parameteric_per_kind = defaultdict(int)
    type_consistency_per_kind = defaultdict(int)
    total_per_kind_for_consistency = defaultdict(int)

    for prediction in data:
        annotation_type = prediction['annotation_type']
        original_annotation = prediction['original_annotation']
        top_predicted = prediction['predicted_annotation_logprob_dist'][0][0]
        if ignore_type_annotation(original_annotation):
            continue
        total_per_kind[annotation_type] += 1
        is_exact_match = type_lattice.are_same_type(original_annotation, top_predicted)
        if is_exact_match:
            correct_per_kind[annotation_type] += 1
        if type_lattice.are_same_type(original_annotation.split("[")[0], top_predicted.split("[")[0]):
            up_to_parameteric_per_kind[annotation_type] += 1

        if is_exact_match:
            type_consistency_per_kind[annotation_type] += 1
            total_per_kind_for_consistency[annotation_type] += 1
        elif original_annotation in type_lattice and top_predicted in type_lattice:
            # Type Consistency
            ground_truth_node_idx = type_lattice.id_of(original_annotation)
            predicted_node_idx = type_lattice.id_of(top_predicted)

            intersection_nodes_idx = type_lattice.intersect(ground_truth_node_idx, predicted_node_idx)
            is_ground_subtype_of_predicted = ground_truth_node_idx in intersection_nodes_idx
            total_per_kind_for_consistency[annotation_type] += 1
            if is_ground_subtype_of_predicted:
                type_consistency_per_kind[annotation_type] += 1

    print('== Exact Match')
    for annot_type in total_per_kind:
        print(f'{annot_type}: {correct_per_kind[annot_type] / total_per_kind[annot_type] :%} ({correct_per_kind[annot_type]}/{total_per_kind[annot_type]})')

    print('== Up to Parametric')
    for annot_type in total_per_kind:
        print(f'{annot_type}: {up_to_parameteric_per_kind[annot_type] / total_per_kind[annot_type] :%} ({up_to_parameteric_per_kind[annot_type]}/{total_per_kind[annot_type]})')

    print('== Consistency')
    for annot_type in total_per_kind:
        print(f'{annot_type}: {type_consistency_per_kind[annot_type] / total_per_kind_for_consistency[annot_type] :%} ({type_consistency_per_kind[annot_type]}/{total_per_kind_for_consistency[annot_type]})')



def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    predictions_path = RichPath.create(arguments['PREDICTIONS_JSONL_GZ'], azure_info_path)
    type_lattice_path = RichPath.create(arguments['TYPE_LATTICE_PATH'], azure_info_path)
    alias_metadata_path = RichPath.create(arguments['ALIAS_METADATA_PATH'], azure_info_path)
    compute(predictions_path, type_lattice_path, alias_metadata_path)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
