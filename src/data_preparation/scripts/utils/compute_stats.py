#!/usr/bin/env python
"""
Usage:
    stats.py [options] GRAPH_PATH OUTPUT_PATH

Options:
    -h --help                  Show this screen.
"""
from dpu_utils.utils import RichPath, run_and_debug
from docopt import docopt
import json
from collections import Counter
from itertools import chain

def run_stats(graph_path: RichPath, output_path: RichPath):
    number_graphs, number_annotations, number_variables = 0, 0, 0
    annotation_table = Counter()
    data_generator = chain(*(g.read_as_jsonl() for g in graph_path.iterate_filtered_files_in_dir('*.jsonl.gz')))
    for data in data_generator:
        number_graphs += 1 if len(data['supernodes']) > 0 else 0
        number_variables += len(data['supernodes'])
        number_annotations += sum(1 for supernode in data['supernodes'].values() if supernode['annotation'] not in {None, 'None', 'Nothing', 'Any'})
        annotation_table.update((supernode['annotation'] for supernode in data['supernodes'].values() if supernode['annotation'] not in {None, 'None', 'Nothing', 'Any'}))
    with open(output_path.to_local_path().path, "a") as f:
        f.write("Statistics for file: " +
                graph_path.to_local_path().path + "\n")
        f.write("Number of graphs: %d\n" % (number_graphs))
        f.write("Number of variables: %d\n" % (number_variables))
        f.write("Number of annotations: %d\n" % (number_annotations))
        f.write("Number of different annotations: %d\n" %
                (len(list(annotation_table))))
        f.write("\nFrequency distribution of annotations type:\n\n")
        for annotation, value in annotation_table.most_common():
            f.write("%s\t%d\n" % (annotation, value))


def run(arguments):
    graph_path = RichPath.create(arguments['GRAPH_PATH'])
    output_path = RichPath.create(arguments['OUTPUT_PATH'])
    run_stats(graph_path, output_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), True)
