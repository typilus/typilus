import json
import sys
import numpy as np

from graph_generator.typeparsing.nodes import parse_type_comment
from graph_generator.typeparsing.visitor import TypeAnnotationVisitor


class NestingCounter(TypeAnnotationVisitor):

    def visit_subscript_annotation(self, node):
        return max(node.value.accept_visitor(self),
            node.slice.accept_visitor(self) if node.slice is not None else 0) + 1

    def visit_tuple_annotation(self, node):
        return max(
            (e.accept_visitor(self) for e in node.elements)
        )

    def visit_name_annotation(self, node):
        return 0

    def visit_list_annotation(self, node):
        if len(node.elements) == 0: return 0
        return max(
            (e.accept_visitor(self) for e in node.elements)
        )

    def visit_attribute_annotation(self, node):
        return node.value.accept_visitor(self)

    def visit_index_annotation(self, node):
        return node.value.accept_visitor(self)

    def visit_elipsis_annotation(self, node):
        return 0

    def visit_name_constant_annotation(self, node):
        return 0

    def visit_unknown_annotation(self, node):
        return 0

if len(sys.argv) != 2:
    print('Usage <file_to_analyze>')
    sys.exit()

with open(sys.argv[1]) as f:
    data = json.load(f)

def get_nesting(type_annotation_str: str) -> int:
    tt = parse_type_comment(type_annotation_str)
    if tt is None:
        return 0
    return tt.accept_visitor(NestingCounter())

all_nesting_levels = []

for type_name, type_data in data['per_type_stats'].items():    
    nesting = get_nesting(type_name)
    all_nesting_levels.extend([nesting] * type_data['count'])

all_nesting_levels = np.array(all_nesting_levels)

print('==== Statistics on All ====')
print(f'Mean {np.mean(all_nesting_levels)} Median {np.median(all_nesting_levels)} P90:{np.percentile(all_nesting_levels, 90)} '
         f'P95:{np.percentile(all_nesting_levels, 95)} P99:{np.percentile(all_nesting_levels, 99)}')

print('==== Statistics on Parametric ====')
print(f'Percent Parametric: {len(all_nesting_levels[all_nesting_levels > 0]) / len(all_nesting_levels):%}')
all_nesting_levels = all_nesting_levels[all_nesting_levels > 0]
print(f'Percent depth 1: {len(all_nesting_levels[all_nesting_levels == 1]) / len(all_nesting_levels):%}')
print(f'Mean {np.mean(all_nesting_levels)} Median {np.median(all_nesting_levels)} P90:{np.percentile(all_nesting_levels, 90)} '
         f'P95:{np.percentile(all_nesting_levels, 95)} P99:{np.percentile(all_nesting_levels, 99)}')


