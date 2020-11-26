import unittest
from glob import iglob
from typing import Dict

from dpu_utils.codeutils import split_identifier_into_parts
from tqdm import tqdm

from graph_generator.graphgenerator import AstGraphGenerator
from graph_generator.graphgenutils import prettyprint_graph
from graph_generator.type_lattice_generator import TypeLatticeGenerator


class TestGraphGenerator(unittest.TestCase):
    lattice = TypeLatticeGenerator('../metadata/typingRules.json')

    @staticmethod
    def __get_generated_graph(fname: str):
        with open(fname) as f:
            b = AstGraphGenerator(f.read(), TestGraphGenerator.lattice)
        return b.build()

    def _validate_adjacent_tokens(self, g: Dict, edge_type: str):
        if edge_type not in g['edges']:
            return

        for v, us in g['edges'][edge_type].items():
            v_token = g['nodes'][v]
            for u in us:
                u_token = g['nodes'][u]
                self.assertEqual(v_token, u_token, f'{v_token} -> {u_token} is incorrect for edge type {edge_type}')

    def _validate_next(self, g: Dict):
        if 'NEXT' not in g['edges']:
            return

        for prev_node, next_node in zip(g['token-sequence'][:-1], g['token-sequence'][1:]):
            self.assertEqual(
                g['edges']['NEXT'][prev_node], [next_node],
                f'Next edges for {prev_node} are not [{next_node}]'
            )

    def _validate_subtoken_of(self, g: Dict):
        if 'SUBTOKEN_OF' not in g['edges']:
            return

        for v, us in g['edges']['SUBTOKEN_OF'].items():
            v_subtokens = split_identifier_into_parts(g['nodes'][v])
            for v_subtoken in v_subtokens:
                for u in us:
                    u_subtokens = split_identifier_into_parts(g['nodes'][u])
                    self.assertIn(v_subtoken, u_subtokens, f'{v_subtokens} is not in {u_subtokens}')

    def _validate_all(self, g: Dict):
        self._validate_next(g)
        self._validate_adjacent_tokens(g, 'NEXT_USE')
        self._validate_adjacent_tokens(g, 'LAST_LEXICAL_USE')
        self._validate_adjacent_tokens(g, 'OCCURRENCE_OF')
        self._validate_subtoken_of(g)

    def test_basic(self):
        g = self.__get_generated_graph('../test_data/basic.py')
        self.assertEqual(1, g['nodes'].count('FunctionDef'))
        self.assertEqual(4, g['nodes'].count('Assign'))
        self._validate_all(g)

    def test_decorated_functions(self):
        g = self.__get_generated_graph('../test_data/decorated_functions.py')
        self.assertEqual(10, g['nodes'].count('FunctionDef'))
        self.assertEqual(7, g['nodes'].count('@'))
        self._validate_all(g)

    def test_nested_comprehension(self):
        g = self.__get_generated_graph('../test_data/nested_comprehension.py')
        self.assertEqual(11, g['nodes'].count('comprehension'))
        self.assertEqual(8, g['nodes'].count('ListComp'))
        self._validate_all(g)

    def test_sanity_on_large_corpus(self):
        fnames = [fname for fname in iglob('../test_data/test_repositories/**/*.py', recursive=True)]
        for fname in tqdm(fnames):
            try:
                g = self.__get_generated_graph(fname)
                self._validate_all(g)
            except SyntaxError:
                pass
            except Exception as e:
                print(f'Failed on {fname}')
                raise e


if __name__ == '__main__':
    unittest.main()
