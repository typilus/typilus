import unittest

from graph_generator.graphgenerator import AstGraphGenerator
from graph_generator.type_lattice_generator import TypeLatticeGenerator


class TestGraphGenerator(unittest.TestCase):
    lattice = TypeLatticeGenerator('../metadata/typingRules.json')

    @staticmethod
    def __get_generated_graph(fname):
        with open(fname) as f:
            b = AstGraphGenerator(f.read(), TestGraphGenerator.lattice)
        return b.build()

    def test_basic(self):
        g = self.__get_generated_graph('../test_data/basic.py')
        self.assertEqual(1, g['nodes'].count('FunctionDef'))
        self.assertEqual(4, g['nodes'].count('Assign'))

    def test_decorated_functions(self):
        g = self.__get_generated_graph('../test_data/decorated_functions.py')
        self.assertEqual(10, g['nodes'].count('FunctionDef'))
        self.assertEqual(7, g['nodes'].count('@'))

    def test_nested_comprehension(self):
        g = self.__get_generated_graph('../test_data/nested_comprehension.py')
        self.assertEqual(11, g['nodes'].count('comprehension'))
        self.assertEqual(8, g['nodes'].count('ListComp'))

if __name__ == '__main__':
    unittest.main()
