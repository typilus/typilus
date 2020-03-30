import heapq
import logging
from collections import defaultdict
from functools import lru_cache
from typing import FrozenSet, List

from dpu_utils.utils import RichPath


class TypeLattice:
    LOGGER = logging.getLogger('TypeLattice')

    def __init__(self, type_lattice_path: RichPath, bottom_symbol: str,
                    built_in_aliases_path: RichPath):
        reference_lattice = type_lattice_path.read_as_json()
        id_aliases = {}
        self.__lattice_node_to_id = {}
        for i, n in enumerate(reference_lattice['nodes']):
            if n in self.__lattice_node_to_id:
                id_aliases[i] = self.__lattice_node_to_id[n]
            else:
                self.__lattice_node_to_id[n] = i

        assert self.__lattice_node_to_id[bottom_symbol] == 0, 'We assume below that bottom is at position 0'
        self.__id_to_node = reference_lattice['nodes']

        def alias_or_self(idx: int) -> int:
            if idx in id_aliases:
                return id_aliases[idx]
            else:
                return idx

        self.__aliases = dict(built_in_aliases_path.read_as_json()['aliasing_rules'])

        self.__is_a_edges = defaultdict(set)
        self.__child_edges = defaultdict(set)
        for from_id, to_id in reference_lattice['edges']:
            from_id = alias_or_self(from_id)
            to_id = alias_or_self(to_id)

            self.__is_a_edges[from_id].add(to_id)
            self.__child_edges[to_id].add(from_id)

        self._force_fix_lattice()

    def __contains__(self, item: str) -> bool:
        return item in self.__lattice_node_to_id

    def id_of(self, type_name: str) -> int:
        return self.__lattice_node_to_id[type_name]

    def _print_all_relationships(self) -> None:
        for child, parents in self.__is_a_edges.items():
            for parent in parents:
                print(f'{self.__id_to_node[child]}->{self.__id_to_node[parent]}')

    def _force_fix_lattice(self) -> None:
        """
        For this to be a (somewhat) valid lattice:
          * If some nodes are bottom (but they shouldn't) manually add them.
        """
        num_fixes = 0
        for from_id in range(len(self.__id_to_node)):
            if from_id == 0:
                continue  # Bottom symbol
            if len(self.__is_a_edges[from_id]) == 0:
                self.__is_a_edges[from_id].add(0)
                self.__child_edges[0].add(from_id)
                num_fixes += 1
        if num_fixes > 0:
            self.LOGGER.warning('%s nodes of %s did not have a supertype/parent. Adding a link to bottom, for now.', num_fixes, len(self.__id_to_node))

    def are_same_type(self, name1: str, name2: str) -> bool:
        if name1 == name2:
            return True

        def unaliased_name(name: str) -> str:
            while name in self.__aliases:
                name = self.__aliases[name]
            return name
        return unaliased_name(name1) == unaliased_name(name2)

    @lru_cache(maxsize=5000)
    def all_implemented_types(self, node_idx: int) -> FrozenSet[int]:
        nodes_to_visit = [node_idx]  # type: List[int]
        reachable_from_node = set()  # type: Set[int]
        while len(nodes_to_visit) > 0:
            next_node_idx = nodes_to_visit.pop()
            reachable_from_node.add(next_node_idx)
            nodes_to_visit.extend((n for n in self.__is_a_edges[next_node_idx] if n not in reachable_from_node))

        return frozenset(reachable_from_node)

    @lru_cache(maxsize=5000)
    def intersect(self, node1_idx: int, node2_idx: int) -> FrozenSet[int]:
        if node1_idx == node2_idx:
            return frozenset({node1_idx})

        # Build a set of all reachable nodes from node 1
        reachable_from_node1 = self.all_implemented_types(node1_idx)
        reachable_from_node2 = self.all_implemented_types(node2_idx)

        all_intersecting_nodes = reachable_from_node1 & reachable_from_node2

        assert len(all_intersecting_nodes), 'The lattice seems to have no bottom.'

        intersecting_nodes = set()  # type: Set[int]
        visited_nodes = set()
        nodes_to_visit = [0]  # type: List[int]
        while len(nodes_to_visit) > 0:
            next_node_idx = nodes_to_visit.pop()
            if next_node_idx in visited_nodes:  # In rare cases there is a loop (by mistake). Skip those examples.
                continue
            else:
                visited_nodes.add(next_node_idx)
            any_children_intersect = False
            for child_idx in self.__child_edges[next_node_idx]:
                if child_idx in all_intersecting_nodes:
                    any_children_intersect = True
                    nodes_to_visit.append(child_idx)
            if not any_children_intersect:
                intersecting_nodes.add(next_node_idx)

        return frozenset(intersecting_nodes)

    @lru_cache(maxsize=5000)
    def get_depth(self, type_idx: int) -> int:
        """
        Find the (min) depth of type_idx to Any.
        """
        if type_idx == 0:
            return 0

        nodes_to_visit = [(0, type_idx)]  # type: List[Tuple[int, int]]
        nodes_seen_or_in_queue = {type_idx}  # type: Set[int]

        while len(nodes_to_visit) > 0:
            current_depth, current_node_idx = heapq.heappop(nodes_to_visit)
            for to_idx in self.__is_a_edges[current_node_idx]:
                if to_idx == 0:
                    return current_depth + 1
                elif to_idx not in nodes_seen_or_in_queue:
                    nodes_seen_or_in_queue.add(to_idx)
                    heapq.heappush(nodes_to_visit, (current_depth + 1, to_idx))

        raise Exception('Should never reach this.')

    @lru_cache(maxsize=5000)
    def find_distance_to_intersection(self, from_idx: int, intersection_idx: int) -> int:
        """
        Find the minimum distance between from_idx to its supertype  intersection_idx.

        If the nodes to not intersect raise an exception.
        """
        if from_idx == intersection_idx:
            return 0

        nodes_to_visit = [(0, from_idx)]
        nodes_seen_or_in_queue = {from_idx}  # type: Set[int]

        while len(nodes_to_visit) > 0:
            current_distance, current_node_idx = heapq.heappop(nodes_to_visit)
            for to_idx in self.__is_a_edges[current_node_idx]:
                if to_idx == intersection_idx:
                    return current_distance + 1
                elif to_idx not in nodes_seen_or_in_queue:
                    nodes_seen_or_in_queue.add(to_idx)
                    heapq.heappush(nodes_to_visit, (current_distance + 1, to_idx))

        raise Exception('Nodes do not intersect.')
