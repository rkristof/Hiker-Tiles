import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    import osmium  # noqa: F401
except ModuleNotFoundError:
    sys.modules['osmium'] = types.SimpleNamespace(SimpleHandler=object, InvalidLocationError=Exception)
try:
    import rasterio  # noqa: F401
except ModuleNotFoundError:
    sys.modules['rasterio'] = types.SimpleNamespace()

sys.path.insert(0, str(Path(__file__).parent))
import extract_hiking_routes as routes
from route_graph import LandmarkCategory, LandmarkIndex, landmark_candidate


class ConstantSampler:
    def __init__(self, elevation=100):
        self.elevation = elevation

    def sample(self, point):
        return self.elevation


class SteepSampler:
    def sample(self, point):
        return 100 if point[0] < 0.00075 else 120


class RouteGraphTests(unittest.TestCase):
    def test_landmark_categories_use_distance_tiers(self):
        highest = landmark_candidate({'information': 'guidepost'})
        direct_access = landmark_candidate({'amenity': 'parking'})
        transport_access = landmark_candidate({'highway': 'bus_stop'})

        self.assertEqual((highest['category'], highest['distance_limit_m']), (LandmarkCategory.HIGHEST, 30))
        self.assertEqual((direct_access['category'], direct_access['distance_limit_m']), (LandmarkCategory.HIGH, 60))
        self.assertEqual((transport_access['category'], transport_access['distance_limit_m']), (LandmarkCategory.MEDIUM, 90))
        self.assertGreater(LandmarkCategory.HIGHEST, LandmarkCategory.HIGH)
        self.assertGreater(LandmarkCategory.HIGH, LandmarkCategory.MEDIUM)

    def test_landmark_collection_only_targets_relations_without_start(self):
        class Tag:
            def __init__(self, key, value):
                self.k = key
                self.v = value

        class Member:
            def __init__(self, member_type, reference, role=''):
                self.type = member_type
                self.ref = reference
                self.role = role

        class Relation:
            def __init__(self, relation_id, members, tags=None):
                self.id = relation_id
                self.members = members
                self.tags = tags or [Tag('type', 'route'), Tag('route', 'hiking')]

        collector = routes.WayRouteCollector()
        with patch.dict(os.environ, {'COUNTRY': 'hungary', 'SYMBOL_TAG': 'osmc:symbol'}):
            collector.relation(Relation(1, [Member('w', 10)]))
            collector.relation(Relation(
                2,
                [Member('n', 20, 'start'), Member('w', 11)],
                [Tag('type', 'route'), Tag('route', 'hiking'), Tag('from', 'A'), Tag('to', 'B')],
            ))

        self.assertTrue(collector.relations[1]['needs_landmark_start'])
        self.assertFalse(collector.relations[2]['needs_landmark_start'])
        self.assertEqual(collector.relations[2]['from'], 'A')
        self.assertEqual(collector.relations[2]['to'], 'B')

    def test_landmark_collection_skips_irrelevant_way_geometry(self):
        class Way:
            id = 10
            tags = []

            @property
            def nodes(self):
                raise AssertionError('irrelevant way geometry was accessed')

        exporter = routes.GeoJSONExporter({}, io.StringIO(), collect_landmarks=True)
        exporter.way(Way())

        self.assertEqual(exporter.landmarks, [])

    def test_landmark_start_matches_existing_graph_node(self):
        relation = {
            'name': 'Gyadai tanösvény',
            'way_ids': [1, 2, 3],
            'node_roles': {},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (1, [0.0000, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start(
            {},
            None,
            [{
                'category': LandmarkCategory.HIGH,
                'distance_limit_m': 60,
                'name': 'Gyadai tanösvény kezdőpontja',
                'points': [[0.0010, 0.0000]],
            }],
        )

        self.assertEqual(start, 2)
        self.assertIn(start, graph._graph)

    def test_landmark_index_returns_only_nearby_points(self):
        index = LandmarkIndex([
            {'points': [[0.0000, 0.0000]], 'distance_limit_m': 30},
            {'points': [[0.0100, 0.0000]], 'distance_limit_m': 30},
        ])

        nearby = list(index.nearby([0.0001, 0.0000]))

        self.assertEqual(len(nearby), 1)
        self.assertEqual(nearby[0][0], 0)

    def test_landmark_distance_limit_is_enforced(self):
        relation = {
            'name': 'Route',
            'way_ids': [1, 2, 3],
            'node_roles': {},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (1, [0.0000, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start(
            {},
            None,
            [{
                'category': LandmarkCategory.HIGHEST,
                'distance_limit_m': 30,
                'points': [[0.00128, 0.0000]],
            }],
        )

        self.assertEqual(start, 1)

    def test_two_endpoints_ignore_landmarks_and_choose_member_order(self):
        relation = {
            'name': 'Route',
            'way_ids': [3, 2, 1],
            'node_roles': {},
            'roundtrip': True,
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
            3: [(3, [0.0020, 0.0000]), (4, [0.0030, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start(
            {},
            None,
            [{
                'category': LandmarkCategory.HIGHEST,
                'distance_limit_m': 30,
                'points': [[0.0030, 0.0000]],
            }],
        )

        finish = graph.resolve_finish(start, {}, None)

        self.assertEqual(start, 4)
        self.assertEqual(finish, 1)

    def test_two_endpoints_reverse_member_order_when_from_follows_to_in_name(self):
        relation = {
            'name': 'Finish - Start',
            'from': 'Start',
            'to': 'Finish',
            'way_ids': [1, 2, 3],
            'node_roles': {},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
            3: [(3, [0.0020, 0.0000]), (4, [0.0030, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start({}, None, ())
        finish = graph.resolve_finish(start, {}, None)

        self.assertEqual(start, 4)
        self.assertEqual(finish, 1)

    def test_explicit_start_ignores_landmarks(self):
        relation = {
            'name': 'Route',
            'way_ids': [1, 2, 3],
            'node_roles': {'start': [1]},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (1, [0.0000, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start(
            {},
            None,
            [{
                'category': LandmarkCategory.HIGHEST,
                'distance_limit_m': 30,
                'points': [[0.0010, 0.0000]],
            }],
        )

        self.assertEqual(start, 1)

    def test_missing_start_uses_closest_endpoint(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {}}
        way_nodes = {
            1: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
            2: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            3: [(3, [0.0020, 0.0000]), (4, [0.0040, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start({}, None, ())

        self.assertEqual(start, 1)
        self.assertEqual(graph._graph.degree(start), 1)

    def test_acyclic_branch_uses_open_traversal(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {'start': [1]}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (4, [0.0010, 0.0010])],
            3: [(2, [0.0010, 0.0000]), (5, [0.0010, -0.0010])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        steps, finish = graph.shortest_traversal(1)
        path, path_finish = graph.traversal_coordinates(1, steps)

        self.assertEqual(finish, 3)
        self.assertEqual(path_finish, 3)
        self.assertEqual(path[0], [0.0, 0.0])
        self.assertEqual(path[-1], [0.002, 0.0])
        self.assertIn([0.001, 0.001], path)
        self.assertIn([0.001, -0.001], path)
        self.assertEqual(len(steps), 6)

    def test_cycle_through_start_prefers_closed_traversal(self):
        relation = {'way_ids': [1, 2, 3, 4], 'node_roles': {'start': [1]}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0005, 0.0010])],
            3: [(3, [0.0005, 0.0010]), (1, [0.0000, 0.0000])],
            4: [(2, [0.0010, 0.0000]), (4, [0.0020, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        steps, finish = graph.shortest_traversal(1)
        path, path_finish = graph.traversal_coordinates(1, steps)

        self.assertEqual(finish, 1)
        self.assertEqual(path_finish, 1)
        self.assertEqual(path[0], [0.0, 0.0])
        self.assertEqual(path[-1], [0.0, 0.0])
        self.assertEqual(len(steps), 5)

    def test_retraced_leaf_does_not_create_roundtrip_cycle(self):
        relation = {
            'way_ids': [2, 3, 4, 1, 1, 5, 6, 6],
            'node_roles': {},
            'roundtrip': True,
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (4, [0.0000, 0.0010])],
            4: [(4, [0.0000, 0.0010]), (2, [0.0010, 0.0000])],
            5: [(3, [0.0010, 0.0010]), (5, [0.0020, 0.0010])],
            6: [(2, [0.0010, 0.0000]), (6, [0.0020, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start({}, None, ())
        self.assertIsNone(graph.resolve_finish(start, {}, None))
        steps, finish = graph.shortest_traversal(start)
        _, path_finish = graph.traversal_coordinates(start, steps)

        self.assertEqual(start, 1)
        self.assertEqual(finish, 5)
        self.assertEqual(path_finish, 5)

    def test_duplicate_members_without_open_endpoint_keep_closed_traversal(self):
        relation = {
            'way_ids': [1, 1, 2, 2, 3, 3, 4, 4],
            'node_roles': {},
            'roundtrip': True,
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (4, [0.0000, 0.0010])],
            4: [(4, [0.0000, 0.0010]), (2, [0.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start({}, None, ())
        self.assertIsNone(graph.resolve_finish(start, {}, None))
        steps, finish = graph.shortest_traversal(start)
        _, path_finish = graph.traversal_coordinates(start, steps)

        self.assertEqual(start, 1)
        self.assertEqual(finish, 1)
        self.assertEqual(path_finish, 1)

    def test_cycle_away_from_start_ignores_roundtrip(self):
        relation = {
            'way_ids': [1, 2, 3, 4],
            'node_roles': {'start': [1]},
            'roundtrip': True,
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (4, [0.0000, 0.0010])],
            4: [(4, [0.0000, 0.0010]), (2, [0.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        finish = graph.resolve_finish(1, {}, None)
        self.assertIsNone(finish)
        steps, finish = graph.shortest_traversal(1, finish)
        _, path_finish = graph.traversal_coordinates(1, steps)

        self.assertEqual(finish, 2)
        self.assertEqual(path_finish, 2)
        self.assertEqual(len(steps), 4)

    def test_roundtrip_returns_to_start(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {'start': [1]}, 'roundtrip': True}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (1, [0.0000, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        finish = graph.resolve_finish(1, {}, None)
        steps, finish = graph.shortest_traversal(1, finish)
        _, path_finish = graph.traversal_coordinates(1, steps)

        self.assertEqual(finish, 1)
        self.assertEqual(path_finish, 1)

    def test_explicit_finish_is_used(self):
        relation = {'way_ids': [1, 2], 'node_roles': {'start': [1], 'end': [3]}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        steps, finish = graph.shortest_traversal(1, 3)

        self.assertEqual(finish, 3)
        self.assertEqual(graph.traversal_coordinates(1, steps)[1], 3)

    def test_repair_requires_both_distance_thresholds(self):
        relation = {'way_ids': [1, 2], 'node_roles': {}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0005, 0.0000])],
            2: [(3, [0.0010, 0.0000]), (4, [0.0015, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        self.assertEqual(graph.repair_disconnected_components(ConstantSampler()), 1)
        self.assertEqual(graph.component_count, 1)

        far_graph = routes.RouteGraph(
            relation,
            {
                1: [(1, [0.0000, 0.0000]), (2, [0.0005, 0.0000])],
                2: [(3, [0.0020, 0.0000]), (4, [0.0025, 0.0000])],
            },
        )
        self.assertEqual(far_graph.repair_disconnected_components(ConstantSampler()), 0)

        steep_graph = routes.RouteGraph(relation, way_nodes)
        self.assertEqual(steep_graph.repair_disconnected_components(SteepSampler()), 0)

    def test_explicit_marker_snaps_to_unique_route_node(self):
        relation = {'way_ids': [1], 'node_roles': {'start': [99]}}
        way_nodes = {1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])]}
        graph = routes.RouteGraph(relation, way_nodes)

        start = graph.resolve_start(
            {99: [0.00101, 0.0000]},
            ConstantSampler(),
        )

        self.assertEqual(start, 2)


if __name__ == '__main__':
    unittest.main()
