import io
import json
import os
import sys
import tempfile
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
from eligible_nodes import LandmarkIndex, landmark_candidate


class ConstantSampler:
    def __init__(self, elevation=100):
        self.elevation = elevation

    def sample(self, point):
        return self.elevation


class SteepSampler:
    def sample(self, point):
        return 100 if point[0] < 0.00075 else 120


def load_route_regression_cases():
    fixture_path = Path(__file__).with_name('route_regression_fixture.json')
    return json.loads(fixture_path.read_text())


class RouteGraphTests(unittest.TestCase):
    def test_route_graph_expands_compressed_traversal_geometry(self):
        relation = {'way_ids': [1], 'node_roles': {}}
        way_nodes = {
            1: [
                (1, [0.0000, 0.0000]),
                (2, [0.0010, 0.0010]),
                (3, [0.0020, 0.0000]),
            ],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        self.assertEqual(
            graph.traversal_coordinates([1, 3]),
            [point for _, point in way_nodes[1]],
        )
        self.assertEqual(
            graph.traversal_coordinates([3, 1]),
            [point for _, point in reversed(way_nodes[1])],
        )

    def test_write_route_lines_calculates_short_route_metadata_from_full_geometry(self):
        class FakeElevation:
            instances = []

            def __init__(self, directory):
                self.route_paths = []
                self.__class__.instances.append(self)

            def route_elevation(self, path):
                self.route_paths.append(path)
                return {
                    'elevation_gain_m': 12,
                    'elevation_loss_m': 3,
                    'profile': {
                        'segments': [{
                            'start_m': 0,
                            'end_m': routes.route_distance_m(path),
                            'elevations': [100, 112],
                            'coordinates': 'profile',
                        }],
                    },
                }

            @staticmethod
            def route_duration_min(distance_m, elevation_gain_m, elevation_loss_m):
                return 30

            def close(self):
                pass

        relation = {
            'name': 'Short route',
            'name_int': '',
            'symbol': '',
            'network': 'lwn',
            'type': 'hiking',
            'way_ids': [1],
            'node_roles': {'start': [1], 'end': [3]},
        }
        way_nodes = {
            1: [
                (1, [0.0000, 0.0000]),
                (2, [0.0010, 0.0010]),
                (3, [0.0020, 0.0000]),
            ],
        }
        collector = types.SimpleNamespace(relations={1: relation})
        exporter = types.SimpleNamespace(
            way_nodes=way_nodes,
            highway_way_ids_by_node={},
            landmarks=[],
        )

        with tempfile.TemporaryDirectory() as directory, \
            patch.object(routes, 'Elevation', FakeElevation), \
            patch.dict(os.environ, {'ELEVATION_DIRECTORY': directory}):
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                routes.write_route_lines(collector, exporter)
            finally:
                os.chdir(previous_directory)

            metadata = json.loads(Path(directory, 'routes-meta.json').read_text())
            feature = json.loads(Path(directory, 'hiking-routes-interaction.geojsonseq').read_text())

        expected_path = [point for _, point in way_nodes[1]]
        expected_distance = routes.route_distance_m(expected_path)
        self.assertEqual(feature['geometry']['coordinates'], expected_path)
        self.assertEqual(metadata[0]['distance_m'], expected_distance)
        self.assertEqual(metadata[0]['elevation_gain_m'], 12)
        self.assertEqual(metadata[0]['elevation_loss_m'], 3)
        self.assertEqual(metadata[0]['duration_min'], 30)
        self.assertEqual(FakeElevation.instances[-1].route_paths, [expected_path])

    def test_write_route_lines_falls_back_to_original_geometry(self):
        class FakeElevation:
            instances = []

            def __init__(self, directory):
                self.route_paths = []
                self.__class__.instances.append(self)

            def route_elevation(self, path):
                self.route_paths.append(path)
                return {
                    'elevation_gain_m': 0,
                    'elevation_loss_m': 0,
                    'profile': {'segments': []},
                }

            @staticmethod
            def route_duration_min(distance_m, elevation_gain_m, elevation_loss_m):
                return 15

            def close(self):
                pass

        relation = {
            'name': 'Fallback route',
            'name_int': '',
            'symbol': '',
            'network': 'lwn',
            'type': 'hiking',
            'way_ids': [1],
            'node_roles': {},
        }
        way_nodes = {
            1: [
                (1, [0.0000, 0.0000]),
                (2, [0.0010, 0.0010]),
                (3, [0.0020, 0.0000]),
            ],
        }
        collector = types.SimpleNamespace(relations={1: relation})
        exporter = types.SimpleNamespace(
            way_nodes=way_nodes,
            highway_way_ids_by_node={},
            landmarks=[],
        )

        with tempfile.TemporaryDirectory() as directory, \
            patch.object(routes, 'Elevation', FakeElevation), \
            patch.object(routes.RouteGraph, 'shortest_complete_traversal_simple', return_value=None), \
            patch.dict(os.environ, {'ELEVATION_DIRECTORY': directory}):
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                routes.write_route_lines(collector, exporter)
            finally:
                os.chdir(previous_directory)

            metadata = json.loads(Path(directory, 'routes-meta.json').read_text())
            feature = json.loads(Path(directory, 'hiking-routes-interaction.geojsonseq').read_text())

        expected_path = [point for _, point in way_nodes[1]]
        self.assertEqual(feature['geometry']['coordinates'], expected_path)
        self.assertEqual(metadata[0]['distance_m'], routes.route_distance_m(expected_path))
        self.assertEqual(metadata[0]['duration_min'], 15)
        self.assertEqual(FakeElevation.instances[-1].route_paths, [expected_path])

    def test_component_graphs_preserve_disconnected_components(self):
        relation = {'way_ids': [1, 2], 'node_roles': {}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(3, [1.0000, 0.0000]), (4, [1.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        components = graph.component_graphs()

        self.assertEqual(len(components), 2)
        self.assertEqual([component.component_count for component in components], [1, 1])
        self.assertEqual(
            [round(component.raw_route_distance_m()) for component in components],
            [routes.route_distance_m([point for _, point in way_nodes[1]]),
             routes.route_distance_m([point for _, point in way_nodes[2]])],
        )

    def test_short_disconnected_route_writes_multilinestring_and_offset_profile(self):
        class FakeElevation:
            PROFILE_MAX_DISTANCE_M = 40_000
            instances = []

            def __init__(self, directory):
                self.route_calls = 0
                self.__class__.instances.append(self)

            def sample(self, point):
                return 100

            def route_elevation(self, path):
                self.route_calls += 1
                distance_m = routes.route_distance_m(path)
                return {
                    'elevation_gain_m': 10,
                    'elevation_loss_m': 4,
                    'profile': {
                        'segments': [{
                            'start_m': 0,
                            'end_m': distance_m,
                            'elevations': [100, 110],
                            'coordinates': 'profile',
                        }],
                    },
                }

            @staticmethod
            def route_duration_min(distance_m, elevation_gain_m, elevation_loss_m):
                return 15

            def close(self):
                pass

        relation = {
            'name': 'Split route',
            'name_int': '',
            'symbol': '',
            'network': 'lwn',
            'type': 'hiking',
            'way_ids': [1, 2],
            'node_roles': {},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(3, [0.0100, 0.0000]), (4, [0.0110, 0.0000])],
        }
        collector = types.SimpleNamespace(relations={1: relation})
        exporter = types.SimpleNamespace(
            way_nodes=way_nodes,
            node_coordinates={},
            highway_way_ids_by_node={},
            landmark_index=None,
            landmarks=[],
        )

        with tempfile.TemporaryDirectory() as directory, \
            patch.object(routes, 'Elevation', FakeElevation), \
            patch.dict(os.environ, {'ELEVATION_DIRECTORY': directory}):
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                routes.write_route_lines(collector, exporter)
            finally:
                os.chdir(previous_directory)

            metadata = json.loads(Path(directory, 'routes-meta.json').read_text())
            feature = json.loads(Path(directory, 'hiking-routes-interaction.geojsonseq').read_text())

        self.assertEqual(len(metadata), 1)
        self.assertEqual(feature['geometry']['type'], 'MultiLineString')
        self.assertEqual(len(feature['geometry']['coordinates']), 2)
        self.assertEqual(metadata[0]['duration_min'], 30)
        self.assertEqual(metadata[0]['elevation_gain_m'], 20)
        self.assertEqual(metadata[0]['elevation_loss_m'], 8)
        self.assertIsNone(metadata[0]['start_lon'])
        self.assertIsNone(metadata[0]['start_lat'])
        self.assertIsNone(metadata[0]['finish_lon'])
        self.assertIsNone(metadata[0]['finish_lat'])
        profile = metadata[0]['elevation_profile']['segments']
        self.assertEqual(len(profile), 2)
        self.assertEqual(profile[0]['start_m'], 0)
        self.assertEqual(profile[1]['start_m'], routes.route_distance_m([point for _, point in way_nodes[1]]))
        self.assertEqual(FakeElevation.instances[-1].route_calls, 2)

    def test_closed_route_writes_finish_at_start(self):
        class FakeElevation:
            PROFILE_MAX_DISTANCE_M = 40_000

            def __init__(self, directory):
                pass

            def route_elevation(self, path):
                return {
                    'elevation_gain_m': 0,
                    'elevation_loss_m': 0,
                    'profile': {'segments': []},
                }

            @staticmethod
            def route_duration_min(distance_m, elevation_gain_m, elevation_loss_m):
                return 15

            def close(self):
                pass

        relation = {
            'name': 'Closed route',
            'name_int': '',
            'symbol': '',
            'network': 'lwn',
            'type': 'hiking',
            'way_ids': [1, 2, 3, 4],
            'node_roles': {'start': [1]},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (4, [0.0000, 0.0010])],
            4: [(4, [0.0000, 0.0010]), (1, [0.0000, 0.0000])],
        }
        collector = types.SimpleNamespace(relations={1: relation})
        exporter = types.SimpleNamespace(
            way_nodes=way_nodes,
            node_coordinates={},
            highway_way_ids_by_node={},
            landmark_index=None,
            landmarks=[],
        )

        with tempfile.TemporaryDirectory() as directory, \
            patch.object(routes, 'Elevation', FakeElevation), \
            patch.dict(os.environ, {'ELEVATION_DIRECTORY': directory}):
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                routes.write_route_lines(collector, exporter)
            finally:
                os.chdir(previous_directory)

            metadata = json.loads(Path(directory, 'routes-meta.json').read_text())
            feature = json.loads(Path(directory, 'hiking-routes-interaction.geojsonseq').read_text())

        self.assertEqual(feature['geometry']['coordinates'][0], [0.0, 0.0])
        self.assertEqual(feature['geometry']['coordinates'][-1], [0.0, 0.0])
        self.assertEqual(metadata[0]['start_lon'], 0.0)
        self.assertEqual(metadata[0]['start_lat'], 0.0)
        self.assertEqual(metadata[0]['finish_lon'], 0.0)
        self.assertEqual(metadata[0]['finish_lat'], 0.0)

    def test_long_connected_route_writes_line_without_duration_or_endpoints(self):
        class FakeElevation:
            PROFILE_MAX_DISTANCE_M = 40_000
            instances = []

            def __init__(self, directory):
                self.route_calls = 0
                self.__class__.instances.append(self)

            def sample(self, point):
                return 100

            def route_elevation(self, path):
                self.route_calls += 1
                raise AssertionError('long route must not sample elevation')

            @staticmethod
            def route_duration_min(distance_m, elevation_gain_m, elevation_loss_m):
                raise AssertionError('long route must not calculate duration')

            def close(self):
                pass

        relation = {
            'name': 'Long route',
            'name_int': '',
            'symbol': '',
            'network': 'lwn',
            'type': 'hiking',
            'way_ids': [1],
            'node_roles': {},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.5000, 0.0000])],
        }
        collector = types.SimpleNamespace(relations={1: relation})
        exporter = types.SimpleNamespace(
            way_nodes=way_nodes,
            node_coordinates={},
            highway_way_ids_by_node={},
            landmark_index=None,
            landmarks=[],
        )

        with tempfile.TemporaryDirectory() as directory, \
            patch.object(routes, 'Elevation', FakeElevation), \
            patch.dict(os.environ, {'ELEVATION_DIRECTORY': directory}):
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                routes.write_route_lines(collector, exporter)
            finally:
                os.chdir(previous_directory)

            metadata = json.loads(Path(directory, 'routes-meta.json').read_text())
            feature = json.loads(Path(directory, 'hiking-routes-interaction.geojsonseq').read_text())

        self.assertEqual(feature['geometry']['type'], 'LineString')
        self.assertIsNone(metadata[0]['duration_min'])
        self.assertIsNone(metadata[0]['elevation_gain_m'])
        self.assertIsNone(metadata[0]['elevation_loss_m'])
        self.assertEqual(metadata[0]['elevation_profile'], {'segments': []})
        self.assertIsNone(metadata[0]['start_lon'])
        self.assertIsNone(metadata[0]['finish_lon'])
        self.assertEqual(FakeElevation.instances[-1].route_calls, 0)

    def test_landmark_catalog_contains_only_highest_tier(self):
        highest = landmark_candidate({'information': 'guidepost'})

        self.assertEqual(highest['tag_key'], 'information')
        self.assertIsNone(landmark_candidate({'amenity': 'parking'}))
        self.assertIsNone(landmark_candidate({'highway': 'bus_stop'}))

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
                [
                    Tag('type', 'route'),
                    Tag('route', 'hiking'),
                    Tag('from', 'A'),
                    Tag('to', 'B'),
                    Tag('roundtrip', 'yes'),
                ],
            ))

        self.assertFalse(collector.relations[1]['node_roles'].get('start'))
        self.assertTrue(collector.relations[2]['node_roles'].get('start'))
        self.assertEqual(collector.relations[2]['node_ids'], [20])
        self.assertEqual(collector.relations[2]['name'], 'A - B')
        self.assertTrue(collector.relations[2]['roundtrip'])

    def test_flatten_nested_routes_merges_recursive_children_and_suppresses_them(self):
        class Tag:
            def __init__(self, key, value):
                self.k = key
                self.v = value

        class Member:
            def __init__(self, member_type, reference):
                self.type = member_type
                self.ref = reference
                self.role = ''

        class Relation:
            def __init__(self, relation_id, members):
                self.id = relation_id
                self.members = members
                self.tags = [Tag('type', 'route'), Tag('route', 'hiking')]

        collector = routes.WayRouteCollector()
        with patch.dict(os.environ, {'COUNTRY': 'hungary', 'SYMBOL_TAG': 'osmc:symbol'}):
            collector.relation(Relation(3, [Member('w', 30)]))
            collector.relation(Relation(2, [Member('w', 20), Member('r', 3)]))
            collector.relation(Relation(1, [Member('w', 10), Member('r', 2), Member('w', 11)]))

        collector.flatten_nested_routes()

        self.assertEqual(list(collector.relations), [1])
        self.assertEqual(collector.relations[1]['way_ids'], [10, 20, 30, 11])
        self.assertEqual(
            collector.relations[1]['route_members'],
            [('w', 10), ('w', 20), ('w', 30), ('w', 11)],
        )

    def test_flatten_nested_routes_preserves_repeated_way_members(self):
        collector = routes.WayRouteCollector()
        collector.relations = {
            1: {
                'way_ids': [10, 11, 10],
                'route_members': [('w', 10), ('w', 11), ('w', 10)],
            },
        }

        collector.flatten_nested_routes()

        self.assertEqual(collector.relations[1]['way_ids'], [10, 11, 10])
        self.assertEqual(
            collector.relations[1]['route_members'],
            [('w', 10), ('w', 11), ('w', 10)],
        )

    def test_landmark_collection_skips_irrelevant_way_geometry(self):
        class Way:
            id = 10
            tags = {}

            @property
            def nodes(self):
                raise AssertionError('irrelevant way geometry was accessed')

        exporter = routes.GeoJSONExporter({}, io.StringIO(), collect_landmarks=True)
        exporter.way(Way())

        self.assertEqual(exporter.landmarks, [])

    def test_highway_access_collector_records_matching_route_nodes(self):
        class Node:
            def __init__(self, node_id):
                self.ref = node_id

        class Way:
            def __init__(self, way_id, highway_type, node_ids):
                self.id = way_id
                self.tags = {'highway': highway_type}
                self.nodes = [Node(node_id) for node_id in node_ids]

        collector = routes.HighwayAccessCollector({2})
        collector.way(Way(20, 'residential', [1, 2]))

        self.assertEqual(collector.accessible_highway_way_ids_by_node(), {2: {20}})

    def test_highway_access_collector_connects_paths_to_access_highways(self):
        class Node:
            def __init__(self, node_id):
                self.ref = node_id

        class Way:
            def __init__(self, way_id, highway_type, node_ids):
                self.id = way_id
                self.tags = {'highway': highway_type}
                self.nodes = [Node(node_id) for node_id in node_ids]

        collector = routes.HighwayAccessCollector({1, 4})
        collector.way(Way(10, 'path', [1, 2]))
        collector.way(Way(20, 'residential', [2, 3]))
        collector.way(Way(30, 'path', [4, 5]))

        self.assertEqual(
            collector.accessible_highway_way_ids_by_node(),
            {1: {10}},
        )

    def test_inferred_simple_line_uses_ordered_leaves(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {}}
        way_nodes = {
            1: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
            2: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            3: [(3, [0.0020, 0.0000]), (4, [0.0040, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start, finish = graph.simple_line_endpoints()
        traversal = graph.shortest_complete_traversal_simple(start, finish)

        self.assertEqual((start, finish), (1, 4))
        self.assertEqual(graph.traversal_coordinates(traversal)[-1], graph.point(finish))

    def test_inferred_closed_loop_uses_accessible_landmark_start(self):
        relation = {
            'name': 'Gyadai tanosveny route',
            'way_ids': [1, 2, 3],
            'node_roles': {},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (1, [0.0000, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        finder = routes.EligibleNodeFinder(
            relation,
            way_nodes,
            {1: {99}, 2: {99}},
            {1, 2},
            [{
                'name': 'Gyadai tanosveny route entrance',
                'points': [[0.0010, 0.0000]],
            }],
        )
        self.assertEqual(finder.rank_eligible_nodes()[0], 2)

    def test_inferred_branch_roundtrip_uses_landmark_start(self):
        relation = {
            'name': 'Branch route',
            'roundtrip': True,
            'way_ids': [1, 2, 3],
            'node_roles': {},
        }
        way_nodes = {
            1: [(0, [0.0000, 0.0000]), (1, [0.0010, 0.0000])],
            2: [(0, [0.0000, 0.0000]), (2, [0.0000, 0.0010])],
            3: [(0, [0.0000, 0.0000]), (3, [-0.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        finder = routes.EligibleNodeFinder(
            relation,
            way_nodes,
            {1: {99}, 3: {99}},
            {1, 3},
            [{
                'name': 'Branch route trailhead',
                'points': [[-0.0010, 0.0000]],
            }],
        )
        self.assertEqual(finder.rank_eligible_nodes()[0], 3)

    def test_inferred_branch_open_uses_shortest_accessible_finish(self):
        relation = {
            'name': 'Branch route',
            'way_ids': [1, 2, 3],
            'node_roles': {},
        }
        way_nodes = {
            1: [(0, [0.0000, 0.0000]), (1, [0.0010, 0.0000])],
            2: [(0, [0.0000, 0.0000]), (2, [0.0000, 0.0010])],
            3: [(0, [0.0000, 0.0000]), (3, [-0.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        traversal = graph.shortest_complete_traversal_simple()

        self.assertEqual(traversal[0], 1)
        self.assertEqual(traversal[-1], 2)

    def test_graph_traversal_does_not_require_access_metadata(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {}}
        way_nodes = {
            1: [(0, [0.0000, 0.0000]), (1, [0.0010, 0.0000])],
            2: [(0, [0.0000, 0.0000]), (2, [0.0000, 0.0010])],
            3: [(0, [0.0000, 0.0000]), (3, [-0.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        self.assertIsNotNone(graph.shortest_complete_traversal_simple())

    def test_graph_returns_shortest_complete_traversal(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(2, [0.0010, 0.0000]), (4, [0.0010, -0.0010])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        traversal = graph.shortest_complete_traversal_simple(1, 4)

        self.assertEqual((traversal[0], traversal[-1]), (1, 4))

    def test_graph_handles_many_route_nodes(self):
        relation = {
            'way_ids': list(range(1, 13)),
            'node_roles': {},
        }
        way_nodes = {
            way_id: [
                (0, [0.0000, 0.0000]),
                (way_id, [way_id * 0.0001, 0.0000]),
            ]
            for way_id in relation['way_ids']
        }
        graph = routes.RouteGraph(relation, way_nodes)
        traversal = graph.shortest_complete_traversal_simple(0, 12)

        self.assertEqual((traversal[0], traversal[-1]), (0, 12))

    def test_landmark_index_returns_only_nearby_points(self):
        index = LandmarkIndex([
            {'points': [[0.0000, 0.0000]]},
            {'points': [[0.0100, 0.0000]]},
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
        finder = routes.EligibleNodeFinder(
            relation,
            way_nodes,
            {1: {99}},
            {1},
            [{
                'points': [[0.00128, 0.0000]],
            }],
        )

        self.assertEqual(finder._landmark_nodes(), set())

    def test_missing_endpoints_use_landmark_for_equal_endpoint_orientations(self):
        relation = {
            'name': 'Route',
            'way_ids': [3, 2, 1],
            'node_roles': {},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
            3: [(3, [0.0020, 0.0000]), (4, [0.0030, 0.0000])],
        }
        finder = routes.EligibleNodeFinder(
            relation,
            way_nodes,
            {1: {99}, 4: {99}},
            {1, 4},
            [{
                'points': [[0.0030, 0.0000]],
            }],
        )

        self.assertEqual(finder.rank_eligible_nodes(), [4, 1])

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

        self.assertIn(1, graph._graph)

    def test_missing_endpoints_use_relation_member_order_without_landmark(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {}}
        way_nodes = {
            1: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
            2: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            3: [(3, [0.0020, 0.0000]), (4, [0.0040, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        start, finish = graph.simple_line_endpoints()

        self.assertEqual(start, 1)
        self.assertEqual(finish, 4)
        self.assertEqual(graph._graph.degree(start), 1)

    def test_simple_line_uses_leaves_even_when_access_is_internal(self):
        relation = {'way_ids': [1], 'node_roles': {}}
        way_nodes = {
            1: [
                (1, [0.0000, 0.0000]),
                (2, [0.0010, 0.0000]),
                (3, [0.0020, 0.0000]),
            ],
        }
        graph = routes.RouteGraph(relation, way_nodes, {2})

        start, finish = graph.simple_line_endpoints()
        traversal = graph.shortest_complete_traversal_simple(start, finish)

        self.assertNotIn(2, graph._graph)
        self.assertEqual((start, finish), (1, 3))
        self.assertEqual(
            graph.traversal_coordinates(traversal)[-1],
            graph.point(finish),
        )

    def test_simple_line_uses_osm_order_over_access_heuristic(self):
        relation = {'way_ids': [1, 2], 'node_roles': {}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes, {3})

        start, finish = graph.simple_line_endpoints()
        traversal = graph.shortest_complete_traversal_simple(start, finish)

        self.assertEqual(start, 1)
        self.assertEqual(finish, 3)
        self.assertEqual(
            graph.traversal_coordinates(traversal)[-1],
            graph.point(finish),
        )

    def test_explicit_start_ignores_external_access_restriction(self):
        relation = {'way_ids': [1, 2], 'node_roles': {'start': [1]}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes, {3})

        self.assertIn(1, graph._graph)

    def test_open_endpoint_search_uses_global_matching(self):
        relation = {'way_ids': [1, 2, 3, 4], 'node_roles': {}}
        way_nodes = {
            1: [(0, [0.0000, 0.0000]), (1, [0.0001, 0.0000])],
            2: [(0, [0.0000, 0.0000]), (2, [-0.0001, 0.0000])],
            3: [(0, [0.0000, 0.0000]), (3, [0.0010, 0.0000])],
            4: [(0, [0.0000, 0.0000]), (4, [-0.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        traversal = graph.shortest_complete_traversal_simple()

        self.assertEqual(set(traversal), {0, 1, 2, 3, 4})

    def test_acyclic_branch_uses_open_traversal(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {'start': [1]}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (4, [0.0010, 0.0010])],
            3: [(2, [0.0010, 0.0000]), (5, [0.0010, -0.0010])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        traversal = graph.shortest_complete_traversal_simple(1, 3)
        path = graph.traversal_coordinates(traversal)

        self.assertEqual((traversal[0], traversal[-1]), (1, 3))
        self.assertEqual(path[-1], graph.point(3))
        self.assertEqual(path[0], [0.0, 0.0])
        self.assertEqual(path[-1], [0.002, 0.0])
        self.assertIn([0.001, 0.001], path)
        self.assertIn([0.001, -0.001], path)
        self.assertEqual(len(traversal), 7)

    def test_traversal_distance_uses_weighted_edges(self):
        relation = {'way_ids': [1, 2, 3, 4], 'node_roles': {}}
        way_nodes = {
            1: [(0, [0.0000, 0.0000]), (1, [0.0010, 0.0000])],
            2: [(0, [0.0000, 0.0000]), (2, [0.0000, 0.0010])],
            3: [(0, [0.0000, 0.0000]), (3, [-0.0010, 0.0000])],
            4: [(0, [0.0000, 0.0000]), (4, [0.0000, -0.0010])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        traversal = graph.shortest_complete_traversal_simple(1, 4)

        self.assertGreater(graph.traversal_distance_simple(traversal), 0)

    def test_cycle_through_start_prefers_closed_traversal(self):
        relation = {'way_ids': [1, 2, 3, 4], 'node_roles': {'start': [1]}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0005, 0.0010])],
            3: [(3, [0.0005, 0.0010]), (1, [0.0000, 0.0000])],
            4: [(2, [0.0010, 0.0000]), (4, [0.0020, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        traversal = graph.shortest_complete_traversal_simple(1, 1)
        path = graph.traversal_coordinates(traversal)

        self.assertEqual((traversal[0], traversal[-1]), (1, 1))
        self.assertEqual(path[-1], graph.point(1))
        self.assertEqual(path[0], [0.0, 0.0])
        self.assertEqual(path[-1], [0.0, 0.0])
        self.assertGreaterEqual(len(traversal), 2)

    def test_explicit_start_can_choose_closed_traversal(self):
        relation = {
            'way_ids': [2, 3, 4, 1, 1, 5, 6, 6],
            'node_roles': {'start': [1]},
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
        graph = routes.RouteGraph(relation, way_nodes, roundtrip=True)
        traversal = graph.eulerian_traversal(1)
        path = graph.traversal_coordinates(traversal)

        self.assertEqual((traversal[0], traversal[-1]), (1, 1))
        self.assertEqual(path[-1], graph.point(1))

    def test_duplicate_members_keep_closed_eulerian_traversal(self):
        relation = {
            'way_ids': [1, 1, 2, 2, 3, 3, 4, 4],
            'node_roles': {},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (4, [0.0000, 0.0010])],
            4: [(4, [0.0000, 0.0010]), (2, [0.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes, roundtrip=True)
        traversal = graph.eulerian_traversal(2)
        path = graph.traversal_coordinates(traversal)

        self.assertEqual((traversal[0], traversal[-1]), (2, 2))
        self.assertEqual(path[-1], graph.point(2))

    def test_explicit_start_on_branch_uses_free_finish(self):
        relation = {
            'way_ids': [1, 2, 3, 4],
            'node_roles': {'start': [1]},
        }
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (4, [0.0000, 0.0010])],
            4: [(4, [0.0000, 0.0010]), (2, [0.0010, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        traversal = graph.shortest_complete_traversal_to_nearest_finish(1)
        path = graph.traversal_coordinates(traversal)

        self.assertEqual(traversal[0], 1)
        self.assertEqual(path[-1], graph.point(traversal[-1]))

    def test_eulerian_route_returns_to_explicit_start(self):
        relation = {'way_ids': [1, 2, 3], 'node_roles': {'start': [1]}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0010, 0.0010])],
            3: [(3, [0.0010, 0.0010]), (1, [0.0000, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)
        traversal = graph.eulerian_traversal(1)
        path = graph.traversal_coordinates(traversal)

        self.assertEqual((traversal[0], traversal[-1]), (1, 1))
        self.assertEqual(path[-1], graph.point(1))

    def test_explicit_finish_is_used(self):
        relation = {'way_ids': [1, 2], 'node_roles': {'start': [1], 'end': [3]}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])],
            2: [(2, [0.0010, 0.0000]), (3, [0.0020, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes)

        traversal = graph.shortest_complete_traversal_simple(1, 3)

        self.assertEqual((traversal[0], traversal[-1]), (1, 3))
        self.assertEqual(graph.traversal_coordinates(traversal)[-1], graph.point(3))

    def test_repair_requires_both_distance_thresholds(self):
        relation = {'way_ids': [1, 2], 'node_roles': {}}
        way_nodes = {
            1: [(1, [0.0000, 0.0000]), (2, [0.0005, 0.0000])],
            2: [(3, [0.0010, 0.0000]), (4, [0.0015, 0.0000])],
        }
        graph = routes.RouteGraph(relation, way_nodes, sampler=ConstantSampler())

        self.assertEqual(graph.component_count, 1)

        far_graph = routes.RouteGraph(
            relation,
            {
                1: [(1, [0.0000, 0.0000]), (2, [0.0005, 0.0000])],
                2: [(3, [0.0020, 0.0000]), (4, [0.0025, 0.0000])],
            },
            sampler=ConstantSampler(),
        )
        self.assertEqual(far_graph.component_count, 2)

        steep_graph = routes.RouteGraph(relation, way_nodes, sampler=SteepSampler())
        self.assertEqual(steep_graph.component_count, 2)

    def test_explicit_marker_outside_route_nodes_is_not_added(self):
        relation = {'way_ids': [1], 'node_roles': {'start': [99]}}
        way_nodes = {1: [(1, [0.0000, 0.0000]), (2, [0.0010, 0.0000])]}
        graph = routes.RouteGraph(relation, way_nodes)

        self.assertNotIn(99, graph._graph)

    # Regression routes included in the fixture (relation ID: name):
    # - 1524566: ZT, Gyadai tanösvény
    # - 2499633: S↺, Thirring körút (Dobogókő, elágazás – Thirring-sziklák – Dobogókő, elágazás)
    # - 14643735: Rám-szakadék tanösvény (Dömös - Rám-szakadék - Dömös)
    # - 4120196: Oszoly-ösvény (Csobánka, Oszoly-pihenő – Oszoly-csúcs)
    # - 5845823: KT, Kékkő tanösvény
    # - 10093867: Hajta természetismereti túra
    # - 16122238: ST209 Esztergom - Dobogókő
    # - 11134335: Sóvirág tanösvény (two-way circular relation)
    # - 15865671: Fent és lent – Tanösvény a Guckler Károly úton
    # - 9273737: Normafa kardioösvény (Normafa – János-hegy)
    # - 14345638: Keltatúra
    # - 7314605: IVV, zöld (Esztergom vá. – Vaskapu Th. – Esztergom vá.)
    # - 5339799: K╱, Szarvaskői geológiai tanösvény
    # - 4120194: Fehér sziklák ösvény (Csobánka, Margitliget – Oszoly-csúcs – Csobánkai templomok)
    # - 14873673: KP Jósvafő
    # - 13008447: Jakobsweg Burgenland – Zubringer 2
    # - 14568269: PB3 Pap-réti túra
    # - 3194149: S (Túrony – Bisse – Tenkes hegy – Csodabogyó tanösvény)
    # - 5695326: PT, Komlóskai Telér tanösvény
    # - 16467089: Német-völgyi tanösvény (Kőhányás – Német-völgy – Kőhányás)
    def test_hungary_routes_preserve_pbf_derived_results(self):
        class RegressionElevation:
            def __init__(self, directory):
                pass

            def sample(self, point):
                return 100

            def route_elevation(self, path):
                return {
                    'elevation_gain_m': 0,
                    'elevation_loss_m': 0,
                    'profile': {'segments': []},
                }

            @staticmethod
            def route_duration_min(distance_m, elevation_gain_m, elevation_loss_m):
                return 0

            def close(self):
                pass

        cases = load_route_regression_cases()
        self.assertEqual(len(cases), 20)

        with tempfile.TemporaryDirectory() as directory, \
            patch.object(routes, 'Elevation', RegressionElevation), \
            patch.dict(os.environ, {'ELEVATION_DIRECTORY': directory}):
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                for case in cases:
                    with self.subTest(route=case['id']):
                        relation = {
                            **case['route'],
                            'name': f"Regression route {case['id']}",
                            'name_int': '',
                            'symbol': '',
                            'network': 'lwn',
                            'type': 'hiking',
                            'route_members': [
                                ('w', way_id)
                                for way_id in case['route']['way_ids']
                            ],
                        }
                        collector = routes.WayRouteCollector()
                        collector.relations = {case['id']: relation}
                        collector.flatten_nested_routes()
                        relation = collector.relations[case['id']]
                        self.assertEqual(relation['way_ids'], case['route']['way_ids'])
                        way_nodes = {
                            int(way_id): nodes
                            for way_id, nodes in case['ways'].items()
                        }
                        route_node_ids = {
                            node_id
                            for nodes in way_nodes.values()
                            for node_id, _ in nodes
                        }
                        external_nodes = set(case.get('externally_reachable_nodes', ()))
                        highway_way_ids_by_node = {
                            node_id: {-1}
                            for node_id in external_nodes
                        }
                        landmark_data = case.get('landmarks')
                        if landmark_data is None and 'landmark' in case:
                            landmark_data = [case['landmark']]
                        exporter = types.SimpleNamespace(
                            way_nodes=way_nodes,
                            highway_way_ids_by_node=highway_way_ids_by_node,
                            landmarks=landmark_data or [],
                        )

                        routes.write_route_lines(
                            types.SimpleNamespace(relations={case['id']: relation}),
                            exporter,
                        )
                        result = json.loads(Path(directory, 'routes-meta.json').read_text())[0]
                        actual = {
                            'start': [result['start_lon'], result['start_lat']]
                            if result['start_lon'] is not None else None,
                            'finish': [result['finish_lon'], result['finish_lat']]
                            if result['finish_lon'] is not None else None,
                            'distance_m': result['distance_m'],
                        }

                        self.assertEqual(actual, case['expected'])
            finally:
                os.chdir(previous_directory)

if __name__ == '__main__':
    unittest.main()
