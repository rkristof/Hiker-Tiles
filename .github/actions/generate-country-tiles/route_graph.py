import math
import re
from enum import IntEnum

import networkx as nx


class LandmarkCategory(IntEnum):
    MEDIUM = 1
    HIGH = 2
    HIGHEST = 3


LANDMARK_RULES = (
    ('highway', 'trailhead', LandmarkCategory.HIGHEST, 30),
    ('information', 'guidepost', LandmarkCategory.HIGHEST, 30),
    ('information', 'map', LandmarkCategory.HIGHEST, 30),
    ('information', 'board', LandmarkCategory.HIGHEST, 30),
    ('tourism', 'information', LandmarkCategory.HIGHEST, 30),
    ('parking', 'entrance', LandmarkCategory.HIGH, 30),
    ('entrance', 'main', LandmarkCategory.HIGH, 30),
    ('amenity', 'parking', LandmarkCategory.HIGH, 60),
    ('tourism', 'visitor_centre', LandmarkCategory.HIGH, 60),
    ('highway', 'bus_stop', LandmarkCategory.MEDIUM, 90),
    ('public_transport', 'platform', LandmarkCategory.MEDIUM, 90),
    ('public_transport', 'stop_position', LandmarkCategory.MEDIUM, 90),
    ('railway', 'station', LandmarkCategory.MEDIUM, 90),
    ('railway', 'halt', LandmarkCategory.MEDIUM, 90),
    ('aerialway', 'station', LandmarkCategory.MEDIUM, 90),
)
LANDMARK_MAX_DISTANCE_M = max(rule[3] for rule in LANDMARK_RULES)
LANDMARK_GRID_SIZE_DEGREES = 0.0005


def landmark_candidate(tags):
    """Return selected metadata for a high-signal OSM landmark, if applicable."""
    for key, value, category, distance_limit_m in LANDMARK_RULES:
        if tags.get(key) == value:
            return {
                'category': category,
                'distance_limit_m': distance_limit_m,
                'tag_key': key,
                'tag_value': value,
                'name': tags.get('name', ''),
                'description': tags.get('description', ''),
                'ref': tags.get('ref', ''),
            }
    return None


class LandmarkIndex:
    """Index landmark points so route start resolution only checks nearby candidates."""

    def __init__(self, landmarks):
        self._landmarks = tuple(landmarks)
        self._cells = {}
        for landmark_index, landmark in enumerate(self._landmarks):
            for point in self._landmark_points(landmark):
                cell = self._cell(point)
                self._cells.setdefault(cell, []).append((landmark_index, point))

    def nearby(self, point):
        """Yield indexed landmark points within the maximum category distance window."""
        latitude_radius = math.degrees(LANDMARK_MAX_DISTANCE_M / 6371000)
        longitude_radius = latitude_radius / max(math.cos(math.radians(point[1])), 1e-6)
        latitude_cell = self._cell(point)[0]
        longitude_cell = self._cell(point)[1]
        latitude_range = range(
            latitude_cell - math.ceil(latitude_radius / LANDMARK_GRID_SIZE_DEGREES),
            latitude_cell + math.ceil(latitude_radius / LANDMARK_GRID_SIZE_DEGREES) + 1,
        )
        longitude_range = range(
            longitude_cell - math.ceil(longitude_radius / LANDMARK_GRID_SIZE_DEGREES),
            longitude_cell + math.ceil(longitude_radius / LANDMARK_GRID_SIZE_DEGREES) + 1,
        )
        for latitude_index in latitude_range:
            for longitude_index in longitude_range:
                yield from self._cells.get((latitude_index, longitude_index), ())

    def landmark(self, landmark_index):
        """Return the landmark stored at an index position."""
        return self._landmarks[landmark_index]

    @staticmethod
    def _cell(point):
        return (
            math.floor(point[1] / LANDMARK_GRID_SIZE_DEGREES),
            math.floor(point[0] / LANDMARK_GRID_SIZE_DEGREES),
        )

    @staticmethod
    def _landmark_points(landmark):
        points = landmark.get('points', [])
        if points:
            return points
        point = landmark.get('point')
        return [point] if point is not None else []


class RouteGraph:
    """Build and traverse one OSM route relation as an edge-preserving graph."""

    def __init__(self, route_relation, way_nodes, node_way_ids=None):
        self._route_relation = route_relation
        self._node_way_ids = node_way_ids
        self._graph = nx.MultiGraph()
        self._simple_graph = None
        self._relation_way_endpoints = []
        self._build(way_nodes)

    @property
    def has_edges(self):
        return self._graph.number_of_edges() > 0

    @property
    def component_count(self):
        return nx.number_connected_components(self._graph)

    def point(self, node_id):
        return self._graph.nodes[node_id]['point']

    def required_distance_m(self):
        return round(sum(
            attributes['distance_m']
            for _, _, attributes in self._graph.edges(data=True)
        ))

    def repair_disconnected_components(self, sampler):
        components = list(nx.connected_components(self._graph))
        if len(components) <= 1:
            return 0

        endpoints = [
            node_id
            for component in components
            for node_id in self._graph_endpoints(component)
        ]
        component_by_node = {
            node_id: component_index
            for component_index, component in enumerate(components)
            for node_id in component
        }
        component_parent = list(range(len(components)))

        def find_component(component_index):
            while component_parent[component_index] != component_index:
                component_parent[component_index] = component_parent[component_parent[component_index]]
                component_index = component_parent[component_index]
            return component_index

        elevations = {
            node_id: sampler.sample(self.point(node_id))
            for node_id in endpoints
        }
        candidates_by_endpoint = {node_id: [] for node_id in endpoints}
        for first_index, first_node in enumerate(endpoints):
            first_elevation = elevations[first_node]
            if first_elevation is None:
                continue
            first_point = self.point(first_node)
            for second_node in endpoints[first_index + 1:]:
                if component_by_node[first_node] == component_by_node[second_node]:
                    continue
                second_elevation = elevations[second_node]
                if second_elevation is None:
                    continue
                second_point = self.point(second_node)
                horizontal_distance = haversine_distance_m(first_point, second_point)
                if horizontal_distance >= 100:
                    continue
                if abs(first_elevation - second_elevation) >= 15:
                    continue
                candidates_by_endpoint[first_node].append((horizontal_distance, second_node))
                candidates_by_endpoint[second_node].append((horizontal_distance, first_node))

        for candidates in candidates_by_endpoint.values():
            candidates.sort()

        repairs = 0
        while True:
            if repairs == len(components) - 1:
                return repairs
            nearest_candidates = {}
            for node_id, candidates in candidates_by_endpoint.items():
                valid_candidates = []
                for candidate in candidates:
                    if (
                        self._graph.degree(node_id) == 1
                        and self._graph.degree(candidate[1]) == 1
                        and find_component(component_by_node[node_id]) != find_component(component_by_node[candidate[1]])
                    ):
                        valid_candidates.append(candidate)
                        if len(valid_candidates) == 2:
                            break
                if valid_candidates:
                    is_unique = (
                        len(valid_candidates) == 1
                        or abs(valid_candidates[0][0] - valid_candidates[1][0]) >= 1e-6
                    )
                    nearest_candidates[node_id] = (valid_candidates[0], is_unique)

            mutual_candidates = []
            for first_node, (candidate, is_unique) in nearest_candidates.items():
                second_node = candidate[1]
                reverse_candidate = nearest_candidates.get(second_node)
                if (
                    is_unique
                    and reverse_candidate is not None
                    and reverse_candidate[1]
                    and reverse_candidate[0][1] == first_node
                ):
                    mutual_candidates.append((candidate[0], first_node, second_node))
            if not mutual_candidates:
                return repairs

            _, first_node, second_node = min(mutual_candidates)
            self._add_edge(
                first_node,
                second_node,
                [self.point(first_node), self.point(second_node)],
            )
            first_component = find_component(component_by_node[first_node])
            second_component = find_component(component_by_node[second_node])
            component_parent[second_component] = first_component
            repairs += 1

    def resolve_start(self, node_coordinates, sampler, landmarks=None):
        explicit_starts = self._route_relation.get('node_roles', {}).get('start', [])
        if explicit_starts:
            if len(explicit_starts) != 1:
                return None
            start_node = explicit_starts[0]
            if start_node in self._graph:
                return start_node
            return self._snap_relation_node(start_node, node_coordinates, sampler)

        leaves = self._graph_endpoints(set(self._graph.nodes))
        if len(leaves) == 1:
            return leaves[0]
        endpoint_pair = self._two_endpoint_pair()
        if endpoint_pair is not None:
            return endpoint_pair[0]
        landmark_index = landmarks
        if landmark_index is not None and not isinstance(landmark_index, LandmarkIndex):
            landmark_index = LandmarkIndex(landmark_index)
        landmark_start = self._resolve_landmark_start(landmark_index)
        if landmark_start is not None:
            return landmark_start
        if leaves:
            anchor_node = next(iter(self._graph), None)
            if anchor_node is not None:
                distances = nx.single_source_dijkstra_path_length(
                    self._simple_weighted_graph(),
                    anchor_node,
                    weight='weight',
                )
                reachable_leaves = [leaf for leaf in leaves if leaf in distances]
                if reachable_leaves:
                    return min(
                        reachable_leaves,
                        key=lambda leaf: (distances[leaf], leaf),
                    )
        return next(iter(self._graph), None)

    def _resolve_landmark_start(self, landmark_index):
        if landmark_index is None or self._graph.number_of_nodes() == 0:
            return None
        route_tokens = self._text_tokens(
            self._route_relation.get('name', ''),
            self._route_relation.get('name_int', ''),
        )
        nearest_by_landmark = {}
        for node_id, graph_attributes in self._graph.nodes(data=True):
            graph_point = graph_attributes['point']
            for landmark_index_value, landmark_point in landmark_index.nearby(graph_point):
                landmark = landmark_index.landmark(landmark_index_value)
                distance_m = haversine_distance_m(landmark_point, graph_point)
                if distance_m > landmark.get('distance_limit_m', 0):
                    continue
                nearest = nearest_by_landmark.get(landmark_index_value)
                candidate = (distance_m, node_id)
                if nearest is None or candidate < nearest:
                    nearest_by_landmark[landmark_index_value] = candidate

        candidates = []
        for landmark_index_value, (distance_m, node_id) in nearest_by_landmark.items():
            landmark = landmark_index.landmark(landmark_index_value)
            name_match_count = len(
                route_tokens & self._text_tokens(
                    landmark.get('name', ''),
                    landmark.get('description', ''),
                    landmark.get('ref', ''),
                )
            )
            candidates.append({
                'node_id': node_id,
                'distance_m': distance_m,
                'category': landmark.get('category'),
                'name_match_count': name_match_count if name_match_count >= 2 else 0,
                'landmark_id': landmark.get('node_id', landmark.get('way_id', 0)),
            })

        if not candidates:
            return None
        candidates.sort(key=lambda candidate: (
            -bool(candidate['name_match_count']),
            -candidate['category'],
            -candidate['name_match_count'],
            candidate['distance_m'],
            candidate['node_id'],
            candidate['landmark_id'],
        ))
        return candidates[0]['node_id']

    @staticmethod
    def _text_tokens(*values):
        return {
            token.casefold()
            for value in values
            for token in re.findall(r'\w+', value, flags=re.UNICODE)
            if len(token) >= 3
        }

    def _two_endpoint_pair(self):
        leaves = set(self._graph_endpoints(set(self._graph.nodes)))
        if len(leaves) != 2:
            return None

        ordered_nodes = [
            node_id
            for first_node, last_node in self._relation_way_endpoints
            for node_id in (first_node, last_node)
        ]
        first_endpoint = next((node_id for node_id in ordered_nodes if node_id in leaves), None)
        last_endpoint = next((node_id for node_id in reversed(ordered_nodes) if node_id in leaves), None)
        if first_endpoint is None or last_endpoint is None or first_endpoint == last_endpoint:
            return tuple(sorted(leaves))

        if self._from_to_reverses_member_order():
            return last_endpoint, first_endpoint
        return first_endpoint, last_endpoint

    def _from_to_reverses_member_order(self):
        from_name = self._route_relation.get('from', '')
        to_name = self._route_relation.get('to', '')
        if not from_name or not to_name:
            return False

        route_tokens = self._text_token_list(
            self._route_relation.get('name', ''),
            self._route_relation.get('name_int', ''),
        )
        from_position = self._find_token_sequence(
            route_tokens,
            self._text_token_list(from_name),
        )
        to_position = self._find_token_sequence(
            route_tokens,
            self._text_token_list(to_name),
        )
        return (
            from_position is not None
            and to_position is not None
            and from_position > to_position
        )

    @staticmethod
    def _text_token_list(*values):
        return [
            token.casefold()
            for value in values
            for token in re.findall(r'\w+', value, flags=re.UNICODE)
            if len(token) >= 3
        ]

    @staticmethod
    def _find_token_sequence(tokens, sequence):
        if not sequence or len(sequence) > len(tokens):
            return None
        sequence_length = len(sequence)
        for index in range(len(tokens) - sequence_length + 1):
            if tokens[index:index + sequence_length] == sequence:
                return index
        return None

    def resolve_finish(self, start_node, node_coordinates, sampler):
        explicit_finishes = self._route_relation.get('node_roles', {}).get('end', [])
        if explicit_finishes:
            if len(explicit_finishes) != 1:
                return None
            finish_node = explicit_finishes[0]
            if finish_node in self._graph:
                return finish_node
            return self._snap_relation_node(finish_node, node_coordinates, sampler)
        endpoint_pair = self._two_endpoint_pair()
        if endpoint_pair is not None and start_node in endpoint_pair:
            return endpoint_pair[1] if start_node == endpoint_pair[0] else endpoint_pair[0]
        if (
            self._route_relation.get('roundtrip')
            and start_node is not None
            and self._is_on_cycle(start_node)
        ):
            return start_node
        return None

    def shortest_traversal(self, start_node, finish_node=None):
        if start_node not in self._graph or not nx.is_connected(self._graph):
            return None

        if finish_node is not None:
            walk = self._shortest_traversal_to(start_node, finish_node)
            return None if walk is None else (walk[0], walk[2])

        if self._is_on_cycle(start_node):
            closed_walk = self._shortest_traversal_to(start_node, start_node)
            if closed_walk is not None:
                return closed_walk[0], closed_walk[2]

        finish_candidates = sorted(
            node_id
            for node_id, degree in self._graph.degree()
            if node_id != start_node and degree % 2 == 1
        )
        if not finish_candidates:
            return None

        open_walk = self._shortest_traversal_to(start_node, None, finish_candidates)
        return None if open_walk is None else (open_walk[0], open_walk[2])

    def traversal_coordinates(self, start_node, steps):
        coordinates = [self.point(start_node)]
        current_node = start_node
        for first_node, second_node, _, edge in steps:
            if current_node == first_node:
                next_node = second_node
            elif current_node == second_node:
                next_node = first_node
            else:
                return coordinates, current_node

            if edge['points'][0] == self.point(current_node):
                edge_points = edge['points']
            else:
                edge_points = list(reversed(edge['points']))
            coordinates.extend(edge_points[1:])
            current_node = next_node
        return coordinates, current_node

    def _build(self, way_nodes):
        relation_way_nodes = [
            (way_id, way_nodes.get(way_id, []))
            for way_id in self._route_relation['way_ids']
        ]
        self._relation_way_endpoints = [
            (nodes[0][0], nodes[-1][0])
            for _, nodes in relation_way_nodes
            if len(nodes) >= 2
        ]
        if self._node_way_ids is None:
            node_way_ids = {}
            for way_id, nodes in relation_way_nodes:
                for node_id, _ in nodes:
                    node_way_ids.setdefault(node_id, set()).add(way_id)
        else:
            node_way_ids = self._node_way_ids

        relation_nodes = {
            node_id
            for role in ('start', 'end')
            for node_id in self._route_relation.get('node_roles', {}).get(role, [])
        }
        for _, nodes in relation_way_nodes:
            if len(nodes) < 2:
                continue
            if nodes[0][0] == nodes[-1][0]:
                for first_node, second_node in zip(nodes, nodes[1:]):
                    self._add_edge(
                        first_node[0],
                        second_node[0],
                        [first_node[1], second_node[1]],
                    )
                continue

            important_indices = [0]
            important_indices.extend(
                index
                for index, (node_id, _) in enumerate(nodes[1:-1], start=1)
                if node_id in relation_nodes or len(node_way_ids[node_id]) > 1
            )
            important_indices.append(len(nodes) - 1)
            for first_index, second_index in zip(important_indices, important_indices[1:]):
                self._add_edge(
                    nodes[first_index][0],
                    nodes[second_index][0],
                    [point for _, point in nodes[first_index:second_index + 1]],
                )

    def _add_edge(self, start_node, end_node, points):
        if start_node == end_node or len(points) < 2:
            return

        self._simple_graph = None
        self._graph.add_node(start_node, point=points[0])
        self._graph.add_node(end_node, point=points[-1])
        self._graph.add_edge(
            start_node,
            end_node,
            points=points,
            distance_m=polyline_distance_m(points),
        )

    def _graph_endpoints(self, component):
        return [node_id for node_id in component if self._graph.degree(node_id) == 1]

    def _is_on_cycle(self, node_id):
        bridge_edges = {
            frozenset((first_node, second_node))
            for first_node, second_node in nx.bridges(self._graph)
        }
        return any(
            frozenset((node_id, neighbor)) not in bridge_edges
            for neighbor in self._graph.neighbors(node_id)
        )

    def _snap_relation_node(self, relation_node_id, node_coordinates, sampler):
        relation_point = node_coordinates.get(relation_node_id)
        if relation_point is None:
            return None
        relation_elevation = sampler.sample(relation_point)
        candidates = []
        for node_id, attributes in self._graph.nodes(data=True):
            point = attributes['point']
            horizontal_distance = haversine_distance_m(relation_point, point)
            if horizontal_distance >= 100:
                continue
            node_elevation = sampler.sample(point)
            if relation_elevation is None or node_elevation is None:
                continue
            if abs(relation_elevation - node_elevation) >= 15:
                continue
            candidates.append((horizontal_distance, node_id))
        if not candidates:
            return None
        candidates.sort()
        if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 1e-6:
            return None
        return candidates[0][1]

    def _simple_weighted_graph(self):
        if self._simple_graph is not None:
            return self._simple_graph

        simple_graph = nx.Graph()
        for first_node, second_node, edge_key, attributes in self._graph.edges(data=True, keys=True):
            if (
                not simple_graph.has_edge(first_node, second_node)
                or attributes['distance_m'] < simple_graph[first_node][second_node]['weight']
            ):
                simple_graph.add_edge(
                    first_node,
                    second_node,
                    weight=attributes['distance_m'],
                    edge_key=edge_key,
                )
        self._simple_graph = simple_graph
        return simple_graph

    def _matching_paths(self, nodes, free_finish_nodes=None):
        simple_graph = self._simple_weighted_graph()
        matching_graph = nx.Graph()
        matching_graph.add_nodes_from(nodes)
        shortest_paths = {}
        shortest_distances = {}
        for source in nodes:
            distances, paths = nx.single_source_dijkstra(
                simple_graph,
                source,
                weight='weight',
            )
            shortest_distances[source] = distances
            shortest_paths[source] = paths

        for first_index, first_node in enumerate(nodes):
            for second_node in nodes[first_index + 1:]:
                if second_node not in shortest_distances[first_node]:
                    return None
                matching_graph.add_edge(
                    first_node,
                    second_node,
                    weight=shortest_distances[first_node][second_node],
                )

        dummy_node = None
        if free_finish_nodes:
            dummy_node = object()
            matching_graph.add_node(dummy_node)
            for finish_index, finish_node in enumerate(free_finish_nodes):
                matching_graph.add_edge(dummy_node, finish_node, weight=finish_index * 1e-12)

        matching = nx.min_weight_matching(matching_graph, weight='weight')
        if len(matching) * 2 != len(matching_graph):
            return None

        paths = []
        selected_finish = None
        for pair in matching:
            first_node, second_node = tuple(pair)
            if dummy_node is not None and first_node is dummy_node:
                selected_finish = second_node
                continue
            if dummy_node is not None and second_node is dummy_node:
                selected_finish = first_node
                continue
            paths.append(shortest_paths[first_node][second_node])
        return paths, simple_graph, selected_finish

    def _shortest_traversal_to(self, start_node, finish_node, free_finish_nodes=None):
        odd_nodes = {node_id for node_id, degree in self._graph.degree() if degree % 2}
        if free_finish_nodes is None:
            target_odd_nodes = set() if finish_node == start_node else {start_node, finish_node}
            matching_nodes = odd_nodes ^ target_odd_nodes
        else:
            matching_nodes = odd_nodes ^ {start_node}
        paths_result = self._matching_paths(sorted(matching_nodes), free_finish_nodes)
        if paths_result is None:
            return None

        paths, simple_graph, selected_finish = paths_result
        if free_finish_nodes is not None:
            finish_node = selected_finish
            if finish_node is None:
                return None
        augmented_graph = self._graph.copy()
        for path in paths:
            for first_node, second_node in zip(path, path[1:]):
                edge_key = simple_graph[first_node][second_node]['edge_key']
                edge_attributes = self._graph.get_edge_data(
                    first_node,
                    second_node,
                    edge_key,
                ).copy()
                augmented_graph.add_edge(first_node, second_node, **edge_attributes)

        if not nx.is_eulerian(augmented_graph) and not nx.is_semieulerian(augmented_graph):
            return None
        walk = [
            (
                first_node,
                second_node,
                edge_key,
                augmented_graph.get_edge_data(first_node, second_node, edge_key),
            )
            for first_node, second_node, edge_key in nx.eulerian_path(
                augmented_graph,
                source=start_node,
                keys=True,
            )
        ]
        distance = sum(edge['distance_m'] for _, _, _, edge in walk)
        return walk, distance, finish_node


def haversine_distance_m(first_point, second_point):
    earth_radius_m = 6371000
    first_latitude = math.radians(first_point[1])
    second_latitude = math.radians(second_point[1])
    delta_latitude = second_latitude - first_latitude
    delta_longitude = math.radians(second_point[0] - first_point[0])
    haversine_term = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(first_latitude)
        * math.cos(second_latitude)
        * math.sin(delta_longitude / 2) ** 2
    )
    return earth_radius_m * 2 * math.asin(math.sqrt(haversine_term))


def polyline_distance_m(points):
    return sum(
        haversine_distance_m(first_point, second_point)
        for first_point, second_point in zip(points, points[1:])
    )
