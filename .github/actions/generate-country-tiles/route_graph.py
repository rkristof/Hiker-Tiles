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
LANDMARK_ORDER_MAX_BONUS = 0.5
NEAR_SHORTEST_TRAVERSAL_TOLERANCE = 0.10


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
            for point in landmark['points']:
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

class RouteGraph:
    """Build and traverse one OSM route relation as an edge-preserving graph."""

    def __init__(self, route_relation, way_nodes):
        self._route_relation = route_relation
        self._graph = nx.MultiGraph()
        self._simple_graph = None
        self._shortest_path_cache = {}
        self._relation_way_endpoints = []
        self._build(way_nodes)

    @property
    def has_edges(self):
        return self._graph.number_of_edges() > 0

    @property
    def component_count(self):
        return nx.number_connected_components(self._graph)

    def component_graphs(self):
        """Return independent graph views for each connected component."""
        components = sorted(
            nx.connected_components(self._graph),
            key=lambda component: min(component),
        )
        component_graphs = []
        for component in components:
            component_relation = {
                **self._route_relation,
                'node_roles': {
                    role: [node_id for node_id in node_ids if node_id in component]
                    for role, node_ids in self._route_relation.get('node_roles', {}).items()
                },
            }
            component_graph = object.__new__(RouteGraph)
            component_graph._route_relation = component_relation
            component_graph._graph = self._graph.subgraph(component).copy()
            component_graph._simple_graph = None
            component_graph._shortest_path_cache = {}
            component_graph._relation_way_endpoints = [
                endpoints
                for endpoints in self._relation_way_endpoints
                if endpoints[0] in component and endpoints[1] in component
            ]
            component_graphs.append(component_graph)
        return component_graphs

    def point(self, node_id):
        return self._graph.nodes[node_id]['point']

    def required_distance_m(self):
        return round(self._edge_distance_m())

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

    def _landmark_candidates(self, landmark_index, allowed_nodes=None):
        if landmark_index is None or self._graph.number_of_nodes() == 0:
            return []
        allowed_nodes = set(allowed_nodes) if allowed_nodes is not None else None
        relation_node_ids = tuple(self._route_relation.get('node_ids', ()))
        relation_node_order = {}
        for order, node_id in enumerate(relation_node_ids):
            relation_node_order.setdefault(node_id, order)
        route_tokens = set(
            self._text_token_list(
                self._route_relation.get('name', ''),
                self._route_relation.get('name_int', ''),
            )
        )
        nearest_by_landmark = {}
        for node_id, graph_attributes in self._graph.nodes(data=True):
            if allowed_nodes is not None and node_id not in allowed_nodes:
                continue
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
            landmark_tokens = set(
                self._text_token_list(
                    landmark.get('name', ''),
                    landmark.get('description', ''),
                    landmark.get('ref', ''),
                )
            )
            name_match_count = len(route_tokens & landmark_tokens)
            candidates.append({
                'node_id': node_id,
                'distance_m': distance_m,
                'category': landmark.get('category'),
                'name_match_count': name_match_count if name_match_count >= 2 else 0,
                'landmark_id': landmark.get('node_id', landmark.get('way_id', 0)),
                'distance_limit_m': landmark.get('distance_limit_m', LANDMARK_MAX_DISTANCE_M),
                'relation_order': relation_node_order.get(landmark.get('node_id')),
            })

        for candidate in candidates:
            distance_score = candidate['distance_m'] / max(candidate['distance_limit_m'], 1)
            order = candidate['relation_order']
            order_score = (
                LANDMARK_ORDER_MAX_BONUS * order / max(len(relation_node_ids) - 1, 1)
                if order is not None and len(relation_node_ids) > 1
                else 0
            )
            candidate['score'] = distance_score + order_score
        return candidates

    def _best_landmark_by_node(self, candidates):
        best_by_node = {}
        for candidate in candidates:
            node_id = candidate['node_id']
            previous = best_by_node.get(node_id)
            if previous is None or self._landmark_sort_key(candidate) < self._landmark_sort_key(previous):
                best_by_node[node_id] = candidate
        return best_by_node

    @staticmethod
    def _landmark_sort_key(candidate):
        return (
            -bool(candidate['name_match_count']),
            -(candidate.get('category') or 0),
            -candidate['name_match_count'],
            candidate['relation_order'] is None,
            candidate['score'],
            candidate['distance_m'],
            candidate['node_id'],
            candidate['landmark_id'],
        )

    def _resolve_explicit_node(self, explicit_nodes, node_coordinates, sampler):
        if len(explicit_nodes) != 1:
            return None
        node_id = explicit_nodes[0]
        if node_id in self._graph:
            return node_id
        return self._snap_relation_node(node_id, node_coordinates, sampler)

    @staticmethod
    def _text_token_list(*values):
        return [
            token.casefold()
            for value in values
            for token in re.findall(r'\w+', value, flags=re.UNICODE)
            if len(token) >= 3
        ]

    def shortest_route(
        self,
        node_coordinates,
        sampler,
        landmarks=None,
    ):
        """Resolve endpoints and return the shortest route inspection traversal."""
        if (
            self._graph.number_of_nodes() == 0
            or not nx.is_connected(self._graph)
        ):
            return None

        node_roles = self._route_relation.get('node_roles', {})
        explicit_start_nodes = node_roles.get('start', [])
        explicit_finish_nodes = node_roles.get('end', [])
        start_node = (
            self._resolve_explicit_node(explicit_start_nodes, node_coordinates, sampler)
            if explicit_start_nodes
            else None
        )
        finish_node = (
            self._resolve_explicit_node(explicit_finish_nodes, node_coordinates, sampler)
            if explicit_finish_nodes
            else None
        )
        if explicit_start_nodes and start_node is None:
            return None
        if explicit_finish_nodes and finish_node is None:
            return None

        if start_node is not None:
            traversal = self._shortest_traversal_from(start_node, finish_node)
            if traversal is None:
                return None
            walk, _, traversal_finish = traversal
            return start_node, walk, traversal_finish

        if finish_node is not None:
            traversal = self._shortest_traversal_from(finish_node)
            if traversal is None:
                return None
            walk, _, inferred_start = traversal
            return inferred_start, list(reversed(walk)), finish_node

        landmark_index = self._landmark_index(landmarks)
        odd_nodes = self._odd_nodes()
        best_landmark_by_node = self._best_landmark_by_node(
            self._landmark_candidates(landmark_index, odd_nodes),
        )
        endpoint_candidates = self._shortest_open_endpoint_candidates(best_landmark_by_node)
        if not endpoint_candidates:
            start_candidates = set(self._graph.nodes)
            landmark_candidates = self._landmark_candidates(landmark_index, start_candidates)
            start = (
                min(landmark_candidates, key=self._landmark_sort_key)['node_id']
                if landmark_candidates
                else self._default_start_node(start_candidates)
            )
            traversal = self._shortest_traversal_to(start, start)
            if traversal is None:
                return None
            return start, traversal[0], traversal[2]

        shortest_distance = min(
            candidate['distance_m']
            for candidate in endpoint_candidates
        )
        maximum_distance = shortest_distance * (1 + NEAR_SHORTEST_TRAVERSAL_TOLERANCE)
        endpoint_candidates = [
            candidate
            for candidate in endpoint_candidates
            if candidate['distance_m'] <= maximum_distance + 1e-6
        ]
        endpoint_nodes = {
            node_id
            for candidate in endpoint_candidates
            for node_id in (candidate['first_node'], candidate['second_node'])
        }
        best_landmark_by_node = {
            node_id: candidate
            for node_id, candidate in best_landmark_by_node.items()
            if node_id in endpoint_nodes
        }
        default_start = self._default_start_node(set(odd_nodes))

        oriented_candidates = []
        for candidate in endpoint_candidates:
            for start, finish in (
                (candidate['first_node'], candidate['second_node']),
                (candidate['second_node'], candidate['first_node']),
            ):
                landmark = best_landmark_by_node.get(start)
                if landmark is not None:
                    selection_key = (
                        0,
                        self._landmark_sort_key(landmark),
                        candidate['distance_m'],
                        start,
                        finish,
                    )
                else:
                    selection_key = (
                        1,
                        candidate['distance_m'],
                        0 if start == default_start else 1,
                        start,
                        finish,
                    )
                oriented_candidates.append((
                    selection_key,
                    start,
                    finish,
                ))

        _, start, finish = min(oriented_candidates)
        traversal = self._shortest_traversal_to(start, finish)
        if traversal is None:
            return None
        return start, traversal[0], traversal[2]

    def _odd_nodes(self):
        return sorted(
            node_id
            for node_id, degree in self._graph.degree()
            if degree % 2
        )

    def _shortest_open_endpoint_candidates(self, landmarks_by_node):
        odd_nodes = self._odd_nodes()
        if len(odd_nodes) < 2:
            return []

        required_distance = self._edge_distance_m()
        global_matching = self._matching_paths(
            odd_nodes,
            odd_nodes,
            free_finish_count=2,
        )
        if global_matching is None:
            return []

        paths, simple_graph, endpoint_pair = global_matching
        endpoint_pair = tuple(endpoint_pair)
        shortest_distance = required_distance + self._matching_distance(paths, simple_graph)
        maximum_distance = shortest_distance * (1 + NEAR_SHORTEST_TRAVERSAL_TOLERANCE)
        endpoint_pair_set = set(endpoint_pair)
        candidates = [{
            'first_node': endpoint_pair[0],
            'second_node': endpoint_pair[1],
            'distance_m': shortest_distance,
        }]
        global_landmark_key = min(
            (
                self._landmark_sort_key(landmarks_by_node[node_id])
                for node_id in endpoint_pair
                if node_id in landmarks_by_node
            ),
            default=None,
        )
        landmark_starts = sorted(
            (
                node_id
                for node_id in landmarks_by_node
                if node_id not in endpoint_pair_set
            ),
            key=lambda node_id: (
                self._landmark_sort_key(landmarks_by_node[node_id]),
                node_id,
            ),
        )
        for start_node in landmark_starts:
            if (
                global_landmark_key is not None
                and self._landmark_sort_key(landmarks_by_node[start_node]) >= global_landmark_key
            ):
                break
            traversal = self._shortest_open_traversal_from(start_node)
            if traversal is None or traversal[1] > maximum_distance + 1e-6:
                continue
            candidates.append({
                'first_node': start_node,
                'second_node': traversal[2],
                'distance_m': traversal[1],
            })
            break
        return candidates

    def _default_start_node(self, nodes):
        ordered_nodes = [
            node_id
            for first_node, last_node in self._relation_way_endpoints
            for node_id in (first_node, last_node)
        ]
        ordered_nodes.extend(self._route_relation.get('node_ids', ()))
        relation_node_order = {}
        for order, node_id in enumerate(ordered_nodes):
            relation_node_order.setdefault(node_id, order)
        return min(
            nodes,
            key=lambda node_id: (
                relation_node_order.get(node_id) is None,
                relation_node_order.get(node_id, 0),
                node_id,
            ),
        )

    def _landmark_index(self, landmarks):
        if landmarks is None or isinstance(landmarks, LandmarkIndex):
            return landmarks
        return LandmarkIndex(landmarks)

    def _free_finish_nodes(self, start_node):
        return [
            node_id
            for node_id in self._odd_nodes()
            if node_id != start_node
        ]

    def _shortest_traversal_from(self, start_node, finish_node=None):
        candidates = (
            [self._shortest_traversal_to(start_node, finish_node)]
            if finish_node is not None
            else [
                self._shortest_traversal_to(start_node, start_node),
                self._shortest_traversal_to(
                    start_node,
                    None,
                    self._free_finish_nodes(start_node),
                ),
            ]
        )
        candidates = [candidate for candidate in candidates if candidate is not None]
        return min(candidates, key=lambda candidate: candidate[1]) if candidates else None

    def _shortest_open_traversal_from(self, start_node):
        matching_nodes = [node_id for node_id in self._odd_nodes() if node_id != start_node]
        matching_result = self._matching_paths(matching_nodes, matching_nodes)
        if matching_result is None:
            return None
        paths, simple_graph, finish_node = matching_result
        if finish_node is None:
            return None
        return (
            start_node,
            self._edge_distance_m() + self._matching_distance(paths, simple_graph),
            finish_node,
        )

    def traversal_coordinates(self, start_node, steps):
        coordinates = [self.point(start_node)]
        current_node = start_node
        for first_node, second_node, _, edge in steps:
            if current_node == first_node:
                next_node = second_node
            elif current_node == second_node:
                next_node = first_node
            else:
                return coordinates

            if edge['points'][0] == self.point(current_node):
                edge_points = edge['points']
            else:
                edge_points = list(reversed(edge['points']))
            coordinates.extend(edge_points[1:])
            current_node = next_node
        return coordinates

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
        node_way_ids = {}
        for way_id, nodes in relation_way_nodes:
            for node_id, _ in nodes:
                node_way_ids.setdefault(node_id, set()).add(way_id)

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
        self._shortest_path_cache = {}
        self._graph.add_node(start_node, point=points[0])
        self._graph.add_node(end_node, point=points[-1])
        self._graph.add_edge(
            start_node,
            end_node,
            points=points,
            distance_m=polyline_distance_m(points),
        )

    def _graph_endpoints(self, component):
        simple_graph = self._simple_weighted_graph()
        return [node_id for node_id in component if simple_graph.degree(node_id) == 1]

    def _snap_relation_node(self, relation_node_id, node_coordinates, sampler):
        relation_point = node_coordinates.get(relation_node_id)
        if relation_point is None or sampler is None:
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

    def _shortest_path_data(self, sources):
        simple_graph = self._simple_weighted_graph()
        for source in sources:
            if source not in self._shortest_path_cache:
                self._shortest_path_cache[source] = nx.single_source_dijkstra(
                    simple_graph,
                    source,
                    weight='weight',
                )
        return (
            {source: self._shortest_path_cache[source][0] for source in sources},
            {source: self._shortest_path_cache[source][1] for source in sources},
        )

    def _edge_distance_m(self):
        return sum(
            attributes['distance_m']
            for _, _, attributes in self._graph.edges(data=True)
        )

    def _matching_paths(self, nodes, free_finish_nodes=None, free_finish_count=1):
        simple_graph = self._simple_weighted_graph()
        matching_graph = nx.Graph()
        matching_graph.add_nodes_from(nodes)
        shortest_distances, shortest_paths = self._shortest_path_data(nodes)

        for first_index, first_node in enumerate(nodes):
            for second_node in nodes[first_index + 1:]:
                if second_node not in shortest_distances[first_node]:
                    return None
                matching_graph.add_edge(
                    first_node,
                    second_node,
                    weight=shortest_distances[first_node][second_node],
                )

        dummy_nodes = []
        if free_finish_nodes:
            for _ in range(free_finish_count):
                dummy_node = object()
                dummy_nodes.append(dummy_node)
                matching_graph.add_node(dummy_node)
                for finish_index, finish_node in enumerate(free_finish_nodes):
                    matching_graph.add_edge(dummy_node, finish_node, weight=finish_index * 1e-12)

        matching = nx.min_weight_matching(matching_graph, weight='weight')
        if len(matching) * 2 != len(matching_graph):
            return None

        paths = []
        selected_finishes = []
        for pair in matching:
            first_node, second_node = tuple(pair)
            if first_node in dummy_nodes:
                selected_finishes.append(second_node)
                continue
            if second_node in dummy_nodes:
                selected_finishes.append(first_node)
                continue
            paths.append(shortest_paths[first_node][second_node])
        selected_finish = (
            None
            if not selected_finishes
            else selected_finishes[0]
            if len(selected_finishes) == 1
            else tuple(sorted(selected_finishes))
        )
        return paths, simple_graph, selected_finish

    @staticmethod
    def _matching_distance(paths, simple_graph):
        return sum(
            simple_graph[first_node][second_node]['weight']
            for path in paths
            for first_node, second_node in zip(path, path[1:])
        )

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
