import math
import re
from enum import IntEnum

import networkx as nx


class LandmarkCategory(IntEnum):
    HIGHEST = 3


LANDMARK_RULES = (
    ('highway', 'trailhead'),
    ('information', 'guidepost'),
    ('information', 'map'),
    ('information', 'board'),
    ('tourism', 'information'),
)
LANDMARK_MAX_DISTANCE_M = 30
LANDMARK_GRID_SIZE_DEGREES = 0.0005


def landmark_candidate(tags):
    """Return selected metadata for a high-signal OSM landmark, if applicable."""
    for key, value in LANDMARK_RULES:
        if tags.get(key) == value:
            return {
                'category': LandmarkCategory.HIGHEST,
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

    def __init__(self, route_relation, way_nodes, externally_reachable_nodes=None):
        self._route_relation = route_relation
        self._externally_reachable_nodes = set(externally_reachable_nodes or ())
        self._graph = nx.MultiGraph()
        self._simple_graph = None
        self._shortest_path_cache = {}
        self._odd_nodes_cache = None
        self._edge_distance_cache = None
        self._relation_node_order = {}
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
            component_graph._externally_reachable_nodes = (
                self._externally_reachable_nodes & set(component)
            )
            component_graph._graph = self._graph.subgraph(component).copy()
            component_graph._simple_graph = None
            component_graph._shortest_path_cache = {}
            component_graph._odd_nodes_cache = None
            component_graph._edge_distance_cache = None
            component_graph._relation_node_order = {
                node_id: order
                for node_id, order in self._relation_node_order.items()
                if node_id in component
            }
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

    def _route_landmark_nodes(self, landmarks):
        landmark_index = self._landmark_index(landmarks)
        if landmark_index is None or self._graph.number_of_nodes() == 0:
            return set()

        route_tokens = set(
            self._text_token_list(
                self._route_relation.get('name', ''),
                self._route_relation.get('name_int', ''),
            )
        )
        if not route_tokens:
            return set()

        landmark_nodes = set()
        for node_id, graph_attributes in self._graph.nodes(data=True):
            for landmark_index_value, landmark_point in landmark_index.nearby(graph_attributes['point']):
                landmark = landmark_index.landmark(landmark_index_value)
                if landmark.get('category') != LandmarkCategory.HIGHEST:
                    continue
                if haversine_distance_m(landmark_point, graph_attributes['point']) > LANDMARK_MAX_DISTANCE_M:
                    continue
                landmark_tokens = set(
                    self._text_token_list(
                        landmark.get('name', ''),
                        landmark.get('description', ''),
                        landmark.get('ref', ''),
                    )
                )
                if len(route_tokens & landmark_tokens) >= 2:
                    landmark_nodes.add(node_id)
                    break
        return landmark_nodes

    def _endpoint_score(self, node_id, landmark_nodes, is_start):
        order = self._relation_node_order[node_id]
        return (node_id in landmark_nodes, -order if is_start else order)

    def _best_endpoint(self, nodes, landmark_nodes, is_start):
        return max(
            nodes,
            key=lambda node_id: self._endpoint_score(node_id, landmark_nodes, is_start),
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
        roundtrip=None,
    ):
        """Resolve endpoints and return the shortest route inspection traversal."""
        if (
            self._graph.number_of_nodes() == 0
            or not nx.is_connected(self._graph)
        ):
            return None

        if roundtrip is None:
            roundtrip = self._route_relation.get('roundtrip', False)
        roundtrip = str(roundtrip).strip().lower() in ('1', 'true', 'yes')
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
            finish_nodes = (
                [finish_node]
                if finish_node is not None
                else [start_node]
                if roundtrip
                else sorted(self._graph)
            )
            traversal = self._shortest_traversal_from(start_node, finish_nodes)
            if traversal is None:
                return None
            walk, _, traversal_finish = traversal
            return start_node, walk, traversal_finish

        if finish_node is not None:
            traversal = self._shortest_traversal_from(finish_node, sorted(self._graph))
            if traversal is None:
                return None
            walk, _, inferred_start = traversal
            return inferred_start, list(reversed(walk)), finish_node

        line_endpoints = self._simple_line_endpoints()
        if line_endpoints:
            start = self._default_start_node(line_endpoints)
            finish = next(node_id for node_id in line_endpoints if node_id != start)
            traversal = self._shortest_traversal_to(start, finish)
            if traversal is None:
                return None
            return start, traversal[0], traversal[2]

        eligible_nodes = self._externally_accessible_nodes()
        if not eligible_nodes:
            return None
        landmark_nodes = self._route_landmark_nodes(landmarks)
        start = self._best_endpoint(eligible_nodes, landmark_nodes, is_start=True)

        finish_nodes = [start]
        if self._odd_nodes() and not roundtrip:
            finish_nodes.extend(sorted(
                eligible_nodes - {start},
                key=lambda node_id: self._endpoint_score(
                    node_id,
                    landmark_nodes,
                    is_start=False,
                ),
                reverse=True,
            ))
        traversal = self._shortest_traversal_from(start, finish_nodes)
        if traversal is None:
            return None
        return start, traversal[0], traversal[2]

    def _odd_nodes(self):
        if self._odd_nodes_cache is None:
            self._odd_nodes_cache = sorted(
                node_id
                for node_id, degree in self._graph.degree()
                if degree % 2
            )
        return self._odd_nodes_cache

    def _externally_accessible_nodes(self):
        return self._externally_reachable_nodes & set(self._graph)

    def _simple_line_endpoints(self):
        simple_graph = self._simple_weighted_graph()
        endpoints = self._graph_endpoints(set(simple_graph))
        if len(endpoints) == 2 and all(simple_graph.degree(node_id) <= 2 for node_id in simple_graph):
            return endpoints
        return None

    def _default_start_node(self, nodes):
        return min(nodes, key=self._relation_node_order.__getitem__)

    def _landmark_index(self, landmarks):
        if landmarks is None or isinstance(landmarks, LandmarkIndex):
            return landmarks
        return LandmarkIndex(landmarks)

    def _shortest_traversal_from(self, start_node, finish_nodes):
        finish_nodes = list(finish_nodes)
        if not finish_nodes:
            return None
        if len(finish_nodes) == 1:
            finish_node = finish_nodes[0]
            traversal_data = self._shortest_traversal_data_to(
                start_node,
                finish_node,
            )
            return (
                None
                if traversal_data is None
                else self._build_traversal(
                    start_node,
                    finish_node,
                    traversal_data,
                )
            )

        finish_node_set = set(finish_nodes)
        candidates = []
        odd_nodes = set(self._odd_nodes())
        if start_node in finish_node_set:
            traversal_data = self._shortest_traversal_data_to(start_node, start_node)
            if traversal_data is not None:
                candidates.append((start_node, traversal_data))

        open_finish_nodes = [
            node_id
            for node_id in finish_nodes
            if node_id != start_node and node_id in odd_nodes
        ]
        if open_finish_nodes:
            traversal_data = self._shortest_open_traversal_data_from(
                start_node,
                open_finish_nodes,
            )
            if traversal_data is not None:
                candidates.append((traversal_data[3], traversal_data[:3]))

        if not candidates:
            return None
        traversal_data = min(
            candidates,
            key=lambda candidate: candidate[1][2],
        )
        return self._build_traversal(
            start_node,
            traversal_data[0],
            traversal_data[1],
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
        for node_id in self._route_relation.get('node_ids', ()):
            self._relation_node_order.setdefault(node_id, len(self._relation_node_order))
        for _, nodes in relation_way_nodes:
            for node_id, _ in nodes:
                self._relation_node_order.setdefault(node_id, len(self._relation_node_order))
        node_way_ids = {}
        for way_id, nodes in relation_way_nodes:
            for node_id, _ in nodes:
                node_way_ids.setdefault(node_id, set()).add(way_id)

        relation_nodes = {
            node_id
            for role in ('start', 'end')
            for node_id in self._route_relation.get('node_roles', {}).get(role, [])
        }
        relation_nodes.update(self._externally_reachable_nodes)
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
        self._odd_nodes_cache = None
        self._edge_distance_cache = None
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
        if self._edge_distance_cache is None:
            self._edge_distance_cache = sum(
                attributes['distance_m']
                for _, _, attributes in self._graph.edges(data=True)
            )
        return self._edge_distance_cache

    def _matching_paths(self, nodes, free_finish_nodes=None):
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

        dummy_node = None
        if free_finish_nodes:
            dummy_node = object()
            matching_graph.add_node(dummy_node)
            for finish_index, finish_node in enumerate(free_finish_nodes):
                matching_graph.add_edge(
                    dummy_node,
                    finish_node,
                    weight=finish_index * 1e-12,
                )

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
        if free_finish_nodes:
            if selected_finish is None:
                return None
            return paths, simple_graph, selected_finish
        return paths, simple_graph

    @staticmethod
    def _matching_distance(paths, simple_graph):
        return sum(
            simple_graph[first_node][second_node]['weight']
            for path in paths
            for first_node, second_node in zip(path, path[1:])
        )

    def _shortest_traversal_to(self, start_node, finish_node):
        traversal_data = self._shortest_traversal_data_to(start_node, finish_node)
        if traversal_data is None:
            return None
        return self._build_traversal(start_node, finish_node, traversal_data)

    def _shortest_traversal_data_to(self, start_node, finish_node):
        odd_nodes = set(self._odd_nodes())
        target_odd_nodes = set() if finish_node == start_node else {start_node, finish_node}
        matching_nodes = odd_nodes ^ target_odd_nodes
        paths_result = self._matching_paths(sorted(matching_nodes))
        if paths_result is None:
            return None

        paths, simple_graph = paths_result
        distance = self._edge_distance_m() + self._matching_distance(paths, simple_graph)
        return paths, simple_graph, distance

    def _shortest_open_traversal_data_from(self, start_node, finish_nodes):
        matching_nodes = sorted(set(self._odd_nodes()) ^ {start_node})
        paths_result = self._matching_paths(matching_nodes, finish_nodes)
        if paths_result is None:
            return None

        paths, simple_graph, finish_node = paths_result
        distance = self._edge_distance_m() + self._matching_distance(paths, simple_graph)
        return paths, simple_graph, distance, finish_node

    def _build_traversal(self, start_node, finish_node, traversal_data):
        paths, simple_graph, distance = traversal_data
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
