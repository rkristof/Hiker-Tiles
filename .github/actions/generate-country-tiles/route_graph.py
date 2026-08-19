import math

import networkx as nx


class RouteGraph:
    """Build and traverse one OSM route relation as an edge-preserving graph."""

    def __init__(self, route_relation, way_nodes):
        self._route_relation = route_relation
        self._graph = nx.MultiGraph()
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
        repairs = 0
        while True:
            components = list(nx.connected_components(self._graph))
            if len(components) <= 1:
                return repairs

            component_by_node = {
                node_id: component_index
                for component_index, component in enumerate(components)
                for node_id in component
            }
            endpoints = [
                node_id
                for component in components
                for node_id in self._graph_endpoints(component)
            ]
            candidates = []
            for first_index, first_node in enumerate(endpoints):
                for second_node in endpoints[first_index + 1:]:
                    if component_by_node[first_node] == component_by_node[second_node]:
                        continue
                    first_point = self.point(first_node)
                    second_point = self.point(second_node)
                    horizontal_distance = haversine_distance_m(first_point, second_point)
                    if horizontal_distance >= 100:
                        continue
                    first_elevation = sampler.sample(first_point)
                    second_elevation = sampler.sample(second_point)
                    if first_elevation is None or second_elevation is None:
                        continue
                    if abs(first_elevation - second_elevation) >= 15:
                        continue
                    candidates.append((horizontal_distance, first_node, second_node))

            if not candidates:
                return repairs

            mutual_candidates = [
                candidate
                for candidate in candidates
                if self._is_unique_nearest(candidate[1], candidate, candidates)
                and self._is_unique_nearest(candidate[2], candidate, candidates)
            ]
            if not mutual_candidates:
                return repairs

            _, first_node, second_node = min(mutual_candidates)
            self._add_edge(
                first_node,
                second_node,
                [self.point(first_node), self.point(second_node)],
            )
            repairs += 1

    def resolve_start(self, node_coordinates, sampler):
        explicit_starts = self._route_relation.get('node_roles', {}).get('start', [])
        if explicit_starts:
            if len(explicit_starts) != 1:
                return None
            start_node = explicit_starts[0]
            if start_node in self._graph:
                return start_node
            return self._snap_relation_node(start_node, node_coordinates, sampler)

        leaves = [
            node_id
            for component in nx.connected_components(self._graph)
            for node_id in self._graph_endpoints(component)
        ]
        if len(leaves) == 1:
            return leaves[0]
        return next(iter(self._graph), None)

    def resolve_finish(self, start_node, node_coordinates, sampler):
        explicit_finishes = self._route_relation.get('node_roles', {}).get('end', [])
        if explicit_finishes:
            if len(explicit_finishes) != 1:
                return None
            finish_node = explicit_finishes[0]
            if finish_node in self._graph:
                return finish_node
            return self._snap_relation_node(finish_node, node_coordinates, sampler)
        if self._route_relation.get('roundtrip'):
            return start_node
        return None

    def shortest_traversal(self, start_node, finish_node=None):
        if start_node not in self._graph or not nx.is_connected(self._graph):
            return None

        finish_candidates = [finish_node] if finish_node is not None else sorted(
            self._graph_endpoints(set(self._graph.nodes)) or [start_node]
        )
        best_walk = None
        for candidate in finish_candidates:
            walk = self._shortest_traversal_to(start_node, candidate)
            if walk is None:
                continue
            if best_walk is None or walk[1] < best_walk[1]:
                best_walk = walk
        return (best_walk[0], best_walk[2]) if best_walk is not None else None

    def traversal_coordinates(self, start_node, steps):
        coordinates = [self.point(start_node)]
        current_node = start_node
        for first_node, second_node, _, edge in steps:
            if first_node == current_node:
                edge_points = edge['points']
                current_node = second_node
            else:
                edge_points = list(reversed(edge['points']))
                current_node = first_node
            coordinates.extend(edge_points[1:])
        return coordinates, current_node

    def _build(self, way_nodes):
        for way_id in self._route_relation['way_ids']:
            nodes = way_nodes.get(way_id, [])
            for first_node, second_node in zip(nodes, nodes[1:]):
                self._add_edge(
                    first_node[0],
                    second_node[0],
                    [first_node[1], second_node[1]],
                )

    def _add_edge(self, start_node, end_node, points):
        if start_node == end_node or len(points) < 2:
            return

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

    @staticmethod
    def _is_unique_nearest(node_id, candidate, candidates):
        distances = [
            distance
            for distance, first_node, second_node in candidates
            if node_id in (first_node, second_node)
        ]
        nearest_distance = min(distances)
        return sum(
            abs(distance - nearest_distance) < 1e-6
            for distance in distances
        ) == 1 and abs(candidate[0] - nearest_distance) < 1e-6

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
        return simple_graph

    def _matching_paths(self, nodes):
        simple_graph = self._simple_weighted_graph()
        matching_graph = nx.Graph()
        paths = {}
        for first_index, first_node in enumerate(nodes):
            for second_node in nodes[first_index + 1:]:
                try:
                    path = nx.shortest_path(
                        simple_graph,
                        first_node,
                        second_node,
                        weight='weight',
                    )
                except nx.NetworkXNoPath:
                    return None
                distance = sum(
                    simple_graph[path[index]][path[index + 1]]['weight']
                    for index in range(len(path) - 1)
                )
                matching_graph.add_edge(first_node, second_node, weight=distance)
                paths[frozenset((first_node, second_node))] = path

        matching = nx.min_weight_matching(matching_graph, weight='weight')
        if len(matching) * 2 != len(nodes):
            return None
        return [paths[frozenset(pair)] for pair in matching], simple_graph

    def _shortest_traversal_to(self, start_node, finish_node):
        target_odd_nodes = set() if finish_node == start_node else {start_node, finish_node}
        odd_nodes = {node_id for node_id, degree in self._graph.degree() if degree % 2}
        paths_result = self._matching_paths(sorted(odd_nodes ^ target_odd_nodes))
        if paths_result is None:
            return None

        paths, simple_graph = paths_result
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
