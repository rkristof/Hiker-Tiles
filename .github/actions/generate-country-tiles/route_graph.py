import math

import networkx as nx


class RouteGraph:
    """Build and traverse one OSM route relation as an edge-preserving graph."""

    def __init__(self, route_relation, way_nodes, node_way_ids=None):
        self._route_relation = route_relation
        self._node_way_ids = node_way_ids
        self._graph = nx.MultiGraph()
        self._simple_graph = None
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

    def resolve_start(self, node_coordinates, sampler):
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

        if finish_node is not None:
            walk = self._shortest_traversal_to(start_node, finish_node)
            return None if walk is None else (walk[0], walk[2])

        closed_walk = self._shortest_traversal_to(start_node, start_node)
        if closed_walk is not None:
            return closed_walk[0], closed_walk[2]

        finish_candidates = sorted(
            self._graph_endpoints(set(self._graph.nodes)) or [start_node]
        )
        best_walk = None
        if start_node in finish_candidates:
            best_walk = self._shortest_traversal_to(start_node, start_node)
            finish_candidates.remove(start_node)
        if finish_candidates:
            walk = self._shortest_traversal_to(start_node, None, finish_candidates)
            if walk is not None and (best_walk is None or walk[1] < best_walk[1]):
                best_walk = walk

        return None if best_walk is None else (best_walk[0], best_walk[2])

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
