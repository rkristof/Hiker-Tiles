import networkx as nx

from utils import haversine_distance_m


class RouteGraph:
    """Build a minimal weighted graph from route ways."""

    def __init__(
        self,
        route_relation,
        way_nodes,
        way_segment_distances,
        connecting_highways_by_node=None,
        sampler=None,
        roundtrip=False,
        inferred_start_node=None,
    ):
        self._route_relation = route_relation
        self._roundtrip = roundtrip
        self._route_way_ids = set(route_relation.get('way_ids', ()))
        self._connecting_highways_by_node = connecting_highways_by_node or {}
        self._relation_node_order = {}
        self._raw_graph = nx.MultiGraph()

        for node_id in route_relation.get('node_ids', ()):
            self._relation_node_order.setdefault(
                node_id,
                len(self._relation_node_order),
            )

        for way_id in route_relation.get('way_ids', ()):
            nodes = way_nodes.get(way_id, ())
            if len(nodes) < 2:
                continue

            for node_id, point in nodes:
                self._relation_node_order.setdefault(node_id, len(self._relation_node_order))
                self._raw_graph.add_node(node_id, point=point)

            segment_distances = way_segment_distances[way_id]
            for segment_index, (first_node, second_node) in enumerate(zip(nodes, nodes[1:])):
                first_node_id, first_point = first_node
                second_node_id, second_point = second_node
                if first_node_id == second_node_id:
                    continue
                self._raw_graph.add_edge(
                    first_node_id,
                    second_node_id,
                    weight=segment_distances[segment_index],
                    points=[first_point, second_point],
                )

        self._raw_route_distance = self._raw_graph.size(weight='weight')
        self._remaining_repair_distance = self._raw_route_distance * 0.10
        self._repair_disconnected_components(sampler)
        self._repair_near_closed_endpoints(sampler)

        if (
            not self._roundtrip
            and self.has_edges
            and nx.is_eulerian(self._raw_graph)
        ):
            self._roundtrip = True

        if self.has_edges:
            self._create_compressed_graph(
                (inferred_start_node,)
                if inferred_start_node is not None
                else ()
            )

    @property
    def has_edges(self):
        return self._raw_graph.number_of_edges() > 0

    @property
    def component_count(self):
        return nx.number_connected_components(self._graph)

    def component_graphs(self):
        """Return independent RouteGraph views for each connected component."""
        components = sorted(
            nx.connected_components(self._graph),
            key=lambda component: min(component),
        )
        component_graphs = []
        for component in components:
            component = set(component)
            component_relation = {
                **self._route_relation,
                'node_roles': {
                    role: [node_id for node_id in node_ids if node_id in component]
                    for role, node_ids in self._route_relation.get('node_roles', {}).items()
                },
            }
            component_graph = object.__new__(RouteGraph)
            component_graph._route_relation = component_relation
            component_graph._roundtrip = self._roundtrip
            component_graph._relation_node_order = {
                node_id: order
                for node_id, order in self._relation_node_order.items()
                if node_id in component
            }
            component_graph._raw_graph = self._raw_graph.subgraph(component)
            component_graph._graph = self._graph.subgraph(component).copy()
            component_graphs.append(component_graph)
        return component_graphs

    def is_simple_line(self):
        """Return whether the graph is one connected line without branches."""
        if not self._graph or not nx.is_connected(self._graph):
            return False
        endpoints = [
            node_id
            for node_id, degree in self._graph.degree()
            if degree == 1
        ]
        return len(endpoints) == 2 and all(
            degree <= 2
            for _, degree in self._graph.degree()
        )

    def simple_line_endpoints(self):
        """Return simple-line start and end nodes in relation order."""
        if not self.is_simple_line():
            return None
        endpoints = [
            node_id
            for node_id, degree in self._graph.degree()
            if degree == 1
        ]
        start_node = min(endpoints, key=self._relation_node_order.__getitem__)
        finish_node = next(node_id for node_id in endpoints if node_id != start_node)
        return start_node, finish_node

    def is_eulerian(self):
        """Return whether the graph supports a closed Eulerian traversal."""
        return bool(self._graph) and nx.is_eulerian(self._graph)

    def point(self, node_id):
        return self._graph.nodes[node_id]['point']

    def traversal_coordinates(self, traversal):
        """Return full route coordinates for a compressed node traversal."""
        if not traversal:
            return []
        graph = self._graph
        if traversal[0] not in graph:
            return []

        coordinates = [graph.nodes[traversal[0]]['point']]
        used_edges = {}
        for first_node, second_node in zip(traversal, traversal[1:]):
            edge_data = graph.get_edge_data(first_node, second_node)
            if edge_data is None:
                return []
            edge_keys = tuple(edge_data)
            edge_index = used_edges.get(frozenset((first_node, second_node)), 0)
            edge_data = edge_data[edge_keys[min(edge_index, len(edge_keys) - 1)]]
            used_edges[frozenset((first_node, second_node))] = edge_index + 1
            points = edge_data['points']
            if points[0] != graph.nodes[first_node]['point']:
                points = list(reversed(points))
            coordinates.extend(points[1:])
        return coordinates

    def eulerian_traversal(self, start_node):
        """Return a closed Eulerian or shortest complete traversal."""
        if start_node is None:
            start_node = next(iter(self._graph), None)
        if (
            start_node not in self._graph
            or not self._graph
            or not nx.is_connected(self._graph)
        ):
            return None
        if not nx.is_eulerian(self._graph):
            return self.shortest_complete_traversal(start_node, start_node)

        return [
            start_node,
            *(
                second_node
                for _, second_node in nx.eulerian_circuit(
                    self._graph,
                    source=start_node,
                )
            ),
        ]

    def traversal_distance(self, traversal):
        """Return the weighted length of a traversal in the graph."""
        distance = 0
        used_edges = {}
        for first_node, second_node in zip(traversal, traversal[1:]):
            edge_data = self._graph.get_edge_data(first_node, second_node)
            edge_keys = tuple(edge_data)
            edge_index = used_edges.get(frozenset((first_node, second_node)), 0)
            edge_data = edge_data[edge_keys[min(edge_index, len(edge_keys) - 1)]]
            used_edges[frozenset((first_node, second_node))] = edge_index + 1
            distance += edge_data['weight']
        return distance

    def shortest_complete_traversal(self, start_node=None, finish_node=None):
        """Return the shortest complete traversal in the graph."""
        if start_node is None and finish_node is None:
            return min(
                (
                    self.shortest_complete_traversal(
                        start_node,
                        None,
                    )
                    for start_node in self._graph
                ),
                key=self.traversal_distance,
                default=None,
            )
        if start_node is None:
            return None
        graph = self._graph
        if (
            start_node not in graph
            or (
                finish_node is not None
                and finish_node not in graph
            )
            or not graph
            or not nx.is_connected(graph)
        ):
            return None

        finish_nodes = (
            [finish_node]
            if finish_node is not None
            else [node_id for node_id in graph if node_id != start_node]
        ) or [start_node]
        shortest_traversal = None
        shortest_distance = float('inf')

        for candidate_finish in finish_nodes:
            odd_nodes = [
                node_id
                for node_id, degree in graph.degree()
                if degree % 2
            ]
            for endpoint in (start_node, candidate_finish):
                if endpoint in odd_nodes:
                    odd_nodes.remove(endpoint)
                else:
                    odd_nodes.append(endpoint)

            matching_graph = nx.Graph()
            shortest_paths = {}
            for first_index, first_node in enumerate(odd_nodes):
                distances, paths = nx.single_source_dijkstra(
                    graph,
                    first_node,
                    weight='weight',
                )
                for second_node in odd_nodes[first_index + 1:]:
                    matching_graph.add_edge(
                        first_node,
                        second_node,
                        weight=distances[second_node],
                    )
                    shortest_paths[first_node, second_node] = paths[second_node]

            euler_graph = graph.copy()
            for first_node, second_node in nx.min_weight_matching(
                matching_graph,
                weight='weight',
            ):
                path = shortest_paths.get((first_node, second_node))
                if path is None:
                    path = reversed(shortest_paths[second_node, first_node])
                for path_first, path_second in nx.utils.pairwise(path):
                    euler_graph.add_edge(
                        path_first,
                        path_second,
                        **self._minimum_edge_data(graph, path_first, path_second),
                    )

            traversal = [start_node]
            traversal.extend(
                second_node
                for _, second_node in nx.eulerian_path(
                    euler_graph,
                    source=start_node,
                )
            )
            distance = self.traversal_distance(traversal)
            if distance < shortest_distance:
                shortest_traversal = traversal
                shortest_distance = distance

        return shortest_traversal

    def shortest_complete_traversal_to_nearest_finish(self, start_node):
        """Return the shortest traversal to a possible finish node."""
        if (
            start_node not in self._graph
            or not self._graph
            or not nx.is_connected(self._graph)
        ):
            return None

        finish_nodes = [
            node_id
            for node_id, degree in self._graph.degree()
            if node_id != start_node and degree % 2
        ]
        if not finish_nodes:
            return self.eulerian_traversal(start_node)

        return min(
            (
                traversal
                for finish_node in finish_nodes
                if (
                    traversal := self.shortest_complete_traversal(
                        start_node,
                        finish_node,
                    )
                )
            ),
            key=self.traversal_distance,
            default=None,
        )

    def _create_compressed_graph(self, additional_nodes=()):
        # Compress topological edges; raw graph retains duplicate route members.
        raw_graph = nx.MultiGraph(nx.Graph(self._raw_graph))
        retained_nodes = {
            node_id
            for node_id, degree in raw_graph.degree()
            if degree == 1 or degree > 2
        }
        retained_nodes.update(
            node_id
            for role in ('start', 'end')
            for node_id in self._route_relation.get('node_roles', {}).get(role, ())
            if node_id in raw_graph
        )
        retained_nodes.update(
            node_id
            for node_id in additional_nodes
            if node_id in raw_graph
        )
        if not retained_nodes and raw_graph:
            retained_nodes.add(next(iter(raw_graph)))

        simple_graph = nx.MultiGraph()
        simple_graph.add_nodes_from(
            (node_id, raw_graph.nodes[node_id])
            for node_id in retained_nodes
        )

        for start_node in retained_nodes:
            while raw_graph.degree(start_node):
                _, current_node, edge_key, edge_data = next(
                    iter(raw_graph.edges(start_node, keys=True, data=True)),
                )
                raw_graph.remove_edge(start_node, current_node, edge_key)
                edge_weight = edge_data['weight']
                edge_points = list(edge_data['points'])
                if edge_points[0] != raw_graph.nodes[start_node]['point']:
                    edge_points.reverse()

                while current_node not in retained_nodes:
                    _, next_node, next_edge_key, next_edge_data = next(
                        iter(raw_graph.edges(current_node, keys=True, data=True)),
                    )
                    raw_graph.remove_edge(current_node, next_node, next_edge_key)
                    edge_weight += next_edge_data['weight']
                    next_points = list(next_edge_data['points'])
                    if next_points[0] != raw_graph.nodes[current_node]['point']:
                        next_points.reverse()
                    edge_points.extend(next_points[1:])
                    current_node = next_node

                simple_graph.add_edge(
                    start_node,
                    current_node,
                    weight=edge_weight,
                    points=edge_points,
                )

        self._graph = simple_graph
        if self._roundtrip and self._graph and nx.is_connected(self._graph):
            self._graph = self._eulerize_graph(self._graph)

    @staticmethod
    def _eulerize_graph(graph):
        eulerized_graph = nx.eulerize(graph)
        for first_node, second_node, edge_key in eulerized_graph.edges(keys=True):
            edge_data = eulerized_graph[first_node][second_node][edge_key]
            if 'weight' not in edge_data:
                edge_data.update(
                    RouteGraph._minimum_edge_data(graph, first_node, second_node),
                )
        return eulerized_graph

    @staticmethod
    def _minimum_edge_data(graph, first_node, second_node):
        edge_data = graph.get_edge_data(first_node, second_node)
        return min(edge_data.values(), key=lambda data: data['weight'])

    def _repair_disconnected_components(self, sampler):
        if sampler is None:
            return

        while True:
            components = list(nx.connected_components(self._raw_graph))
            if len(components) < 2:
                return

            endpoints = self._degree_one_endpoints()
            repaired = False
            for first_index, first_component in enumerate(components):
                first_endpoints = [
                    node_id
                    for node_id in endpoints
                    if node_id in first_component
                ]
                first_distance = self._raw_graph.subgraph(first_component).size(
                    weight='weight',
                )
                for second_component in components[first_index + 1:]:
                    second_endpoints = [
                        node_id
                        for node_id in endpoints
                        if node_id in second_component
                    ]
                    max_distance = (
                        first_distance
                        + self._raw_graph.subgraph(second_component).size(
                            weight='weight',
                        )
                    ) * 0.10
                    for first_node, second_node in self._ordered_endpoint_pairs(
                        first_endpoints,
                        second_endpoints,
                        max_distance,
                    ):
                        if self._repair_endpoint_pair(
                            sampler,
                            first_node,
                            second_node,
                            max_distance,
                        ):
                            repaired = True
                            break
                    if repaired:
                        break
                if repaired:
                    break
            if not repaired:
                return

    def _repair_near_closed_endpoints(self, sampler):
        if sampler is None or self._raw_route_distance <= 0:
            return

        max_distance = self._raw_route_distance * 0.10
        min_graph_distance = self._raw_route_distance * 0.60
        while True:
            first_endpoints = self._degree_one_endpoints()
            second_endpoints = set(first_endpoints)
            second_endpoints.update(
                node_id
                for node_id in self._raw_graph
                if any(
                    way_id not in self._route_way_ids
                    for way_id in self._connecting_highways_by_node.get(node_id, {})
                )
            )
            if not first_endpoints or len(second_endpoints) < 2:
                return

            repaired = False
            for component in nx.connected_components(self._raw_graph):
                component_first_endpoints = [
                    node_id
                    for node_id in first_endpoints
                    if node_id in component
                ]
                component_second_endpoints = [
                    node_id
                    for node_id in second_endpoints
                    if node_id in component
                ]
                component_graph = self._raw_graph.subgraph(component)
                shortest_paths = {
                    first_node: nx.single_source_dijkstra_path_length(
                        component_graph,
                        first_node,
                        cutoff=min_graph_distance,
                        weight='weight',
                    )
                    for first_node in component_first_endpoints
                }
                for first_node, second_node in self._ordered_endpoint_pairs(
                    component_first_endpoints,
                    component_second_endpoints,
                    max_distance,
                ):
                    if second_node in shortest_paths[first_node]:
                        continue
                    if self._repair_endpoint_pair(
                        sampler,
                        first_node,
                        second_node,
                        max_distance,
                    ):
                        repaired = True
                        break
                if repaired:
                    break
            if not repaired:
                return

    def _ordered_endpoint_pairs(self, first_endpoints, second_endpoints, max_distance):
        pairs = {}
        for first_node in first_endpoints:
            for second_node in second_endpoints:
                if first_node != second_node:
                    pairs.setdefault(
                        frozenset((first_node, second_node)),
                        (first_node, second_node),
                    )
        pairs = pairs.values()
        pairs = [
            (
                haversine_distance_m(
                    self._raw_graph.nodes[first_node]['point'],
                    self._raw_graph.nodes[second_node]['point'],
                ),
                first_node,
                second_node,
            )
            for first_node, second_node in pairs
        ]
        pairs = [pair for pair in pairs if pair[0] < max_distance]
        return [
            (first_node, second_node)
            for _, first_node, second_node in sorted(pairs, key=lambda pair: pair[0])
        ]

    def _degree_one_endpoints(self):
        return [
            node_id
            for node_id in self._raw_graph
            if self._raw_graph.degree(node_id) == 1
        ]

    def _repair_endpoint_pair(
        self,
        sampler,
        first_node,
        second_node,
        max_distance,
    ):
        max_distance = min(max_distance, self._remaining_repair_distance)
        if max_distance <= 0:
            return False

        first_elevation = sampler.sample(self._raw_graph.nodes[first_node]['point'])
        second_elevation = sampler.sample(self._raw_graph.nodes[second_node]['point'])
        if (
            first_elevation is None
            or second_elevation is None
            or abs(first_elevation - second_elevation) >= 15
        ):
            return False

        edge_data = self._repair_edge(
            first_node,
            second_node,
            max_distance,
        )
        if edge_data is None:
            return False

        self._raw_graph.add_edge(
            first_node,
            second_node,
            **edge_data,
        )
        self._remaining_repair_distance -= edge_data['weight']
        return True

    def _repair_edge(self, first_node, second_node, max_distance):
        initial_distance = haversine_distance_m(
            self._raw_graph.nodes[first_node]['point'],
            self._raw_graph.nodes[second_node]['point'],
        )
        first_extension = self._best_highway_extension(first_node, second_node)
        second_extension = self._best_highway_extension(second_node, first_node)
        first_point = (
            first_extension['point']
            if first_extension is not None
            else self._raw_graph.nodes[first_node]['point']
        )
        if first_point == self._raw_graph.nodes[second_node]['point']:
            second_extension = None
        second_point = (
            second_extension['point']
            if second_extension is not None
            else self._raw_graph.nodes[second_node]['point']
        )
        remaining_distance = haversine_distance_m(first_point, second_point)
        extension_distance = sum(
            extension['distance']
            for extension in (first_extension, second_extension)
            if extension is not None
        )
        if (
            extension_distance + remaining_distance <= max_distance
            and remaining_distance < max_distance * 0.5
        ):
            connection_points = []
            if first_extension is not None:
                connection_points.extend(first_extension['points'])
            else:
                connection_points.append(self._raw_graph.nodes[first_node]['point'])
            if connection_points[-1] != second_point:
                connection_points.append(second_point)
            if second_extension is not None:
                connection_points.extend(reversed(second_extension['points'][:-1]))
            return {
                'weight': extension_distance + remaining_distance,
                'points': connection_points,
            }

        if initial_distance >= max_distance * 0.5:
            return None
        return {
            'weight': initial_distance,
            'points': [
                self._raw_graph.nodes[first_node]['point'],
                self._raw_graph.nodes[second_node]['point'],
            ],
        }

    def _best_highway_extension(self, node_id, other_node):
        candidates = []
        for way_id, highway in self._connecting_highways_by_node.get(node_id, {}).items():
            if way_id in self._route_way_ids:
                continue
            nodes = highway['nodes']
            if len(nodes) < 2:
                continue
            node_indexes = [
                index
                for index, (candidate_node_id, _) in enumerate(nodes)
                if candidate_node_id == node_id
            ]
            for node_index in node_indexes:
                for target_index in range(len(nodes)):
                    if target_index == node_index:
                        continue
                    step = 1 if target_index > node_index else -1
                    path_nodes = [
                        nodes[index]
                        for index in range(node_index, target_index + step, step)
                    ]
                    if not path_nodes:
                        continue
                    extension_distance = sum(
                        haversine_distance_m(first[1], second[1])
                        for first, second in zip(path_nodes, path_nodes[1:])
                    )
                    remaining_distance = haversine_distance_m(
                        path_nodes[-1][1],
                        self._raw_graph.nodes[other_node]['point'],
                    )
                    candidates.append({
                        'distance': extension_distance,
                        'point': path_nodes[-1][1],
                        'points': [point for _, point in path_nodes],
                        'remaining_distance': remaining_distance,
                    })
        return min(
            candidates,
            key=lambda candidate: (
                candidate['remaining_distance'],
                candidate['distance'],
            ),
            default=None,
        )
