import networkx as nx

from utils import haversine_distance_m


class RouteGraph:
    """Build a minimal weighted graph from route ways."""

    def __init__(
        self,
        route_relation,
        way_nodes,
        sampler=None,
        roundtrip=False,
    ):
        self._route_relation = route_relation
        self.eligible_nodes = ()
        self._roundtrip = roundtrip
        self._relation_node_order = {}
        self._raw_graph = nx.MultiGraph()
        self._complex_graph = None

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

            for first_node, second_node in zip(nodes, nodes[1:]):
                first_node_id, first_point = first_node
                second_node_id, second_point = second_node
                if first_node_id == second_node_id:
                    continue
                self._raw_graph.add_edge(
                    first_node_id,
                    second_node_id,
                    weight=haversine_distance_m(first_point, second_point),
                    points=[first_point, second_point],
                )

        self._repair_disconnected_components(sampler)

        if (
            not self._roundtrip
            and self._raw_graph
            and self._raw_graph.number_of_edges() > 0
            and nx.is_eulerian(self._raw_graph)
        ):
            self._roundtrip = True

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
            raw_component = nx.node_connected_component(
                self._raw_graph,
                next(iter(component)),
            )
            component_graph._raw_graph = self._raw_graph.subgraph(raw_component)
            component_graph._graph = self._graph.subgraph(component).copy()
            component_graph._complex_graph = None
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

    def traversal_coordinates(self, traversal, graph=None):
        """Return full route coordinates for a compressed node traversal."""
        if not traversal:
            return []
        graph = self._graph if graph is None else graph
        if traversal[0] not in graph:
            return []

        coordinates = [graph.nodes[traversal[0]]['point']]
        used_edges = {}
        for first_node, second_node in zip(traversal, traversal[1:]):
            edge_data = graph.get_edge_data(first_node, second_node)
            if edge_data is None:
                return []
            if graph.is_multigraph():
                edge_keys = tuple(edge_data)
                edge_index = used_edges.get(frozenset((first_node, second_node)), 0)
                edge_data = edge_data[edge_keys[min(edge_index, len(edge_keys) - 1)]]
                used_edges[frozenset((first_node, second_node))] = edge_index + 1
            points = edge_data['points']
            if points[0] != graph.nodes[first_node]['point']:
                points = list(reversed(points))
            coordinates.extend(points[1:])
        return coordinates

    def complex_eulerian_traversal(self):
        """Return an Eulerian traversal from an eligible or ordered graph node."""
        graph = self._complex_graph
        if not graph or not nx.is_connected(graph) or not nx.is_eulerian(graph):
            return None

        start_node = next(
            (node_id for node_id in self.eligible_nodes if node_id in graph),
            None,
        )
        if start_node is None:
            start_node = min(graph, key=self._relation_node_order.__getitem__)

        return [
            start_node,
            *(
                second_node
                for _, second_node in nx.eulerian_circuit(
                    graph,
                    source=start_node,
                )
            ),
        ]

    def eulerian_traversal(self, start_node):
        """Return a closed Eulerian or shortest complete traversal."""
        if (
            start_node not in self._graph
            or not self._graph
            or not nx.is_connected(self._graph)
        ):
            return None
        if not nx.is_eulerian(self._graph):
            return self.shortest_complete_traversal_simple(start_node, start_node)

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

    def traversal_distance_simple(self, traversal):
        """Return the weighted length of a traversal in the simple graph."""
        return self._traversal_distance(traversal, self._graph)

    def traversal_distance_complex(self, traversal):
        """Return the weighted length of a traversal in the complex graph."""
        return self._traversal_distance(traversal, self._complex_graph)

    @staticmethod
    def _traversal_distance(traversal, graph):
        distance = 0
        used_edges = {}
        for first_node, second_node in zip(traversal, traversal[1:]):
            edge_data = graph.get_edge_data(first_node, second_node)
            if graph.is_multigraph():
                edge_keys = tuple(edge_data)
                edge_index = used_edges.get(frozenset((first_node, second_node)), 0)
                edge_data = edge_data[edge_keys[min(edge_index, len(edge_keys) - 1)]]
                used_edges[frozenset((first_node, second_node))] = edge_index + 1
                distance += edge_data['weight']
            else:
                distance += edge_data['weight']
        return distance

    def shortest_complete_traversal_simple(self, start_node=None, finish_node=None):
        """Return the shortest complete traversal in the simple graph."""
        if start_node is None and finish_node is None:
            return min(
                (
                    self._shortest_complete_traversal(
                        start_node,
                        None,
                        self._graph,
                    )
                    for start_node in self._graph
                ),
                key=self.traversal_distance_simple,
                default=None,
            )
        if start_node is None or finish_node is None:
            return None
        return self._shortest_complete_traversal(
            start_node,
            finish_node,
            self._graph,
        )

    def shortest_complete_traversal_complex(self, start_node, finish_node):
        """Return the shortest complete traversal in the complex graph."""
        return self._shortest_complete_traversal(
            start_node,
            finish_node,
            self._complex_graph,
        )

    def _shortest_complete_traversal(self, start_node, finish_node, graph):
        """Return a shortest weighted walk covering every graph edge."""
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

            euler_graph = nx.MultiGraph(graph)
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
            distance = self._traversal_distance(traversal, graph)
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
                    traversal := self.shortest_complete_traversal_simple(
                        start_node,
                        finish_node,
                    )
                )
            ),
            key=self.traversal_distance_simple,
            default=None,
        )

    def raw_route_distance_m(self):
        return sum(
            edge_data['weight']
            for _, _, edge_data in self._raw_graph.edges(data=True)
        )

    def _create_simple_graph(self):
        self._graph = RouteGraph._create_compressed_graph(
            self._raw_graph,
            self._route_relation,
        )
        if self._roundtrip and self._graph and nx.is_connected(self._graph):
            self._graph = self._eulerize_graph(self._graph)

    def _create_complex_graph(self, eligible_nodes):
        """Create a compressed graph that also retains eligible route nodes."""
        self.eligible_nodes = tuple(
            dict.fromkeys(
                node_id
                for node_id in eligible_nodes
                if node_id in self._raw_graph
            )
        )
        self._complex_graph = RouteGraph._create_compressed_graph(
            self._raw_graph,
            self._route_relation,
            self.eligible_nodes,
        )
        if self._roundtrip and self._complex_graph and nx.is_connected(self._complex_graph):
            self._complex_graph = self._eulerize_graph(self._complex_graph)

    @staticmethod
    def _create_compressed_graph(raw_graph, route_relation, additional_nodes=()):
        # Compress topological edges; raw graph retains duplicate route members.
        raw_graph = nx.MultiGraph(nx.Graph(raw_graph))
        retained_nodes = {
            node_id
            for node_id, degree in raw_graph.degree()
            if degree == 1 or degree > 2
        }
        retained_nodes.update(
            node_id
            for role in ('start', 'end')
            for node_id in route_relation.get('node_roles', {}).get(role, ())
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
                first_node, current_node, edge_key, edge_data = next(
                    iter(raw_graph.edges(start_node, keys=True, data=True)),
                )
                raw_graph.remove_edge(start_node, current_node, edge_key)
                edge_weight = edge_data['weight']
                edge_points = list(edge_data['points'])
                if edge_points[0] != raw_graph.nodes[start_node]['point']:
                    edge_points.reverse()

                while current_node not in retained_nodes:
                    first_node, next_node, next_edge_key, next_edge_data = next(
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

        return simple_graph

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
        if graph.is_multigraph():
            return min(edge_data.values(), key=lambda data: data['weight'])
        return edge_data

    def _repair_disconnected_components(self, sampler):
        components = list(nx.connected_components(self._raw_graph))
        if sampler is None or len(components) < 2:
            return

        component_by_node = {
            node_id: component_index
            for component_index, component in enumerate(components)
            for node_id in component
        }
        endpoints = [
            node_id
            for node_id in self._raw_graph
            if self._raw_graph.degree(node_id) == 1
        ]
        elevations = {
            node_id: sampler.sample(self._raw_graph.nodes[node_id]['point'])
            for node_id in endpoints
        }
        candidates = {node_id: [] for node_id in endpoints}
        for first_index, first_node in enumerate(endpoints):
            if elevations[first_node] is None:
                continue
            first_point = self._raw_graph.nodes[first_node]['point']
            for second_node in endpoints[first_index + 1:]:
                if (
                    component_by_node[first_node]
                    == component_by_node[second_node]
                    or elevations[second_node] is None
                ):
                    continue
                if abs(elevations[first_node] - elevations[second_node]) >= 15:
                    continue
                distance = haversine_distance_m(
                    first_point,
                    self._raw_graph.nodes[second_node]['point'],
                )
                if distance >= 100:
                    continue
                candidates[first_node].append((distance, second_node))
                candidates[second_node].append((distance, first_node))

        for node_candidates in candidates.values():
            node_candidates.sort()

        parent = list(range(len(components)))

        def find(component_index):
            while parent[component_index] != component_index:
                parent[component_index] = parent[parent[component_index]]
                component_index = parent[component_index]
            return component_index

        active_endpoints = set(endpoints)
        while active_endpoints:
            nearest = {}
            for node_id in active_endpoints:
                valid = [
                    candidate
                    for candidate in candidates[node_id]
                    if candidate[1] in active_endpoints
                    and find(component_by_node[candidate[1]])
                    != find(component_by_node[node_id])
                ][:2]
                if valid:
                    nearest[node_id] = (
                        valid[0],
                        len(valid) == 1 or valid[1][0] - valid[0][0] >= 1e-6,
                    )

            mutual = [
                (candidate[0][0], node_id, candidate[0][1])
                for node_id, candidate in nearest.items()
                if candidate[1]
                and nearest.get(candidate[0][1], (None, False))[1]
                and nearest[candidate[0][1]][0][1] == node_id
            ]
            if not mutual:
                return

            distance, first_node, second_node = min(mutual)
            self._raw_graph.add_edge(
                first_node,
                second_node,
                weight=distance,
                points=[
                    self._raw_graph.nodes[first_node]['point'],
                    self._raw_graph.nodes[second_node]['point'],
                ],
            )
            first_component = find(component_by_node[first_node])
            second_component = find(component_by_node[second_node])
            parent[second_component] = first_component
            active_endpoints.remove(first_node)
            active_endpoints.remove(second_node)
