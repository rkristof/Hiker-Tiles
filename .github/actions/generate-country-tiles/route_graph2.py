import math

import networkx as nx


class RouteGraph2:
    """Build a minimal weighted graph from route ways."""

    def __init__(
        self,
        route_relation,
        way_nodes,
        externally_reachable_nodes=None,
        sampler=None,
    ):
        self._route_relation = route_relation
        self._externally_reachable_nodes = set(externally_reachable_nodes or ())
        self._relation_node_order = {}
        raw_graph = nx.MultiGraph()

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
                raw_graph.add_node(node_id, point=point)

            for first_node, second_node in zip(nodes, nodes[1:]):
                first_node_id, first_point = first_node
                second_node_id, second_point = second_node
                if first_node_id == second_node_id:
                    continue
                raw_graph.add_edge(
                    first_node_id,
                    second_node_id,
                    weight=self._haversine_distance_m(first_point, second_point),
                )

        self._repair_disconnected_components(raw_graph, sampler)

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
        if not retained_nodes and raw_graph:
            retained_nodes.add(next(iter(raw_graph)))

        self._graph = nx.Graph()
        self._graph.add_nodes_from(
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

                while current_node not in retained_nodes:
                    _, next_node, next_edge_key, next_edge_data = next(
                        iter(raw_graph.edges(current_node, keys=True, data=True)),
                    )
                    raw_graph.remove_edge(current_node, next_node, next_edge_key)
                    edge_weight += next_edge_data['weight']
                    current_node = next_node

                if self._graph.has_edge(start_node, current_node):
                    self._graph[start_node][current_node]['weight'] += edge_weight
                else:
                    self._graph.add_edge(start_node, current_node, weight=edge_weight)

    @property
    def has_edges(self):
        return self._graph.number_of_edges() > 0

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

    def point(self, node_id):
        return self._graph.nodes[node_id]['point']

    def shortest_traversal(self, start_node, finish_node):
        """Return the weighted shortest traversal as an ordered node path."""
        if start_node not in self._graph or finish_node not in self._graph:
            return None
        try:
            return nx.shortest_path(
                self._graph,
                start_node,
                finish_node,
                weight='weight',
            )
        except nx.NetworkXNoPath:
            return None

    def shortest_traversal_to_nearest_finish(self, start_node):
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
            finish_nodes = [start_node]

        return min(
            (
                traversal
                for finish_node in finish_nodes
                if (traversal := self.shortest_traversal(start_node, finish_node))
            ),
            key=lambda traversal: nx.path_weight(self._graph, traversal, weight='weight'),
            default=None,
        )

    def raw_route_distance_m(self):
        return sum(
            edge_data['weight']
            for _, _, edge_data in self._graph.edges(data=True)
        )

    @classmethod
    def _repair_disconnected_components(cls, graph, sampler):
        components = list(nx.connected_components(graph))
        if sampler is None or len(components) < 2:
            return

        component_by_node = {
            node_id: component_index
            for component_index, component in enumerate(components)
            for node_id in component
        }
        endpoints = [
            node_id
            for node_id in graph
            if graph.degree(node_id) == 1
        ]
        elevations = {
            node_id: sampler.sample(graph.nodes[node_id]['point'])
            for node_id in endpoints
        }
        candidates = {node_id: [] for node_id in endpoints}
        for first_index, first_node in enumerate(endpoints):
            if elevations[first_node] is None:
                continue
            first_point = graph.nodes[first_node]['point']
            for second_node in endpoints[first_index + 1:]:
                if (
                    component_by_node[first_node]
                    == component_by_node[second_node]
                    or elevations[second_node] is None
                ):
                    continue
                if abs(elevations[first_node] - elevations[second_node]) >= 15:
                    continue
                distance = cls._haversine_distance_m(
                    first_point,
                    graph.nodes[second_node]['point'],
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
            graph.add_edge(
                first_node,
                second_node,
                weight=distance,
            )
            first_component = find(component_by_node[first_node])
            second_component = find(component_by_node[second_node])
            parent[second_component] = first_component
            active_endpoints.remove(first_node)
            active_endpoints.remove(second_node)

    @staticmethod
    def _haversine_distance_m(first_point, second_point):
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