import json
import os

import osmium

from eligible_nodes import (
    EligibleNodeFinder,
    LandmarkIndex,
    SettlementIndex,
    landmark_candidate,
)
from elevation import Elevation, offset_elevation_profile
from route_graph import RouteGraph
from utils import haversine_distance_m, polyline_distance_m


NETWORK_GROUP_BY_TAG = {
    'iwn': 'iwn', 'icn': 'iwn', 'international': 'iwn', 'int': 'iwn',
    'nwn': 'nwn', 'ncn': 'nwn', 'national': 'nwn', 'nat': 'nwn',
    'rwn': 'rwn', 'rcn': 'rwn', 'regional': 'rwn', 'reg': 'rwn',
    'lwn': 'lwn', 'lcn': 'lwn', 'local': 'lwn', 'loc': 'lwn',
}
NETWORK_RANK = {
    'iwn': 0,
    'nwn': 1,
    'rwn': 2,
    'lwn': 3,
}
ROUTE_TYPES = ('hiking', 'foot', 'mtb')
PROPOSED_STATES = frozenset(('planned', 'proposed'))
INACTIVE_STATES = frozenset((
    'abandoned',
    'cancelled',
    'demolished',
    'destroyed',
    'disused',
    'inactive',
    'obsolete',
    'razed',
    'removed',
))
INACTIVE_ROUTE_TAGS = (
    'abandoned:route',
    'demolished:route',
    'destroyed:route',
    'disused:route',
    'razed:route',
)
ROUTE_MINZOOM = {
    'iwn': 5,
    'nwn': 5,
    'rwn': 8,
    'lwn': 10,
}
SYMBOL_MINZOOM = {
    'iwn': 7,
    'nwn': 9,
    'rwn': 11,
    'lwn': 13,
}
MAX_TRAVERSAL_DISTANCE_M = 40_000


def write_feature(output, geometry, properties, minzoom=None):
    """Write one GeoJSON feature in sequence format, with optional zoom metadata."""
    feature = {'type': 'Feature'}
    if minzoom is not None:
        feature['tippecanoe'] = {'minzoom': minzoom, 'maxzoom': 13}
    feature['geometry'] = geometry
    feature['properties'] = properties
    output.write(json.dumps(feature) + '\n')


def build_route_properties(route_type, name, network, difficulty=0):
    route_properties = {'type': route_type, 'name': name, 'network': network}
    if difficulty:
        route_properties['difficulty'] = difficulty
    return route_properties


class WayRouteCollector(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.way_routes = {}  # way_id -> route attributes from each relation
        self.relations = {}   # relation_id -> route attributes and members

    def relation(self, relation):
        tags = {tag.k: tag.v for tag in relation.tags}
        if tags.get('type') != 'route':
            return
        route_type = tags.get('route', '')
        if route_type not in ROUTE_TYPES:
            return
        if self.is_lifecycle_excluded(tags):
            return
        # Keep relation attributes on each member way; way export later combines
        # attributes when a way belongs to multiple route relations.
        route_attributes = {
            'type': route_type,
            'name': self._make_route_title(tags, relation.id),
            'name_int': tags.get('name:en', '') or tags.get('int_name', ''),
            'network': self._network_group(tags.get('network', '')),
            'symbol': self._route_symbol(tags, os.environ['SYMBOL_TAG']),
            'difficulty': self._route_difficulty(os.environ['COUNTRY'], tags),
            'roundtrip': tags.get('roundtrip', '').strip().lower() in ('1', 'true', 'yes'),
        }
        node_roles = {}
        for member in relation.members:
            if member.type == 'n' and member.role in ('start', 'end'):
                node_roles.setdefault(member.role, {})[member.ref] = None

        self.relations[relation.id] = {
            **route_attributes,
            'way_ids': [member.ref for member in relation.members if member.type == 'w'],
            'node_ids': [member.ref for member in relation.members if member.type == 'n'],
            'route_members': [
                (member.type, member.ref)
                for member in relation.members
                if member.type in ('w', 'r')
            ],
            'node_roles': node_roles,
        }
        for member in relation.members:
            if member.type == 'w':
                self.way_routes.setdefault(member.ref, []).append(route_attributes)

    def flatten_nested_routes(self):
        """Expand supported child routes into parent relations and suppress children."""
        source_relations = self.relations
        expanded_way_ids = {}
        absorbed_relation_ids = set()

        def expand(relation_id, visiting):
            if relation_id in expanded_way_ids:
                return expanded_way_ids[relation_id]
            if relation_id in visiting:
                return []

            relation = source_relations[relation_id]
            way_ids = []
            next_visiting = visiting | {relation_id}
            for member_type, member_id in relation['route_members']:
                if member_type == 'w':
                    member_way_ids = (member_id,)
                elif member_id in source_relations and member_id not in next_visiting:
                    member_way_ids = expand(member_id, next_visiting)
                    absorbed_relation_ids.add(member_id)
                else:
                    member_way_ids = ()

                for way_id in member_way_ids:
                    way_ids.append(way_id)

            expanded_way_ids[relation_id] = way_ids
            return way_ids

        for relation_id in source_relations:
            expand(relation_id, set())

        self.relations = {
            relation_id: {
                **source_relations[relation_id],
                'way_ids': expanded_way_ids[relation_id],
                'route_members': [('w', way_id) for way_id in expanded_way_ids[relation_id]],
            }
            for relation_id in source_relations
            if relation_id not in absorbed_relation_ids
        }

    @staticmethod
    def _make_itinerary(tags):
        itinerary = []
        from_name = tags.get('from')
        if from_name is not None:
            itinerary.append(from_name)

        via = tags.get('via')
        if via is not None:
            separator = ';' if ';' in via else ' - ' if ' - ' in via else ','
            itinerary.extend(value.strip() for value in via.split(separator))

        to_name = tags.get('to')
        if to_name is not None:
            itinerary.append(to_name)
        return itinerary

    @classmethod
    def _make_route_title(cls, tags, relation_id):
        """Choose the most useful route title from increasingly weak OSM tags."""
        name = tags.get('name', '')
        if name:
            return name

        itinerary = cls._make_itinerary(tags)
        if itinerary:
            return ' - '.join(itinerary)

        symbol_description = tags.get('symbol', '')
        if symbol_description:
            return symbol_description

        ref = tags.get('ref', '')
        if ref:
            return f'[{ref}]'

        return f'({relation_id})'

    @staticmethod
    def _difficulty_from_symbol(symbol):
        for marker, difficulty in (('red:white:red_bar', 2), ('blue:white:blue_bar', 3)):
            if symbol.startswith(marker):
                return difficulty
        return None

    @classmethod
    def _route_difficulty(cls, country, tags):
        """Normalize country-specific difficulty tags; standard routes have no class."""
        if tags.get('route') not in ('hiking', 'foot'):
            return None

        network = tags.get('network', '').strip().lower()
        symbol = tags.get('osmc:symbol', '').strip().lower()

        if country == 'switzerland' and network == 'lwn':
            return cls._difficulty_from_symbol(symbol)

        if country == 'austria':
            if (
                tags.get('operator', '').strip().lower() == 'land vorarlberg'
                and network == 'rwn'
                and tags.get('network:type', '').strip().lower() == 'basic_network'
            ):
                return cls._difficulty_from_symbol(symbol)

        if country == 'italy' and network == 'lwn' and symbol.startswith('red:'):
            return {'E': 1, 'EE': 2, 'EEA*': 3}.get(tags.get('cai_scale', '').strip().upper())

        return None

    @staticmethod
    def _network_group(network):
        """Return highest-priority network represented by semicolon-separated tags."""
        group = 'lwn'
        for network_tag in network.split(';'):
            group_candidate = NETWORK_GROUP_BY_TAG.get(network_tag.strip().lower())
            if group_candidate and NETWORK_RANK[group_candidate] < NETWORK_RANK[group]:
                group = group_candidate
        return group

    @staticmethod
    def _route_symbol(tags, symbol_tag):
        if symbol_tag != 'kct':
            return tags.get(symbol_tag) or tags.get('osmc:symbol', '')

        kct_key = next((key for key in tags if key.startswith('kct_')), None)
        if kct_key is not None:
            return f'kct_{kct_key[4:]}:{tags[kct_key]}'

        if tags.get('operator', '').lower() == 'kst':
            colour = tags.get('colour', '')
            symbol = tags.get('symbol', '')
            if colour and symbol:
                return f'kct_{colour}:{symbol}'

        return tags.get('osmc:symbol', '')

    @staticmethod
    def is_lifecycle_excluded(tags):
        """Return whether lifecycle tags exclude a route relation."""
        state = tags.get('state', '').strip().lower()
        if state in PROPOSED_STATES:
            return True
        if state in INACTIVE_STATES:
            return True

        for status_key in ('status', 'route:status'):
            status = tags.get(status_key, '').strip().lower()
            if status in PROPOSED_STATES or status in INACTIVE_STATES:
                return True

        proposed_type = tags.get('proposed:type', '').strip().lower()
        if proposed_type == 'route':
            return True

        for tag_key in INACTIVE_ROUTE_TAGS:
            if tags.get(tag_key, '').strip():
                return True

        return False


class ConnectingHighwayCollector(osmium.SimpleHandler):
    def __init__(self, route_node_ids):
        super().__init__()
        self._route_node_ids = set(route_node_ids)
        self._direct_way_ids = set()
        self._direct_node_ids = set()
        self._highways_by_id = {}
        self._collecting_geometry = False

    def way(self, way):
        highway_type = way.tags.get('highway')
        if highway_type is None:
            return

        try:
            osmium_nodes = tuple(way.nodes)
            node_ids = tuple(node.ref for node in osmium_nodes)
            if not self._collecting_geometry:
                if not any(node_id in self._route_node_ids for node_id in node_ids):
                    return
                self._direct_way_ids.add(way.id)
                self._direct_node_ids.update(node_ids)
                return
            if (
                way.id not in self._direct_way_ids
                and not any(node_id in self._direct_node_ids for node_id in node_ids)
            ):
                return
            nodes = [(node.ref, [node.lon, node.lat]) for node in osmium_nodes]
        except osmium.InvalidLocationError:
            return

        self._highways_by_id[way.id] = {'highway_type': highway_type, 'nodes': nodes}

    def collect_highways(self, filename):
        self.apply_file(filename, locations=False)
        self._collecting_geometry = True
        try:
            self.apply_file(filename, locations=True)
        finally:
            self._collecting_geometry = False

    def connecting_highways_by_node(self):
        highways_by_node = {}
        for way_id in self._direct_way_ids:
            highway = self._highways_by_id[way_id]
            for node_id, _ in highway['nodes']:
                if node_id in self._route_node_ids:
                    highways_by_node.setdefault(node_id, {})[way_id] = highway
        return highways_by_node

    def highway_index(self):
        highways_by_node = {}
        for way_id, highway in self._highways_by_id.items():
            nodes = highway['nodes']
            for node_index, (node_id, _) in enumerate(nodes):
                highways_by_node.setdefault(node_id, []).append(
                    (way_id, node_index),
                )
        return HighwayIndex(
            {
                node_id: tuple(highways)
                for node_id, highways in highways_by_node.items()
            },
            self._highways_by_id,
        )


class HighwayIndex(dict):
    def __init__(self, highways_by_node, highways_by_id):
        super().__init__(highways_by_node)
        self._highways_by_id = highways_by_id
        self._segment_distance_cache = {}

    def highway(self, way_id):
        return self._highways_by_id[way_id]

    def segment_distance(self, way_id, segment_index):
        cache_key = (way_id, segment_index)
        if cache_key not in self._segment_distance_cache:
            highway = self.highway(way_id)
            first_point = highway['nodes'][segment_index][1]
            second_point = highway['nodes'][segment_index + 1][1]
            self._segment_distance_cache[cache_key] = haversine_distance_m(
                first_point,
                second_point,
            )
        return self._segment_distance_cache[cache_key]


class GeoJSONExporter(osmium.SimpleHandler):
    def __init__(
        self,
        way_routes,
        points_file,
        collect_landmarks=False,
        endpoint_maps_by_node=None,
    ):
        super().__init__()
        self.way_routes = way_routes
        self.points_file = points_file
        self.collect_landmarks = collect_landmarks
        self.endpoint_maps_by_node = endpoint_maps_by_node or {}
        self.way_count = 0
        self.point_count = 0
        self.route_symbols = set()  # unique route symbol values seen
        self.symbol_groups_buffer = []  # buffered (coordinates, route_properties, symbol_entries)
        self.way_groups_buffer = []  # buffered (coordinates, route_properties), merged after the pass
        self.way_nodes = {}  # way_id -> [(node_id, coordinates)] for route traversal
        self.way_segment_distances = {}
        self.landmarks = []  # candidate landmarks used only by relations without starts
        self.settlements = []  # settlement points used only by relations without starts
        self.landmark_index = None
        self.settlement_index = None

    def node(self, node):
        tags = node.tags
        landmark = landmark_candidate(tags) if self.collect_landmarks else None
        settlement = tags.get('place') if self.collect_landmarks else None
        point_properties = self._point_properties(tags)
        if (
            node.id not in self.endpoint_maps_by_node
            and landmark is None
            and settlement is None
            and point_properties is None
        ):
            return
        try:
            point = [node.lon, node.lat]
        except osmium.InvalidLocationError:
            return
        for endpoint_nodes in self.endpoint_maps_by_node.get(node.id, ()):
            endpoint_nodes[node.id] = point
        if landmark is not None:
            landmark['node_id'] = node.id
            landmark['points'] = [point]
            self.landmarks.append(landmark)
        if settlement is not None:
            self.settlements.append({'place': settlement, 'points': [point]})
        if point_properties is None:
            return
        write_feature(
            self.points_file,
            {'type': 'Point', 'coordinates': point},
            point_properties,
        )
        self.point_count += 1

    def way(self, way):
        route_attributes = self.way_routes.get(way.id)
        tags = way.tags
        landmark = None
        if self.collect_landmarks:
            landmark = landmark_candidate(tags)
        if not route_attributes and landmark is None:
            return
        try:
            nodes = [(node.ref, [node.lon, node.lat]) for node in way.nodes]
        except osmium.InvalidLocationError:
            return
        if len(nodes) < 2:
            return
        if landmark is not None:
            landmark['way_id'] = way.id
            landmark['points'] = [point for _, point in nodes]
            self.landmarks.append(landmark)
        if not route_attributes:
            return
        coordinates = [point for _, point in nodes]
        self.way_nodes[way.id] = nodes
        self.way_segment_distances[way.id] = segment_distances = [
            haversine_distance_m(first_point, second_point)
            for (_, first_point), (_, second_point) in zip(nodes, nodes[1:])
        ]
        # A way may belong to several relations. Use the highest-ranked network
        # for shared line properties and retain the most demanding difficulty.
        primary_route = min(route_attributes, key=lambda attributes: NETWORK_RANK[attributes['network']])
        difficulty = max((attributes['difficulty'] or 0 for attributes in route_attributes), default=0)
        symbol_entries = list(dict.fromkeys(
            (attributes['symbol'], attributes['network'])
            for attributes in route_attributes
            if attributes['symbol']
        ))
        # Symbol entries stay in network order because their index becomes the
        # rendered symbol slot in the separate symbol layer.
        symbol_entries.sort(key=lambda entry: NETWORK_RANK[entry[1]])
        route_properties = build_route_properties(primary_route['type'], primary_route['name'], primary_route['network'], difficulty)
        self.way_groups_buffer.append((coordinates, route_properties))
        if symbol_entries:
            self.route_symbols.update(symbol for symbol, _ in symbol_entries)
            self.symbol_groups_buffer.append((coordinates, route_properties, symbol_entries))
        self.way_count += 1

    @staticmethod
    def _point_properties(tags):
        natural = tags.get('natural', '')
        if natural in ('peak', 'cave_entrance', 'volcano'):
            name = tags.get('name', '')
            elevation = tags.get('ele')
            if name != '' and elevation:
                name = f'{name}\n{elevation} m'
            return {'type': natural, 'name': name}

        if tags.get('checkpoint', '') == 'hiking' and tags.get('checkpoint:type', '') == 'stamp':
            name = tags.get('name', '')
            if tags.get('ref'):
                name = f'{name} ({tags["ref"]})'
            return {'type': 'stamp', 'name': name}

        return None


def chain_lines(lines):
    """Chain segments through unique unused endpoint matches."""
    if not lines:
        return []

    def extend_chain(chain, at_start):
        while True:
            endpoint = chain[0] if at_start else chain[-1]
            next_segments = [
                (segment_index, edge)
                for segment_index, edge in adjacency.get(tuple(endpoint), [])
                if not used[segment_index]
            ]
            if len(next_segments) != 1:
                return

            segment_index, edge = next_segments[0]
            used[segment_index] = True
            line = lines[segment_index]
            if at_start:
                chain[0:0] = line[:-1] if edge == 'end' else list(reversed(line[1:]))
            else:
                chain.extend(line[1:] if edge == 'start' else list(reversed(line[:-1])))

    line_count = len(lines)
    used = [False] * line_count
    adjacency = {}
    # Ambiguous junctions and closed loops stop extension instead of guessing
    # an order that could change route geometry.
    for segment_index, line in enumerate(lines):
        adjacency.setdefault(tuple(line[0]), []).append((segment_index, 'start'))
        adjacency.setdefault(tuple(line[-1]), []).append((segment_index, 'end'))
    chains = []
    for start_index in range(line_count):
        if used[start_index]:
            continue
        used[start_index] = True
        chain = list(lines[start_index])
        extend_chain(chain, at_start=False)
        extend_chain(chain, at_start=True)
        if len(chain) >= 2:
            chains.append(chain)
    return chains


def route_distance_m(points):
    return round(polyline_distance_m(points))


def merge_way_groups(way_groups):
    groups_by_key = {}
    for coordinates, route_properties in way_groups:
        group_key = (route_properties['type'], route_properties['name'], route_properties['network'], route_properties.get('difficulty', 0))
        groups_by_key.setdefault(group_key, []).append(coordinates)

    merged_groups = []
    for (route_type, name, network, difficulty), line_segments in groups_by_key.items():
        route_properties = build_route_properties(route_type, name, network, difficulty)
        for chain in chain_lines(line_segments):
            merged_groups.append((chain, route_properties))
    return merged_groups


def merge_symbol_groups(symbol_groups):
    groups_by_key = {}
    for coordinates, route_properties, symbol_entries in symbol_groups:
        group_key = (route_properties['type'], route_properties['name'], route_properties['network'], tuple(symbol_entries))
        groups_by_key.setdefault(group_key, []).append(coordinates)

    merged_groups = []
    for (route_type, name, network, symbol_entries_tuple), line_segments in groups_by_key.items():
        route_properties = build_route_properties(route_type, name, network)
        symbol_entries = list(symbol_entries_tuple)
        for chain in chain_lines(line_segments):
            merged_groups.append((chain, route_properties, symbol_entries))
    return merged_groups


def write_route_layers(exporter):
    """Merge way segments and write route and symbol GeoJSON sequence layers."""
    with open('hiking-routes.geojsonseq', 'w') as route_file, \
        open('hiking-symbols.geojsonseq', 'w') as symbol_file:
        merged_route_groups = merge_way_groups(exporter.way_groups_buffer)
        print(f'Way segments merged: {len(exporter.way_groups_buffer)} -> {len(merged_route_groups)} chains')
        for coordinates, route_properties in merged_route_groups:
            write_feature(
                route_file,
                {'type': 'LineString', 'coordinates': coordinates},
                route_properties,
                ROUTE_MINZOOM[route_properties['network']],
            )

        symbol_groups_before = len(exporter.symbol_groups_buffer)
        symbol_groups = merge_symbol_groups(exporter.symbol_groups_buffer)
        print(f'Symbol way-segments merged: {symbol_groups_before} -> {len(symbol_groups)} chains')

        symbol_feature_count = 0
        for coordinates, route_properties, symbol_entries in symbol_groups:
            for slot, (symbol, network) in enumerate(symbol_entries[:9]):
                write_feature(
                    symbol_file,
                    {'type': 'LineString', 'coordinates': coordinates},
                    {**route_properties, 'network': network, 'symbol': symbol, 'slot': slot},
                    SYMBOL_MINZOOM[network],
                )
                symbol_feature_count += 1
        print(f'Symbol lines written: {symbol_feature_count} (from {len(symbol_groups)} merged chains)')


def export_route_features(collector):
    def snap_relation_endpoints(exporter):
        for route_relation in collector.relations.values():
            route_nodes = {
                node_id: point
                for way_id in route_relation['way_ids']
                for node_id, point in exporter.way_nodes.get(way_id, ())
            }
            for role in ('start', 'end'):
                endpoint_nodes = route_relation.get('node_roles', {}).get(role, ())
                if not endpoint_nodes:
                    continue
                route_relation['node_roles'][role] = [
                    node_id
                    if node_id in route_nodes
                    else min(
                        route_nodes,
                        key=lambda candidate: haversine_distance_m(
                            endpoint_nodes[node_id],
                            route_nodes[candidate],
                        ),
                    )
                    for node_id in endpoint_nodes
                    if node_id in route_nodes or (
                        endpoint_nodes[node_id] is not None and route_nodes
                    )
                ]

    with open('natural-points.geojsonseq', 'w') as points_file:
        collect_landmarks = any(
            not route_relation.get('node_roles', {}).get('start')
            for route_relation in collector.relations.values()
        )
        endpoint_maps_by_node = {}
        for route_relation in collector.relations.values():
            for role in ('start', 'end'):
                endpoint_nodes = route_relation.get('node_roles', {}).get(role, ())
                for node_id in endpoint_nodes:
                    endpoint_maps_by_node.setdefault(node_id, []).append(endpoint_nodes)
        exporter = GeoJSONExporter(
            collector.way_routes,
            points_file,
            collect_landmarks=collect_landmarks,
            endpoint_maps_by_node=endpoint_maps_by_node,
        )
        exporter.apply_file('tiles-filtered.osm.pbf', locations=True)
        for relation in collector.relations.values():
            relation['raw_distance_m'] = sum(
                sum(exporter.way_segment_distances.get(way_id, ()))
                for way_id in relation['way_ids']
            )
        exporter.landmark_index = LandmarkIndex(exporter.landmarks)
        exporter.settlement_index = SettlementIndex(exporter.settlements)
        snap_relation_endpoints(exporter)
        route_node_ids = {
            node_id
            for relation in collector.relations.values()
            if relation['raw_distance_m'] < MAX_TRAVERSAL_DISTANCE_M
            for way_id in relation['way_ids']
            for node_id, _ in exporter.way_nodes.get(way_id, ())
        }
        collector = ConnectingHighwayCollector(route_node_ids)
        collector.collect_highways('highways-filtered.osm.pbf')
        exporter.connecting_highways_by_node = collector.connecting_highways_by_node()
        exporter.highway_index = collector.highway_index()
        print(f'Route ways matched: {exporter.way_count}')
        print(f'Natural points written: {exporter.point_count}')
    write_route_layers(exporter)
    return exporter


def write_route_lines(collector, exporter):
    def component_traversal(
        route_graph,
        start_node=None,
        finish_node=None,
        inferred_start_node=None,
    ):
        traversal = None
        if start_node is not None and finish_node is not None:
            if start_node == finish_node:
                traversal = route_graph.eulerian_traversal(start_node)
            else:
                traversal = route_graph.shortest_complete_traversal(start_node, finish_node)
        elif start_node is not None:
            if route_graph._roundtrip:
                traversal = route_graph.eulerian_traversal(start_node)
            else:
                traversal = route_graph.shortest_complete_traversal_to_nearest_finish(start_node)
                if not route_graph.is_eulerian():
                    route_graph._graph = route_graph._eulerize_graph(route_graph._graph)
                roundtrip_traversal = route_graph.eulerian_traversal(start_node)
                if (
                    traversal is None
                    or route_graph.traversal_distance(roundtrip_traversal)
                    < route_graph.traversal_distance(traversal)
                ):
                    traversal = roundtrip_traversal
        elif (line_endpoints := route_graph.simple_line_endpoints()) is not None:
            start_node, finish_node = line_endpoints
            traversal = route_graph.shortest_complete_traversal(start_node, finish_node)
        else:
            if route_graph.is_eulerian():
                traversal = route_graph.eulerian_traversal(inferred_start_node)
            else:
                traversal = route_graph.shortest_complete_traversal(
                    inferred_start_node,
                    finish_node,
                )

        return traversal

    route_metadata = []
    route_feature_count = 0
    elevation = Elevation(os.environ['ELEVATION_DIRECTORY'])
    try:
        with open('hiking-routes-interaction.geojsonseq', 'w') as route_lines_file:
            for relation_id, route_relation in collector.relations.items():
                node_roles = route_relation.get('node_roles', {})
                start_node = next(iter(node_roles.get('start', ())), None)
                finish_node = next(iter(node_roles.get('end', ())), None)
                roundtrip = route_relation.get('roundtrip', False)
                
                original_lines = [
                    [point for _, point in exporter.way_nodes.get(way_id, ())]
                    for way_id in route_relation['way_ids']
                ]

                distance_m = route_relation['raw_distance_m']
                is_short_route = distance_m < MAX_TRAVERSAL_DISTANCE_M
                if is_short_route:
                    inferred_start_node = None
                    if start_node is None:
                        eligible_node_finder = EligibleNodeFinder(
                            route_relation,
                            exporter.way_nodes,
                            exporter.connecting_highways_by_node,
                            {
                                node_id
                                for way_id in route_relation['way_ids']
                                for node_id, _ in exporter.way_nodes.get(way_id, ())
                            },
                            exporter.landmark_index,
                            exporter.settlement_index,
                        )
                        inferred_start_node = next(
                            iter(eligible_node_finder.rank_eligible_nodes()),
                            None,
                        )
                    route_graph = RouteGraph(
                        route_relation,
                        exporter.way_nodes,
                        way_segment_distances=exporter.way_segment_distances,
                        connecting_highways_by_node=exporter.connecting_highways_by_node,
                        highway_index=exporter.highway_index,
                        sampler=elevation,
                        roundtrip=roundtrip,
                        inferred_start_node=inferred_start_node,
                    )
                    if not route_graph.has_edges:
                        continue
                    component_results = []
                    for component_graph in route_graph.component_graphs():
                        component_start_node = (
                            start_node
                            if start_node in component_graph._graph
                            else None
                        )
                        component_finish_node = (
                            finish_node
                            if finish_node in component_graph._graph
                            else None
                        )
                        component_inferred_start_node = (
                            inferred_start_node
                            if inferred_start_node in component_graph._graph
                            else None
                        )
                        traversal = component_traversal(
                            component_graph,
                            component_start_node,
                            component_finish_node,
                            component_inferred_start_node,
                        )
                        if traversal is None:
                            component_results = None
                            break
                        path = component_graph.traversal_coordinates(traversal)
                        if len(path) < 2:
                            component_results = None
                            break
                        component_results.append({
                            'start_point': component_graph._graph.nodes[traversal[0]]['point'],
                            'finish_point': component_graph._graph.nodes[traversal[-1]]['point'],
                            'path': path,
                            'distance_m': route_distance_m(path),
                        })

                    if component_results is None:
                        lines = original_lines
                        endpoints = None
                    else:
                        lines = [result['path'] for result in component_results]
                        endpoints = (
                            component_results[0]
                            if len(component_results) == 1
                            else None
                        )
                else:
                    lines = original_lines
                    endpoints = None

                lines = [line for line in lines if len(line) >= 2]

                if not lines:
                    continue

                elevation_gain_m = None
                elevation_loss_m = None
                elevation_profile = {'segments': []}
                duration_min = None

                if is_short_route:
                    distance_m = sum(route_distance_m(line) for line in lines)
                    elevation_gain_m = 0
                    elevation_loss_m = 0
                    component_durations = []
                    distance_offset_m = 0
                    for line in lines:
                        line_distance_m = route_distance_m(line)
                        elevation_data = elevation.route_elevation(line)
                        elevation_gain_m += elevation_data['elevation_gain_m']
                        elevation_loss_m += elevation_data['elevation_loss_m']
                        elevation_profile['segments'].extend(
                            offset_elevation_profile(
                                elevation_data['profile'],
                                distance_offset_m,
                            )['segments']
                        )
                        component_durations.append(elevation.route_duration_min(
                            line_distance_m,
                            elevation_data['elevation_gain_m'],
                            elevation_data['elevation_loss_m'],
                        ))
                        distance_offset_m += line_distance_m
                    duration_min = sum(component_durations)

                all_points = [point for line in lines for point in line]
                route_metadata.append({
                    'id': relation_id,
                    'name': route_relation['name'],
                    'name_int': route_relation.get('name_int', ''),
                    'symbol': route_relation['symbol'],
                    'network': route_relation['network'],
                    'type': route_relation['type'],
                    'min_lon': min(point[0] for point in all_points),
                    'min_lat': min(point[1] for point in all_points),
                    'max_lon': max(point[0] for point in all_points),
                    'max_lat': max(point[1] for point in all_points),
                    'distance_m': distance_m,
                    'elevation_gain_m': elevation_gain_m,
                    'elevation_loss_m': elevation_loss_m,
                    'elevation_profile': elevation_profile,
                    'duration_min': duration_min,
                    'start_lon': endpoints['start_point'][0] if endpoints else None,
                    'start_lat': endpoints['start_point'][1] if endpoints else None,
                    'finish_lon': endpoints['finish_point'][0] if endpoints else None,
                    'finish_lat': endpoints['finish_point'][1] if endpoints else None,
                })

                write_feature(
                    route_lines_file,
                    {
                        'type': 'LineString' if len(lines) == 1 else 'MultiLineString',
                        'coordinates': lines[0] if len(lines) == 1 else lines,
                    },
                    {
                        'relation_id': relation_id,
                        'name': route_relation['name'],
                        'symbol': route_relation['symbol'],
                        'network': route_relation['network'],
                        'route_type': route_relation['type'],
                    },
                )
                route_feature_count += 1
    finally:
        elevation.close()

    with open('routes-meta.json', 'w') as metadata_file:
        json.dump(route_metadata, metadata_file)
    print(f'Route lines: {len(route_metadata)} routes, {route_feature_count} route features')

def write_symbol_catalog(exporter):
    route_symbols = sorted(exporter.route_symbols)
    with open('osmc-symbols.txt', 'w') as symbols_file:
        symbols_file.write('\n'.join(route_symbols) + '\n' if route_symbols else '')
    print(f'Unique route symbol values: {len(route_symbols)}')


def main():
    collector = WayRouteCollector()
    collector.apply_file('tiles-filtered.osm.pbf')
    collector.flatten_nested_routes()
    print(f'Route member ways collected: {len(collector.way_routes)}')

    exporter = export_route_features(collector)
    write_route_lines(collector, exporter)
    write_symbol_catalog(exporter)


if __name__ == '__main__':
    main()
