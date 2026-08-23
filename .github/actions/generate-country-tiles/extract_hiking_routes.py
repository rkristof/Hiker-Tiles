import json
import os

import osmium

from elevation import Elevation
from route_graph import LandmarkIndex, RouteGraph, landmark_candidate, polyline_distance_m


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
        self.relations = {}   # relation_id -> {route attributes, way_ids}

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
        }
        node_roles = {}
        for member in relation.members:
            if member.type == 'n' and member.role in ('start', 'end'):
                node_roles.setdefault(member.role, []).append(member.ref)

        self.relations[relation.id] = {
            **route_attributes,
            'way_ids': [member.ref for member in relation.members if member.type == 'w'],
            'node_roles': node_roles,
            'from': tags.get('from', '').strip(),
            'to': tags.get('to', '').strip(),
            'roundtrip': tags.get('roundtrip', '').strip().lower() == 'yes',
            'needs_landmark_start': not bool(node_roles.get('start')),
        }
        for member in relation.members:
            if member.type == 'w':
                self.way_routes.setdefault(member.ref, []).append(route_attributes)

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


class GeoJSONExporter(osmium.SimpleHandler):
    def __init__(self, way_routes, points_file, collect_landmarks=False):
        super().__init__()
        self.way_routes = way_routes
        self.points_file = points_file
        self.collect_landmarks = collect_landmarks
        self.way_count = 0
        self.point_count = 0
        self.route_symbols = set()  # unique route symbol values seen
        self.symbol_groups_buffer = []  # buffered (coordinates, route_properties, symbol_entries)
        self.way_groups_buffer = []  # buffered (coordinates, route_properties), merged after the pass
        self.way_nodes = {}  # way_id -> [(node_id, coordinates)] for route traversal
        self.node_way_ids = {}  # node_id -> route way IDs, shared across relations
        self.node_coordinates = {}  # node_id -> coordinates for relation endpoint markers
        self.landmarks = []  # candidate landmarks used only by relations without starts
        self.landmark_index = None

    def node(self, node):
        try:
            self.node_coordinates[node.id] = [node.lon, node.lat]
        except osmium.InvalidLocationError:
            return
        tags = {tag.k: tag.v for tag in node.tags}
        if self.collect_landmarks:
            landmark = landmark_candidate(tags)
            if landmark is not None:
                landmark['node_id'] = node.id
                landmark['points'] = [[node.lon, node.lat]]
                self.landmarks.append(landmark)
        point_properties = self._point_properties(tags)
        if point_properties is None:
            return
        write_feature(
            self.points_file,
            {'type': 'Point', 'coordinates': [node.location.lon, node.location.lat]},
            point_properties,
        )
        self.point_count += 1

    def way(self, way):
        route_attributes = self.way_routes.get(way.id)
        landmark = None
        if self.collect_landmarks:
            landmark = landmark_candidate({tag.k: tag.v for tag in way.tags})
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
        for node_id, _ in nodes:
            self.node_way_ids.setdefault(node_id, set()).add(way.id)
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
    with open('natural-points.geojsonseq', 'w') as points_file:
        collect_landmarks = any(
            route_relation['needs_landmark_start']
            for route_relation in collector.relations.values()
        )
        exporter = GeoJSONExporter(collector.way_routes,points_file,collect_landmarks)
        exporter.apply_file('tiles-filtered.osm.pbf', locations=True)
        exporter.landmark_index = LandmarkIndex(exporter.landmarks) if collect_landmarks else None
        print(f'Route ways matched: {exporter.way_count}')
        print(f'Natural points written: {exporter.point_count}')
    write_route_layers(exporter)
    return exporter


def write_route_lines(collector, exporter):
    route_metadata = []
    route_feature_count = 0
    elevation = Elevation(os.environ['ELEVATION_DIRECTORY'])
    try:
        with open('hiking-routes-interaction.geojsonseq', 'w') as route_lines_file:
            for relation_id, route_relation in collector.relations.items():
                route_graph = RouteGraph(route_relation, exporter.way_nodes, exporter.node_way_ids)
                if not route_graph.has_edges:
                    continue

                route_graph.repair_disconnected_components(elevation)
                landmark_index = exporter.landmark_index if route_relation['needs_landmark_start'] else None
                start_node = route_graph.resolve_start(exporter.node_coordinates, elevation, landmark_index)
                finish_node = route_graph.resolve_finish(start_node, exporter.node_coordinates, elevation)
                explicit_finish = bool(route_relation.get('node_roles', {}).get('end'))
                lower_bound_distance_m = route_graph.required_distance_m()

                if route_graph.component_count > 1:
                    continue
                if start_node is None:
                    continue
                if explicit_finish and finish_node is None:
                    continue
                if lower_bound_distance_m >= Elevation.PROFILE_MAX_DISTANCE_M:
                    continue

                traversal = route_graph.shortest_traversal(start_node, finish_node)
                if traversal is None:
                    continue

                steps, finish_node = traversal
                path, _ = route_graph.traversal_coordinates(start_node, steps)
                distance_m = route_distance_m(path)
                elevation_data = elevation.route_elevation(
                    path,
                    include_profile=distance_m < Elevation.PROFILE_MAX_DISTANCE_M,
                )
                elevation_gain_m = elevation_data['elevation_gain_m']
                elevation_loss_m = elevation_data['elevation_loss_m']
                elevation_profile = elevation_data['profile']
                duration_min = elevation.route_duration_min(
                    distance_m,
                    elevation_gain_m,
                    elevation_loss_m,
                )
                route_metadata.append({
                    'id': relation_id,
                    'name': route_relation['name'],
                    'name_int': route_relation.get('name_int', ''),
                    'symbol': route_relation['symbol'],
                    'network': route_relation['network'],
                    'type': route_relation['type'],
                    'min_lon': min(point[0] for point in path),
                    'min_lat': min(point[1] for point in path),
                    'max_lon': max(point[0] for point in path),
                    'max_lat': max(point[1] for point in path),
                    'distance_m': distance_m,
                    'elevation_gain_m': elevation_gain_m,
                    'elevation_loss_m': elevation_loss_m,
                    'elevation_profile': elevation_profile,
                    'duration_min': duration_min,
                    'start_lon': route_graph.point(start_node)[0],
                    'start_lat': route_graph.point(start_node)[1],
                    'finish_lon': route_graph.point(finish_node)[0],
                    'finish_lat': route_graph.point(finish_node)[1],
                })
                route_properties = {
                    'relation_id': relation_id,
                    'name': route_relation['name'],
                    'symbol': route_relation['symbol'],
                    'network': route_relation['network'],
                    'route_type': route_relation['type'],
                }
                write_feature(
                    route_lines_file,
                    {'type': 'LineString', 'coordinates': path},
                    route_properties,
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
    print(f'Route member ways collected: {len(collector.way_routes)}')

    exporter = export_route_features(collector)
    write_route_lines(collector, exporter)
    write_symbol_catalog(exporter)


if __name__ == '__main__':
    main()
