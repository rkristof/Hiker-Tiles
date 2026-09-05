import mmap
import math
import os
import struct
import subprocess
import tempfile
from bisect import bisect_left

import numpy

from utils import haversine_distance_m


class NativeNeighborView:
    def __init__(self, nodes, edges, start, end, source_index):
        self._nodes = nodes
        self._edges = edges
        self._start = start
        self._end = end
        self._source_index = source_index

    def __len__(self):
        return self._end - self._start

    def __iter__(self):
        source = self._nodes[self._source_index]
        source_point = [float(source['longitude']), float(source['latitude'])]
        for edge in self._edges[self._start:self._end]:
            target = self._nodes[int(edge['target_node_index'])]
            yield (
                int(edge['way_id']),
                int(target['node_id']),
                source_point,
                [float(target['longitude']), float(target['latitude'])],
                float(edge['distance_m']),
            )


class NativeHighwayIndex:
    _node_dtype = numpy.dtype([
        ('node_id', '<i8'),
        ('longitude', '<f8'),
        ('latitude', '<f8'),
        ('edge_offset', '<u8'),
        ('edge_count', '<u4'),
        ('reserved', '<u4'),
    ])
    _edge_dtype = numpy.dtype([
        ('target_node_index', '<u4'),
        ('reserved', '<u4'),
        ('way_id', '<i8'),
        ('distance_m', '<f8'),
    ])
    _way_dtype = numpy.dtype([
        ('way_id', '<i8'),
        ('node_count', '<u4'),
        ('reserved', '<u4'),
        ('highway_type', 'S32'),
    ])
    _cell_dtype = numpy.dtype([
        ('key', '<u8'),
        ('entry_offset', '<u8'),
        ('entry_count', '<u4'),
        ('reserved', '<u4'),
    ])
    _spatial_entry_dtype = numpy.dtype('<u4')
    _cell_size = 0.01

    def __init__(
        self,
        nodes,
        edges,
        way_records,
        cells,
        spatial_entries,
    ):
        self._nodes = nodes
        self._node_ids = nodes['node_id']
        self._edges = edges
        self._cells = cells
        self._cell_keys = cells['key']
        self._spatial_entries = spatial_entries
        self._way_node_counts = {
            int(record['way_id']): int(record['node_count'])
            for record in way_records
        }
        self._way_highway_types = {
            int(record['way_id']): bytes(record['highway_type']).split(b'\0', 1)[0].decode()
            for record in way_records
        }

    def contains_node(self, node_id):
        position = self._node_position(node_id)
        return (
            position < len(self._nodes)
            and int(self._node_ids[position]) == node_id
        )

    def neighbors(self, node_id):
        position = self._node_position(node_id)
        if (
            position >= len(self._nodes)
            or int(self._node_ids[position]) != node_id
        ):
            return ()
        node = self._nodes[position]
        return NativeNeighborView(
            self._nodes,
            self._edges,
            int(node['edge_offset']),
            int(node['edge_offset']) + int(node['edge_count']),
            position,
        )

    def _node_position(self, node_id):
        return bisect_left(self._node_ids, node_id)

    def way_node_count(self, way_id):
        return self._way_node_counts[way_id]

    def highway_type(self, way_id):
        return self._way_highway_types[way_id]

    @staticmethod
    def _cell_key(longitude_cell, latitude_cell):
        return (
            ((int(longitude_cell) & 0xffffffff) << 32)
            | (int(latitude_cell) & 0xffffffff)
        )

    def nodes_within_distance(self, point, max_distance):
        latitude_delta = max_distance / 111320.0
        longitude_scale = max(math.cos(math.radians(point[1])), 1e-12)
        longitude_delta = max_distance / (111320.0 * longitude_scale)
        min_longitude_cell = math.floor((point[0] - longitude_delta) / self._cell_size)
        max_longitude_cell = math.floor((point[0] + longitude_delta) / self._cell_size)
        min_latitude_cell = math.floor((point[1] - latitude_delta) / self._cell_size)
        max_latitude_cell = math.floor((point[1] + latitude_delta) / self._cell_size)

        for longitude_cell in range(min_longitude_cell, max_longitude_cell + 1):
            for latitude_cell in range(min_latitude_cell, max_latitude_cell + 1):
                key = self._cell_key(longitude_cell, latitude_cell)
                cell_position = int(numpy.searchsorted(self._cell_keys, key))
                if (
                    cell_position >= len(self._cells)
                    or int(self._cell_keys[cell_position]) != key
                ):
                    continue
                cell = self._cells[cell_position]
                start = int(cell['entry_offset'])
                end = start + int(cell['entry_count'])
                for node_index in self._spatial_entries[start:end]:
                    node = self._nodes[int(node_index)]
                    candidate = [float(node['longitude']), float(node['latitude'])]
                    if haversine_distance_m(point, candidate) <= max_distance:
                        yield int(node['node_id']), candidate


class NativeConnectingHighwaysByNode:
    def __init__(self, highway_index):
        self._highway_index = highway_index
        self._cache = {}

    def get(self, node_id, default=None):
        highways = self._cache.get(node_id)
        if highways is not None:
            return highways

        adjacency = self._highway_index.neighbors(node_id)
        if not adjacency:
            return default
        highways = {}
        for way_id, _, _, _, _ in adjacency:
            highways[way_id] = {
                'node_count': self._highway_index.way_node_count(way_id),
                'highway_type': self._highway_index.highway_type(way_id),
            }
        self._cache[node_id] = highways
        return highways


class NativeHighwayCollector:
    """Collect connecting highways with the native libosmium collector."""

    _counts_format = '<QQQQQ'
    _counts_size = struct.calcsize(_counts_format)
    _node_dtype = NativeHighwayIndex._node_dtype
    _edge_dtype = NativeHighwayIndex._edge_dtype
    _way_dtype = NativeHighwayIndex._way_dtype
    _cell_dtype = NativeHighwayIndex._cell_dtype
    _spatial_entry_dtype = NativeHighwayIndex._spatial_entry_dtype

    def __init__(
        self,
        route_node_ids,
        binary_path=None,
    ):
        self._route_node_ids = set(route_node_ids)
        self._binary_path = binary_path or os.environ.get(
            'HIGHWAY_COLLECTOR_BINARY',
            'native-highway-collector',
        )
        self._native_highway_index = None

    def collect_highways(self, filename):
        with tempfile.TemporaryDirectory(prefix='hiker-highway-') as directory:
            directory = os.path.abspath(directory)
            route_nodes_path = os.path.join(directory, 'route-node-ids.txt')
            output_path = os.path.join(directory, 'highways.bin')
            with open(route_nodes_path, 'w') as route_nodes_file:
                route_nodes_file.write(
                    ''.join(f'{node_id}\n' for node_id in sorted(self._route_node_ids))
                )
            command = [
                self._binary_path,
                '--input', filename,
                '--route-nodes', route_nodes_path,
                '--output', output_path,
            ]
            subprocess.run(command, check=True)
            self._read_output(output_path)

    def _read_output(self, filename):
        with open(filename, 'rb') as output:
            mapping = mmap.mmap(output.fileno(), 0, access=mmap.ACCESS_READ)
        if len(mapping) < self._counts_size:
            mapping.close()
            raise RuntimeError('Native highway collector output is truncated')

        (
            node_count,
            edge_count,
            way_count,
            cell_count,
            spatial_entry_count,
        ) = struct.unpack_from(self._counts_format, mapping)

        expected_size = self._counts_size + (
            node_count * self._node_dtype.itemsize
            + edge_count * self._edge_dtype.itemsize
            + way_count * self._way_dtype.itemsize
            + cell_count * self._cell_dtype.itemsize
            + spatial_entry_count * self._spatial_entry_dtype.itemsize
        )
        if len(mapping) < expected_size:
            mapping.close()
            raise RuntimeError('Native highway collector output is truncated')

        offset = self._counts_size
        node_records = numpy.frombuffer(
            mapping,
            dtype=self._node_dtype,
            count=node_count,
            offset=offset,
        )
        offset += node_count * self._node_dtype.itemsize
        edge_records = numpy.frombuffer(
            mapping,
            dtype=self._edge_dtype,
            count=edge_count,
            offset=offset,
        )
        offset += edge_count * self._edge_dtype.itemsize
        way_records = numpy.frombuffer(
            mapping,
            dtype=self._way_dtype,
            count=way_count,
            offset=offset,
        )
        offset += way_count * self._way_dtype.itemsize
        cell_records = numpy.frombuffer(
            mapping,
            dtype=self._cell_dtype,
            count=cell_count,
            offset=offset,
        )
        offset += cell_count * self._cell_dtype.itemsize
        spatial_entries = numpy.frombuffer(
            mapping,
            dtype=self._spatial_entry_dtype,
            count=spatial_entry_count,
            offset=offset,
        )
        offset += spatial_entry_count * self._spatial_entry_dtype.itemsize
        self._native_node_records = node_records
        self._native_edge_records = edge_records
        self._native_way_records = way_records
        self._native_cell_records = cell_records
        self._native_spatial_entries = spatial_entries
        self._native_highway_index = None

    def connecting_highways_by_node(self):
        return NativeConnectingHighwaysByNode(self.highway_index())

    def highway_index(self):
        if self._native_highway_index is None:
            self._native_highway_index = NativeHighwayIndex(
                self._native_node_records,
                self._native_edge_records,
                self._native_way_records,
                self._native_cell_records,
                self._native_spatial_entries,
            )
        return self._native_highway_index
