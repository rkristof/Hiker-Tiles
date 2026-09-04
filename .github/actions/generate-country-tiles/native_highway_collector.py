import mmap
import os
import struct
import subprocess
import tempfile
from bisect import bisect_left
from collections.abc import Mapping

import numpy

from utils import haversine_distance_m


class NativeNodeSequence:
    _dtype = numpy.dtype([
        ('node_id', '<i8'),
        ('longitude', '<f8'),
        ('latitude', '<f8'),
    ])

    def __init__(self, data):
        self._records = numpy.frombuffer(data, dtype=self._dtype)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        record = self._records[index]
        return int(record['node_id']), [
            float(record['longitude']),
            float(record['latitude']),
        ]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class NativeAdjacencyView:
    def __init__(self, entries, start, end, way_records):
        self._entries = entries
        self._start = start
        self._end = end
        self._way_records = way_records

    def __len__(self):
        return self._end - self._start

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry = self._entries[self._start + index]
        return (
            int(self._way_records[int(entry['way_index'])]['way_id']),
            int(entry['node_index']),
        )

    def __iter__(self):
        for entry in self._entries[self._start:self._end]:
            yield (
                int(self._way_records[int(entry['way_index'])]['way_id']),
                int(entry['node_index']),
            )


class NativeWayView(Mapping):
    def __init__(
        self,
        highway_type,
        node_data,
        node_data_offset,
        node_offset,
        node_count,
    ):
        self._highway_type = highway_type
        self._node_data = node_data
        self._node_data_offset = node_data_offset
        self._node_offset = node_offset
        self._node_count = node_count
        self._nodes = None

    def __getitem__(self, key):
        if key == 'highway_type':
            return self._highway_type
        if key == 'nodes':
            if self._nodes is None:
                start = self._node_data_offset + self._node_offset * NativeNodeSequence._dtype.itemsize
                end = start + self._node_count * NativeNodeSequence._dtype.itemsize
                self._nodes = NativeNodeSequence(self._node_data[start:end])
            return self._nodes
        raise KeyError(key)

    def __iter__(self):
        return iter(('highway_type', 'nodes'))

    def __len__(self):
        return 2


class NativeHighwayIndex(Mapping):
    def __init__(
        self,
        node_records,
        entries,
        way_records,
        node_data,
        node_data_offset,
        highway_types,
    ):
        self._node_records = node_records
        self._node_ids = node_records['node_id']
        self._entries = entries
        self._way_records = way_records
        self._node_data = memoryview(node_data)
        self._node_data_offset = node_data_offset
        self._highway_types = highway_types
        self._segment_distance_cache = {}
        self._way_indexes_by_id = {
            int(record['way_id']): index
            for index, record in enumerate(way_records)
        }
        self._way_views = {}

    def __contains__(self, node_id):
        position = self._node_position(node_id)
        return (
            position < len(self._node_records)
            and int(self._node_ids[position]) == node_id
        )

    def __getitem__(self, node_id):
        position = self._node_position(node_id)
        if (
            position >= len(self._node_records)
            or int(self._node_ids[position]) != node_id
        ):
            raise KeyError(node_id)
        record = self._node_records[position]
        start = int(record['entry_offset'])
        end = start + int(record['entry_count'])
        return NativeAdjacencyView(
            self._entries,
            start,
            end,
            self._way_records,
        )

    def _node_position(self, node_id):
        return bisect_left(self._node_ids, node_id)

    def __iter__(self):
        return (int(node_id) for node_id in self._node_records['node_id'])

    def __len__(self):
        return len(self._node_records)

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

    def highway(self, way_id):
        way_index = self._way_indexes_by_id[way_id]
        highway = self._way_views.get(way_id)
        if highway is None:
            record = self._way_records[way_index]
            highway = NativeWayView(
                self._highway_types[int(record['type_index'])],
                self._node_data,
                self._node_data_offset,
                int(record['node_offset']),
                int(record['node_count']),
            )
            self._way_views[way_id] = highway
        return highway


class NativeConnectingHighwaysByNode:
    def __init__(self, highway_index, route_node_ids):
        self._highway_index = highway_index
        self._route_node_ids = route_node_ids
        self._cache = {}

    def get(self, node_id, default=None):
        if node_id not in self._route_node_ids:
            return default
        highways = self._cache.get(node_id)
        if highways is not None:
            return highways

        adjacency = self._highway_index.get(node_id)
        if not adjacency:
            return default
        highways = {}
        for way_id, _ in adjacency:
            highways[way_id] = self._highway_index.highway(way_id)
        self._cache[node_id] = highways
        return highways


class NativeHighwayCollector:
    """Collect connecting highways with the native libosmium collector."""

    _header_format = '<QQQQII'
    _header_size = struct.calcsize(_header_format)
    _way_dtype = numpy.dtype([
        ('way_id', '<i8'),
        ('node_offset', '<u8'),
        ('node_count', '<u4'),
        ('type_index', '<u4'),
    ])
    _type_dtype = numpy.dtype([('offset', '<u4'), ('size', '<u4')])
    _index_node_dtype = numpy.dtype([
        ('node_id', '<i8'),
        ('entry_offset', '<u8'),
        ('entry_count', '<u8'),
    ])
    _index_entry_dtype = numpy.dtype([
        ('way_index', '<u4'),
        ('node_index', '<u4'),
    ])

    def __init__(
        self,
        route_node_ids,
        binary_path=None,
        excluded_way_ids=(),
    ):
        self._route_node_ids = set(route_node_ids)
        self._excluded_way_ids = set(excluded_way_ids)
        self._binary_path = binary_path or os.environ.get(
            'HIGHWAY_COLLECTOR_BINARY',
            'native-highway-collector',
        )
        self._native_highway_index = None
        self._native_mapping = None

    def collect_highways(self, filename):
        with tempfile.TemporaryDirectory(prefix='hiker-highway-') as directory:
            directory = os.path.abspath(directory)
            route_nodes_path = os.path.join(directory, 'route-node-ids.txt')
            excluded_ways_path = os.path.join(directory, 'excluded-way-ids.txt')
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
            if self._excluded_way_ids:
                with open(excluded_ways_path, 'w') as excluded_ways_file:
                    excluded_ways_file.write(
                        ''.join(f'{way_id}\n' for way_id in sorted(self._excluded_way_ids))
                    )
                command.extend(['--exclude-way-ids', excluded_ways_path])
            subprocess.run(command, check=True)
            self._read_output(output_path)

    def _read_output(self, filename):
        with open(filename, 'rb') as output:
            mapping = mmap.mmap(output.fileno(), 0, access=mmap.ACCESS_READ)
        if len(mapping) < self._header_size:
            mapping.close()
            raise RuntimeError('Native highway collector output is truncated')

        (
            selected_way_count,
            node_count,
            index_node_count,
            index_entry_count,
            type_count,
            type_data_size,
        ) = struct.unpack_from(self._header_format, mapping)

        offset = self._header_size
        way_records = numpy.frombuffer(
            mapping,
            dtype=self._way_dtype,
            count=selected_way_count,
            offset=offset,
        )
        offset += selected_way_count * self._way_dtype.itemsize
        node_data_offset = offset
        offset += node_count * NativeNodeSequence._dtype.itemsize
        type_records = numpy.frombuffer(
            mapping,
            dtype=self._type_dtype,
            count=type_count,
            offset=offset,
        )
        offset += type_count * self._type_dtype.itemsize
        type_data_offset = offset
        offset += type_data_size
        index_nodes = numpy.frombuffer(
            mapping,
            dtype=self._index_node_dtype,
            count=index_node_count,
            offset=offset,
        )
        offset += index_node_count * self._index_node_dtype.itemsize
        index_entries = numpy.frombuffer(
            mapping,
            dtype=self._index_entry_dtype,
            count=index_entry_count,
            offset=offset,
        )
        offset += index_entry_count * self._index_entry_dtype.itemsize
        if len(mapping) < offset:
            mapping.close()
            raise RuntimeError('Native highway collector output is truncated')

        highway_types = [
            bytes(
                mapping[
                    type_data_offset + int(record['offset']):
                    type_data_offset + int(record['offset']) + int(record['size'])
                ]
            ).decode('utf-8')
            for record in type_records
        ]

        self._native_mapping = mapping
        self._native_index_node_records = index_nodes
        self._native_index_entries = index_entries
        self._native_way_records = way_records
        self._native_node_data_offset = node_data_offset
        self._native_highway_types = highway_types
        self._native_highway_index = None

    def connecting_highways_by_node(self):
        return NativeConnectingHighwaysByNode(
            self.highway_index(),
            self._route_node_ids,
        )

    def highway_index(self):
        if self._native_highway_index is None:
            self._native_highway_index = NativeHighwayIndex(
                self._native_index_node_records,
                self._native_index_entries,
                self._native_way_records,
                self._native_mapping,
                self._native_node_data_offset,
                self._native_highway_types,
            )
        return self._native_highway_index
