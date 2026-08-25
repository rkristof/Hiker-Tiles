import math
import os

import rasterio

from route_graph import haversine_distance_m


def offset_elevation_profile(profile, distance_m):
    return {
        'segments': [
            {
                **segment,
                'start_m': segment['start_m'] + distance_m,
                'end_m': segment['end_m'] + distance_m,
            }
            for segment in profile['segments']
        ],
    }


class Elevation:
    PROFILE_MAX_DISTANCE_M = 40_000
    PROFILE_SAMPLE_INTERVAL_M = 40
    PROFILE_COORDINATE_PRECISION = 5
    REVERSAL_THRESHOLD_M = 5.0

    def __init__(self, directory=None, sampler=None):
        self.directory = directory
        self._sampler = sampler
        self.tiles = {}

    def sample(self, point):
        if self._sampler is not None:
            return self._sampler.sample(point)

        longitude, latitude = point
        tile_name = self._tile_name(math.floor(latitude), math.floor(longitude))
        path = os.path.join(self.directory, f'{tile_name}.tif')
        tile_data = self._load(path)
        if tile_data is None:
            return None

        elevation_band, inverse_transform, no_data_value = tile_data
        column, row_index = inverse_transform * (longitude, latitude)
        row_index, column = int(row_index), int(column)
        if (
            row_index < 0
            or row_index >= elevation_band.shape[0]
            or column < 0
            or column >= elevation_band.shape[1]
        ):
            return None

        elevation = elevation_band[row_index, column]
        if no_data_value is not None and elevation == no_data_value:
            return None
        elevation = float(elevation)
        return elevation if math.isfinite(elevation) else None

    def route_elevation(self, path):
        """Sample route elevation once and calculate totals with a profile."""
        elevation_gain_m = 0.0
        elevation_loss_m = 0.0
        profile_segments = []
        current_segment = []

        def close_segment():
            nonlocal current_segment, elevation_gain_m, elevation_loss_m
            if len(current_segment) < 2:
                current_segment = []
                return

            segment_gain_m, segment_loss_m = self.calculate_elevation_change(current_segment)
            elevation_gain_m += segment_gain_m
            elevation_loss_m += segment_loss_m
            profile_segments.append(self.build_elevation_profile_segment([
                (path_distance, point, round(elevation))
                for path_distance, point, elevation in current_segment
            ]))
            current_segment = []

        for path_distance, point in self.sample_path_points(
            path,
            self.PROFILE_SAMPLE_INTERVAL_M,
        ):
            elevation = self.sample(point)
            if elevation is None:
                close_segment()
                continue

            current_segment.append((round(path_distance), point, elevation))

        close_segment()
        return {
            'elevation_gain_m': elevation_gain_m,
            'elevation_loss_m': elevation_loss_m,
            'profile': {'segments': profile_segments},
        }

    def close(self):
        self.tiles.clear()

    @staticmethod
    def route_duration_min(distance_m, elevation_gain_m, elevation_loss_m):
        """Calculate route duration from horizontal and vertical travel."""
        horizontal_minutes = distance_m / 1000 * 60 / 4
        vertical_minutes = (elevation_gain_m / 300 * 60) + (elevation_loss_m / 500 * 60)
        total_minutes = max(horizontal_minutes, vertical_minutes) + min(horizontal_minutes, vertical_minutes) / 2
        return int(round(total_minutes / 15) * 15)

    @staticmethod
    def sample_path_points(path, interval_m):
        """Return route coordinates sampled at fixed distances plus the endpoint."""
        if len(path) < 2:
            return []

        samples = [(0.0, path[0])]
        path_distance = 0.0
        next_sample_distance = interval_m
        previous_point = path[0]
        for current_point in path[1:]:
            segment_distance = haversine_distance_m(previous_point, current_point)
            segment_end_distance = path_distance + segment_distance
            while next_sample_distance <= segment_end_distance:
                if segment_distance == 0:
                    break
                fraction = (next_sample_distance - path_distance) / segment_distance
                samples.append((
                    next_sample_distance,
                    Elevation.interpolate_point(previous_point, current_point, fraction),
                ))
                next_sample_distance += interval_m
            path_distance = segment_end_distance
            previous_point = current_point

        if path_distance > samples[-1][0]:
            samples.append((path_distance, path[-1]))
        return samples

    @staticmethod
    def interpolate_point(first_point, second_point, fraction):
        return [
            first_point[0] + (second_point[0] - first_point[0]) * fraction,
            first_point[1] + (second_point[1] - first_point[1]) * fraction,
        ]

    @classmethod
    def encode_polyline(cls, points, precision=None):
        """Encode longitude/latitude points with Google polyline delta encoding."""
        if precision is None:
            precision = cls.PROFILE_COORDINATE_PRECISION
        factor = 10 ** precision
        previous_latitude = 0
        previous_longitude = 0
        encoded = []

        def encode_value(value):
            value = ~(value << 1) if value < 0 else value << 1
            characters = []
            while value >= 0x20:
                characters.append(chr((0x20 | (value & 0x1F)) + 63))
                value >>= 5
            characters.append(chr(value + 63))
            return ''.join(characters)

        for longitude, latitude in points:
            quantized_latitude = math.floor(latitude * factor + 0.5)
            quantized_longitude = math.floor(longitude * factor + 0.5)
            encoded.append(encode_value(quantized_latitude - previous_latitude))
            encoded.append(encode_value(quantized_longitude - previous_longitude))
            previous_latitude = quantized_latitude
            previous_longitude = quantized_longitude

        return ''.join(encoded)

    @classmethod
    def build_elevation_profile_segment(cls, samples):
        """Pack sampled elevations and coordinates into one compact profile segment."""
        return {
            'start_m': samples[0][0],
            'end_m': samples[-1][0],
            'elevations': [elevation for _, _, elevation in samples],
            'coordinates': cls.encode_polyline([point for _, point, _ in samples]),
        }

    @classmethod
    def calculate_elevation_change(cls, samples):
        """Calculate gain and loss after filtering insignificant reversals."""
        if len(samples) < 2:
            return 0, 0

        elevation_gain_m = 0.0
        elevation_loss_m = 0.0
        anchor_elevation = samples[0][2]
        candidate_elevation = anchor_elevation
        direction = 0

        for _, _, elevation in samples[1:]:
            if direction == 0:
                if elevation > candidate_elevation:
                    direction = 1
                    candidate_elevation = elevation
                elif elevation < candidate_elevation:
                    direction = -1
                    candidate_elevation = elevation
                continue

            if direction == 1:
                if elevation >= candidate_elevation:
                    candidate_elevation = elevation
                elif candidate_elevation - elevation >= cls.REVERSAL_THRESHOLD_M:
                    elevation_delta_m = candidate_elevation - anchor_elevation
                    if elevation_delta_m >= cls.REVERSAL_THRESHOLD_M:
                        elevation_gain_m += elevation_delta_m
                        anchor_elevation = candidate_elevation
                    direction = -1
                    candidate_elevation = elevation
            elif direction == -1:
                if elevation <= candidate_elevation:
                    candidate_elevation = elevation
                elif elevation - candidate_elevation >= cls.REVERSAL_THRESHOLD_M:
                    elevation_delta_m = anchor_elevation - candidate_elevation
                    if elevation_delta_m >= cls.REVERSAL_THRESHOLD_M:
                        elevation_loss_m += elevation_delta_m
                        anchor_elevation = candidate_elevation
                    direction = 1
                    candidate_elevation = elevation

        if direction == 1:
            elevation_delta_m = candidate_elevation - anchor_elevation
            if elevation_delta_m >= cls.REVERSAL_THRESHOLD_M:
                elevation_gain_m += elevation_delta_m
        elif direction == -1:
            elevation_delta_m = anchor_elevation - candidate_elevation
            if elevation_delta_m >= cls.REVERSAL_THRESHOLD_M:
                elevation_loss_m += elevation_delta_m

        return round(elevation_gain_m), round(elevation_loss_m)

    @staticmethod
    def _tile_name(latitude, longitude):
        lat_prefix = 'N' if latitude >= 0 else 'S'
        lon_prefix = 'E' if longitude >= 0 else 'W'
        return (
            f'Copernicus_DSM_COG_10_{lat_prefix}{abs(latitude):02d}_00_'
            f'{lon_prefix}{abs(longitude):03d}_00_DEM'
        )

    def _load(self, path):
        if path not in self.tiles:
            if os.path.exists(path):
                with rasterio.open(path) as dataset:
                    self.tiles[path] = (dataset.read(1), ~dataset.transform, dataset.nodata)
            else:
                self.tiles[path] = None
        return self.tiles[path]
