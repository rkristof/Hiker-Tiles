import sys
import types
import unittest
from pathlib import Path


try:
    import osmium  # noqa: F401
except ModuleNotFoundError:
    sys.modules['osmium'] = types.SimpleNamespace(SimpleHandler=object, InvalidLocationError=Exception)
try:
    import rasterio  # noqa: F401
except ModuleNotFoundError:
    sys.modules['rasterio'] = types.SimpleNamespace()

sys.path.insert(0, str(Path(__file__).parent))
import extract_hiking_routes as routes


class ConstantSampler:
    def sample(self, point):
        return 100


class SpatialNoiseSampler:
    def __init__(self):
        self.sample_count = 0

    def sample(self, point):
        self.sample_count += 1
        spatial_band = int(round(point[0] * 100_000)) // 10
        return 100 + 4 * (spatial_band % 2)


class ExtractHikingRoutesTests(unittest.TestCase):
    def test_profile_includes_off_grid_endpoint(self):
        path = [[0.0000, 0.0000], [0.0010, 0.0000]]
        samples = routes.sample_path_points(path, 40)
        profile = routes.route_elevation(path, ConstantSampler())['profile']

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0][0], 0)
        self.assertGreater(samples[-1][0], samples[-2][0])
        profile_segment = profile['segments'][0]
        self.assertEqual(profile_segment['start_m'], 0)
        self.assertEqual(profile_segment['end_m'], round(samples[-1][0]))
        self.assertEqual(len(profile_segment['elevations']), 4)
        self.assertEqual(profile_segment['elevations'][-1], 100)
        self.assertTrue(profile_segment['coordinates'])
        self.assertNotIn('longitude', profile_segment)
        self.assertNotIn('latitude', profile_segment)

    def test_profile_keeps_dem_gaps_as_separate_segments(self):
        class GapSampler:
            def sample(self, point):
                return None if 0.0012 < point[0] < 0.0020 else 100

        path = [[0.0000, 0.0000], [0.0030, 0.0000]]
        profile = routes.route_elevation(path, GapSampler())['profile']

        self.assertEqual(len(profile['segments']), 2)
        self.assertEqual(
            (profile['segments'][0]['start_m'], profile['segments'][0]['end_m']),
            (0, 120),
        )
        self.assertEqual(
            (profile['segments'][1]['start_m'], profile['segments'][1]['end_m']),
            (240, 334),
        )

    def test_profile_coordinates_are_quantized_to_five_decimal_places(self):
        points = [[12.345674, 47.123454], [12.345686, 47.123466]]

        encoded = routes.encode_polyline(points)

        self.assertEqual(encoded, 'qxr~GmgjjACC')

    def test_elevation_change_uses_profile_sample_interval(self):
        path = [[index * 0.0001, 0.0] for index in range(41)]
        sampler = SpatialNoiseSampler()
        elevation = routes.route_elevation(path, sampler)
        profile = elevation['profile']

        expected_gain = 0
        expected_loss = 0
        for segment in profile['segments']:
            elevations = segment['elevations']
            for previous, current in zip(elevations, elevations[1:]):
                delta = current - previous
                if delta >= 2:
                    expected_gain += delta
                elif delta <= -2:
                    expected_loss -= delta

        self.assertEqual(
            (elevation['elevation_gain_m'], elevation['elevation_loss_m']),
            (expected_gain, expected_loss),
        )
        self.assertEqual(sampler.sample_count, len(routes.sample_path_points(path, 40)))


if __name__ == '__main__':
    unittest.main()
