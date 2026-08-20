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


class ExtractHikingRoutesTests(unittest.TestCase):
    def test_profile_includes_off_grid_endpoint(self):
        path = [[0.0000, 0.0000], [0.0010, 0.0000]]
        samples = routes.sample_path_points(path, 40)
        profile = routes.route_elevation_profile(path, ConstantSampler())

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0][0], 0)
        self.assertGreater(samples[-1][0], samples[-2][0])
        profile_samples = profile['segments'][0]['samples']
        self.assertEqual(len(profile_samples), 4)
        self.assertEqual(profile_samples[0]['distance_m'], 0)
        self.assertEqual(profile_samples[-1]['distance_m'], round(samples[-1][0]))
        self.assertEqual(profile_samples[0]['longitude'], path[0][0])
        self.assertEqual(profile_samples[0]['latitude'], path[0][1])
        self.assertEqual(profile_samples[-1]['longitude'], path[-1][0])
        self.assertEqual(profile_samples[-1]['latitude'], path[-1][1])
        self.assertEqual(profile_samples[-1]['elevation_m'], 100)

    def test_profile_keeps_dem_gaps_as_separate_segments(self):
        class GapSampler:
            def sample(self, point):
                return None if 0.0012 < point[0] < 0.0020 else 100

        path = [[0.0000, 0.0000], [0.0030, 0.0000]]
        profile = routes.route_elevation_profile(path, GapSampler())

        self.assertEqual(len(profile['segments']), 2)
        self.assertEqual(
            [sample['distance_m'] for sample in profile['segments'][0]['samples']],
            [0, 40, 80, 120],
        )
        self.assertEqual(
            [sample['distance_m'] for sample in profile['segments'][1]['samples']],
            [240, 280, 320, 334],
        )


if __name__ == '__main__':
    unittest.main()
