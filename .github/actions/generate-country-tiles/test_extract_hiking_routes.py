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
        self.assertEqual(len(profile['segments'][0]['elevations']), 4)


if __name__ == '__main__':
    unittest.main()
