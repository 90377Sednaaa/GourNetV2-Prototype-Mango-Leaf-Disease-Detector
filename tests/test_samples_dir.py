import pathlib
import unittest

class TestSamplesDir(unittest.TestCase):
    def test_samples_directory_exists(self):
        root = pathlib.Path(__file__).parent.parent
        samples_dir = root / "samples"
        self.assertTrue(samples_dir.is_dir())
        readme = samples_dir / "README.md"
        self.assertTrue(readme.is_file())

if __name__ == "__main__":
    unittest.main()
