import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import resolve_data_dir


class DatasetPathTests(unittest.TestCase):
    def test_relative_data_dir_resolves_to_project_data_folder(self):
        resolved = Path(resolve_data_dir("./data"))
        expected = Path(__file__).resolve().parents[1] / "data"
        self.assertEqual(resolved, expected)


if __name__ == "__main__":
    unittest.main()
