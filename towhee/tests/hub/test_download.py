import unittest
from pathlib import Path
from PIL import Image
from shutil import rmtree
from towhee import pipeline

cache_path = Path(__file__).parent.parent.resolve()


class TestDownload(unittest.TestCase):
    """
    Simple hub download and run test.
    """
    def test_pipeline(self):
        p = pipeline('towhee/ci_test', cache=str(cache_path), force_download=True)
        img = Image.open(str(cache_path / 'towhee/ci_test/towhee_logo.png'))
        res = p(img)
        self.assertEqual(res[0].size, 1000)
        rmtree(str(cache_path) + '/towhee')


if __name__ == '__main__':
    unittest.main()
