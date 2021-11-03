import unittest
from pathlib import Path
from PIL import Image
from shutil import rmtree

from towhee import pipeline
from towhee.hub.file_manager import FileManagerConfig

cache_path = Path(__file__).parent.resolve()

class TestDownload(unittest.TestCase):
    """
    Simple hub download and run test.
    """
    def test_pipeline(self):
        fmc = FileManagerConfig()
        fmc.change_default_cache(default_path=cache_path/'test_cache')
        p = pipeline('towhee/ci_test', fmc=fmc)
        img = Image.open(str(cache_path / 'test_cache/ci_test&towhee$main/towhee_logo.png'))
        res = p(img)
        self.assertEqual(res[0].size, 1000)
        rmtree(str(cache_path) + '/test_cache')


if __name__ == '__main__':
    unittest.main()
