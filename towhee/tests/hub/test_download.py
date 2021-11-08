import unittest
from pathlib import Path
from PIL import Image
# from shutil import rmtree

from towhee import pipeline
from towhee.hub.file_manager import FileManagerConfig, FileManager
from towhee.tests import CACHE_PATH



cache_path = Path(__file__).parent.parent.resolve()


class TestDownload(unittest.TestCase):
    """
    Simple hub download and run test.
    """

    @classmethod
    def setUpClass(cls):
        new_cache = (CACHE_PATH/'test_cache')
        pipeline_cache = (CACHE_PATH/'test_util')
        operator_cache = (CACHE_PATH/'mock_operators')
        fmc = FileManagerConfig()
        fmc.change_default_cache(new_cache)
        pipelines = list(pipeline_cache.rglob('*.yaml'))
        operators = [f for f in operator_cache.iterdir() if f.is_dir()]
        fmc.cache_local_pipeline(pipelines)
        fmc.cache_local_operator(operators)
        fm = FileManager(fmc) # pylint: disable=unused-variable

    def test_pipeline(self):
        p = pipeline('towhee/ci_test')
        img = Image.open(CACHE_PATH / 'test_cache/ci_test&towhee$main/towhee_logo.png')
        res = p(img)
        self.assertEqual(res[0].size, 1000)


if __name__ == '__main__':
    unittest.main()
