import sys
import unittest

sys.path.append('../..')
from research.utils.input_reader import InputReader

class InputReaderTest(unittest.TestCase):
    def setUp(self):
        self.reader_video = InputReader('research/test/inputs/input_video_file.json')
        self.reader_folder_videos = InputReader('research/test/inputs/input_folder_videos.json')
        self.reader_folder_images = InputReader('research/test/inputs/input_folder_images.json')

    def test_get_video_file_path(self):
        self.assertEqual(self.reader_video.get_video_file_path(), "data/test/test_video.mp4")

    def test_get_record_dration(self):
        self.assertEqual(self.reader_video.get_record_duration(), 100)

    def test_get_base(self):
        self.assertEqual(self.reader_video.get_base(), 1.1)
    
    def test_get_smin(self):
        self.assertEqual(self.reader_video.get_smin(), 8)

    def test_get_cache_path(self):
        self.assertEqual(self.reader_video.get_cache_path(), "data/cache/test_video.npz")

    def test_get_result_path(self):
        self.assertEqual(self.reader_video.get_result_path(), "data/results/test_video.mp4")

    def test_get_folder_videos_path(self):
        self.assertEqual(self.reader_folder_videos.get_folder_videos_path(), "data/test/test_folder_videos")

    def test_get_folder_images_path(self):
        self.assertEqual(self.reader_folder_images.get_folder_images_path(), "data/test/test_folder_images")

    

if __name__ == "__main__":
    unittest.main()