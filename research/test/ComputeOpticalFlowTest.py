import sys
import numpy as np

sys.path.append('../..')
from research.utils.input_reader import InputReader
from research.utils.data_generator import VideoReader
from research.utils.optical_flow import ComputeOpticalFlow

import unittest

class ComputeOpticalFlowTest(unittest.TestCase):
    def setUp(self):
        self.input_reader = InputReader('research/test/inputs/input_video_file.json')
        self.video_path = self.input_reader.get_video_file_path()
        self.computed_cache_path = self.input_reader.get_cache_path()
        self.loaded_cache_path = 'data/cache/test_video_Farneback.npz'

        self.video_reader = VideoReader(self.video_path)
        self.generator = self.video_reader.frames_generator()

        self.cof_loaded = ComputeOpticalFlow(self.generator, self.loaded_cache_path)
        self.cof_loaded.load_optical_flow_from_cache()

        self.cof_computed = ComputeOpticalFlow(self.generator, self.computed_cache_path)

    def test_compute_optical_flow_by_Farneback(self):
        horizontal_components, vertical_components = self.cof_computed.compute_optical_flow_by_Farneback()
        self.assertEqual(len(horizontal_components), 99, "Неверное количество горизонтальных компонент!")
        self.assertEqual(len(vertical_components), 99, "Неверное количество вертикальных компонент!")

    def test_compute_pixel_offset(self):
        pixel_offset = self.cof_loaded.compute_pixel_offset()
        self.assertEqual(len(pixel_offset), 1040, "Неверное количество компонентов, которые превышают порог (thresh=0.5)!")

    def test_downsample_vertical_magnitude(self):
        downsampled_vertical_magnitude = self.cof_loaded.downsample_vertical_magnitude()
        self.assertEqual(len(downsampled_vertical_magnitude), 19, "Неверное количество выбранных вертикальных величин для отображения!")
        
    def test_downsample_horizontal_magnitude(self):
        downsampled_horizontal_magnitude = self.cof_loaded.downsample_horizontal_magnitude()
        self.assertEqual(len(downsampled_horizontal_magnitude), 19, "Неверное количество выбранных горизонтальных величин!")

    def test_compute_step_between_vectors(self):
        step = self.cof_loaded.compute_step_between_vectors()
        self.assertEqual(step, 55, "Неверное значение шага для отображения 25 векторов (num_displayed_vectors=25)!")

    def test_get_image_height(self):
        height = self.cof_loaded.get_image_height()
        self.assertEqual(height, 1040)

    def test_get_image_width(self):
        width = self.cof_loaded.get_image_width()
        self.assertEqual(width, 1388)

    def test_compute_vector_field(self):
        vector_field_load = self.cof_loaded.compute_vector_field()
        self.assertEqual(len(vector_field_load), 99)
        
if __name__ == "__main__":
    unittest.main()
