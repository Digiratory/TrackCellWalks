import sys

sys.path.append('../..')
from research.utils.input_reader import InputReader
from research.utils.data_generator import VideoReader
from research.utils.optical_flow import ComputeOpticalFlow
from research.utils.visualizer import OpticalFlowVisualizer
from research.utils.analyzer import OpticalFlowAnalyzer

import unittest

class OpticalFlowAlgorithmVideoFileTest(unittest.TestCase):
    def setUp(self):
        self.params_reader = InputReader('research/test/inputs/input_video_file.json')

        self.input_path = self.params_reader.get_input_path()
        self.cache_path = self.params_reader.get_cache_path()
        self.result_path = self.params_reader.get_result_path()

        self.base = self.params_reader.get_base()
        self.smin = self.params_reader.get_smin()
        self.record_duration = self.params_reader.get_record_duration()
        self.smax = self.record_duration/2

    def test_ComputeOpticalFlowByFarneback(self):

        video_reader = VideoReader(self.input_path)
        frame_generator = video_reader.frames_generator()

        cof = ComputeOpticalFlow(frame_generator, self.cache_path)
        vertical_components, horizontal_components = cof.compute_optical_flow_by_Farneback()

    def test_LoadOpticalFlow(self): 
        cof = ComputeOpticalFlow(self.input_path, self.cache_path)
        vertical_components, horizontal_components = cof.load_optical_flow_from_cahce()

        analyzer = OpticalFlowAnalyzer(horizontal_components, vertical_components)

        height = analyzer.get_image_height()
        width = analyzer.get_image_width()
        step = analyzer.compute_step_between_vectors()
        
        visulaizer = OpticalFlowVisualizer(height, width, step)

        pixel_offset = analyzer.compute_pixel_offset()
        selected_horizontal_magnitude = analyzer.select_horizontal_magnitude()
        selected_vertical_magnitude = analyzer.select_vertical_magnitude()

        visulaizer.plot_optical_flow_magnitude_and_vector_field(pixel_offset, selected_horizontal_magnitude, selected_vertical_magnitude)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(OpticalFlowAlgorithmVideoFileTest('test_hello'))



