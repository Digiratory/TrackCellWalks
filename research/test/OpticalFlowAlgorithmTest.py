from research.utils.input_reader import InputReader
from research.utils.data_generator import VideoReader
from research.utils.optical_flow import ComputeOpticalFlow
from research.utils.visualizer import OpticalFlowVisualizer
from research.utils.analyzer import OpticalFlowAnalyzer

import unittest

class OpticalFlowAlgorithmVideoFileTest(unittest.TestCase):
    def setUp(self):
        self.input_reader = InputReader('research/test/inputs/input_video_file.json')

        self.video = self.input_reader.get_video_file_path()
        self.cache_path = self.input_reader.get_cache_path()

    def test_ComputeOpticalFlowByFarneback(self):

        video_reader = VideoReader(self.video)
        frame_generator = video_reader.frames_generator()

        cof = ComputeOpticalFlow(frame_generator, self.cache_path)
        cof.compute_optical_flow_by_Farneback()

        analyzer = OpticalFlowAnalyzer(self.input_reader, cof)

        visualizer = OpticalFlowVisualizer(cof, analyzer)
        visualizer.plot_optical_flow_magnitude_and_vector_field()
        visualizer.plot_fluctuation_analysis()

    def test_LoadOpticalFlow(self):
        cof = ComputeOpticalFlow(self.video, self.cache_path)
        cof.load_optical_flow_from_cache()

        analyzer = OpticalFlowAnalyzer(self.input_reader, cof)

        visualizer = OpticalFlowVisualizer(cof, analyzer)
        visualizer.plot_optical_flow_magnitude_and_vector_field()
        visualizer.plot_fluctuation_analysis()

    def test_ComputeOpticalFlowByTVL1(self):
        video_reader = VideoReader(self.video)
        frame_generator = video_reader.frames_generator()

        cof = ComputeOpticalFlow(frame_generator, self.cache_path)
        cof.compute_optical_flow_by_tvl1()

        analyzer = OpticalFlowAnalyzer(self.input_reader, cof)

        visualizer = OpticalFlowVisualizer(cof, analyzer)
        visualizer.plot_optical_flow_magnitude_and_vector_field()
        visualizer.plot_fluctuation_analysis()


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(OpticalFlowAlgorithmVideoFileTest('test_hello'))



