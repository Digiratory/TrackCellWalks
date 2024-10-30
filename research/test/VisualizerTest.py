import unittest
from unittest.mock import patch

from research.utils.input_reader import InputReader
from research.utils.data_generator import VideoReader
from research.utils.optical_flow import ComputeOpticalFlow
from research.utils.analyzer import OpticalFlowAnalyzer
from research.utils.visualizer import OpticalFlowVisualizer

class OpticalFlowVisualizerTest(unittest.TestCase):
    def setUp(self):
        input_reader = InputReader('research/test/inputs/input_video_file.json')
        video_path = input_reader.get_video_file_path()
        cache_path = input_reader.get_cache_path()

        video_reader = VideoReader(video_path)
        generator = video_reader.frames_generator()

        cof = ComputeOpticalFlow(generator, cache_path)
        cof.load_optical_flow_from_cache()

        analyzer = OpticalFlowAnalyzer(input_reader, cof)

        self.visualizer = OpticalFlowVisualizer(cof, analyzer)

    @patch('matplotlib.pyplot.show')
    def test_plot_optical_flow_magnitude_and_vector_field(self, mock_show):
        try:
            self.visualizer.plot_optical_flow_magnitude_and_vector_field()
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"test_plot_optical_flow_magnitude_and_vector_field failed with exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_plot_fluctuation_analysis(self, mock_show):
        try:
            self.visualizer.plot_fluctuation_analysis()
            mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"test_plot_fluctuation_analysis failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
