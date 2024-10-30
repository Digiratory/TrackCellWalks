import sys
sys.path.append('../..')

from research.utils.input_reader import InputReader
from research.utils.data_generator import VideoReader
from research.utils.optical_flow import ComputeOpticalFlow
from research.utils.visualizer import OpticalFlowVisualizer
from research.utils.analyzer import OpticalFlowAnalyzer

project_path = "/Users/glebsolanik/work/TrackCellWalks/"

input_reader = InputReader('params.json')
video = input_reader.get_video_file_path()
cache_path = project_path + "data/cache/test_video_TVL1.npz"

video_reader = VideoReader(video)
frame_generator = video_reader.frames_generator()

cof = ComputeOpticalFlow(frame_generator, cache_path)
cof.load_optical_flow_from_cache()

analyzer = OpticalFlowAnalyzer(input_reader, cof)

visualizer = OpticalFlowVisualizer(cof, analyzer)
visualizer.plot_optical_flow_magnitude_and_vector_field()
visualizer.plot_fluctuation_analysis()
