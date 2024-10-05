import sys
import numpy as np

sys.path.append('../..')
from research.utils.input_reader import InputReader
from research.utils.data_generator import VideoReader
from research.utils.optical_flow import ComputeOpticalFlow

import unittest

class ComputeOpticalFlowTest(unittest.TestCase):
    def setUp(self):
        self.params_reader = InputReader('research/test/inputs/input_video_file.json')
        self.video_path = self.params_reader.get_input_path()
        self.computed_cache_path = self.params_reader.get_cache_path()
        self.loaded_cache_path = 'data/cache/test_video_Farneback.npz'

        self.video_reader = VideoReader(self.video_path)
        self.generator = self.video_reader.frames_generator()

        self.cof_loaded = ComputeOpticalFlow(self.generator, self.loaded_cache_path)
        self.cof_computed = ComputeOpticalFlow(self.generator, self.computed_cache_path)

    def test_compute_optical_flow_by_Farneback(self):
        loaded_horzinotal_components, loaded_vertical_components = self.cof_loaded.load_optical_flow_from_cahce()
    
        computed_horzinotal_components, computed_vertical_components = self.cof_computed.compute_optical_flow_by_Farneback()

        isHorizontalEqual = self.isArraysEqual(loaded_horzinotal_components, computed_horzinotal_components)
        isVerticalEqual = self.isArraysEqual(loaded_vertical_components, computed_vertical_components)

        self.assertTrue(isHorizontalEqual)
        self.assertTrue(isVerticalEqual)

    def isArraysEqual(self, arr1, arr2) -> bool:
        equal_array = arr1 == arr2
        unique_elements_of_array = np.unique(equal_array)
        if False not in unique_elements_of_array:
            return True
        return False    
        

if __name__ == "__main__":
    unittest.main()
