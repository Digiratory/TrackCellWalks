import unittest

from research.utils.input_reader import InputReader
from research.utils.data_generator import VideoReader
from research.utils.optical_flow import ComputeOpticalFlow
from research.utils.analyzer import OpticalFlowAnalyzer


class OpticalFlowAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        input_reader = InputReader("research/test/inputs/input_video_file.json")
        video_file_path = input_reader.get_video_file_path()
        cache_path = input_reader.get_cache_path()
        
        video_reader = VideoReader(video_file_path)
        
        cof = ComputeOpticalFlow(video_reader, cache_path)
        cof.load_optical_flow_from_cache()

        self.analyzer = OpticalFlowAnalyzer(input_reader, cof)

    def test_compute_temporal_scales(self):
        temporal_scales = self.analyzer._compute_temporal_scales()
        self.assertEqual(temporal_scales, [7, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25, 28, 30, 34, 37, 41, 45], "Временные масштабы построены неверно!")

    def test_calc_fluctuation_stds(self):
        fluctuation_stds = self.analyzer.fluctuation_stds
        self.assertEqual(len(fluctuation_stds), 19, "Неверное количество компонент стандартного отклонения суммы значений в окнах!")
    
    def test_calculate_regression_errors(self):
        errs = self.analyzer._calculate_regression_errors()
        self.assertEqual(len(errs), 11, "Неверное количество ошибок!")

    def test_find_change_point(self):
        errs = self.analyzer._calculate_regression_errors()
        cross = self.analyzer._find_change_point(errs)
        self.assertEqual(cross, 7, "Точка пересечения найдена неверно")

    def test_fluctuation_analysis(self):
        _, res_low, res_high = self.analyzer.fluctuation_analysis()

        self.assertEqual(res_low.slope,  0.8993755012909069, 'Наклон линии регрессии для низких временных масштабов не совпадает!')
        self.assertEqual(res_high.slope, 0.8622907299971683, 'Наклон линии регрессии для высоких временных масштабов не совпадает!')
    





