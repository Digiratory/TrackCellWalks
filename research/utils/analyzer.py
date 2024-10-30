import math
import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view

from research.utils.optical_flow import ComputeOpticalFlow
from research.utils.input_reader import InputReader

class OpticalFlowAnalyzer:
    def __init__(self, input_reader: InputReader, optical_flow: ComputeOpticalFlow):
        self.base = input_reader.get_base()
        self.smin = input_reader.get_smin()
        self.smax = input_reader.get_record_duration() / 2

        self.optical_flow = optical_flow

        self.temporal_scales = self._compute_temporal_scales()
        self.fluctuation_stds = self._calc_fluctuation_stds()

        self.n_sigms = 3    # количество стандартных отклонений для доверительного интервала

    def _compute_temporal_scales(self) -> list[int]:
        """
        Функция вычисляет временные масштабы для анализа многомерных временных рядов с использованием алгоритма DCCA.

        :param base: float, база логарифма.
        :param smin: int, минимальный размер временного масштаба.
        :param smax: int, максимальный размер временного масштаба.
        :return: список временных масштабов.
        """

        temporal_scales = []
        for degree in range(int(math.log2(self.smin)/math.log2(self.base)), int(math.log2(self.smax)/math.log2(self.base))):
            new = int(self.base**degree)
            
            if new not in temporal_scales:
                temporal_scales.append(new)

        return temporal_scales
    
    def _calc_fluctuation_stds(self):
        fluctuation_stds = []
        vector_field = self.optical_flow.compute_vector_field()

        for window_scale in self.temporal_scales:
            
            window = sliding_window_view(vector_field, window_scale, axis=0)[::window_scale//4, ...]
            
            fluctuation_std = np.nanstd(np.sum(window, axis=-1))

            fluctuation_stds.append(fluctuation_std)
        return fluctuation_stds

    def fluctuation_analysis(self):
        """
        Returns:
            cross (int): Индекс точки изменения режима.
            res_low (object): Результаты регрессионного анализа для низких временных масштабов.
            res_high (object): Результаты регрессионного анализа для высоких временных масштабов.
        """
        errs = self._calculate_regression_errors()

        cross = self._find_change_point(errs)

        res_low, res_high = self._perform_regression_analysis(cross)

        return cross, res_low, res_high

    def _calculate_regression_errors(self):
        errs = []
        for cp in range(4, len(self.temporal_scales)-4):
            res1 = stats.linregress(np.log10(self.temporal_scales[cp:]), np.log10(self.fluctuation_stds[cp:]))
            res2 = stats.linregress(np.log10(self.temporal_scales[:cp]), np.log10(self.fluctuation_stds[:cp]))

            errs.append(res1.stderr + res2.stderr)

        return errs

    def _find_change_point(self, errs):
        cross = np.argmin(np.array(errs)) + 4
        return cross
    
    def _perform_regression_analysis(self, cross):
        res_low = stats.linregress(np.log10(self.temporal_scales[:cross]), np.log10(self.fluctuation_stds[:cross]))
        res_high = stats.linregress(np.log10(self.temporal_scales[cross:]), np.log10(self.fluctuation_stds[cross:]))

        return res_low, res_high

    def calculate_log_regression(self, linear_regression):
        """Вычисление логарифмической регрессии

            Формула: y = a + b * ln(x)
            где y - зависимая переменная, 
                x - независимая переменная, 
                a - перехват,
                b - наклон.

            Params:
                linear_regression - линейная регрессия, рассчитанная по методу наименьших квадратов.
        """

        log_regression = 10**(linear_regression.intercept + linear_regression.slope * np.array(np.log10(self.temporal_scales))) 
        return log_regression
    
    def calculate_boundaries_confidence_interval(self, log_regression, linear_regression, n_sigms=3):
        """Вычисление нижней и верхней границ доверительного интервала
            
            Params:
                n_sigms(int): количество стандартных отклонений для доверительного интервала. Default 3.

        """
        lower_bound =  log_regression - n_sigms * linear_regression.stderr
        upper_bound =  log_regression + n_sigms * linear_regression.stderr

        return lower_bound, upper_bound





