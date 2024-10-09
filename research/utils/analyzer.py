import math
import numpy as np
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view

from research.utils.optical_flow import ComputeOpticalFlow
from research.utils.input_reader import InputReader

class OpticalFlowAnalyzer:
    def __init__(self, InputReaderObj: InputReader, ComputeOpticalFlowObj: ComputeOpticalFlow):
        self.InputReaderObj = InputReaderObj
        self.ComputeOpticalFlowObj = ComputeOpticalFlowObj

    

    
    def analyze_hs(self, hs, title="H(S)"):
        S = self.compute_temporal_scales()
    
        errs = []
        for cp in range(4, len(S)-4):
            res1 = stats.linregress(np.log10(S[cp:]), np.log10(hs[cp:]))
            res2 = stats.linregress(np.log10(S[:cp]), np.log10(hs[:cp]))

            print(f"S={S[cp]}; Err={(res1.stderr + res2.stderr):.3f}")

            errs.append(res1.stderr + res2.stderr)

        cross = np.argmin(np.array(errs)) + 4

        res_l = stats.linregress(np.log10(S[:cross]), np.log10(hs[:cross]))
        res_h = stats.linregress(np.log10(S[cross:]), np.log10(hs[cross:]))
        
        with open(output_file, 'w') as f:
            f.write("Time Scale,Fluctuation\n")
            for t, e in zip(S, errs):
                f.write(f"{t},{e}\n")
        return cross, res_l.slope, res_h.slope

    def compute_temporal_scales(self, base: float, smin: float, smax: float) -> list[int]:
        """
        Функция вычисляет временные масштабы для анализа многомерных временных рядов с использованием алгоритма DCCA.

        :param base: float, база логарифма.
        :param smin: int, минимальный размер временного масштаба.
        :param smax: int, максимальный размер временного масштаба.
        :return: список временных масштабов.
        """

        temporal_scales = []
        for degree in range(int(math.log2(smin)/math.log2(base)), int(math.log2(smax)/math.log2(base))):
            new = int(base**degree)
            if new not in temporal_scales:
                temporal_scales.append(new)
        return temporal_scales
