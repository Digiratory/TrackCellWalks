import numpy as np
import matplotlib.pyplot as plt

from research.utils.optical_flow import ComputeOpticalFlow 
from research.utils.analyzer import OpticalFlowAnalyzer


class OpticalFlowVisualizer:
    def __init__(self, ComputeOpticalFlowObj: ComputeOpticalFlow, analyzer: OpticalFlowAnalyzer):
        self.ComputeOpticalFlowObj = ComputeOpticalFlowObj
        self.analyzer = analyzer

        self.__height = self.ComputeOpticalFlowObj.get_image_height()
        self.__width = self.ComputeOpticalFlowObj.get_image_width()

        self.__step_between_vectors = self.ComputeOpticalFlowObj.compute_step_between_vectors()

    def plot_optical_flow_magnitude_and_vector_field(self):
        pixel_offset = self.ComputeOpticalFlowObj.compute_pixel_offset()

        selected_horizontal_magnitude = self.ComputeOpticalFlowObj.downsample_horizontal_magnitude()
        selected_vertical_magnitude = self.ComputeOpticalFlowObj.downsample_vertical_magnitude()

        y_grid, x_grid = np.mgrid[:self.__height:self.__step_between_vectors, :self.__width:self.__step_between_vectors]

        plt.figure(figsize=(8, 8))

        plt.imshow(pixel_offset)
        
        plt.quiver(
            x_grid, y_grid, 
            selected_horizontal_magnitude, selected_vertical_magnitude, 
            color='r', 
            units='dots',
            angles='xy', 
            scale_units='xy', 
            lw=3
        )
        plt.title("Optical flow magnitude and vector field")
        plt.axis('off')
        plt.tight_layout()

        plt.show()

    def plot_fluctuation_analysis(self, title="H(S)"):
        
        plt.figure()
        plt.title(title)

        cross, linear_regr_low, linear_regr_high = self.analyzer.fluctuation_analysis()

        print(
            f"Optimal temporal scale (S) = {self.analyzer.temporal_scales[cross]};\n"
            f"Regression slope for low temporal scales (H_l(S)) = {linear_regr_low.slope:.5f};\n"
            f"Regression slope for high temporal scales (H_h(S)) = {linear_regr_high.slope:.5f}"
        )

        self._plot_low_log_regression(
            linear_regression=linear_regr_low,
            cross=cross
        )

        self._plot_high_log_regression(
            linear_regression=linear_regr_high,
            cross=cross
        )

        plt.legend()
        plt.axvline(self.analyzer.temporal_scales[cross])
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='both')
        plt.plot()
        plt.show()

    def _plot_low_log_regression(self, linear_regression, cross):
        """Построение логарифмической регрессии низких временных рядов"""

        temporal_scales = self.analyzer.temporal_scales
        fluctuation_stds = self.analyzer.fluctuation_stds
        n_sigms = self.analyzer.n_sigms

        log_regression = self.analyzer.calculate_log_regression(linear_regression)
        
        lower_bound, upper_bound = self.analyzer.calculate_boundaries_confidence_interval(
            log_regression=log_regression, 
            linear_regression=linear_regression
        )

        plt.plot(temporal_scales[:cross+1], fluctuation_stds[:cross+1], 'o', color='blue', label=f"$H_l(S)$")   # Точки данных для низких временных масштабов.

        plt.plot(temporal_scales, log_regression, lw=1, label=f'$H_l(S) = {linear_regression.slope:.2f}, S<{temporal_scales[cross]}$', color='blue', ls='--') # Регрессия для низких временных рядов

        plt.fill_between(temporal_scales, lower_bound, upper_bound, facecolor='blue', alpha=0.25,
                label=f'${n_sigms} \sigma(H_l(S))$')  # Заполнение области между верхней и нижней границами для низких временных масштабов.
        
    def _plot_high_log_regression(self, linear_regression, cross):
        """Построение логарифмической регрессии высоких временных рядов"""
        temporal_scales = self.analyzer.temporal_scales
        fluctuation_stds = self.analyzer.fluctuation_stds
        n_sigms = self.analyzer.n_sigms

        log_regression = self.analyzer.calculate_log_regression(linear_regression)
        
        lower_bound, upper_bound = self.analyzer.calculate_boundaries_confidence_interval(
            log_regression=log_regression, 
            linear_regression=linear_regression
        )

        plt.plot(temporal_scales[cross:], fluctuation_stds[cross:], 'o', color='red', label=f"$H_h(S)$")

        plt.plot(temporal_scales, log_regression, lw=1, label=f'$H_h(S) = {linear_regression.slope:.2f}, S>={temporal_scales[cross]}$', color='red', ls='--')

        plt.fill_between(temporal_scales, lower_bound, upper_bound, facecolor='red', alpha=0.25,
                label=f'${n_sigms} \sigma(H_l(S))$')
        
        
    
    