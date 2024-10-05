import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import math

class OpticalFlowAnalyzer:
    def __init__(self, horizontal_components:np.array, vertical_components:np.array, num_displayed_vectors:int=25):
        self.__horizontal_components:np.array = horizontal_components
        self.__vertical_components:np.array = vertical_components

        self.__thresh:float = 0.5

        self.__num_displayed_vectors = num_displayed_vectors

    # def _compute_vector_field(self) -> np.array:
    #     vector_field = self.__vertical_components + 1j * self.__horizontal_components
    #     return vector_field
    
    def compute_pixel_offset(self): 
        vertical_magnitude = self.compute_vertical_magnitude()
        horizontal_mangnitude = self.compute_horizontal_magnitude()

        pixel_offset = np.sqrt(vertical_magnitude ** 2 + horizontal_mangnitude ** 2)

        return pixel_offset
    
    def compute_vertical_magnitude(self):
        vertical_magnitude = self._compute_mean_components(self.__vertical_components)
        return vertical_magnitude
    
    def _compute_mean_components(self, components):
        components_quantiled = np.quantile(components, self.__thresh)
        mean = components.mean(
            axis=0, 
            where=components > components_quantiled
        )
        return mean
    
    def compute_horizontal_magnitude(self):
        horizontal_magnitude = self._compute_mean_components(self.__horizontal_components)
        return horizontal_magnitude
    
    def select_vertical_magnitude(self):
        step = self.compute_step_between_vectors()
        vertical_mangintude = self.compute_vertical_magnitude()
        selected_vertical_magnitude = vertical_mangintude[::step, ::step]
        return selected_vertical_magnitude

    def compute_step_between_vectors(self):
        height = self.get_image_height()
        width = self.get_image_width()
        step = max(height//self.__num_displayed_vectors, width//self.__num_displayed_vectors)
        return step
    
    def get_image_height(self):
        height = self.__horizontal_components.shape[1]
        return height
    
    def get_image_width(self):
        width = self.__horizontal_components.shape[2]
        return width

    def select_horizontal_magnitude(self):
        step = self.compute_step_between_vectors()
        horiznotal_magnitude = self.compute_horizontal_magnitude()
        selected_horizontal_magnitude = horiznotal_magnitude[::step, ::step]
        return selected_horizontal_magnitude

    def analyze_std_of_sum_sliding_window(self, std_of_sum_sliding_window, temporal_scales, output_file, plot=True, title="H(S)") -> tuple:
        """
        Функция для анализа масштабирования флуктуаций сигнала.

        Args:
            hs (list): Массив значений флуктуации.
            S (list[int]): Массив временных масштабов.
            output_file (str): Путь файла с результатами анализа.
            plot (bool, optional): Генерировать и отображать график. Defaults to True.
            title (str, optional): Заголовок графика. Defaults to "H(S)". 
        
        Returns:
            tuple: Кортеж, содержащий индекс перекреста, наклон низкочастотного режима масштабирования и 
                наклон высокочастотного режима масштабирования

        """
        
        regression_errs = []
        for separation_point in range(4, len(temporal_scales)-4):
            
            res_before_separation_point = stats.linregress(np.log10(temporal_scales[:separation_point]), np.log10(std_of_sum_sliding_window[:separation_point]))
            res_after_separation_point = stats.linregress(np.log10(temporal_scales[separation_point:]), np.log10(std_of_sum_sliding_window[separation_point:]))
            
            print(f"S={temporal_scales[separation_point]}; Err={(res_before_separation_point.stderr + res_after_separation_point.stderr):.3f}")
            
            regression_errs.append(res_before_separation_point.stderr + res_after_separation_point.stderr)

        optimal_separation_point = np.argmin(np.array(regression_errs)) + 4

        res_before_optimal_separation_point = stats.linregress(np.log10(temporal_scales[optimal_separation_point:]), np.log10(std_of_sum_sliding_window[optimal_separation_point:]))
        res_after_optimal_separation_point = stats.linregress(np.log10(temporal_scales[:optimal_separation_point]), np.log10(std_of_sum_sliding_window[:optimal_separation_point]))

    # def save(self, path_to_save='.'):
    #     output_path = os.path.join(path_to_save, 'fluctuation.csv')
    #     with open(output_path, 'w') as f:
    #         f.write("Time Scale,Fluctuation\n")
    #         for t, e in zip(temporal_scales, regression_errs):
    #             f.write(f"{t},{e}\n")

    def plot(self):
        n_sigm = 3
        print(f"Opt S = {S[cross]}; H_l(S) = {res_l.slope}; H_h(S) = {res_h.slope}")
        plt.figure()
        plt.title(title)
        print(res_l)
        regr_l =  10**(res_l.intercept+res_l.slope*np.array(np.log10(S)))
        regr_l_upper_bound =  10**(res_l.intercept+res_l.slope*np.array(np.log10(S))+n_sigm*res_l.stderr)
        regr_l_lower_bound =  10**(res_l.intercept+res_l.slope*np.array(np.log10(S))-n_sigm*res_l.stderr)
        plt.plot(S[:cross+1], hs[:cross+1], 'o', color='blue', label=f"$H_l(S)$")
        plt.plot(S, regr_l, lw=1, label=f'$H_l(S) = {res_l.slope:.2f}, S<{S[cross]}$', color='blue', ls='--')
        plt.fill_between(S, regr_l_lower_bound, regr_l_upper_bound, facecolor='blue', alpha=0.25,
                label=f'${n_sigm} \sigma(H_l(S))$')
        
        
        regr_h =  10**(res_h.intercept+res_h.slope*np.array(np.log10(S)))
        regr_h_upper_bound =  10**(res_h.intercept+res_h.slope*np.array(np.log10(S))+n_sigm*res_h.stderr)
        regr_h_lower_bound =  10**(res_h.intercept+res_h.slope*np.array(np.log10(S))-n_sigm*res_h.stderr)
        plt.plot(S[cross:], hs[cross:], 'o', color='red', label=f"$H_h(S)$")
        plt.plot(S, regr_h, lw=1, label=f'$H_h(S) = {res_h.slope:.2f}, S>={S[cross]}$', color='red', ls='--')
        plt.fill_between(S, regr_h_lower_bound, regr_h_upper_bound, facecolor='red', alpha=0.25,
                label=f'${n_sigm} \sigma(H_l(S))$')
        
        plt.legend()
        plt.axvline(S[cross])
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which='both')
        plt.plot()

    return cross, res_l.slope, res_h.slope

    def make_animation(vs_np: np.array, us_np: np.array, output: str):
        """
        Создает анимацию оптического потока и векторного поля.

        Args:
            vs_np (np.array): Массив вертикальных компонент оптического потока.
            us_np (np.array): Массив горизонтальных компонент оптического потока.
            output (str): Путь к файлу для сохранения анимации.
        """
        nl, nc = vs_np.shape[1:]
        nvec = 25  # Number of vectors to be displayed along each image dimension
        step = max(nl//nvec, nc//nvec)

        low_perc = 0.1
        hig_perc = 0.9
        v_05 = np.quantile(vs_np, low_perc)# y direction    
        v_95 = np.quantile(vs_np, hig_perc)# y direction    
        u_05 = np.quantile(us_np, low_perc)# x direction
        u_95 = np.quantile(us_np, hig_perc)# x direction
        # --- Compute flow magnitude
        magn_05 = np.sqrt(v_05 ** 2 + u_05 ** 2)
        magn_95 = np.sqrt(v_95 ** 2 + u_95 ** 2)

        
        plt.style.use("ggplot")
        fig = plt.figure(figsize=(10,15))
        plt.axis('off')
        writervideo = FFMpegWriter(fps=10) 
        with writervideo.saving(fig, output, 100):
            for v_np, u_np in tqdm(zip(vs_np, us_np)):
                fig.clf()
                # --- Display
                # --- Quiver plot arguments
                y, x = np.mgrid[:nl:step, :nc:step]
                u_ = u_np[::step, ::step]
                v_ = v_np[::step, ::step]

                norm = np.sqrt(v_np ** 2 + u_np ** 2)
                fig.gca().imshow(norm, vmin=magn_05, vmax=magn_95, cmap="jet")
                fig.gca().quiver(x, y, u_, v_, color='r', units='dots',
                        angles='xy', scale_units='xy', lw=3)
                writervideo.grab_frame()

    def compute_temporal_scales(base: float, smin: float, smax: float) -> list[int]:
        """
        Вычисляет временные масштабы для анализа многомерных временных рядов.

        Args:
            base (float): База логарифма.
            smin (float): Минимальный размер временного масштаба.
            smax (float): Максимальный размер временного масштаба.

        Returns:
            list[int]: Список временных масштабов.
        """

        temporal_scales = []
        for degree in range(int(math.log2(smin)/math.log2(base)), int(math.log2(smax)/math.log2(base))):
            new = int(base**degree)
            if new not in temporal_scales:
                temporal_scales.append(new)
        return temporal_scales

