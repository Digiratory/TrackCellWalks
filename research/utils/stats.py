import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import math

def plot_entire_stat_tresh(shape, vs_np, us_np, title="Sequence image sample", thresh = 0.95):
    """
    Функция для построения графика оптического потока и векторного поля.

    :param shape: tuple, форма изображения (высота, ширина).
    :param vs_np: np.array, массив вертикальных компонент оптического потока.
    :param us_np: np.array, массив горизонтальных компонент оптического потока.
    :param title: str, заголовок графика (по умолчанию "Sequence image sample").
    :param thresh: float, порог для определения среднего значения оптического потока (по умолчанию 0.95).
    """
    v_mean = vs_np.mean(axis=0, where=vs_np>np.quantile(vs_np, thresh))# y direction    
    u_mean = us_np.mean(axis=0, where=us_np>np.quantile(us_np, thresh))# x direction    
    # --- Compute flow magnitude
    norm = np.sqrt(u_mean ** 2 + v_mean ** 2)
    # --- Display
    plt.figure(figsize=(8, 8))
    # --- Quiver plot arguments

    nvec = 25  # Number of vectors to be displayed along each image dimension
    nl, nc = shape
    step = max(nl//nvec, nc//nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u_mean[::step, ::step]
    v_ = v_mean[::step, ::step]

    plt.imshow(norm)
    plt.quiver(x, y, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)
    plt.title("Optical flow magnitude and vector field")
    plt.axis('off')
    plt.tight_layout()

    plt.show()
    
def plot_entire_stat_mask(shape, vs_np, us_np, mask, title="Sequence image sample"):
    v_mean = vs_np.mean(axis=0, where=mask)# y direction    
    u_mean = us_np.mean(axis=0, where=mask)# x direction    
    # --- Compute flow magnitude
    norm = np.sqrt(u_mean ** 2 + v_mean ** 2)
    # --- Display
    plt.figure(figsize=(8, 8))
    # --- Quiver plot arguments

    nvec = 25  # Number of vectors to be displayed along each image dimension
    nl, nc = shape
    step = max(nl//nvec, nc//nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u_mean[::step, ::step]
    v_ = v_mean[::step, ::step]

    plt.imshow(norm)
    plt.quiver(x, y, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)
    plt.title("Optical flow magnitude and vector field")
    plt.axis('off')
    plt.tight_layout()

    plt.show()
    
def analyze_hs(hs, S, plot=True, title="H(S)"):
    errs = []
    for cp in range(4, len(S)-4):
        res1 = stats.linregress(np.log10(S[cp:]), np.log10(hs[cp:]))
        res2 = stats.linregress(np.log10(S[:cp]), np.log10(hs[:cp]))
        print(f"S={S[cp]}; Err={(res1.stderr + res2.stderr):.3f}")
        errs.append(res1.stderr + res2.stderr)
    cross = np.argmin(np.array(errs)) + 4
    res_l = stats.linregress(np.log10(S[:cross]), np.log10(hs[:cross]))
    res_h = stats.linregress(np.log10(S[cross:]), np.log10(hs[cross:]))
    if plot:
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

def make_animation(vs_np, us_np, output):
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
