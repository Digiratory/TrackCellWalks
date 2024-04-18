import numpy as np
from IPython.display import clear_output
from skimage.registration import optical_flow_tvl1
import sys

from tqdm.notebook import tqdm
import os

sys.path.append('..')
from utils.data_generator import zvi_based_frame_generator
from utils.data_generator import video_based_frame_generator

RECORD_DURATION = 100

def frame_generator(video_path:str, start_frame=0, step=1, blur_sigma=None, frame_count=np.iinfo(int).max):
    """
    Функция определяет тип видеофайла и вызывает соответствующий генератор кадров.

    :param video_path: str, путь к видеофайлу.
    :param start_frame: int, опциональный параметр, номер первого кадра для чтения.
    :param step: int, опциональный параметр, шаг между кадрами для чтения.
    :param blur_sigma: float, опциональный параметр, стандартное отклонение для размытия изображения (если требуется).
    :param frame_count: int, опциональный параметр, максимальное количество кадров для чтения.
    :return: generator, генератор кадров из видео.
    """
    if video_path.endswith("zvi"):
        return zvi_based_frame_generator(video_path, start_frame, step, blur_sigma, frame_count)
    else:
        return video_based_frame_generator(video_path, start_frame, step, blur_sigma, frame_count)


def compute_optical_flow(generator, radius=None, gen_length=None):
    """
    Функция для вычисления оптического потока.
    :param generator: генератор кадров из видео.
    :param radius: int, радиус для вычисления оптического потока.
    :param gen_length: int, общее количество кадров в генераторе.
    :return: tuple, компоненты оптического потока: vs (вертикальная компонента) и us (горизонтальная компонента).
    """

    vs, us = [], []
    image0 = None
    for frame_np in tqdm(generator, total=gen_length):
        clear_output(wait=True)
        #Build masked Image
        frame_blur = frame_np # cv2.blur(frame_np,(5,5))
        if image0 is None:
            image0 = frame_blur
        else:
            # --- Compute the optical flow
            image1 = frame_blur
            v, u = optical_flow_tvl1(image0, image1)
            image0 = image1.copy()
            vs.append(v)
            us.append(u)
    return np.array(vs), np.array(us)

def get_vid_opt_flow(video_path, cache_path, start_frame=0, step=1):
    """
    Функция получает оптический поток из видеофайла или кэша.

    :param video_path: str, путь к видеофайлу.
    :param cache_path: str, путь к файлу кэша.
    :param start_frame: int, опциональный параметр, номер первого кадра для чтения (по умолчанию 0).
    :param step: int, опциональный параметр, шаг между кадрами для чтения (по умолчанию 1).
    :return: tuple, компоненты оптического потока: vs (вертикальная компонента) и us (горизонтальная компонента).
    """

    if not os.path.exists(cache_path):
        vs_np, us_np = compute_optical_flow(frame_generator(video_path, blur_sigma=1, start_frame=start_frame, step=step, frame_count=RECORD_DURATION)) # 3 sec for performance
        np.savez(cache_path, vs=vs_np, us=us_np)
    else:
        with np.load(cache_path) as npzfile:
            vs_np = npzfile["vs"]
            us_np = npzfile["us"]
    return vs_np, us_np





