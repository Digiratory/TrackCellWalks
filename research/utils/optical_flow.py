# --- Compute the optical flow

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from skimage.registration import optical_flow_tvl1
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from tqdm.notebook import tqdm
import os
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
import math
from glob import glob
import slideio
import pdb

RECORD_DURATION = 100

def video_based_frame_generator(video_path, start_frame=0, step=1, blur_sigma=None, frame_count=np.iinfo(int).max):
    """
    Функция генерирует кадры видео из файлового источника.

    :param video_path: str, путь к видеофайлу.
    :param start_frame: int, опциональный параметр, номер первого кадра для чтения.
    :param step: int, опциональный параметр, шаг между кадрами для чтения.
    :param blur_sigma: float, опциональный параметр, стандартное отклонение для размытия изображения (если требуется).
    :param frame_count: int, опциональный параметр, максимальное количество кадров для чтения.
    :return: generator, генератор кадров из видео.
    """

    cap = cv2.VideoCapture(video_path) 
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    count = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    while True:
        ret, frame = cap.read()
        if not ret or count>= frame_count:
            print("Can't receive frame (stream end?). Exiting ...")
            cap.release()
            break
        grey_np = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur_sigma is not None:
            grey_np = gaussian_filter(grey_np, sigma=blur_sigma)
        count += step 
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        yield grey_np
    cap.release()
    
def zvi_based_frame_generator(video_path, start_frame=0, step=1, blur_sigma=None, frame_count=np.iinfo(int).max):
    """
    Функция генерирует кадры изображений из файла формата ZVI.

    :param video_path: str, путь к файлу ZVI.
    :param start_frame: int, опциональный параметр, номер первого кадра для чтения.
    :param step: int, опциональный параметр, шаг между кадрами для чтения.
    :param blur_sigma: float, опциональный параметр, стандартное отклонение для размытия изображения (если требуется).
    :param frame_count: int, опциональный параметр, максимальное количество кадров для чтения.
    :return: generator, генератор кадров изображений из файла ZVI.
    """

    slide = slideio.open_slide(video_path,"ZVI")
    scene = slide.get_scene(0)
    for frame_i in range(0, frame_count):
        i = frame_i % scene.num_t_frames
        yield scene.read_block(rect=(0,0,0,0), size=(0,0), channel_indices=[0], slices=(0,1), frames=(i,i+1))
    
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

def bacterial_ds_generator(video_folder, cache_folder):
    """
    Генератор для создания путей к видеофайлам и их файлам кэша.

    :param video_folder: str, путь к папке с видеофайлами.
    :param cache_folder: str, путь к папке, где будут храниться файлы кэша.
    :return: tuple, кортеж из двух строк, представляющих путь к видеофайлу и путь к файлу кэша.
    """

    videos = glob(os.path.join(video_folder, f"*.avi"))
    for vid in videos:
        vid_name = os.path.split(vid)[1][:-4]
        cache_path = os.path.join(cache_folder, vid_name+".npz")
        yield vid, cache_path






