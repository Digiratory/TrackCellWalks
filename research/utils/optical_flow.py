import numpy as np
from IPython.display import clear_output
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
import cv2
from cv2 import calcOpticalFlowFarneback
import sys
import pdb

from tqdm.notebook import tqdm
import os
import json

sys.path.append('..')
from utils.data_generator import frame_generator


with open('params.json') as f:
        params = json.load(f)

RECORD_DURATION = params['record_duration']

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
        
            flow = calcOpticalFlowFarneback(prev=image0, 
                                            next=image1,
                                            flow=None,
                                            pyr_scale=0.5,
                                            levels=1,
                                            winsize=15,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=1.2,
                                            flags=0)
            u = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
            v = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
            
            vs.append(v)
            us.append(u)
    return np.array(vs), np.array(us)

def get_vid_opt_flow(input_file, cache_file, start_frame=0, step=1):
    """
    Функция получает оптический поток из видеофайла или кэша.

    :param video_path: str, путь к видеофайлу.
    :param cache_path: str, путь к файлу кэша.
    :param start_frame: int, опциональный параметр, номер первого кадра для чтения (по умолчанию 0).
    :param step: int, опциональный параметр, шаг между кадрами для чтения (по умолчанию 1).
    :return: tuple, компоненты оптического потока: vs (вертикальная компонента) и us (горизонтальная компонента).
    """

    if not os.path.exists(cache_file):
        vs_np, us_np = compute_optical_flow(frame_generator(input_file, 
                                                            blur_sigma=1, 
                                                            start_frame=start_frame, 
                                                            step=step, 
                                                            frame_count=RECORD_DURATION),
                                            gen_length=RECORD_DURATION)
        np.savez(cache_file, vs=vs_np, us=us_np)
    else:
        with np.load(cache_file) as npzfile:
            vs_np = npzfile["vs"]
            us_np = npzfile["us"]
    return vs_np, us_np