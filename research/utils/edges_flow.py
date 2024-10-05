import numpy as np
import cv2
from tqdm import tqdm
from IPython.display import clear_output
import os
import json
import sys

sys.path.append('..')
from utils.data_generator import frame_generator

with open('params.json') as f:
        params = json.load(f)

RECORD_DURATION = params['record_duration']

def mean_window(data: np.array, axis: int) -> np.array:
    """
    Вычисляет среднее значение по заданной оси.

    Args:
        data (np.array): Входные данные.
        axis (int): Ось, по которой вычисляется среднее значение.

    Returns:
        np.array: Массив средних значений.

    """
    res = np.sum(data, axis=axis)
    return res


def std_window(data: np.array, axis: int) -> np.array:
    """
    Вычисляет стандартное отклонение по заданной оси.

    Args:
        data (np.array): Входные данные.
        axis (int): Ось, по которой вычисляется стандартное отклонение.

    Returns:
        np.array: Массив стандартных отклонений.

    """
    res = np.std(data, axis=axis)
    return res


def roll(array: np.array, win_shape: tuple, dx: int = 1, dy: int = 1) -> np.array:
    """
    Выполняет скользящее окно для массива.

    Args:
        array (np.array): Исходный массив.
        win_shape (tuple): Размеры окна скольжения.
        dx (int, optional): Горизонтальный шаг. Defaults to 1.
        dy (int, optional): Вертикальный шаг. Defaults to 1.

    Returns:
        np.array: Результат скользящего окна.

    """

    shape = array.shape[:-2] + \
        ((array.shape[-2] - win_shape[-2]) // dy + 1,) + \
        ((array.shape[-1] - win_shape[-1]) // dx + 1,) + \
        win_shape  # sausage-like shape with 2D cross-section
    strides = array.strides[:-2] + \
        (array.strides[-2] * dy,) + \
        (array.strides[-1] * dx,) + \
        array.strides[-2:]
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def sliding_window(data: np.array, win_shape: tuple, fcn, dx: int = 1, dy: int = 1) -> np.array:
    """
    Выполняет скользящее окно для массива данных.

    Args:
        data (np.array): Входные данные.
        win_shape (tuple): Форма окна.
        fcn: Функция, применяемая к окну.
        dx (int, optional): Горизонтальный шаг. Defaults to 1.
        dy (int, optional): Вертикальный шаг. Defaults to 1.

    Returns:
        np.array: Результат скользящего окна.

    """

    n = data.ndim  # number of dimensions
    # np.all over 2 dimensions of the rolling 2D window for 4D array
    result = fcn(roll(data, win_shape, dx, dy), axis=(n, n+1))
    return result


def tile_array(array: np.array, b0: int, b1: int) -> np.array:
    """
    Создает "плитку" массива изображения.

    Args:
        array (np.array): Исходный массив.
        b0 (int): Размер по первой оси.
        b1 (int): Размер по второй оси.

    Returns:
        np.array: Результат "плитки" массива.

    """
    r, c = array.shape                                    # number of rows/columns
    rs, cs = array.strides                                # row/column strides
    # view a as larger 4D array
    x = np.lib.stride_tricks.as_strided(array, (r, b0, c, b1), (rs, 0, cs, 0))
    return x.reshape(r*b0, c*b1)                      # create new 2D array


def edge_density(img_np: np.array,
                 win_size: int,
                 win_step: int = 10) -> np.array:
    """Method is implemented algorithm for local edge density estimation, 
    proposed in "Sinitca, A. M., Kayumov, A. R., Zelenikhin, P. V., 
    Porfiriev, A. G., Kaplun, D. I., & Bogachev, M. I. (2023). 
    Segmentation of patchy areas in biomedical images based on local edge
    density estimation. Biomedical Signal Processing and Control, 79, 104189."

    https://www.sciencedirect.com/science/article/abs/pii/S1746809422006437

    Args:
        img_np (np.array): Gray scale image
        win_size (int): Size of averaging windows
        win_step (int, optional): Step for windows sliding. Defaults to 10.
        canny_1 (float, optional): 1st threshold for canny. Defaults to 41.
        canny_2 (float, optional): 2nd threshold for canny. Defaults to 207.

    Returns:
        np.array: Edge dencity map
    """
    dxy = win_step
    ddepth = cv2.CV_16S
    mid = cv2.Sobel(img_np, ddepth, 1, 1, ksize=5)
    mid = cv2.convertScaleAbs(mid)
    
    result = sliding_window(mid, (win_size, win_size),
                            mean_window, dx=dxy, dy=dxy) // ((win_size*win_size))
    result = tile_array(result, dxy, dxy)
    
    h_pad = img_np.shape[0] - result.shape[0]
    w_pad = img_np.shape[1] - result.shape[1]
    result = np.pad(result, ((
        h_pad//2, h_pad//2+img_np.shape[0] % 2), (w_pad//2, w_pad//2+img_np.shape[1] % 2)), 'edge')

    
    return result

def compute_edges_flow(generator, win_size: int, win_step: int = 10) -> np.array:
    """
    Вычисляет плотность границ для каждого кадра видео.

    Args:
        generator: Генератор кадров из видео.
        win_size (int): Размер окна усреднения.
        win_step (int, optional): Шаг скольжения окна. Defaults to 10.

    Returns:
        np.array: Массив, содержащий плотность границ для каждого кадра видео.

    """

    edges = []
    image0 = None
    for frame_np in tqdm(generator):
        clear_output(wait=True)
        #Build masked Image
        edge_density_np = edge_density(frame_np, win_size, win_step)
        edges.append(edge_density_np)
        # break
    return np.array(edges)

def get_vid_edges_flow(video_path: str, 
                       cache_path: str, 
                       start_frame: int = 0, 
                       step: int = 1, 
                       win_size: int = 50, 
                       win_step: int = 10) -> np.array:
    """
    Получает плотность границ из видеофайла или кэша.

    Args:
        video_path (str): Путь к видеофайлу.
        cache_path (str): Путь к файлу кэша.
        start_frame (int, optional): Номер первого кадра для чтения. Defaults to 0.
        step (int, optional): Шаг между кадрами для чтения. Defaults to 1.
        win_size (int, optional): Размер окна усреднения. Defaults to 50.
        win_step (int, optional): Шаг скольжения окна. Defaults to 10.

    Returns:
        np.array: Массив, содержащий плотность границ для каждого кадра видео.

    """
    if not os.path.exists(cache_path):
        edges = compute_edges_flow(frame_generator(video_path, 
                                                   start_frame=start_frame, 
                                                   step=step, 
                                                   frame_count=RECORD_DURATION),
                                         win_size,
                                         win_step = 10) 
        np.savez(cache_path, edges=edges)
    else:
        with np.load(cache_path) as npzfile:
            edges = npzfile["edges"]
    return edges