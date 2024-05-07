import cv2
from scipy.ndimage import gaussian_filter
from glob import glob
import os
import slideio
import numpy as np


def bacterial_ds_generator(input_dir, cache_dir, output_dir):
    """
    Генератор для создания путей к видеофайлам, файлам кэша и анимациям.

    :param input_dir: str, путь к папке с видеофайлами.
    :param cache_dir: str, путь к папке, где будут храниться файлы кэша.
    :param output_dir: str, путь к папке, где будут храниться файлы анимаций.
    :return: tuple, кортеж из двух строк, представляющих путь к видеофайлу и путь к файлу кэша.
    """

    videos = glob(os.path.join(input_dir, f"*.avi"))
    for input_file in videos:
        path_file, _ = os.path.splitext(input_file)
        file_name = os.path.basename(path_file) 
        cache_file = os.path.join(cache_dir, file_name+".npz")
        output_file = os.path.join(output_dir, file_name+".mp4")
        fluctuation_file = os.path.join(output_dir, file_name+".csv")
        yield input_file, cache_file, output_file, fluctuation_file

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