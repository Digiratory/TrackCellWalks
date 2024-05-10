import cv2
from scipy.ndimage import gaussian_filter
from glob import glob
import os
import slideio
import numpy as np


def bacterial_ds_generator(input_dir: str, 
                           cache_dir: str, 
                           output_dir: str, 
                           cache_suffix: str = ""):
    """
    Генератор для создания путей к видеофайлам, файлам кэша и анимациям.

    Args:
        input_dir (str): Путь к папке с видеофайлами.
        cache_dir (str): Путь к папке, где будут храниться файлы кэша.
        output_dir (str): Путь к папке, где будут храниться файлы анимаций.
        cache_suffix (str, optional): Суффикс для файлов кэша. Defaults to "".

    Returns:
        tuple: Кортеж из двух строк, представляющих путь к видеофайлу и путь к файлу кэша.
    """

    videos = glob(os.path.join(input_dir, f"*.avi"))
    for input_file in videos:
        path_file, _ = os.path.splitext(input_file)
        file_name = os.path.basename(path_file) 
        cache_file = os.path.join(cache_dir, file_name + cache_suffix + ".npz")
        
        if cache_suffix == "":
            output_file = os.path.join(output_dir, file_name + ".mp4")
            fluctuation_file = os.path.join(output_dir, file_name + ".csv")
            yield input_file, cache_file, output_file, fluctuation_file
        else:
            yield input_file, cache_file

def zvi_based_frame_generator(video_path: str, 
                              start_frame: int = 0, 
                              step: int = 1, 
                              blur_sigma: float = None, 
                              frame_count: int = np.iinfo(int).max):
    """
    Генерирует кадры изображений из файла формата ZVI.

    Args:
        video_path (str): Путь к файлу ZVI.
        start_frame (int, optional): Номер первого кадра для чтения. Defaults to 0.
        step (int, optional): Шаг между кадрами для чтения. Defaults to 1.
        blur_sigma (float, optional): Стандартное отклонение для размытия изображения (если требуется). Defaults to None.
        frame_count (int, optional): Максимальное количество кадров для чтения. Defaults to максимальное значение для int.

    Yields:
        generator: Генератор кадров изображений из файла ZVI.
    """

    slide = slideio.open_slide(video_path,"ZVI")
    scene = slide.get_scene(0)
    for frame_i in range(0, frame_count):
        i = frame_i % scene.num_t_frames
        yield scene.read_block(rect=(0,0,0,0), size=(0,0), channel_indices=[0], slices=(0,1), frames=(i,i+1))


def video_based_frame_generator(video_path: str, 
                                start_frame: int = 0, 
                                step: int = 1, 
                                blur_sigma: float = None, 
                                frame_count: int = np.iinfo(int).max):
    """
    Генерирует кадры видео из файлового источника.

    Args:
        video_path (str): Путь к видеофайлу.
        start_frame (int, optional): Номер первого кадра для чтения. Defaults to 0.
        step (int, optional): Шаг между кадрами для чтения. Defaults to 1.
        blur_sigma (float, optional): Стандартное отклонение для размытия изображения (если требуется). Defaults to None.
        frame_count (int, optional): Максимальное количество кадров для чтения. Defaults to максимальное значение для int.

    Yields:
        generator: Генератор кадров из видео.
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

def frame_generator(video_path: str, 
                    start_frame: int = 0, 
                    step: int = 1, 
                    blur_sigma: float = None, 
                    frame_count: int = np.iinfo(int).max):
    """
    Определяет тип видеофайла и вызывает соответствующий генератор кадров.

    Args:
        video_path (str): Путь к видеофайлу.
        start_frame (int, optional): Номер первого кадра для чтения. Defaults to 0.
        step (int, optional): Шаг между кадрами для чтения. Defaults to 1.
        blur_sigma (float, optional): Стандартное отклонение для размытия изображения (если требуется). Defaults to None.
        frame_count (int, optional): Максимальное количество кадров для чтения. Defaults to максимальное значение для int.

    Returns:
        generator: Генератор кадров из видео.
    """
    if video_path.endswith("zvi"):
        return zvi_based_frame_generator(video_path, start_frame, step, blur_sigma, frame_count)
    else:
        return video_based_frame_generator(video_path, start_frame, step, blur_sigma, frame_count)