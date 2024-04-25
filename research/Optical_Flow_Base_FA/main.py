from sys import argv
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import math
import sys
import pdb
import getopt
import re
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.optical_flow import get_vid_opt_flow
from utils.stats import plot_entire_stat_tresh, analyze_hs, make_animation, compute_temporal_scales
from utils.data_generator import bacterial_ds_generator

def video_process(input_file: str, 
                  cache_file: str, 
                  output_animation_file: str, 
                  output_fluctuation_file: str, 
                  temporal_scales: list):
    """Функция обработки видео. 
    - Вычисление оптического потока
    - Построение графиков
    - Выполнение анализа 
    - Сохранение анимации
    
    :param input_file: путь до видео
    :param cache_file: путь до кэша видео
    :param output_file_animation: путь сохранения анимации видео
    :param output_fluctuation_file: путь сохранения файла с флуктационной характеристикой
    """
    
    vs_np, us_np = get_vid_opt_flow(input_file, cache_file)
    vector_field = vs_np + 1j * us_np

    plot_entire_stat_tresh((vs_np.shape[1],vs_np.shape[2]), vs_np, us_np, thresh=0.5)
    
    vector_field = vs_np + 1j * us_np
    compl_vars_ = []
    for w_size in tqdm(temporal_scales):
        window = sliding_window_view(vector_field, w_size, axis=0)[::w_size//4, ...]
        H = np.std(np.sum(window, axis=-1))
        compl_vars_.append(H)
    
    
    analyze_hs(hs=compl_vars_, 
               S=np.array(temporal_scales), 
               output_file=output_fluctuation_file,
               plot=True,
               title=f"H(S): {input_file}")

    make_animation(vs_np, us_np, output_animation_file)


if __name__ == '__main__':

    project_path = str(Path(__file__).parent.parent.parent) + '/'
    output_path = project_path + 'data/output/'
    cache_path = project_path + 'data/cache/'

    with open('params.json') as f:
        params = json.load(f)
    
    input_type_bool = None
    if os.path.isfile(params['input']):
        input_type_bool = True
        input_file = os.path.abspath(params['input'])
        path_file, ext = os.path.splitext(input_file)
        file_name = os.path.basename(path_file) 
        file_types = ['.avi']
        if ext not in file_types:
            print('The video format is incorrect. Supported:', *file_types)
            sys.exit()
    elif os.path.isdir(params['input']):
        input_type_bool = False
        input_dir = os.path.abspath(params['input']) + '/'
    else:
        print('Invalid input type. Chose video file or path with video files')
        sys.exit()
        
    output_animation_file = params['output_animation']
    
    if output_animation_file is None:
        if input_type_bool:
            output_animation_file = output_path + file_name + '.mp4'
    
    output_fluctuation_file = params['output_fluctuation_characteristic_file']
    if output_fluctuation_file is None:
        if input_type_bool:
            output_fluctuation_file = output_path + file_name + '.csv'

    base = params['base']
    smin = params['smin']
    smax = params['record_duration']/2

    temporal_scales = compute_temporal_scales(base, smin, smax)

    if input_type_bool:
        print('Start video processing', file_name)
        cache_file = cache_path + file_name + '.npz'
        video_process(input_file=input_file, 
                      cache_file=cache_file, 
                      output_animation_file=output_animation_file, 
                      output_fluctuation_file=output_fluctuation_file, 
                      temporal_scales=temporal_scales)
    else: 
        print('Start dir processing')
        for input_file, \
            cache_file, \
            output_animation_file, \
            output_fluctuation_file in bacterial_ds_generator(input_dir=input_dir, 
                                                              cache_dir=cache_path, 
                                                              output_dir=output_path):
            video_process(input_file=input_file,
                          cache_file=cache_file, 
                          output_animation_file=output_animation_file, 
                          output_fluctuation_file= output_fluctuation_file,
                          temporal_scales=temporal_scales)