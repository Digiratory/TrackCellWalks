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
from utils.edges_flow import get_vid_edges_flow

def video_process(input_file: str, 
                  cache_file: str,
                  cache_edges: str, 
                  output_animation_file: str, 
                  output_fluctuation_file: str, 
                  temporal_scales: list):
    """
    Функция обработки видео. 
    
    Args:
        input_file (str): Путь до видео.
        cache_file (str): Путь до кэша видео.
        cache_edges (str): Путь до кэша границ.
        output_file_animation (str): Путь сохранения анимации видео.
        output_fluctuation_file (str): Путь сохранения файла с флуктационной характеристикой.
        temporal_scales (list[int]): Список временных масштабов.
    """

    vs_np, us_np = get_vid_opt_flow(input_file, cache_file)
    edges = get_vid_edges_flow(input_file, cache_edges, win_size=50, win_step=25)[1:]
    
    mask = edges > np.quantile(edges, 0.3)

    vector_field = vs_np + 1j * us_np
    vector_field_std = np.diff(np.var(vector_field, axis=(1,2)))

    std_ = np.std(vector_field_std)
    mean_ = np.mean(vector_field_std)

    candidate_top = np.argwhere(vector_field_std> mean_ + 2*std_).flatten()
    candidate_bottom = np.argwhere(vector_field_std < mean_ - 2*std_).flatten()

    to_remove = set(candidate_top+1) & set(candidate_bottom)

    vector_field = np.delete(vector_field, list(to_remove), 0)
    mask = np.delete(mask, list(to_remove), 0)

    vector_field_std = np.diff(np.var(vector_field, axis=(1,2)))

    std_ = np.std(vector_field_std)
    mean_ = np.mean(vector_field_std)

    vector_field[np.invert(mask)] =  np.nan

    plot_entire_stat_tresh((vs_np.shape[1],vs_np.shape[2]), vs_np, us_np, thresh=0.5)
    
    compl_vars_ = []
    for w_size in tqdm(temporal_scales):
        window = sliding_window_view(vector_field, w_size, axis=0)[::w_size//4, ...]
        H = np.nanstd(np.sum(window, axis=-1))
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
        file_types = ['.avi', '.mp4']
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
        cache_file = cache_path + file_name + '.npz'
        cache_edges = cache_path + file_name + "_edges" + '.npz'
        video_process(input_file=input_file, 
                      cache_edges=cache_edges,
                      cache_file=cache_file, 
                      output_animation_file=output_animation_file, 
                      output_fluctuation_file=output_fluctuation_file, 
                      temporal_scales=temporal_scales)
    else: 
        for ((input_file, cache_file, output_animation_file, output_fluctuation_file), 
             (input_file_edges, cache_edges)) in zip(bacterial_ds_generator(input_dir=input_dir, 
                                                                                 cache_dir=cache_path, 
                                                                                 output_dir=output_path),
                                                          bacterial_ds_generator(input_dir=input_dir, 
                                                                                 cache_dir=cache_path, 
                                                                                 output_dir=output_path, 
                                                                                 cache_suffix="_edges")):
            video_process(input_file=input_file,
                          cache_file=cache_file, 
                          cache_edges= cache_edges,
                          output_animation_file=output_animation_file, 
                          output_fluctuation_file= output_fluctuation_file,
                          temporal_scales=temporal_scales)