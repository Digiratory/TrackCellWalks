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

# local imports
sys.path.append('..')
from utils.optical_flow import get_vid_opt_flow
from utils.stats import plot_entire_stat_tresh, analyze_hs, make_animation, compute_temporal_scales
from utils.data_generator import bacterial_ds_generator
from utils.optical_flow import RECORD_DURATION


def main(argv):
    input_file = None
    output_file = None
    input_dir = None
    output_dir = None
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile=", "idir=", "odir="])
    except getopt.GetoptError:
        print ('main.py -i <inputfile> -o <outputfile>, -d <inputdir>, -od <outputdir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -i <inputfile> -o <outputfile>, -d <inputdir>, -od <outputdir>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in("-id", "--idir"):
            input_dir = arg
        elif opt in("-od", "--odir"):
            output_dir = arg
    return input_file, output_file, input_dir, output_dir

def video_process(input_file, cache_file, output_file):
    file_name, _ = os.path.splitext(input_file)
    
    vs_np, us_np = get_vid_opt_flow(input_file, cache_file)
    vector_field = vs_np + 1j * us_np

    plot_entire_stat_tresh((vs_np.shape[1],vs_np.shape[2]), vs_np, us_np, thresh=0.5)
    
    vector_field = vs_np + 1j * us_np
    compl_vars_ = []
    for w_size in tqdm(temporal_scales):
        window = sliding_window_view(vector_field, w_size, axis=0)[::w_size//4, ...]
        H = np.std(np.sum(window, axis=-1))
        compl_vars_.append(H)
        
    analyze_hs(compl_vars_, np.array(temporal_scales), title=f"H(S): {file_name}")

    make_animation(vs_np, us_np, output_file)

if __name__ == '__main__':

    input_file, output_file, input_dir, output_dir = main(sys.argv[1:])

    if (input_file is None) and (input_dir is None):
        print("The input file is missing, plesase use -i or -d param")
    else:
        base = 1.1
        smin = 8
        smax = RECORD_DURATION/2

        temporal_scales = compute_temporal_scales(base, smin, smax)

        if input_dir is None:
            file_name, _ = os.path.splitext(input_file)
            if output_file is None:
                output_file = file_name + '.mp4'
            cache_file = file_name + '.npz'
            video_process(input_file, cache_file, output_file)
        else:
            if output_dir is None: 
                output_dir = input_dir

            for video_file_, cache_file_, ouput_file_ in bacterial_ds_generator(video_folder=input_dir, cache_folder=input_dir, output_folder=output_dir):
                video_process(video_file_, cache_file_, output_file)
    

    