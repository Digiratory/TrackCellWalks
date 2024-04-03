from sys import argv
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import math
import sys
import pdb
import getopt
import re

# local imports
sys.path.append('..')
from utils.optical_flow import get_vid_opt_flow, compute_optical_flow, frame_generator
from utils.stats import plot_entire_stat_tresh, analyze_hs, make_animation
from utils.optical_flow import bacterial_ds_generator
from utils.optical_flow import RECORD_DURATION


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('main.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    return inputfile, outputfile
    

if __name__ == '__main__':

    input_file, output_file = main(sys.argv[1:])
    if not output_file:
        output_file = input_file[:-4] + '.mp4'
    cache_file = input_file[:-4] + '.npz'
    input_file_name = re.search(r'[^/\\]+$', input_file).group(0)
      
    base = 1.1
    smin = 8
    L = RECORD_DURATION
    smax = L/2

    S = []
    for degree in range(int(math.log2(smin)/math.log2(base)), int(math.log2(smax)/math.log2(base))):
        new = int(base**degree)
        if not new in S:
            S.append(new)

    vs_np, us_np = get_vid_opt_flow(input_file, cache_file)
    vector_field = vs_np + 1j * us_np

    THRESH = 0.5
    plot_entire_stat_tresh((vs_np.shape[1],vs_np.shape[2]), vs_np, us_np, thresh=THRESH)
    
    compl_vars_ = []
    for w_size in tqdm(S):
        window = sliding_window_view(vector_field, w_size, axis=0)[::w_size//4, ...]
        H = np.std(np.sum(window, axis=-1))
        compl_vars_.append(H)
        
    cross, slope_l, slope_h = analyze_hs(compl_vars_, np.array(S), title=f"H(S): {input_file_name}")

    make_animation(vs_np, us_np, output_file)