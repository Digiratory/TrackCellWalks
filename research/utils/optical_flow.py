# --- Compute the optical flow
from tqdm import tqdm
from IPython.display import clear_output
from skimage.registration import optical_flow_tvl1
import numpy as np



def compute_optical_low(generator, radius=None, gen_length=None):
    vs, us = [], []
    image0 = None
    for frame_np in tqdm(generator, total=gen_length):
        clear_output(wait=True)
        #Build masked Image
        frame_blur = frame_np#cv2.blur(frame_np,(5,5))
        if image0 is None:
            image0 = frame_blur
        else:
            # --- Compute the optical flow
            image1 = frame_blur
            #v, u = optical_flow_ilk(image0, image1, radius=25)
            v, u = optical_flow_tvl1(image0, image1)
            image0 = image1.copy()
            vs.append(v)
            us.append(u)
    return np.array(vs), np.array(us)


