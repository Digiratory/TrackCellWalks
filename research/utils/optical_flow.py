import numpy as np
import cv2
from cv2 import calcOpticalFlowFarneback
from skimage.registration import optical_flow_tvl1 
import sys

sys.path.append('..')

class ComputeOpticalFlow:
    def __init__(self, generator, cache_path):
        self.__generator = generator
        self.__cache_path:str = cache_path
        
        self.__vertical_components = []
        self.__horizontal_components = []

    def compute_optical_flow_by_Farneback(self):
        image_previous = next(self.__generator)
        for frame in self.__generator:
            image_next = frame

            flow = calcOpticalFlowFarneback(
                prev=image_previous, 
                next=image_next,
                flow=None,
                pyr_scale=0.5,
                levels=1,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            horizontal_component = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
            vertical_component = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)

            image_previous = image_next.copy()
            
            self.__horizontal_components.append(horizontal_component)
            self.__vertical_components.append(vertical_component)

        self._save_to_cache()

        return np.array(self.__vertical_components), np.array(self.__horizontal_components)

    def _save_to_cache(self):
        np.savez(
            self.__cache_path, 
            vertical_components=self.__vertical_components, 
            horizontal_components=self.__horizontal_components
        )

    def compute_optical_flow_by_tvl1(self):
        image_previous = next(self.__generator)
        for frame in self.__generator:
            image_next = frame
            
            horizontal_component, vertical_component = optical_flow_tvl1(image_previous, image_next)

            image_previous = image_next.copy()

            self.__horizontal_components.append(horizontal_component)
            self.__vertical_components.append(vertical_component)

        self._save_to_cache()

        return np.array(self.__vertical_components), np.array(self.__horizontal_components)

    def load_optical_flow_from_cahce(self):
        with np.load(self.__cache_path) as npzfile:
                self.__vertical_components = npzfile["vertical_components"]
                self.__horizontal_components = npzfile["horizontal_components"]

        return self.__vertical_components, self.__horizontal_components
    
if __name__ == '__main__':
    # TO-DO using algo
    pass