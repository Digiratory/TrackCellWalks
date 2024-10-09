import numpy as np
import cv2
from cv2 import calcOpticalFlowFarneback
from skimage.registration import optical_flow_tvl1 

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
    
    def compute_pixel_offset(self): 
        vertical_magnitude = self.compute_vertical_magnitude()
        horizontal_mangnitude = self.compute_horizontal_magnitude()

        pixel_offset = np.sqrt(vertical_magnitude ** 2 + horizontal_mangnitude ** 2)

        return pixel_offset
    
    def compute_vertical_magnitude(self):
        vertical_magnitude = self._compute_mean_components(self.__vertical_components)
        return vertical_magnitude
    
    def _compute_mean_components(self, components):
        components_quantiled = np.quantile(components, self.__thresh)
        mean = components.mean(
            axis=0, 
            where=components > components_quantiled
        )
        return mean
    
    def compute_horizontal_magnitude(self):
        horizontal_magnitude = self._compute_mean_components(self.__horizontal_components)
        return horizontal_magnitude
    
    def select_vertical_magnitude(self):
        step = self.compute_step_between_vectors()
        vertical_mangintude = self.compute_vertical_magnitude()
        selected_vertical_magnitude = vertical_mangintude[::step, ::step]
        return selected_vertical_magnitude

    def compute_step_between_vectors(self):
        height = self.get_image_height()
        width = self.get_image_width()
        step = max(height // self.__num_displayed_vectors, width // self.__num_displayed_vectors)
        return step
    
    def get_image_height(self):
        height = self.__horizontal_components.shape[1]
        return height
    
    def get_image_width(self):
        width = self.__horizontal_components.shape[2]
        return width

    def select_horizontal_magnitude(self):
        step = self.compute_step_between_vectors()
        horiznotal_magnitude = self.compute_horizontal_magnitude()
        selected_horizontal_magnitude = horiznotal_magnitude[::step, ::step]
        return selected_horizontal_magnitude

    def compute_vector_field(self):
        return self.__vertical_components + 1j * self.__horizontal_components
        