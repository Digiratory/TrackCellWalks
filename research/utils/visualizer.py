import numpy as np
import matplotlib.pyplot as plt

class OpticalFlowVisualizer:
    def __init__(self, height, width, step):
        self.__height = height
        self.__width = width
        self.__step = step

    def plot_optical_flow_magnitude_and_vector_field(self, pixel_offset, selected_horizontal_magnitude, selected_vertical_magnitude):
        y_grid, x_grid = np.mgrid[:self.__height:self.__step, :self.__width:self.__step]

        plt.figure(figsize=(8, 8))

        plt.imshow(pixel_offset)
        
        plt.quiver(
            x_grid, y_grid, 
            selected_horizontal_magnitude, selected_vertical_magnitude, 
            color='r', 
            units='dots',
            angles='xy', 
            scale_units='xy', 
            lw=3
        )
        plt.title("Optical flow magnitude and vector field")
        plt.axis('off')
        plt.tight_layout()

        plt.show()
    
    