import cv2
from scipy.ndimage import gaussian_filter
from typing import Generator
import numpy as np
import os

class ReaderConfig:
    def __init__(self, frame_count:int=100, start_frame:int=0 , step:int=1, blur_sigma:float=None):
        self.frame_count = frame_count
        self.start_frame = start_frame
        self.step = step
        self.blur_sigma = blur_sigma

class VideoReader(ReaderConfig):
    def __init__(self, video_path: str) -> None:
        super().__init__()
        self.video_path: str = video_path
        self.cap: cv2.VideoCapture = None

    def frames_generator(self):
        """Функция возвращает генератор кадров видео."""

        self._open_video()

        self._setStartFrameInVideo(self.start_frame)
        
        for next_frame in range(self.start_frame + 1, self.frame_count + 1, self.step):
            nextFrameExists, frame = self.cap.read()
            
            if not nextFrameExists:
                self._close_video()
                break

            gray_frame = self._frameToGrayColor(frame)

            gray_frame = self._noiseReduction(gray_frame)

            self._setStartFrameInVideo(next_frame)

            yield gray_frame
        self._close_video()

    def _open_video(self) -> None:    
        self.cap = cv2.VideoCapture(self.video_path)
        if not self._isVideoOppend():
            raise Exception("Open video error!")
        
    def _isVideoOppend(self) -> bool:
        if self.cap.isOpened():
            return True
        return False
        
    def _setStartFrameInVideo(self, frame) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def _close_video(self) -> None:
        self.cap.release()
    
    def _frameToGrayColor(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

    def _noiseReduction(self, grey_frame) -> np.array:
        if self.blur_sigma is not None:
            grey_frame = gaussian_filter(grey_frame, sigma=self.blur_sigma)
        return grey_frame

class FolderWithVideosReader(ReaderConfig):
    def __init__(self, folder_path:str) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.videos = os.scandir(self.folder_path)

    def frames_generator(self, video):
        video_reader = VideoReader(video.path)
        return video_reader.frames_generator()
        
class FolderWithImagesReader(ReaderConfig):
    def __init__(self, folder_path:str) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.images = os.scandir(self.folder_path)

    def frames_generator(self):
        for i, image in enumerate(self.images):
            if i + 1 >= self.frame_count:   # i+1, так как enumerate считает с нуля
                break
            grey_frame = cv2.imread(image.path, cv2.COLOR_BGR2GRAY)
            grey_frame = self._noiseReduction(grey_frame)
            yield grey_frame

    def _noiseReduction(self, grey_frame) -> np.array:
        if self.blur_sigma is not None:
            grey_frame = gaussian_filter(grey_frame, sigma=self.blur_sigma)
        return grey_frame