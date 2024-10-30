import unittest

from research.utils.input_reader import InputReader
from research.utils.data_generator import VideoReader, FolderWithVideosReader, FolderWithImagesReader

path_to_tests = 'research/test/inputs/'

class VideoReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        reader_video = InputReader(path_to_tests + 'input_video_file.json')
        video_path = reader_video.get_video_file_path()
        self.video_generator = VideoReader(video_path)
    
    def test_frame_generate(self):
        frames = {}
        generator = self.video_generator.frames_generator()
        for i, frame in enumerate(generator):
            frames[i] = frame
        
        self.assertEqual(len(frames), 100, 'Должны быть прочитаны все кадры')

class FolderWithVideosReaderTest(unittest.TestCase):
    def setUp(self):
        reader = InputReader(path_to_tests + 'input_folder_videos.json')
        folder_path = reader.get_folder_videos_path()
        self.reader_folder_videos = FolderWithVideosReader(folder_path)

    def test_frame_generate(self):
        videos_frames = {}
        
        videos = self.reader_folder_videos.videos
        for i, video in enumerate(videos):
            generator = self.reader_folder_videos.frames_generator(video)
    
            frames = {}
            for j, frame in enumerate(generator):
                frames[j] = frame
            
            videos_frames[i] = frames
        
        self.assertEqual(len(videos_frames), 2, 'Должно быть два прочитанных видео')
        self.assertEqual(len(videos_frames[0]), 100, 'Должны быть прочитаны все кадры')
        self.assertEqual(len(videos_frames[1]), 100, 'Должны быть прочитаны все кадры')

class FolderWithImagesReaderTest(unittest.TestCase):
    def setUp(self):
        reader = InputReader(path_to_tests + 'input_folder_images.json')
        folder_path = reader.get_folder_images_path()
        self.reader_folder_images = FolderWithImagesReader(folder_path)

    def test_frame_generator(self):
        generator = self.reader_folder_images.frames_generator()
        frames = {}
        for i, frame in enumerate(generator):
            frames[i] = frame
        
        self.assertEqual(len(frames), 96, 'Должны быть прочитаны все кадры')
            

if __name__ == '__main__':
    unittest.main()
        

