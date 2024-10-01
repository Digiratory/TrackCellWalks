import os
import json

class InputReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

        self.__file_content = self._read_file_content()

        self.__validator = InputValidator(self.__file_content)

    def _read_file_content(self):
        with open(self.file_path) as f:
            params = json.load(f)
        return params

    def get_input_path(self):
        path_input = self.__file_content['input']
        return path_input

    def get_record_duration(self) -> int:
        return self.__file_content['record_duration']

    def get_base(self) -> int:        
        return self.__file_content['base']

    def get_smin(self) -> int:        
        return self.__file_content['smin']
    
    def get_cache_path(self):
        input_path_name = self._getInputPathName()
        ext = '.npz'
        cache_path = 'data/cache/' + input_path_name + ext
        return cache_path

    def _getInputPathName(self):
        input_path = self.get_input_path()
        name_with_ext = os.path.basename(input_path)
        name, _ = os.path.splitext(name_with_ext)
        return name

    def get_result_path(self):
        input_path_name = self._getInputPathName()
        ext = '.mp4'
        cache_path = 'data/results/' + input_path_name + ext
        return cache_path
    
    def get_video_file_path(self):
        self.__validator.validate_video_file()
        return self.__file_content['input']

    def get_folder_videos_path(self):
        self.__validator.validate_folder_videos()
        return self.__file_content['input']

    def get_folder_images_path(self):
        self.__validator.validate_folder_images()
        return self.__file_content['input']

    def _isInputVideoFile(self) -> bool:
        input_path = self.get_input_path()
        if (os.path.isfile(input_path)):
            _, ext = os.path.splitext(input_path)
            if ext in ['.mp4', '.avi']:
                return True
        return False
    
    def _isInputFolderWithImages(self) -> bool:
        input_path = self.get_input_path()
        if (os.path.isdir(input_path)):
            files = os.listdir(input_path)
            _, ext = os.path.splitext(files[0])
            if ext in []: # TODO добавить разрешения изображений
                return True
        return False

    def _isInputFolderWithVideos(self) -> bool:
        input_path = self.get_input_path()
        if (os.path.isdir(input_path)):
            files = os.listdir(input_path)
            _, ext = os.path.splitext(files[0])
            if ext in ['.mp4', '.avi']:
                return True
        return False
    
class InputValidator:
    def __init__(self, file_content:dict):
        self.__file_content = file_content
    
        self.__file_structure = ['input', 'record_duration', 'base', 'smin']

        self.__video_exts = ['.mp4', '.avi']
        self.__image_exts = ['.jpg']

        self._validate_file_structure()

    def _validate_file_structure(self) -> None:
        for key in self.__file_structure:
            self._isKeyInFileContent(key)

    def _isKeyInFileContent(self, key) -> None:
        if key not in self.__file_content.keys():
                raise KeyError(f"The input data must contain a key '{key}'")
        
    def validate_video_file(self):
        file = self.__file_content['input']
        
        if os.path.isfile(file):
            _, ext = os.path.splitext(file)
            self._checkExtInSupportedExts(ext, self.__video_exts)
        else:
            raise KeyError('The path is not a file')
    
    def _checkExtInSupportedExts(self, ext, exts):
        if ext not in exts:
            raise KeyError(f'The file format is not supported\nSupported exts: {exts}')
        
    def validate_folder_videos(self):
        folder = self.__file_content['input']

        if os.path.isdir(folder):
            self._checkFolderFilesExtsInSupportedExts(folder, self.__video_exts)
        else:
            raise KeyError('The path is not a folder')
    
    def _checkFolderFilesExtsInSupportedExts(self, folder, exts):
        for file in os.scandir(folder):
            if os.path.isfile(file.path):
                _, ext = os.path.splitext(file.name)
                self._checkExtInSupportedExts(ext, exts)
            else:
                raise KeyError(f'The {file.name} is not file')

    
    def validate_folder_images(self):
        folder = self.__file_content['input']

        if os.path.isdir(folder):
            self._checkFolderFilesExtsInSupportedExts(folder, self.__image_exts)
        else:
            raise KeyError('The path is not a folder')
    
    
