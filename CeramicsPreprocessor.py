import os 
from typing import List, Tuple
from scipy.ndimage import label
import numpy as np 
import cv2 as cv 



class CeramicsPreprocessorBase:
    """
    Base class to import and preprocess the ceramics images
    """
    def __init__(self, path: str, file_endings: List):
        self.path = path
        self.image_path_list = []
        self.file_endings = file_endings

    def run(self) ->Tuple:
        raise NotImplementedError

    def createImagePathList(self, dir_name: str, file_endings: List) ->List:
        """ returns a list of filenames in given directory and sub directories. 
        List contains all files with given file ending"""
        list_of_directories = os.listdir(dir_name)
        all_files = list()
        for entry in list_of_directories:
            full_path = os.path.join(dir_name, entry)
            if os.path.isdir(full_path):
                all_files = all_files + self.createImagePathList(full_path, file_endings)
            else:
                all_files.append(full_path)
     
        #remove file strings which do not have a valid file ending 
        [ all_files.remove(i) if (lambda x: os.path.splitext(x)[1])(i) not in file_endings else i for i in all_files]    
        return all_files

    def thresholdImage(self, image) -> np.ndarray:
        blur = cv.GaussianBlur(image, (5, 5), 0)
        _, thr_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        return thr_img

    def erodeImage(self, image: np.ndarray, kernel_size: Tuple, iterations: int) ->np.ndarray:
        kernel = np.ones(kernel_size, np.uint8)
        return cv.erode(image, kernel, iterations=iterations)

    def extractCeramicShapeInBinaryImage(self, image) ->np.ndarray:
        labeled_array, _ = label(image/255.0)
        _, counts_elements = np.unique(labeled_array, return_counts=True)
       
        # find second largest region (largest region is background)
        sorted_arg = np.flip(np.argsort(counts_elements))
        label_nr = sorted_arg[1]

        # create new array and mask with labeled_array with
        # specified label
        out_array = np.ones_like(labeled_array)*255
        out_array *= (labeled_array == label_nr)
        return out_array
    