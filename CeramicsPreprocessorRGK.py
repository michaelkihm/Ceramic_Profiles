#import numpy as np 
import cv2 as cv 
from CeramicsPreprocessor import CeramicsPreprocessorBase
from typing import Tuple


class CeramicsPreprocessorRGK(CeramicsPreprocessorBase):
    """
    Class to import RGK conspectus dataset
    """
    def run(self) ->Tuple:
        binary_images = []
        image_path_list = self.createImagePathList(self.path, self.file_endings)
        for img_path in image_path_list:
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            img = self.thresholdImage(img)
            img = self.erodeImage(img, (4, 4),2)
            binary_images.append(self.extractCeramicShapeInBinaryImage(img))
        return binary_images, image_path_list        