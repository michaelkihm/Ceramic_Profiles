import numpy as np
#import cv2 as cv
#import matplotlib.pyplot as plt
from typing import List
from FourierDescriptorBase import FourierDescriptorBase


class FourierDescriptorMagnitude(FourierDescriptorBase):
    """
    Class to construct Magnitude Fourier descriptor of given binary images \n
    @param list of filenames or list of images
    @param number of fourier descriptor (FD) harmonics -> FD[-m_n, -m_n-1, ...-m_1, m_0, m_1, ...., m_n-1, m_n ] for n harmonics
    """

    def __init__(self, images: List, descriptor_harmonics: int):
        super(FourierDescriptorMagnitude, self).__init__(
            images, descriptor_harmonics)

    def normalizeDescriptor(self, descriptor: np.ndarray) -> np.ndarray:
        descriptor[0] = 0  # translation invariant
        # scale invariant
        descriptor /= descriptor[-1 if np.abs(descriptor[0])
                                 < np.abs(descriptor[-1]) else 0]
        # rotation and mirror invariant
        return np.abs(descriptor)
