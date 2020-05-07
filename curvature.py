import numpy as np
from typing import List
from FourierDescriptorMagnitude import FourierDescriptorMagnitude
from fourier_utils import reconstruct_fourier_descriptor


class Curvature:
    """
    @brief Computes curvature of given images. 
    @param images can be a list of filenames or a list of preporcessed binary images
    @param descriptor_harmonics Number of frequency harmonics
    @param contour_points number of points of the smoothed contour shape
    """

    def __init__(self, images: List, descriptor_harmonics: int = 30, contour_points: int = 100):

        self.curvatures = []
        self._reconstructed_fds = []
        self._contour_points = contour_points
        self._descriptors = []

        self._fd = FourierDescriptorMagnitude(
            images, descriptor_harmonics, normalize=False)

    def run(self,signed:bool=False) -> np.ndarray:
        """
        @brief  Computes curvature of given binary images. \n
                1. Transforms shape vector in Fourier space to filter shape vector \n
                2. Transforms filtered shape vector back in spatial domain
                3. Selects top shape pixel as origin of the shape
                4. Computes curvature
        """

        # Bring contours in Fourier space
        self._descriptors = self._fd.run()

        # Compute the curvature of the shape vectors
        for descriptor in self._descriptors:
            temp = reconstruct_fourier_descriptor(
                descriptor, self._contour_points)
            self._reconstructed_fds.append(self._sort_contour(temp))
            # self.curvatures.append(self._curvature_splines(
            #     self._reconstructed_fds[-1][:, 0], self._reconstructed_fds[-1][:, 1]))
            self.curvatures.append(self._curvature(self._reconstructed_fds[-1], signed))

        return self.curvatures

    def get_curvatures(self) -> np.ndarray:
        """ @brief returns curvatures of the class """
        return self.curvatures

    def get_reconstructed_fds(self):
        """ @brief return curves """
        return self._reconstructed_fds

    def _sort_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        @brief  Returns contour array with starting point being at the top y position
                1. Find top most y-coordinate pixel 
                2. Find array index of the pixel found in step 1
                3. Roll array to put top most pixel at array element 0
        @param contour contour array to be sorted by the function
        """
        cont = contour.astype(np.int)
        sort = sorted(cont, key=lambda x: x[1])
        index = np.where((cont[:, 0] == sort[0][0]) &
                         (cont[:, 1] == sort[0][1]))[0]
        out = np.roll(cont, -index, axis=0)
        return out

    def _curvature(self, contour: np.ndarray,signed:bool=False) -> np.array:
        """
        @brief  Calculate the signed or unsigned curvature of a 2D curve at each point
        @param contour 2D array of x and y coordinates of the coordinate
        @param signed indicates if signed curvature should be calculated
        """

        dx_dt = np.gradient(contour[:, 0])
        dy_dt = np.gradient(contour[:, 1])
 
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        numerator = d2x_dt2 * dy_dt - dx_dt * d2y_dt2
        curvature =  numerator if signed else np.abs(numerator)
        curvature /= (dx_dt**2 + dy_dt**2)**1.5

        assert len(contour) == len(curvature)
        return curvature
