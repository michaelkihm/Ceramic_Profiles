import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from typing import Tuple

def plot_fourier_spectrum(fourier_descriptor):
    """
    Plots absolute values of the Fourier spectrum. Spectrum is shown zero centered
    """
    M = len(fourier_descriptor)
    x = [i for i in range(-(M//2), M//2+1)]
    plt.title("Magnitude Fourier descriptor")
    plt.xlabel('harmonics')
    plt.plot(x, np.abs(np.fft.fftshift(fourier_descriptor)))


def truncate_descriptor(fourier_descriptor, new_length):
    """@brief truncate an unshifted fourier descriptor array to given length length
       @param fourier_descriptor
       @param new_length new length of given fourier descriptor
    """
    fourier_descriptor = np.fft.fftshift(fourier_descriptor)
    center_index = len(fourier_descriptor) // 2
    fourier_descriptor = fourier_descriptor[
        center_index - new_length // 2:center_index + new_length // 2]
    return np.fft.ifftshift(fourier_descriptor)

def draw_reconsturcted_descriptor(descriptor:np.ndarray, cont_points:int, image_shape:Tuple=(200,400), image:np.ndarray=None,thickness:int=2):
    """
    @brief Returns shape of reconstructed Fourier Descriptor. FD is either drawn in given image or is drawn in black image with given shape.
    @remark Given FD should not be normalized
    @param descriptor: Numpy array with Fourier descriptor
    @param cont_point: Number of points of the reconstructed contour
    @param image_shape: 2d tuple with shape of image to in which reconstructed FD will be drawn. Only used of no image is given
    @param image: Image in which reconstructed FD will be drawan
    """
    contour_reconstruct = reconstruct_fourier_descriptor(descriptor, cont_points)
    contour_reconstruct = contour_reconstruct.astype(np.int)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    
    if image is None:
        img = np.zeros((*image_shape, 3), np.float)
    else:
        img = cv.cvtColor(image.astype(np.uint8),cv.COLOR_GRAY2RGB)

    # draw
    for i in range(len(contour_reconstruct)-1):
        cv.line(img, tuple(contour_reconstruct[i, 0, :]), tuple(
            contour_reconstruct[i+1, 0, :]), (0, 255, 0), thickness=thickness)
    
    return img


def reconstruct_fourier_descriptor(descriptor:np.ndarray, cont_points:int ):
    """
    @brief Reconstructs uniform Fourier descriptor with N contour points
    @param descriptor: Uniform FD
    @param cont_points: Number of points of the reconstructed contour
    """
    recon_descriptor = np.zeros((cont_points,2))
    for  i  in range(cont_points):
        t =  float(i) / cont_points
        recon_descriptor[i] = get_reconstruction_of_fourier_descriptor_point(descriptor,t)

    return recon_descriptor

def get_reconstruction_of_fourier_descriptor_point(descriptor:np.ndarray,coefficient:int):
    """
    @brief  Reconstruct x and y coordinates of given Fourier coefficient. Helper function for
            reconstruct_fourier_descriptor      
    @param descriptor: FD descriptor array
    @param coefficient: FD coefficient
    """
    harmonics = (len(descriptor) - 1) // 2
    harmonics_min = harmonics*-1
    
    x = 0.0
    y = 0.0

    for m in range(harmonics_min,harmonics+1):
        A = descriptor[m].real
        B = descriptor[m].imag
        phi = 2*np.pi*m*coefficient
        x += A * np.cos(phi) - B*np.sin(phi)
        y += A * np.sin(phi) + B*np.cos(phi)

    return x, y


def phase_descriptor_distance(g1:np.ndarray, g2:np.ndarray) ->float:
    g1_0 = g1[0:len(g1)//2]
    g2_0 = g2[0:len(g1)//2]
    g2_1 = g2[len(g1)//2:]
    assert len(g2_0) == len(g2_1)
    return min( complex_descriptor_distance(g1_0,g2_0), complex_descriptor_distance(g1_0, g2_1) )

def complex_descriptor_distance(g1:np.ndarray, g2:np.ndarray) ->float:
    s = 0
    for m in range( len(g1)//2 ):
        s += np.power(g1[m] - g2[m],2) + np.power(g1[m+len(g1)//2] - g2[m+len(g1)//2],2)
    
    return np.sqrt(s)


    
