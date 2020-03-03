import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def reconstruct_fourier_descriptor(fourier_descriptor: np.ndarray):
    """
    Reconstracts the shape from a Fourier Descriptor.
    Only temp version
    """
    contour_reconstruct = np.fft.ifft(fourier_descriptor)
    contour_reconstruct = np.array(
        [contour_reconstruct.real, contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    # make positive
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    # normalization
    contour_reconstruct *= 800 / contour_reconstruct.max()
    # type cast to int32
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)
    black = np.zeros((800, 800, 3), np.uint8)
    # draw
    for i in range(len(contour_reconstruct)-1):
        cv.line(black, tuple(contour_reconstruct[i, 0, :]), tuple(
            contour_reconstruct[i+1, 0, :]), (0, 255, 0), thickness=1)

    # save image
    cv.imwrite("reconstruct_result_funtion.jpg", black)


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
