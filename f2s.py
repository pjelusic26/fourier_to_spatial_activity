# https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/

import numpy as np
import matplotlib.pyplot as plt

def f2s(wavelength, angle, display = 'NO'):

    # Setup sinusoid parameters
    array_x = np.arange(-128, 129, 1)
    x, y = np.meshgrid(array_x, array_x)
    sinusoid = np.sin(
        2*np.pi*(x * np.cos(angle) + y * np.sin(angle)) / wavelength
    )

    # Calculate Fourier transform of sinusoid
    fourier = np.fft.ifftshift(sinusoid)
    fourier = np.fft.fft2(fourier)
    fourier = np.fft.fftshift(fourier)

    # Inverse the FFT!
    spatial = np.fft.ifftshift(fourier)
    spatial = np.fft.ifft2(spatial)
    spatial = np.fft.fftshift(spatial)
    spatial = spatial.real  # Take only the real part

    if display == 'YES':

        plt.set_cmap("gray")
        plt.subplot(121)

        plt.subplot(121)
        plt.imshow(abs(fourier))
        plt.xlim([0, 256])
        plt.ylim([256, 0])  # Note, order is reversed for y

        plt.subplot(122)
        plt.imshow(spatial)

        plt.show()

    return fourier, spatial