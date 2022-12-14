# https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/

import numpy as np
import matplotlib.pyplot as plt

def f2s(wavelength, angle, multi = 'NO', display = 'NO'):

    # Setup sinusoid parameters
    array_x = np.arange(-256, 256, 1)
    x, y = np.meshgrid(array_x, array_x)
    sinusoid = np.sin(
        2*np.pi*(x * np.cos(angle) + y * np.sin(angle)) / wavelength
    )

    if multi == 'YES':

        array_x = np.arange(-256, 256, 1)
        x, y = np.meshgrid(array_x, array_x)
        angle_2 = np.pi/2
        sinusoid_angle = np.sin(
            2*np.pi*(x * np.cos(angle_2) + y * np.sin(angle_2)) / wavelength
        )

        sinusoid = sinusoid + sinusoid_angle
        

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
        plt.xlim([0, 512])
        plt.ylim([512, 0])  # Note, order is reversed for y

        plt.subplot(122)
        plt.imshow(spatial)

        plt.show()

    return fourier, spatial

def s2f(spatial):
    
    fourier = np.fft.ifftshift(spatial)
    fourier = np.fft.fft2(fourier)
    fourier = np.fft.fftshift(fourier)

    return fourier