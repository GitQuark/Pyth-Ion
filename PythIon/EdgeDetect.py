import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
# Going to try doing some actual signal processing
# Basically this is a 1d canny filter_and_plot
#
# Convolve with a box
# Perform gaussian smoothing
# Use np.diff


def canny_1d(data, window_width, sigma, truncate=3):
    """
Processes 1d array in the same way as a canny filter_and_plot.
Uses gaussian smoothing after convolution to eliminate noisy derivative
    :param window_width:
    :param truncate:
    :param sigma:
    :param data:
    :return:
    """
    # TODO: Make as fast as possible
    if sigma is None:
        sigma = window_width / 10  # 10 was chosen arbitrarily
    zeroed_data = np.array(data) - np.mean(data[np.where(np.array(data))])
    window = np.ones(window_width)
    convolved = signal.fftconvolve(zeroed_data, window, mode='valid')
    smoothed = gaussian_filter1d(convolved, sigma=window_width, truncate=truncate)
    final = gaussian_filter1d(np.diff(smoothed), sigma=sigma, truncate=truncate)
    return final

# TODO: Subtract box_length (in number of data points)/ output_sample_rate from the trailing edge of rise and fall
# i.e. for box 500 data poins wide, 500/output_sample_rate would be subtracted from fall_end and rise_end






