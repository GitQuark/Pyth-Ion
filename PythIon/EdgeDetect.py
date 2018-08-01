import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
# Going to try doing some actual signal processing
# Basically this is a 1d canny filter
#
# Convolve with a box
# Perform gaussian smoothing
# Use np.diff


def canny_1d(data, window_width, sigma=None, truncate=3):
    """
Processes 1d array in the same way as a canny filter.
Uses gaussian smoothing after convolution to eliminate noisy derivative
    :param truncate:
    :param sigma:
    :param data:
    :param window_width:
    :return:
    """
    if sigma is None:
        sigma = window_width / 10  # 10 was chosen arbitrarily
    data_length = len(data)
    zeroed_data = np.array(data) - np.mean(data[np.where(np.array(data) > 0.1e-9)])
    # (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0)
    window = np.hstack((np.zeros(data_length), np.ones(window_width), np.zeros(data_length)))
    convolved = signal.fftconvolve(zeroed_data, window, mode='same')
    smoothed = gaussian_filter1d(convolved, sigma=sigma, truncate=truncate)
    final = np.diff(smoothed)
    return final

# TODO: Subtract box_length (in number of data points)/ output_sample_rate from the trailing edge of rise and fall
# i.e. for box 500 data poins wide, 500/output_sample_rate would be subtracted from fall_end and rise_end

