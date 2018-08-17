import peakutils
import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from PythIon.Utility import Event
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
    # TODO: Make as fast as possible
    if sigma is None:
        sigma = window_width / 10  # 10 was chosen arbitrarily
    # data_length = len(data)
    zeroed_data = np.array(data) - np.mean(data[np.where(np.array(data) > 0.1e-9)])
    # (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0)
    window = np.ones(window_width)
    convolved = signal.fftconvolve(zeroed_data, window, mode='valid')
    smoothed = gaussian_filter1d(convolved, sigma=sigma, truncate=truncate)
    final = np.diff(smoothed)
    return final

# TODO: Subtract box_length (in number of data points)/ output_sample_rate from the trailing edge of rise and fall
# i.e. for box 500 data poins wide, 500/output_sample_rate would be subtracted from fall_end and rise_end


def peak_detect(edge_data, step_size=15, order=100):
    maxima = signal.argrelextrema(edge_data[::step_size], np.greater, order=order)[0]
    # noinspection PyTypeChecker
    sorted_peaks = np.sort(edge_data[maxima])
    sorted_peaks_deriv = np.diff(sorted_peaks)
    threshold = sorted_peaks[np.argmax(sorted_peaks_deriv) + 1]
    extrema_locations = np.where(np.abs(edge_data) > threshold)[0]
    return extrema_locations


# Returns the actual event list properly nested
def edge_detection(data, output_sample_rate):
    event_list = []
    edge_data = canny_1d(data, 250, 3)
    extrema_locations = peak_detect(edge_data)
    start_and_end = np.diff(extrema_locations)
    transitions = np.where(start_and_end > 1)[0]
    start_idxs = np.concatenate([[0], transitions + 1])
    end_idxs = np.concatenate([transitions, [len(extrema_locations) - 1]])
    intervals = list(zip(extrema_locations[start_idxs], extrema_locations[end_idxs]))
    for idx, interval in enumerate(intervals):
        # IGNORE: Following python conventions of data point at end not included; interval[1] is in event
        print(idx)
        start = interval[0]
        end = interval[1] + 1
        new_event = Event(data[start:end], start, end, np.mean(data[start:end]), output_sample_rate)
        event_list.append(new_event)
    return event_list
