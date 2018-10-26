import math
from itertools import chain
from typing import List

import pandas as pd
# from PythIon.plotguiuniversal import *
from scipy import signal
from PythIon.Loading import *


def bound(num, lower, upper):
    if lower and upper:
        num = max(lower, min(upper, num))
    elif lower:
        num = max(lower, num)
    elif upper:
        num = min(num, upper)

    return num


def get_file(file_type, idx_offset, next_or_prev, mat_file_name):
    start_index = mat_file_name[-idx_offset::]
    shift = 1
    mystery_offset = 1000  # No idea what this is for
    if next_or_prev == "prev":
        shift *= -1
        mystery_offset *= -1

    def next_idx(file_type, start_idx):
        if file_type == ".log":
            new_idx = str(int(start_idx) + shift)
        elif file_type == ".abf":
            new_idx = str(int(start_idx) + shift).zfill(idx_offset)
        else:
            return
        return new_idx

    file_base = mat_file_name[0:len(mat_file_name) - idx_offset]
    next_index = next_idx(file_type, start_index)

    file_name = file_base + next_index + file_type
    while not os.path.isfile(file_name):
        next_index = next_idx(file_type, next_index)
        if shift * (int(next_index)) > shift * (int(start_index) + mystery_offset):
            print('no such file: ' + file_name)
            break
    if os.path.isfile(file_name):
        return file_name


def num_from_text_element(element, lower=None, upper=None, default=0):
    # TODO: Check element is correct type
    try:
        num_input = int(element.value())
    except ValueError:
        # User tried to enter something other than a number into event num text box
        element.setText(str(default))
        return default
    bounded_num_input = bound(num_input, lower, upper)
    return bounded_num_input


def generate_event_pts(start, end, buffer, baseline, delta):
    event_pts = np.concatenate((
        np.repeat(np.array([baseline]), buffer),
        np.repeat(np.array([baseline - delta]), end - start),
        np.repeat(np.array([baseline]), buffer)), 0)
    return event_pts


def calc_dt(events):
    return np.concatenate([[0], np.diff([event.start for event in events]) / events[0].output_sample_rate])


def calc_frac(events):
    return [event.local_baseline / events[0].baseline for event in events]


# # Takes the data and returns list of event objects.
# def analyze(data, threshold, output_sample_rate):
#     # Find all the points below thrshold
#     below = np.where(data < threshold)[0]
#     start_and_end = np.diff(below)
#     transitions = np.where(start_and_end > 1)[0]
#     # Assuming that record starts and end at baseline
#     # below[transitions] give starting points, below[transitions + 1] gives event end points
#     start_idxs = np.concatenate([[0], transitions + 1])
#     end_idxs = np.concatenate([transitions, [len(below) - 1]])
#     events_intervals = list(zip(below[start_idxs], below[end_idxs]))
#     baseline = np.mean(data)
#     events = []
#     for interval in events_intervals:
#         events.append(Event(data, interval[0], interval[1], output_sample_rate, baseline))
#     return events


def save_batch_info(events, batch_info, info_file_name):
    del_i = [event.delta for event in events]
    frac = calc_frac(events)
    dwell = [event.duration for event in events]
    dt = calc_dt(events)
    noise = [event.noise for event in events]
    start_points = [event.start for event in events]
    end_points = [event.end for event in events]
    cut_start = batch_info["cutstart"]
    cut_end = batch_info["cutend"]
    batch_info = pd.DataFrame({'cutstart': cut_start, 'cutend': cut_end})
    batch_info = batch_info.dropna()
    batch_info = batch_info.append(pd.DataFrame({'deli': del_i, 'frac': frac, 'dwell': dwell, 'dt': dt,
                                                 'noise': noise, 'startpoints': start_points,
                                                 'endpoints': end_points}),
                                   ignore_index=True)
    batch_info.to_pickle(info_file_name + 'batchinfo.pkl')
    return batch_info


def save_cat_data(dataset, file_name):
    # TODO: Finish this function
    with open(file_name, 'w') as file:
        file.write(dataset)


# def event_info_update(data, info_file_name, cb, events, ui, sdf, time_plot, durations_plot,
#                       w1, w2, w3, w4, w5, sample_rate):
#     frac = calc_frac(events)
#     dt = calc_dt(events)
#     num_events = len(events)
#     # Plotting starts after this
#     durations = [event.duration for event in events]
#     deltas = [event.delta for event in events]
#     start_points = [event.start for event in events]
#     end_points = [event.end for event in events]
#     noise = [event.noise for event in events]
#
#     # skips plotting first and last two points, there was a weird spike issue
#     #        self.time_plot.plot(self.t[::10][2:][:-2],data[::10][2:][:-2],pen='b')
#     time_plot.clear()
#     t = np.arange(0, len(data)) / sample_rate
#     time_plot.plot(t[2:][:-2], data[2:][:-2], pen='b')
#     if num_events >= 2:
#         # Plotting start and end points
#         time_plot.plot(t[start_points], data[start_points], pen=None, symbol='o', symbolBrush='g',
#                        symbolSize=10)
#         time_plot.plot(t[end_points], data[end_points], pen=None, symbol='o', symbolBrush='r', symbolSize=10)
#     time_plot.autoRange()

    # # Updating satistics text
    # mean_delta = round(np.mean(deltas) * BILLION, 2)
    # median_duration = round(float(np.median(durations)), 2)
    # event_rate = round(num_events / t[-1], 1)
    # ui.eventcounterlabel.setText('Events:' + str(num_events))
    # ui.meandelilabel.setText('Deli:' + str(mean_delta) + ' nA')
    # ui.meandwelllabel.setText('Dwell:' + str(median_duration) + u' μs')
    # ui.meandtlabel.setText('Rate:' + str(event_rate) + ' events/s')
    #
    # # Dataframe containing all information
    # sdf = sdf[sdf.fn != info_file_name]
    # fn = pd.Series([info_file_name] * num_events)
    # color = pd.Series([pg.colorTuple(cb.color())] * num_events)

    # sdf = sdf.append(pd.DataFrame({'fn': fn, 'color': color, 'deli': deltas,
    #                                'frac': frac, 'durations': durations,
    #                                'dt': dt, 'stdev': noise, 'startpoints': start_points,
    #                                'endpoints': end_points}), ignore_index=True)
    #
    # # I think below should be trying to show only the points associated with current file
    # # But I'm not really sure
    # # try:
    # #     durations_plot.data = durations_plot.data[np.where(np.array(sdf.fn) != info_file_name)]
    # # except Exception as e:
    # #     print(e)
    # #     raise IndexError
    # durations_plot.addPoints(x=np.log10(durations), y=frac, symbol='o', brush=(cb.color()), pen=None, size=10)
    # # w1 is window 1???
    # w1.addItem(durations_plot)
    # w1.setLogMode(x=True, y=False)
    # w1.autoRange()
    # w1.setRange(yRange=[0, 1])
    #
    # ui.scatterplot.update()
    #
    # update_histograms(sdf, ui, events, w2, w3, w4, w5)
    # return sdf


def converging_baseline(data, sigma=1, tol=0.001):
    baseline = np.mean(data)
    stdev = np.std(data)
    for n in range(10):
        zeroed_data = data - baseline
        baseline_points = abs(zeroed_data) - (sigma * stdev) < 0
        old_baseline = baseline
        baseline = np.mean(data[baseline_points])
        stdev = np.std(data[baseline_points])
        if abs(old_baseline - baseline) < abs(baseline * tol):
            break
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    return baseline, stdev


def stdfilt(data, filter_size=5):
    # Use itterative method for calulating standard deviation
    running_stdev = [math.sqrt(x) for _, x in running_stats(data, filter_size)]
    return running_stdev


def load_data(data_path):
    file_dir = os.path.dirname(data_path)
    file_name, file_type = os.path.basename(data_path).split('.')
    if file_type == 'log':
        data, data_params = load_log_file(file_name, file_dir)
    elif file_type == 'opt':
        data, data_params = load_opt_file(file_name, file_dir)
    elif file_type == 'txt':
        data, data_params = load_txt_file(file_name, file_dir)
    elif file_type == 'npy':
        data, data_params = load_npy_file(file_name, file_dir)
    elif file_type == 'abf':
        data, data_params = load_abf_file(file_name, file_dir)
    else:
        return
    return data, data_params


def running_stats(data, filter_size=5):
    # TODO: Generalize to skew, kurtosis, etc.
    mean = data[0]  # These are put up here so pycharm doesn't complain
    variance = 0
    for n in range(len(data)):
        if n >= filter_size:
            diff = data[n] - data[n - filter_size]
            mean_change = diff / filter_size
            new_pt = data[n]
            old_pt = data[n - filter_size]
            old_mean = mean
            new_mean = mean + mean_change
            variance = variance + ((new_pt - new_mean) ** 2 - (old_pt - old_mean) ** 2) / (filter_size - 1)
            mean = new_mean
        elif 1 <= n < filter_size:
            mean = np.mean(data[:n + 1])
            variance = np.var(data[:n + 1])
        yield mean, variance


def threshold_data(data, threshold, comparison='greater'):
    if comparison is 'greater':
        event_points = np.greater(abs(data), threshold)
    elif comparison is 'less':
        event_points = np.less(abs(data), threshold)
    else:
        return
    event_bounries = np.diff(event_points.astype(int))
    if sum(event_bounries) % 2 == 1:
        print("Uneven number of boundaries")
        event_bounds = []  # This section should never happen
    elif any(event_bounries):
        boundary_idxs = list(np.concatenate(np.argwhere(event_bounries)))
        event_bounds = list(zip(boundary_idxs[::2], boundary_idxs[1::2]))
    else:
        event_bounds = []
    return event_bounds


def peak_detect(edge_data, thresh_multiplier=math.sqrt(2)):
    # thresh_multiplier tries to correct the standard deviation to represent the peaks
    # extrema = signal.argrelextrema(edge_data[::step_size], np.greater, order=order)[0]
    # # noinspection PyTypeChecker
    # sorted_peaks = np.sort(np.abs(edge_data[extrema * step_size]))
    # sorted_peaks_deriv = np.diff(sorted_peaks)
    # threshold = sorted_peaks[np.argmax(sorted_peaks_deriv) + 1]
    extrema = signal.argrelextrema(abs(edge_data), np.greater)
    if len(extrema) >= 2:
        threshold = np.std(edge_data[extrema]) * thresh_multiplier
    else:
        threshold = np.std(edge_data) * thresh_multiplier
    extrema_intervals = threshold_data(np.abs(edge_data), threshold)
    return extrema_intervals


def identify_start_and_end(data, start, threshold):
    def separate_if_list(arg):
        if type(arg) is List or type(arg) is tuple:
            assert len(arg) == 2
            lower, upper = arg
        else:
            lower = upper = arg
        return lower, upper

    start_point, end_point = separate_if_list(start)
    start_threshold, end_threshold = separate_if_list(threshold)

    while data[start_point] < start_threshold:
        start_point -= 1
    while data[end_point] < end_threshold:
        end_point += 1

    return start_point, end_point


def threshold_search(data, sample_rate, cutoff_freq):
    # Crude implementation to be improved
    # Look at using scipy optimization
    global_min, global_max = np.min(data), np.max(data)
    # start_offset = (np.max(data) - base) / 2
    if len(data) < 10000:
        spacing = 1
    else:
        # Has some relation to nyquist rate, don't care to look into it right now
        spacing = int(sample_rate / (2 * cutoff_freq))
    thresholds = np.linspace(global_min, global_max, 100)
    counts = [sum((data[::spacing] < threshold).astype(int)) for threshold in thresholds]
    turning_point = np.argmax(np.diff(counts) / (np.arange(len(counts) - 1) + 1))
    n = turning_point
    # Crawl down hill to find minimum
    while counts[n] > counts[n - 1]:
        n -= 1
    return thresholds[n]


def find_extrema(data, order=1, mode='clip'):  # 1d only
    # argrelmax/min output tuples event for 1d data
    # TODO: maybe replace with find peaks
    mins = signal.argrelmin(data, order=order, mode=mode)[0]  # Pulling arrays out of tuples
    maxes = signal.argrelmax(data, order=order, mode=mode)[0]
    extrema = np.array(list(chain.from_iterable(zip(mins, maxes))) + [mins[-1]])
    return extrema


def roc_pts(data, num_pts=50):
    x_vals = np.linspace(min(data), max(data), num_pts)
    y_vals = [sum((abs(data) < x_val).astype(int)) for x_val in x_vals]
    return x_vals, y_vals


def slope_crawl(data, start, direction='forward', comparison='max'):
    if direction is 'forward':
        change = 1
    else:
        change = -1
    if comparison is 'max':
        while data[start + change] > data[start]:
            start += change
    elif comparison is 'min':
        while data[start + change] < data[start]:
            start += change
    return start


BILLION = 10 ** 9  # Hopefully makes reading clearer
