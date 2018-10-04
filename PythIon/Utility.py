import math
import os
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
import pyqtgraph as pg

# from PythIon.plotguiuniversal import *
from qtpy import QtWidgets
from scipy import signal

from PythIon import EdgeDetect
from PythIon.SetupUtilities import load_log_file, load_txt_file, load_npy_file, load_abf_file, load_opt_file


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
        num_input = int(element.text())
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


# Takes the data and returns list of event objects.
def analyze(data, threshold, output_sample_rate):
    # Find all the points below thrshold
    below = np.where(data < threshold)[0]
    start_and_end = np.diff(below)
    transitions = np.where(start_and_end > 1)[0]
    # Assuming that record starts and end at baseline
    # below[transitions] give starting points, below[transitions + 1] gives event end points
    start_idxs = np.concatenate([[0], transitions + 1])
    end_idxs = np.concatenate([transitions, [len(below) - 1]])
    events_intervals = list(zip(below[start_idxs], below[end_idxs]))
    baseline = np.mean(data)
    events = []
    for interval in events_intervals:
        events.append(Event(data, interval[0], interval[1], baseline, output_sample_rate))
    return events


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
    with open(file_name, 'w') as file:
        file.write()


def update_histograms(sdf, ui, events, w2, w3, w4, w5):
    colors = sdf.color.unique()
    for i, x in enumerate(colors):
        frac = calc_frac(events)
        dt = calc_dt(events)
        num_events = len(events)
        # Plotting starts after this
        durations = [event.duration for event in events]
        deltas = [event.delta for event in events]
        start_points = [event.start for event in events]
        end_points = [event.end for event in events]
        noise = [event.noise for event in events]
        frac_bins = np.linspace(0, 1, int(ui.fracbins.text()))
        delta_bins = np.linspace(float(ui.delirange0.text()) * 10 ** -9, float(ui.delirange1.text()) * 10 ** -9,
                                 int(ui.delibins.text()))
        duration_bins = np.linspace(float(ui.dwellrange0.text()), float(ui.dwellrange1.text()),
                                    int(ui.dwellbins.text()))
        dt_bins = np.linspace(float(ui.dtrange0.text()), float(ui.dtrange1.text()), int(ui.dtbins.text()))

        frac_y, frac_x = np.histogram(frac, bins=frac_bins)
        deli_y, deli_x = np.histogram(deltas, bins=delta_bins)
        dwell_y, dwell_x = np.histogram(np.log10(durations), bins=duration_bins)
        dt_y, dt_x = np.histogram(dt, bins=dt_bins)

        hist = pg.BarGraphItem(height=frac_y, x0=frac_x[:-1], x1=frac_x[1:], brush=x)
        w2.addItem(hist)
        hist = pg.BarGraphItem(height=deli_y, x0=deli_x[:-1], x1=deli_x[1:], brush=x)
        w3.addItem(hist)
        w3.setRange(xRange=[float(ui.delirange0.text()) * 10 ** -9, float(ui.delirange1.text()) * 10 ** -9])
        hist = pg.BarGraphItem(height=dwell_y, x0=dwell_x[:-1], x1=dwell_x[1:], brush=x)
        w4.addItem(hist)
        hist = pg.BarGraphItem(height=dt_y, x0=dt_x[:-1], x1=dt_x[1:], brush=x)
        w5.addItem(hist)


def event_info_update(data, info_file_name, cb, events, ui, sdf, time_plot, durations_plot,
                      w1, w2, w3, w4, w5, sample_rate):
    frac = calc_frac(events)
    dt = calc_dt(events)
    num_events = len(events)
    # Plotting starts after this
    durations = [event.duration for event in events]
    deltas = [event.delta for event in events]
    start_points = [event.start for event in events]
    end_points = [event.end for event in events]
    noise = [event.noise for event in events]

    # skips plotting first and last two points, there was a weird spike issue
    #        self.time_plot.plot(self.t[::10][2:][:-2],data[::10][2:][:-2],pen='b')
    time_plot.clear()
    t = np.arange(0, len(data)) / sample_rate
    time_plot.plot(t[2:][:-2], data[2:][:-2], pen='b')
    if num_events >= 2:
        # TODO: Figure out why a single point can't be plotted
        # Plotting start and end points
        time_plot.plot(t[start_points], data[start_points], pen=None, symbol='o', symbolBrush='g',
                       symbolSize=10)
        time_plot.plot(t[end_points], data[end_points], pen=None, symbol='o', symbolBrush='r', symbolSize=10)
    time_plot.autoRange()

    # Updating satistics text
    mean_delta = round(np.mean(deltas) * BILLION, 2)
    median_duration = round(float(np.median(durations)), 2)
    event_rate = round(num_events / t[-1], 1)
    ui.eventcounterlabel.setText('Events:' + str(num_events))
    ui.meandelilabel.setText('Deli:' + str(mean_delta) + ' nA')
    ui.meandwelllabel.setText('Dwell:' + str(median_duration) + u' μs')
    ui.meandtlabel.setText('Rate:' + str(event_rate) + ' events/s')

    # Dataframe containing all information
    sdf = sdf[sdf.fn != info_file_name]
    fn = pd.Series([info_file_name] * num_events)
    color = pd.Series([pg.colorTuple(cb.color())] * num_events)

    sdf = sdf.append(pd.DataFrame({'fn': fn, 'color': color, 'deli': deltas,
                                   'frac': frac, 'durations': durations,
                                   'dt': dt, 'stdev': noise, 'startpoints': start_points,
                                   'endpoints': end_points}), ignore_index=True)

    # I think below should be trying to show only the points associated with current file
    # But I'm not really sure
    # try:
    #     durations_plot.data = durations_plot.data[np.where(np.array(sdf.fn) != info_file_name)]
    # except Exception as e:
    #     print(e)
    #     raise IndexError
    durations_plot.addPoints(x=np.log10(durations), y=frac, symbol='o', brush=(cb.color()), pen=None, size=10)
    # w1 is window 1???
    w1.addItem(durations_plot)
    w1.setLogMode(x=True, y=False)
    w1.autoRange()
    w1.setRange(yRange=[0, 1])

    ui.scatterplot.update()

    update_histograms(sdf, ui, events, w2, w3, w4, w5)
    return sdf


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


class Event(object):
    subevents: List

    def __init__(self, data, start, end, sample_rate, baseline=None):
        # Start and end are the indicies in the data
        print(start, end)
        self.sample_rate = sample_rate
        start = slope_crawl(data, start, 'backward')
        end = slope_crawl(data, end, 'forward')
        self.start, self.end = start, end  # start and end are relative to the whole dataset
        self.data = data[self.start: self.end]
        extrema = find_extrema(self.data)
        self.main_bounds = (extrema[0], extrema[-1] + 1)
        main_event = self.data[self.main_bounds[0]: self.main_bounds[1]]
        self.local_baseline, self.local_stedev = np.mean(main_event), np.std(main_event)
        # Intervald sre relative to the internal data
        self.intervals = self.interval_detect(main_event, num_stdevs=1.75, min_length=1000)
        # True can false are converted to 1 and 0, np.argmax returns index of first ture value

        if baseline is not None:
            self.baseline = baseline
        else:
            self.baseline = np.concatenate(data[start - 1000:start], data[end:end + 1000]).mean()

        # TODO: Fix slight error in calculations; Too much past the tails is included in fall and rise
        # Rise_end seems to have consistent issues
        self.noise = np.std(self.data)
        self.durations = [(start - end) / sample_rate for start, end in self.intervals]  # In seconds
        self.levels = [np.mean(main_event[start: end]) for start, end in self.intervals]  # In Volts
        self.deltas = [baseline - level for level in self.levels]

    def __repr__(self):
        return str(self.levels)

    # Returns the actual event list properly nested
    @staticmethod
    def interval_detect(data, num_stdevs=1.75, min_length=500):
        # TODO: set a maximum rise time
        if len(data) < min_length:
            return [(0, len(data))]
        edge_data = EdgeDetect.canny_1d(data, 250, 3)
        change_intervals = peak_detect(edge_data)  # Intervals where slope is significant
        index_list = [0] + [idx for interval in change_intervals for idx in interval] + [len(data)]
        intervals = [(index_list[2 * n], index_list[2 * n + 1]) for n in range(int(len(index_list) / 2))]
        intervals.reverse()
        interval = intervals.pop()
        while intervals and interval[1] - interval[0] < min_length:
            if not intervals:
                event_list = [(0, len(data))]
                break
            _, new_end = intervals.pop()
            interval = (0, new_end)
        else:
            event_list = [interval]
        intervals.reverse()
        for interval in intervals[1:]:
            # IGNORE: Following python conventions of data point at end not included; interval[1] is in event
            old_start, old_end = event_list[-1]
            start, end = interval
            too_short = end - start < min_length
            # print(old_start, old_end, start, end)
            # print(np.mean(data[start:end]) - np.mean(data[old_start:old_end]))
            level_change = np.mean(data[start:end]) - np.mean(data[old_start:old_end])
            change_too_small = abs(level_change) < num_stdevs * np.std(data)
            if too_short or change_too_small:
                event_list[-1] = (old_start, end)
            else:
                prev_start, prev_end = event_list[-1]
                if data[prev_end] < data[start]:
                    new_end_of_prev_interval = slope_crawl(data, start, 'backward', comparison='min')
                else:  # data[prev_end] > data[start]
                    new_end_of_prev_interval = slope_crawl(data, start, 'backward', comparison='max')
                event_list[-1] = (prev_start, new_end_of_prev_interval)

                event_list.append((start, end))
        return event_list

    def piecewise_fit(self, offset_fit=False):
        offset = 0
        if offset_fit:
            offset += self.start
        fit_points = [(offset, self.data[0])]
        offset += self.main_bounds[0]
        for idx, interval in enumerate(self.intervals):
            start, end = interval
            level = self.levels[idx]
            fit_points.append((start + offset, level))
            fit_points.append((end + offset, level))
        fit_points.append((offset + len(self.data) - 1, self.data[-1]))
        fit_points.append(fit_points[-1])  # Added since pyqtgraph does not display the last line for some reason
        x, y = zip(*fit_points)
        return np.array(x), np.array(y)

    def generate_level_entries(self):
        data_list = []
        event_length = self.end - self.start
        for idx, interval in enumerate(self.intervals):
            data_list.append({
                'start': (self.start + interval[0]) / self.sample_rate,
                'end': (self.start + interval[1]) / self.sample_rate,
                'current_level': self.levels[idx],
                'frac': (interval[1] - interval[0]) / event_length,
                'duration': (interval[1] - interval[0]) / self.sample_rate
            })
        return data_list

    def shift_by(self, shift):
        self.start += shift
        self.end += shift


class CurrentData(object):
    data_params: dict
    events: List[Event]

    def __init__(self, data_path, edge_buffer=1000):
        self.file_dir = os.path.dirname(data_path)
        self.file_name, self.file_type = os.path.basename(data_path).split('.')
        self.data, data_info = load_data(data_path)
        baseline, _ = converging_baseline(self.data)
        self.data_params = {
            'inverted': False,
            'edge_buffer': edge_buffer,
            'baseline': baseline,
            **data_info}  # Python 3.5+ trick to combine dictionary values with an existing dictionary
        self.processed_data = self.data
        self.events = []
        self.cuts = []

    def detect_events(self, threshold=None, min_length=1000):
        self.events = []
        if not threshold:
            sample_rate = self.data_params.get('sample_rate')
            low_pass_cutoff = self.data_params.get('low_pass_cutoff')
            threshold = threshold_search(self.processed_data, sample_rate, low_pass_cutoff)
        self.data_params['threshold'] = threshold
        bounds = threshold_data(self.processed_data, threshold, 'less')
        baseline = self.data_params.get('baseline')
        sample_rate = self.data_params.get('sample_rate')
        for bound in bounds:
            start, end = bound
            if end - start < min_length:
                continue
            event = Event(self.processed_data, start, end, baseline, sample_rate)
            self.events.append(event)

    def process_data(self, low_pass_cutoff=None):
        self.processed_data = self.data

        if self.data_params.get('inverted'):
            self.processed_data = -self.processed_data

        if self.cuts:
            slice_list = [slice(0, self.cuts[0][0])]  # Initial interval of included points
            for idx, cut in enumerate(self.cuts[1:-1]):
                # TODO: Make more efficient
                # No effort to figure out an 'abosute position of the cuts
                _, include_start = cut
                # one increment for first cut being skipped, the other for acessing next cut
                include_end, _ = self.cuts[idx + 1 + 1]
                slice_list.append(slice(include_start, include_end))
            slice_list.append(slice(self.cuts[-1][1], len(self.data)))
            included_pts = np.r_[slice_list]
            self.processed_data = self.processed_data[included_pts]

        if not low_pass_cutoff:
            low_pass_cutoff = self.data_params.get('low_pass_cutoff')
        sample_rate = self.data_params.get('sample_rate')
        if low_pass_cutoff and sample_rate:
            self.data_params['low_pass_cutoff'] = low_pass_cutoff
            nyquist_freq = sample_rate / 2
            wn = low_pass_cutoff / nyquist_freq
            # noinspection PyTupleAssignmentBalance
            b, a = signal.bessel(4, wn, btype='low')
            self.processed_data = signal.filtfilt(b, a, self.processed_data)

        edge_buffer = self.data_params.get('edge_buffer')
        self.processed_data = self.processed_data[edge_buffer: -edge_buffer]

    def reset(self):
        self.processed_data = self.data
        self.cuts = []
        self.events = []

    def event_fits(self):
        fits = []
        for event in self.events:
            fit = event.piecewise_fit(offset_fit=True)
            fits.append(fit)
        return fits  # List of fits fo each event

    def get_event_prop(self, prop_name):
        if not self.events:
            return
        if prop_name not in self.events[0].__dict__.keys():
            return

        return [event.__getattribute__(prop_name) for event in self.events]

    def generate_event_table(self):
        event_data_list = []
        for idx, event in enumerate(self.events):
            event_entries = event.generate_level_entries()
            for entry in event_entries:
                entry['event_number'] = idx
                event_data_list.append(entry)
        return pd.DataFrame(event_data_list)

    def add_cut(self, interval):
        start, end = interval
        print(interval)
        for event in self.events:
            if event.start > end:
                event.shift_by(-(end - start))
        new_cuts = []
        insert_point_found = False
        self.cuts.reverse()
        while self.cuts:
            cut_start, cut_end = self.cuts.pop()
            print(cut_start, cut_end)
            offset = cut_end - cut_start
            if cut_start < start:
                start += offset
                end += offset
                new_cuts.append((cut_start, cut_end))
            elif start < cut_start < end:
                end += offset
                while self.cuts and start < self.cuts[-1][0] < end:
                    cut = self.cuts.pop()
                    offset = cut[1] - cut[0]
                    end += offset
                new_cuts.append((start, end))
                insert_point_found = True
            else:  # cut is interiely after new cut
                if not insert_point_found:
                    new_cuts.append((start, end))
                    insert_point_found = True
                new_cuts.append((cut_start, cut_end))
        if not insert_point_found:
            new_cuts.append((start, end))
        self.cuts = new_cuts


def update_signal_plot(dataset: CurrentData, signal_plot: pg.PlotItem, current_hist: pg.PlotItem):
    signal_plot.clear()  # This might be unnecessary
    sample_rate = dataset.data_params.get('sample_rate')
    baseline = dataset.data_params.get('baseline')
    threshold = dataset.data_params.get('threshold')
    data = dataset.processed_data
    if sample_rate:
        t = np.arange(0, len(data)) / sample_rate
        signal_plot.plot(t, data, pen='b')
    else:
        signal_plot.plot(data, pen='b')
    if dataset.events:
        fits = dataset.event_fits()
        for x, y in fits:
            # x = np.concatenate([x, [0]])
            # y = np.concatenate([y, [0]])
            if sample_rate:
                x = x / sample_rate
            signal_plot.plot(x, y, pen='r')
    if dataset.file_type != '.abf':
        signal_plot.addLine(y=baseline, pen='g')
        signal_plot.addLine(y=threshold, pen='r')

    current_hist.clear()
    aph_y, aph_x = np.histogram(data, bins=1000)
    aph_hist = pg.PlotCurveItem(aph_x, aph_y, stepMode=True, fillLevel=0, brush='b')
    current_hist.addItem(aph_hist)
    current_hist.setXRange(np.min(data), np.max(data))
