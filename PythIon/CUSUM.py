import math
from typing import List, Tuple
from PythIon.Utility import load_log_file

from scipy import signal
from scipy import io as spio
import numpy as np


class Event(object):
    subevents: List

    def __init__(self, data, start, end, baseline, output_sample_rate, subevents=[].copy()):
        # Start and end are the indicies in the data
        self.baseline = baseline
        self.output_sample_rate = output_sample_rate

        self.local_baseline = np.mean(data[start:end + 1])  # Could be made more precise, but good enough
        # True can false are converted to 1 and 0, np.argmax returns index of first ture value
        true_start = start - np.argmax(data[start::-1] > self.local_baseline) + 1
        true_end = end - np.argmax(data[end::-1] < self.local_baseline)
        self.start = true_start
        self.end = true_end

        self.data = data[self.start:self.end + 1]
        # TODO: Fix slight error in calculations; To much past the tails is included in fall and rise
        # Rise_end seems to have consistent issues
        self.noise = np.std(self.data)
        self.duration = (self.end - self.start) / output_sample_rate  # In seconds
        self.delta = baseline - np.min(self.data)  # In Volts
        fall_offset = np.argmax(data[self.start::-1] > baseline)
        rise_offset = 0
        if self.end < len(data):  # Condition can fail if voltage has not returned to normal by the end of the data
            rise_offset = np.argmax(data[self.end:] > baseline)

        self.fall_start = self.start - fall_offset + 1
        self.rise_end = self.end + rise_offset - 1
        self.full_data = data[self.fall_start:self.rise_end + 1]

        self.subevents = subevents

    def __repr__(self):
        return str((self.start/self.output_sample_rate, self.end/self.output_sample_rate))


# Recursive function that will detect events and output a nested data structure
# arranged as [mean, start, [mean, start, end], [mean, start, end], end]
# Where the nested pairs are sub events. Those events may themselves contain sub-events
def cusum(data, base_sd, output_sample_rate, baseline=None,
          deviation_length=1000, stepsize=3, deviation_size=3.5, anchor=0, level=0, max_level=3):
    """

    :param max_level:
    :param output_sample_rate:
    :param baseline:
    :param deviation_length:
    :param level:
    :param data: Voltage data in a list
    :param base_sd: standard deviation of selected portion of
    :param stepsize: Will only look at one point ever step size; saves compute time
    :param deviation_size: Size of deviation to detect in standard deviations
    :param anchor: Index of last detected deviation
    :return: Returns a nested list of events. Format TBD
    """
    # Note: edges of event go with event ie. \_/ is the event, not \_, _/, or _.
    # Event duration is marked for _ (low volatage) section of event. fall_start and rise_end mark entire \_/ region
    # Adapted from https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
    deviation_length = round(deviation_length / stepsize)
    alpha = 0.0027  # Probability of false alarm
    beta = 0.01  # Probability of failing to detect shift
    k = deviation_size * base_sd / 2
    d = (2 / deviation_size ** 2) * math.log((1 - beta) / alpha)
    h = d * k * deviation_length  # Threshold that can't be exceeded by by either control limit (both are positive)
    events = []
    # end_idx = 0

    # Using a running mean instead of overall mean
    def new_control_limit(current_limits, deviation, k):
        current_max = current_limits[1]
        current_min = current_limits[0]
        new_max = max(0, current_max + deviation - k)
        new_min = max(0, current_min - deviation - k)
        return new_min, new_max

    # (Lower Limit, Upper Limit)
    limits: List[Tuple[float, float]] = [(0, 0)]  # List of ordered pairs
    subevent_offset = 0  # Number of points contained in subevents
    running_mean = data[anchor]
    if not baseline:
        baseline = running_mean
    running_variance = base_sd ** 2
    running_sd = base_sd
    n = anchor + 1  # Index containing current location in data
    cum_sum = 0  # Cummulative sum of all deviations; equivalent to CUSUM col on bottom of NIST page
    # Implmented from wikipedia and NIST page
    while n < len(data) - 1:
        pt = data[n]
        old_mean = running_mean
        running_mean = running_mean + (pt - running_mean) / (n + 1 - anchor)
        running_variance = running_variance + (pt - old_mean) * (pt - running_mean)

        if n >= anchor + 2:
            # print(n, anchor, subevent_offset, n - anchor - subevent_offset - 1, level)
            running_sd = math.sqrt(running_variance / (n - anchor - subevent_offset - 1))
            k = deviation_size * running_sd / 2
            h = d * k * deviation_length

        deviation = pt - running_mean
        cum_sum += deviation
        old_limit = limits[-1]
        limit = new_control_limit(old_limit, deviation, k)
        limits.append(limit)
        # Assumes the data starts at universal baseline
        if limit[0] > h:  # Handles dipping into event
            too_deep = level > max_level
            if too_deep:  # Don't get carried away
                break
            limits[-1] = (0, 0)  # Reset the accumulated errors
            event = cusum(data, running_sd, output_sample_rate, baseline=running_mean, anchor=n, level=level+1,
                          stepsize=stepsize, max_level=max_level)
            events.append(event)
            n = event.rise_end  # No need to add anything since n is incremented below
            subevent_offset += event.rise_end - event.fall_start
        elif limit[1] > h:  # Handles returning to baseline
            break
        n += stepsize

    if level == 0:
        return events
    else:
        return Event(data, anchor, n, baseline, output_sample_rate, events)


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


if __name__ == "__main__":
    # Mat file is matlab data. Should be in a more portable format (JSON?); Was changed to info_file_name
    info_file_name = r"C:\Users\Noah PC\PycharmProjects\Pyth-Ion\PythIon\Sample Data\3500bp-200mV.mat"
    data_file_name = r"C:\Users\Noah PC\PycharmProjects\Pyth-Ion\PythIon\Sample Data\3500bp-200mV.log"
    lp_filter_cutoff = 100000
    out_sample_rate = 4166670
    threshold = 0.3e-9
    data, sample_rate = load_log_file(info_file_name, data_file_name, lp_filter_cutoff, out_sample_rate)
    data = data[20:-20]  # Removing weird spikes from data
    small_data = data[:int(1e6)]  # First million points contain one event
    base_sd = np.std(data[:200000])
    baseline = np.mean(data)
    # events = analyze(data, threshold, out_sample_rate)
    # num_events = len(events)
    cusum_result = cusum(data, base_sd, out_sample_rate)
    # frac = [event.local_baseline / baseline for event in events]
    # Time between the starts of events?
    # dt = np.concatenate([[0], np.diff([event.start for event in events]) / out_sample_rate])



