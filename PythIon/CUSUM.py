import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from PythIon.Utility import running_stats, Event, CurrentData
from PythIon.SetupUtilities import load_log_file


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
    :param stepsize: Will only look at one point ever step size  saves compute time
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
    cum_sum = 0  # Cummulative sum of all deviations  equivalent to CUSUM col on bottom of NIST page
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
        return Event(data, anchor, n, baseline, output_sample_rate)


def correct_cusum(x, delta, h):
    """
 (Detection of abrupt changes in the mean ; optimal for Gaussian signals)
    :param x: signal samples
    :param delta: most likely jump to be detected
    :param h: threshold for the detection test
    :return: mc   : piecewise constant segmented signal
             kd   : detection times (in samples)
             krmv : estimated change times (in samples)
    """
    # Algo initialization
    detection_number = 0  # detection number
    kd = []  # detection time( in samples)
    krmv = []  # estimated change time( in samples)
    k0 = 0  # initial sample
    k = 0  # current sample
    mean = [x[k0]] * len(x)  # mean value estimation
    variance = [0] * len(x)  # variance estimation
    # (negative decision, positive decision)
    initial_array = [0, 0]
    log_liklihoods = [initial_array] * len(x)  # instantaneous log - likelihood ratio
    cumulative_sums = [initial_array] * len(x)  # cumulated sum for positive jumps
    decision_values = [initial_array] * len(x)  # decision function for positive jumps

    # Global Loop
    while k < len(x) - 1:
        # current sample
        k += 1
        # mean and variance estimation(from initial to current sample)
        mean[k] = mean[k - 1] + (x[k] - mean[k - 1]) / (k - k0 + 1)
        variance[k] = variance[k - 1] + (x[k] - mean[k - 1]) * (x[k] - mean[k])
        # instantaneous log - likelihood ratios
        negative_liklihood = -delta / variance[k] * (x[k] - mean[k] + delta / 2)
        positive_liklihood = delta / variance[k] * (x[k] - mean[k] - delta / 2)
        log_liklihoods[k] = [negative_liklihood, positive_liklihood]
        # cumulated sums
        cumulative_sums[k] = [cumulative_sums[k - 1][0] + log_liklihoods[k][0],
                              cumulative_sums[k - 1][1] + log_liklihoods[k][1]]
        # decision functions
        decision_values[k] = [max(decision_values[k - 1][0] + log_liklihoods[k][0], 0),
                              max(decision_values[k - 1][1] + log_liklihoods[k][1], 0)]
        if decision_values[k][0] > h or decision_values[k][1] > h:
            # detection number and detection time update
            kd.append(k)
            # change time estimation
            neg_cumulative_sums = [cum_sum[0] for cum_sum in cumulative_sums[k0:]]
            pos_cumulative_sums = [cum_sum[1] for cum_sum in cumulative_sums[k0:]]
            kmin = np.argmin(neg_cumulative_sums)
            krmv.append(kmin + k0)
            if decision_values[k][1] > h:
                kmin = np.argmin(pos_cumulative_sums)
                krmv[detection_number] = kmin + k0
            detection_number = detection_number + 1

            # algorithm reinitialization
            k0 = k
            mean[k0] = x[k0]
            variance[k0] = 0
            log_liklihoods[k0] = initial_array
            cumulative_sums[k0] = initial_array
            decision_values[k0] = initial_array
        # waitbar(k / length(x), w)

    data_list = []
    if detection_number == 0:
        mean_voltage = np.mean(x)
        mc = np.repeat(mean_voltage, len(x))
        data_list.append([0, len(x), len(x), mean_voltage])
    elif detection_number == 1:
        mc = np.concatenate((np.repeat(mean[krmv[0]], krmv[0]), np.repeat(mean[k], k - krmv[0])))
        data_list.append([0, krmv[0], krmv[0] - 1, mean[krmv[0]]])
        data_list.append([krmv[0], k + 1, k - krmv[0] + 1, mean[k]])
    else:
        # print(krmv[0])
        mc = np.repeat(mean[krmv[0]], krmv[0])
        data_list.append([0, krmv[0], krmv[0], mean[krmv[0]]])
        for idx in range(1, detection_number):
            new_piece = np.repeat(mean[krmv[idx]], krmv[idx] - krmv[idx - 1])
            mc = np.concatenate((mc, new_piece))
            data_list.append([krmv[idx - 1], krmv[idx], krmv[idx] - krmv[idx - 1], mean[krmv[idx]]])
        new_piece = np.repeat(mean[k], k - krmv[-1])
        mc = np.concatenate((mc, new_piece))
        data_list.append([krmv[-1], k + 1, k - krmv[-1] + 1, mean[k]])

    table_labels = ['Start Index', 'End Index', 'Duration', 'Voltage Level']
    event_table = pd.DataFrame(data=data_list, columns=table_labels)
    return mc, kd, krmv, event_table


if __name__ == "__main__":
    # Mat file is matlab data. Should be in a more portable format (JSON?)  Was changed to info_file_name
    file_name = '3500bp-200mV'
    working_dir = r"C:\Users\Noah PC\PycharmProjects\Pyth-Ion\PythIon\Sample Data"
    data_path = os.path.join(working_dir, file_name) + '.log'
    dataset = CurrentData(data_path)
    low_pass_cutoff = 7500
    dataset.process_data(low_pass_cutoff)
    dataset.detect_events()

    out_sample_rate = 4166670
    threshold = 0.3e-9
    data, sample_rate = load_log_file(file_name, working_dir)
    data = data[20:-20]  # Removing weird spikes from data
    correct_test = correct_cusum(data[:12000000], 3e-10, 1e-10)
    # base_sd = np.std(data[:200000])
    # baseline = np.mean(data)
    # events = analyze(data, threshold, out_sample_rate)
    # num_events = len(events)
    # cusum_result = cusum(data, base_sd, out_sample_rate)
    # frac = [event.local_baseline / baseline for event in events]
    # # Time between the starts of events?
    # dt = np.concatenate([[0], np.diff([event.start for event in events]) / out_sample_rate])



