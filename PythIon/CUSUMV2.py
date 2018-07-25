import numpy as np


# Recursive function that will detect events and output a nested data structure
# arranged as [mean, start, [mean, start, end], [mean, start, end], end]
# Where the nested pairs are sub events. Those events may themselves contain sub-events
def detect_events(data, num_starting_samples, event_list=[].copy(), start_idx=0):
    # num_starting_samples = 20  # Used to get a an initial mean; update once more familiar with data
    # Determine starting mean
    mean = data[0]
    sample_var = 0
    sub_event_offset = 0
    if event_list:
        # Gather information about previous state so we know when phase is over
        baseline = event_list[0]
    else:
        event_list.append(start_idx)
        baseline = np.mean(data[start_idx: start_idx + num_starting_samples])
    current_idx = start_idx

    # TODO: Make this memory efficient and make sure it one affects the data inside function; no side-effects
    for pt, idx in enumerate(data[start_idx:]):
        # Run through data, looking at every point
        old_mean = mean
        mean = mean + ((pt - mean) / idx)  # Rolling mean calculation
        # Rolling variance kind of; true_sd = sqrt(variance/(n - 1)), or true_sd = sqrt(variance/idx)
        sample_var = sample_var + (pt - old_mean) * (pt - mean)
        # Calculate rolling mean and standard deviation

        # If the deviation list large enough, mark it, and then make a recursive call
        deviation = True
        if deviation:
            # Recursive call that will detect more events
            # TODO: Data passed in recursive call should have the transition data points removed
            # Have function trim the beginning of data until it is within 1.5 sd of mean
            sub_event = detect_events(data[start_idx], num_starting_samples, event_list, start_idx)
            sub_event_offset += sub_event[-1] - sub_event[1]
            event_list.append(sub_event)
        else:
            event_list.append([baseline, start_idx, idx + start_idx])
            return event_list
    return event_list

# Consider the cumulative average as np.cumsum(data)/np.array(range(1, len(data))
# def moving_stats(data, anchor, stepsize):
#     basesd = np.std(data)
#     n_states = 0
#     var_m = data[0]
#     var_s = 0
#     mean = data[0]
#     for k in range(len(data)):
#         # algorithm to calculate running variance,
#         # details here: http://www.johndcook.com/blog/standard_deviation/
#         # From above source, M is moving average, S is moving
#         var_old_m = var_m
#         var_m = var_m + (data[k] - var_m) / float(k + 1 - anchor)
#         var_s = var_s + (data[k] - var_old_m) * (data[k] - var_m)
#         variance = var_s / float(k + 1 - anchor)
#         mean = ((k - anchor) * mean + data[k]) / float(k + 1 - anchor)
#         print(mean, variance)
#         if variance == 0:
#             # with low-precision data sets it is possible that two adjacent
#             # values are equal, in which case there is zero variance for the two-vector
#             # of sample if this occurs next to a detected jump. This is very, very rare, but it does happen.
#             variance = basesd * basesd
#             # in that case, we default to the local baseline variance,
#             # which is a good an estimate as any.
#             print('entered')
#
#         # instantaneous log-likelihood for current sample assuming local baseline has jumped in positive direction
#         logp = stepsize * basesd / variance * (data[k] - mean - stepsize * basesd / 2)
#         # instantaneous log-likelihood for current sample assuming local baseline has jumped in negative direction
#         logn = -stepsize * basesd / variance * (data[k] - mean + stepsize * basesd / 2)
#         c_pos[k] = c_pos[k - 1] + logp  # accumulate positive log-likelihoods
#         c_neg[k] = c_neg[k - 1] + logn  # accumulate negative log-likelihoods
#         g_pos[k] = max(g_pos[k - 1] + logp, 0)  # accumulate or reset positive decision function
#         g_neg[k] = max(g_neg[k - 1] + logn, 0)  # accumulate or reset negative decision function
#
#         if g_pos[k] > threshhold or g_neg[k] > threshhold:
#             if g_pos[k] > threshhold:  # significant positive jump detected
#                 jump = anchor + np.argmin(c_pos[anchor:k + 1])  # find the location of the start of the jump
#                 if jump - edges[n_states] > minlength:
#                     edges = np.append(edges, jump)
#                     n_states += 1
#             if g_neg[k] > threshhold:  # significant negative jump detected
#                 jump = anchor + np.argmin(c_neg[anchor:k + 1])
#                 if jump - edges[n_states] > minlength:
#                     edges = np.append(edges, jump)
#                     n_states += 1
#             anchor = k
#             c_pos[0:len(c_pos)] = 0  # reset all decision arrays
#             c_neg[0:len(c_neg)] = 0
#             g_pos[0:len(g_pos)] = 0
#             g_neg[0:len(g_neg)] = 0
#             mean = data[anchor]
#             var_m = data[anchor]
#             var_s = 0
#         if max_states > 0:
#             if n_states > max_states:
#                 print('too sensitive')
#                 print(threshhold, stepsize)
#                 n_states = 0
#                 k = 0
#                 stepsize = stepsize * 1.1
#                 threshhold = threshhold * 1.1
#                 logp = 0  # instantaneous log-likelihood for positive jumps
#                 logn = 0  # instantaneous log-likelihood for negative jumps
#                 c_pos = np.zeros(len(data), dtype='float64')  # cumulative log-likelihood function for positive jumps
#                 c_neg = np.zeros(len(data), dtype='float64')  # cumulative log-likelihood function for negative jumps
#                 g_pos = np.zeros(len(data), dtype='float64')  # decision function for positive jumps
#                 g_neg = np.zeros(len(data), dtype='float64')  # decision function for negative jumps
#                 edges = np.array([0], dtype='int64')  # init array w/ the pos of the first subevent - start of event
#                 anchor = 0  # the last detected change
#                 length = len(data)
#                 mean = data[0]
#                 variance = basesd ** 2
#                 k = 0
#                 n_states = 0
#                 var_m = data[0]
#                 var_s = 0
#                 mean = data[0]


def detect_cusum(data, base_sd, dt, threshhold=10, stepsize=3,
                 minlength=1000, max_states=-1):

    # dt = 1
    print('base_sd = ' + str(base_sd))
    # log_p = 0  # instantaneous log-likelihood for positive jumps
    # log_n = 0  # instantaneous log-likelihood for negative jumps
    c_pos = np.zeros(len(data), dtype='float64')  # cumulative log-likelihood function for positive jumps
    c_neg = np.zeros(len(data), dtype='float64')  # cumulative log-likelihood function for negative jumps
    g_pos = np.zeros(len(data), dtype='float64')  # decision function for positive jumps
    g_neg = np.zeros(len(data), dtype='float64')  # decision function for negative jumps
    edges = np.array([0], dtype='int64')  # init array with the position of the first subevent - the start of the event
    anchor = 0  # the last detected change
    length = len(data)
    # mean = data[0]
    # variance = base_sd ** 2
    k = 0
    n_states = 0
    var_m = data[0]
    var_s = 0
    mean = data[0]

    # sThreshhold = threshhold
    # sStepSize = stepsize

    while k < length - 1:
        if k % 1000 == 0:
            pass
        k += 1
        # algorithm to calculate running variance,
        # details here: http://www.johndcook.com/blog/standard_deviation/
        # From above source, M is moving average, S is cumulative variance
        var_old_m = var_m
        var_m = var_m + (data[k] - var_m) / float(k + 1 - anchor)
        var_s = var_s + (data[k] - var_old_m) * (data[k] - var_m)
        variance = var_s / float(k + 1 - anchor)
        mean = ((k - anchor) * mean + data[k]) / float(k + 1 - anchor)
        # with low-precision data sets it is possible that two adjacent values are equal, in which case there is zero
        # variance for the two-vector of sample if this occurs next to a detected jump. This is very, very rare, but it
        # does happen. in that case, we default to the local baseline variance, which is a good an estimate as any.
        if variance == 0:
            variance = base_sd * base_sd
            print('Zero Variance Point Detected')

        # instantaneous log-likelihood for current sample assuming local baseline has jumped in the positive direction
        log_p = stepsize * base_sd * ((data[k] - mean) - (stepsize * base_sd / 2)) / variance
        # instantaneous log-likelihood for current sample assuming local baseline has jumped in the negative direction
        log_n = stepsize * base_sd * ((mean - data[k]) - (stepsize * base_sd / 2)) / variance
        c_pos[k] = c_pos[k - 1] + log_p  # accumulate positive log-likelihoods
        c_neg[k] = c_neg[k - 1] + log_n  # accumulate negative log-likelihoods
        g_pos[k] = max(g_pos[k - 1] + log_p, 0)  # accumulate or reset positive decision function
        g_neg[k] = max(g_neg[k - 1] + log_n, 0)  # accumulate or reset negative decision function
        if g_pos[k] > threshhold or g_neg[k] > threshhold:
            if g_pos[k] > threshhold:  # significant positive jump detected
                jump = anchor + np.argmin(c_pos[anchor:k + 1])  # find the location of the start of the jump
            else:  # significant negative jump detected
                jump = anchor + np.argmin(c_neg[anchor:k + 1])
            if jump - edges[n_states] > minlength:
                edges = np.append(edges, jump)
                n_states += 1
            anchor = k
            c_pos[0:len(c_pos)] = 0  # reset all decision arrays
            c_neg[0:len(c_neg)] = 0
            g_pos[0:len(g_pos)] = 0
            g_neg[0:len(g_neg)] = 0
            mean = data[anchor]
            var_m = data[anchor]
            var_s = 0
        if max_states > 0:
            if n_states > max_states:
                print('too sensitive')
                print(threshhold, stepsize)
                n_states = 0
                k = 0
                stepsize = stepsize * 1.1
                threshhold = threshhold * 1.1
                log_p = 0  # instantaneous log-likelihood for positive jumps
                log_n = 0  # instantaneous log-likelihood for negative jumps
                c_pos = np.zeros(len(data), dtype='float64')  # cumulative log-likelihood function for positive jumps
                c_neg = np.zeros(len(data), dtype='float64')  # cumulative log-likelihood function for negative jumps
                g_pos = np.zeros(len(data), dtype='float64')  # decision function for positive jumps
                g_neg = np.zeros(len(data), dtype='float64')  # decision function for negative jumps
                edges = np.array([0], dtype='int64')  # init array w/ the pos of the first subevent - start of event
                anchor = 0  # the last detected change
                length = len(data)
                mean = data[0]
                variance = base_sd ** 2
                k = 0
                n_states = 0
                var_m = data[0]
                var_s = 0
                mean = data[0]

    edges = np.append(edges, len(data))  # mark the end of the event as an edge
    n_states += 1

    cusum = dict()
    # detect current levels during detected sub-events
    cusum['CurrentLevels'] = [np.average(data[int(edges[i] + minlength):int(edges[i + 1])]) for i in range(n_states)]
    cusum['EventDelay'] = edges * dt  # locations of sub-events in the data
    cusum['Threshold'] = threshhold  # record the threshold used
    cusum['stepsize'] = stepsize
    cusum['jumps'] = np.diff(cusum['CurrentLevels'])
    # self.__recordevent(cusum)

    return cusum
