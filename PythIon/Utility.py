import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pandas.io.parsers

from PythIon.abfheader import read_header
from typing import List
from inspect import getmembers
from scipy import ndimage
from scipy import signal
from scipy import io as spio

# from PythIon.plotguiuniversal import *
from PythIon.Widgets.PlotGUI import *


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
    except ValueError as e:
        print(e)
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


def generate_data_pts(data, start, end):
    return data[start:end]


# Loads the log file data and outputs a vector of voltages and the sample rate
def load_log_file(info_file_name, data_file_name, lp_filter_cutoff, output_sample_rate):
    def data_to_amps(raw_data, adc_bits, adc_vref, closedloop_gain, current_offset):
        # Computations to turn uint16 data into amps
        bitmask = (2 ** 16) - (1 + ((2 ** (16 - adc_bits)) - 1))
        raw_data = -adc_vref + ((2 * adc_vref * (raw_data & bitmask)) / 2 ** 16)
        raw_data = (raw_data / closedloop_gain + current_offset)
        data = raw_data[0]  # Retrurns the list to a single level: [[data]] -> [data]
        return data

    mat = spio.loadmat(info_file_name)
    # ADC is analog to digital converter
    # Loading in data about file from matlab data file
    sample_rate = mat['ADCSAMPLERATE'][0][0]
    ti_gain = mat['SETUP_TIAgain']
    pre_adc_gain = mat['SETUP_preADCgain'][0][0]
    current_offset = mat['SETUP_pAoffset'][0][0]
    adc_vref = mat['SETUP_ADCVREF'][0][0]
    adc_bits = mat['SETUP_ADCBITS']
    closedloop_gain = ti_gain * pre_adc_gain
    # Info has been loaded

    chimera_file = np.dtype('uint16')  # Was <u2 "Little-endian 2 byte unsigned integer"
    raw_data = np.fromfile(data_file_name, chimera_file)
    # Part to handle low sample rate
    if sample_rate < 4000e3:
        raw_data = raw_data[::round(sample_rate / output_sample_rate)]
    data = data_to_amps(raw_data, adc_bits, adc_vref, closedloop_gain, current_offset)
    # Data has been loaded

    # Fow filtering data (NG: Don't know why or what this does)
    wn = round(lp_filter_cutoff / (sample_rate / 2), 4)  #
    # noinspection PyTupleAssignmentBalance
    b, a = signal.bessel(4, wn, btype='low')
    # FIXME: scipy gives warning for this function due to internal implementation
    data = signal.filtfilt(b, a, data)

    return data, sample_rate


def load_opt_file(data_file_name, mat_file_name, ui, output_sample_rate, lp_filter_cutoff):
    data = np.fromfile(data_file_name, dtype=np.dtype('>d'))
    try:
        mat = spio.loadmat(mat_file_name + '_inf')
        mat_struct = mat[os.path.basename(mat_file_name)]
        # matstruct.shape
        mat = mat_struct[0][0]
        sample_rate = np.float64(mat['sample_rate'])
        filt_rate = np.float64(mat['filterfreq'])
    except TypeError as e:
        print(e)
        # try to load NFS file
        try:
            matfile = os.path.basename(mat_file_name)
            mat = spio.loadmat(mat_file_name)[matfile]
            sample_rate = np.float64(mat['sample_rate'])
            filt_rate = np.float64(mat['filterfreq'] * 1000)
            # potential = np.float64(mat['potential'])
            # pre_trigger_time_ms = np.float64(mat['pretrigger_time'])
            # post_trigger_time_ms = np.float64(mat['posttrigger_time'])

            # trigger_data = mat['triggered_pulse']
            # start_voltage = trigger_data[0].initial_value
            # final_voltage = trigger_data[0].ramp_target_value
            # ramp_duration_ms = trigger_data[0].duration
            # eject_voltage = trigger_data[1].initial_value
            # eject_duration_ms = np.float64(trigger_data[1].duration)
        except TypeError as e:
            print(e)
            return

    if sample_rate < output_sample_rate:
        print("data sampled at lower rate than requested, reverting to original sampling rate")
        ui.outputsamplerateentry.setText(str((round(sample_rate) / 1000)))
        output_sample_rate = sample_rate

    elif output_sample_rate > 250e3:
        print('sample rate can not be >250kHz for axopatch files, displaying with a rate of 250kHz')
        output_sample_rate = 250e3

    if lp_filter_cutoff >= filt_rate:
        print('Already LP filtered lower than or at entry, data will not be filtered')
        lp_filter_cutoff = filt_rate
        ui.LPentry.setText(str((round(lp_filter_cutoff) / 1000)))

    elif lp_filter_cutoff < 100e3:
        wn = round(lp_filter_cutoff / (100 * 10 ** 3 / 2), 4)
        # noinspection PyTupleAssignmentBalance
        b, a = signal.bessel(4, wn, btype='low')
        data = signal.filtfilt(b, a, data)
    else:
        print('Filter value too high, data not filtered')

    return data, output_sample_rate


def load_txt_file(data_file_name):
    data = pandas.io.parsers.read_csv(data_file_name, skiprows=1)
    # data = np.reshape(np.array(data),np.size(data))*10**9
    data = np.reshape(np.array(data), np.size(data))
    return data


def load_npy_file(data_file_name):
    data = np.load(data_file_name)
    return data


def load_abf_file(data_file_name, output_sample_rate, ui, lp_filter_cutoff, p1):
    f = open(data_file_name, "rb")  # reopen the file
    f.seek(6144, os.SEEK_SET)
    data = np.fromfile(f, dtype=np.dtype('<i2'))
    header = read_header(data_file_name)
    sample_rate = 1e6 / header['protocol']['fADCSequenceInterval']
    telegraph_mode = int(header['listADCInfo'][0]['nTelegraphEnable'])
    if telegraph_mode == 1:
        ab_flow_pass = header['listADCInfo'][0]['fTelegraphFilter']
        gain = header['listADCInfo'][0]['fTelegraphAdditGain']
    else:
        gain = 1
        ab_flow_pass = sample_rate
    data = data.astype(float) * (20. / (65536 * gain)) * 10 ** -9
    if len(header['listADCInfo']) == 2:
        # v = data[1::2] * gain / 10
        data = data[::2]
    else:
        pass
        # v = []

    if output_sample_rate > sample_rate:
        print('output sample_rate can not be higher than sample_rate, resetting to original rate')
        output_sample_rate = sample_rate
        ui.outputsamplerateentry.setText(str((round(sample_rate) / 1000)))
    if lp_filter_cutoff >= ab_flow_pass:
        print('Already LP filtered lower than or at entry, data will not be filtered')
        lp_filter_cutoff = ab_flow_pass
        ui.LPentry.setText(str((round(lp_filter_cutoff) / 1000)))
    else:
        wn = round(lp_filter_cutoff / (100 * 10 ** 3 / 2), 4)
        # noinspection PyTupleAssignmentBalance
        b, a = signal.bessel(4, wn, btype='low')
        data = signal.filtfilt(b, a, data)

    tags = header['listTag']
    for tag in tags:
        if tag['sComment'][0:21] == "Holding on 'Cmd 0' =>":
            cmdv = tag['sComment'][22:]
            # cmdv = [int(s) for s in cmdv.split() if s.isdigit()]
            cmdt = tag['lTagTime'] / output_sample_rate
            p1.addItem(pg.InfiniteLine(cmdt))
            # cmdtext = pg.TextItem(text = str(cmdv)+' mV')
            cmd_text = pg.TextItem(text=str(cmdv))
            p1.addItem(cmd_text)
            cmd_text.setPos(cmdt, np.max(data))

    return data, sample_rate, output_sample_rate, lp_filter_cutoff, p1


def update_p1(instance, t, data, baseline, threshold, file_type='.log'):
    instance.signal_plot.clear()  # This might be unnecessary
    instance.signal_plot.plot(t, data, pen='b')
    if file_type != '.abf':
        instance.signal_plot.addLine(y=baseline, pen='g')
        instance.signal_plot.addLine(y=threshold, pen='r')


def plot_on_load(instance, data, baseline, threshold, file_type, t, p1, p3):
    p1.clear()
    p1.setDownsampling(ds=True)
    # skips plotting first and last two points, there was a weird spike issue
    # p1.plot(t[2:][:-2], data[2:][:-2], pen='b')

    update_p1(instance, t, data, baseline, threshold, file_type)

    p1.autoRange()

    p3.clear()
    aph_y, aph_x = np.histogram(data, bins=1000)
    aph_hist = pg.PlotCurveItem(aph_x, aph_y, stepMode=True, fillLevel=0, brush='b')
    p3.addItem(aph_hist)
    p3.setXRange(np.min(data), np.max(data))


def calc_dt(events):
    return np.concatenate([[0], np.diff([event.start for event in events]) / events[0].output_sample_rate])


def calc_frac(events):
    return [event.local_baseline / events[0].baseline for event in events]


# Function to hide ui setup boilerplate
def setupUi(form):
    # TODO: Maybe move the top two lines out of function
    ui = Ui_PythIon()
    ui.setupUi(form)

    # Linking buttons to main functions
    ui.loadbutton.clicked.connect(form.get_file)
    ui.analyzebutton.clicked.connect(form.analyze)
    ui.cutbutton.clicked.connect(form.cut)
    ui.baselinebutton.clicked.connect(form.set_baseline)
    ui.clearscatterbutton.clicked.connect(form.clear_scatter)
    ui.deleteeventbutton.clicked.connect(form.delete_event)
    ui.invertbutton.clicked.connect(form.invert_data)
    ui.concatenatebutton.clicked.connect(form.concatenate_text)
    ui.nextfilebutton.clicked.connect(form.next_file)
    ui.previousfilebutton.clicked.connect(form.previous_file)
    ui.showcatbutton.clicked.connect(form.show_cat_trace)
    ui.savecatbutton.clicked.connect(form.save_cat_trace)
    ui.gobutton.clicked.connect(form.inspect_event)
    ui.previousbutton.clicked.connect(form.previous_event)
    ui.nextbutton.clicked.connect(form.next_event)
    ui.savefitsbutton.clicked.connect(form.save_event_fits)
    ui.fitbutton.clicked.connect(form.cusum)
    ui.Poresizeraction.triggered.connect(form.size_pore)
    ui.actionBatch_Process.triggered.connect(form.batch_info_dialog)

    # Setting up plotting elements and their respective options
    # TODO: Make list of plots (Setting all to white I assume)
    ui.signalplot.setBackground('w')
    ui.scatterplot.setBackground('w')
    ui.eventplot.setBackground('w')
    ui.frachistplot.setBackground('w')
    ui.delihistplot.setBackground('w')
    ui.dwellhistplot.setBackground('w')
    ui.dthistplot.setBackground('w')
    # ui.PSDplot.setBackground('w')
    return ui


def setup_signal_plot(ui):
    signal_plot = ui.signalplot.addPlot()
    signal_plot.setLabel('bottom', text='Time', units='s')
    signal_plot.setLabel('left', text='Current', units='A')
    # signal_plot.enableAutoRange(axis='x')
    signal_plot.setClipToView(clip=True)  # THIS IS THE MOST IMPORTANT LINE!!!!
    signal_plot.setDownsampling(ds=True, auto=True, mode='peak')
    return signal_plot


def setup_event_plot(clicked):
    event_plot = pg.ScatterPlotItem()
    event_plot.sigClicked.connect(clicked)
    return event_plot


def setup_scatter_plot(instance, p2):
    w1 = instance.ui.scatterplot.addPlot()
    w1.addItem(p2)
    w1.setLabel('bottom', text='Time', units=u'μs')
    w1.setLabel('left', text='Fractional Current Blockage')
    w1.setLogMode(x=True, y=False)
    w1.showGrid(x=True, y=True)
    return w1


def setup_cb(ui):
    cb = pg.ColorButton(ui.scatterplot, color=(0, 0, 255, 50))
    cb.setFixedHeight(30)
    cb.setFixedWidth(30)
    cb.move(0, 210)
    cb.show()
    return cb


def setup_plots(instance):  # TODO: revisit name
    frac_hist = instance.ui.frachistplot.addPlot()
    frac_hist.setLabel('bottom', text='Fractional Current Blockage')
    frac_hist.setLabel('left', text='Counts')

    delta_hist = instance.ui.delihistplot.addPlot()
    delta_hist.setLabel('bottom', text='ΔI', units='A')
    delta_hist.setLabel('left', text='Counts')

    duration_hist = instance.ui.dwellhistplot.addPlot()
    duration_hist.setLabel('bottom', text='Log Dwell Time', units='μs')
    duration_hist.setLabel('left', text='Counts')

    dt_hist = instance.ui.dthistplot.addPlot()
    dt_hist.setLabel('bottom', text='dt', units='s')
    dt_hist.setLabel('left', text='Counts')
    return frac_hist, delta_hist, duration_hist, dt_hist


def load_logo():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logo = ndimage.imread(dir_path + os.sep.join(["", "Assets", "pythionlogo.png"]))
    logo = np.rot90(logo, -1)
    logo = pg.ImageItem(logo)
    return logo


def setup_voltage_hist(ui, logo):
    p3 = ui.eventplot.addPlot()
    p3.hideAxis('bottom')
    p3.hideAxis('left')
    p3.addItem(logo)
    p3.setMouseEnabled(x=True, y=False)
    # Maybe make log scaled on y-axis
    p3.setAspectLocked(True)
    return p3


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


def save_cat_data(data, events, ui, info_file_name):
    event_buffer = np.int(ui.eventbufferentry.text())
    baseline = events[0].baseline
    cat_data = np.array([])
    cat_fits = np.array([])
    # NOTE: Below was changed from idx in range(num_events - 1) due to assumed off-by-one error
    # Section with similar code was changed as well
    for idx, event in enumerate(events[:-1]):
        adj_start = event.start - event_buffer
        adj_end = event.end + event_buffer
        if adj_end > events[idx + 1].start:
            print('overlapping event')
            continue
        data_pts = generate_data_pts(data, adj_start, adj_end)
        event_pts = generate_event_pts(adj_start, adj_end, event_buffer, baseline, event.delta)
        cat_data = np.concatenate((cat_data, data_pts), 0)
        cat_fits = np.concatenate((cat_fits, event_pts), 0)

    # cat_data = cat_data[::10]
    cat_data.astype('d').tofile(info_file_name + '_cattrace.bin')


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


def event_info_update(data, info_file_name, t, cb, events, ui, sdf, time_plot, durations_plot, w1, w2, w3, w4, w5):
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


BILLION = 10 ** 9  # Hopefully makes reading clearer


class Event(object):
    subevents: List

    def __init__(self, data, start, end, baseline, output_sample_rate, subevents=None):
        # Start and end are the indicies in the data
        if subevents is None:
            subevents = []
        self.baseline = baseline
        self.output_sample_rate = output_sample_rate

        self.local_baseline = np.mean(data[start:end])  # Could be made more precise, but good enough
        # True can false are converted to 1 and 0, np.argmax returns index of first ture value
        true_start = start - np.argmax(data[start::-1] > self.local_baseline) + 1
        true_end = end - np.argmax(data[end - 1::-1] < self.local_baseline)
        self.start = true_start
        self.end = true_end

        self.data = data[self.start:self.end + 1]
        # TODO: Fix slight error in calculations; Too much past the tails is included in fall and rise
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


class VoltageData(object):
    events: List[Event]

    def __init__(self, data, output_sample_rate, data_file_name=None, info_file_name=None, file_type=None,
                 lp_filter_cutoff=None, sample_rate=None, max_states=None, baseline=None):
        self.baseline = baseline
        self.max_states = max_states
        self.sample_rate = sample_rate
        self.lp_filter_cutoff = lp_filter_cutoff
        self.file_type = file_type
        self.info_file_name = info_file_name
        self.data = data
        self.data_file_name = data_file_name
        self.output_sample_Rate = output_sample_rate
        self.events = []

    def delete_event(self):
        pass

    def get_event_prop(self, prop_name):
        if not self.events:
            return
        if prop_name not in self.events[0].__dict__.keys():
            return

        return [event.__getattribute__(prop_name) for event in self.events]