import pandas as pd
import pandas.io.parsers
import pyqtgraph as pg
from scipy import io as spio
from scipy import ndimage
from scipy import signal
from typing import List

from PythIon.Utility import *

from PythIon import CUSUM
from PythIon.PoreSizer import *
from PythIon.abfheader import *
from PythIon.batchinfo import *
# plotguiuniversal works well for mac and laptops,
# for larger screens try PlotGUI
# from PlotGUI import *
from PythIon.plotguiuniversal import *

from scipy.ndimage.filters import gaussian_filter1d
from scipy.fftpack import fft


class GUIForm(QtWidgets.QMainWindow):
    events: List[Event]

    def __init__(self, width, height, master=None):
        # Setup GUI and draw elements from UI file
        QtWidgets.QMainWindow.__init__(self, master)
        pg.setConfigOptions(antialias=True)

        self.ui = setup_ui(self)
        self.p1 = setup_p1(self.ui)
        self.p2 = setup_p2(self.clicked)
        self.w1 = setup_w1(self, self.p2)
        self.cb = setup_cb(self.ui)
        self.w2, self.w3, self.w4, self.w5 = setup_plots(self)
        self.logo = load_logo()
        self.p3 = setup_p3(self.ui, self.logo)

        # Initializing various variables used for analysis
        self.data_file_name = None
        self.base_region = pg.LinearRegionItem()
        self.base_region.hide()  # Needed since LinearRegionItems is loaded as visible
        self.cut_region = pg.LinearRegionItem(brush=(198, 55, 55, 75))
        self.cut_region.hide()
        self.last_event = []
        self.last_clicked = []
        self.has_baseline_been_set = False
        self.last_event = 0
        self.cat_data = []
        self.sdf = pd.DataFrame(columns=['fn', 'color', 'deli', 'frac',
                                         'dwell', 'dt', 'startpoints', 'endpoints'])
        self.analyze_type = 'coarse'
        self.num_events = 0
        self.batch_processor = None
        self.file_list = None
        self.data = None
        self.baseline = None
        self.var = None
        self.sample_rate = None
        self.output_sample_rate = None
        self.mat_file_name = None
        self.t = None
        self.cat_fits = None
        self.min_level_t = None
        self.LPfiltercutoff = None
        self.max_states = 1
        self.width = width
        self.height = height
        self.file_type = None
        self.events = None

        self.batch_info = pd.DataFrame(columns=list(['cutstart', 'cutend']))
        self.total_plot_points = len(self.p2.data)
        self.threshold = np.float64(self.ui.thresholdentry.text()) * 10 ** -9

    def load(self, load_and_plot=True):
        sdf = self.sdf
        ui = self.ui
        p3 = self.p3
        p2 = self.p2
        p1 = self.p1
        baseline = self.baseline
        data_file_name = self.data_file_name
        var = self.var
        sample_rate = self.sample_rate
        has_baseline_been_set = self.has_baseline_been_set

        p3.clear()
        p3.setLabel('bottom', text='Current', units='A', unitprefix='n')
        p3.setLabel('left', text='', units='Counts')
        p3.setAspectLocked(False)

        colors = np.array(sdf.color)
        for i in range(len(colors)):
            colors[i] = pg.Color(colors[i])

        p2.setBrush(colors, mask=None)

        ui.eventinfolabel.clear()
        ui.eventcounterlabel.clear()
        ui.meandelilabel.clear()
        ui.meandwelllabel.clear()
        ui.meandtlabel.clear()
        ui.eventnumberentry.setText(str(1))

        float_tol = 10 ** -9  # I may be completely misunderstanding this
        threshold = np.float64(ui.thresholdentry.text()) * float_tol
        ui.filelabel.setText(data_file_name)
        print(data_file_name)
        # TODO: Remove the magic numbers (1000 specifically)
        lp_filter_cutoff = np.float64(ui.LPentry.text()) * 1000
        # use integer multiples of 4166.67 ie 2083.33 or 1041.67
        output_sample_rate = np.float64(ui.outputsamplerateentry.text()) * 1000

        # noinspection PyTypeChecker
        mat_file_name = str(os.path.splitext(data_file_name)[0])
        # noinspection PyTypeChecker
        file_type = str(os.path.splitext(data_file_name)[1])
        if file_type == '.log':
            data, sample_rate = load_log_file(mat_file_name, data_file_name, lp_filter_cutoff, output_sample_rate)
        elif file_type == '.opt':
            data, output_sample_rate = load_opt_file(data_file_name, mat_file_name, ui,
                                                     output_sample_rate, lp_filter_cutoff)
        elif file_type == '.txt':
            data = load_txt_file(data_file_name)
        elif file_type == '.npy':
            data = load_npy_file(data_file_name)
        elif file_type == '.abf':
            data, sample_rate, output_sample_rate, lp_filter_cutoff, p1 = \
                load_abf_file(data_file_name, output_sample_rate, ui, lp_filter_cutoff, p1)
        else:
            return

        # zeroed_data = np.array(data) - np.mean(data[np.where(np.array(data) > 0.1e-9)])
        # step = np.hstack((np.ones(len(data)), -1 * np.ones(len(data))))
        # box = np.hstack((np.zeros(len(data)), np.ones(500), (np.zeros(len(data)))))
        # data = np.diff(signal.fftconvolve(zeroed_data, box, mode='valid'))
        t = np.arange(0, len(data)) / output_sample_rate

        # TODO: Separate function

        if not has_baseline_been_set:
            baseline = np.median(data)
            var = np.std(data)
        ui.eventcounterlabel.setText('Baseline=' + str(round(baseline * BILLION, 2)) + ' nA')

        if load_and_plot:
            plot_on_load(self, data, baseline, threshold, file_type, t, p1, p3)

        self.sample_rate = sample_rate
        self.LPfiltercutoff = lp_filter_cutoff
        self.output_sample_rate = output_sample_rate
        self.mat_file_name = mat_file_name
        self.var = var
        self.baseline = baseline
        self.threshold = threshold
        self.t = t
        self.data = data
        self.file_type = file_type

    #        if self.v != []:
    #            self.p1.plot(self.t[2:][:-2],self.v[2:][:-2],pen='r')

    #        self.w6.clear()
    #        f, Pxx_den = signal.welch(data*10**12, self.outputsamplerate, nperseg = self.outputsamplerate)
    #        self.w6.plot(x = f[1:], y = Pxx_den[1:], pen = 'b')
    #        self.w6.setXRange(0,np.log10(self.outputsamplerate))

    # Static
    def get_file(self):
        wd = os.getcwd()
        try:
            # attempt to open dialog from most recent directory
            data_file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', wd,
                                                                   "*.log;*.opt;*.npy;*.abf")
            if data_file_name == ('', ''):
                self.data_file_name = data_file_name
                return

            data_file_name = data_file_name[0]
            self.data_file_name = data_file_name
            wd = os.path.dirname(data_file_name)
            self.load()

        except IOError as e:
            # if user cancels during file selection, exit loop
            print(e)

    def analyze(self):
        # start_points, end_points, mins = None, None, None  # unused
        data = self.data
        if data is None:
            return
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4
        w5 = self.w5
        output_sample_rate = self.output_sample_rate
        t = self.t

        sdf = self.sdf
        time_plot = self.p1
        durations_plot = self.p2
        info_file_name = self.mat_file_name

        cb = self.cb

        analyze_type = 'coarse'
        w2.clear()
        w3.clear()
        w4.clear()
        w5.clear()
        ui = self.ui
        threshold = np.float64(ui.thresholdentry.text()) * 10 ** -9
        # find all points below threshold

        # Setup happens above
        events = analyze(data, threshold, output_sample_rate)
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
            time_plot.plot(t[start_points], data[start_points], pen=None, symbol='o', symbolBrush='g', symbolSize=10)
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

        # self.save()
        # self.save_target()
        # Reassigning class variables
        self.events = events
        self.analyze_type = analyze_type
        self.sdf = sdf

    def save(self):
        info_file_name = self.mat_file_name
        events = self.events
        deltas = [event.delta for event in events]
        durations = [event.duration for event in events]
        frac = calc_frac(events)
        dt = calc_dt(events)
        noise = [event.noise for event in events]
        col_stack = np.column_stack((deltas, frac, durations, dt, noise))
        header_names = '\t'.join(["deli", "frac", "dwell", "dt", 'stdev'])
        np.savetxt(info_file_name + 'DB.txt', col_stack, delimiter='\t', header=header_names)

    # Called by Go button
    # TODO: Continue cleanump and input sanitizing
    def inspect_event(self, clicked=[].copy()):
        events = self.events
        if not events:
            return
        num_events = len(events)
        baseline = events[0].baseline
        sdf = self.sdf
        ui = self.ui
        data = self.data
        p2 = self.p2
        p3 = self.p3
        t = self.t
        mat_file_name = self.mat_file_name

        # Reset plot
        p3.setLabel('bottom', text='Time', units='s')
        p3.setLabel('left', text='Current', units='A')
        p3.clear()

        # Correct for user error if non-extistent number is entered
        event_buffer = num_from_text_element(ui.eventbufferentry, default=1000)
        first_index = sdf.fn[sdf.fn == mat_file_name].index[0]
        if not clicked:
            event_number = int(ui.eventnumberentry.text())
        else:
            # noinspection PyUnresolvedReferences
            event_number = clicked - first_index
            ui.eventnumberentry.setText(str(event_number))
        if event_number >= num_events:
            event_number = num_events - 1
            ui.eventnumberentry.setText(str(event_number))

        # plot event trace
        event = events[event_number]
        adj_start = int(event.start - event_buffer)
        adj_end = int(event.end + event_buffer)
        p3.plot(t[adj_start:adj_end], data[adj_start:adj_end], pen='b')

        event_fit = np.concatenate((
            np.repeat(np.array([baseline]), event_buffer),
            np.repeat(np.array([baseline - event.delta]), data[event.start] - data[event.end]),
            np.repeat(np.array([baseline]), event_buffer)), 0)
        # plot event fit
        p3.plot(t[adj_start:adj_end], event_fit, pen=pg.mkPen(color=(173, 27, 183), width=3))
        p3.autoRange()

        # Mark event that is being viewed on scatter plot

        colors = np.array(sdf.color)
        for i in range(len(colors)):
            colors[i] = pg.Color(colors[i])
        colors[first_index + event_number] = pg.mkColor('r')

        p2.setBrush(colors, mask=None)

        # Mark event start and end points
        p3.plot([t[event.start], t[event.start]],
                [data[event.start], data[event.start]], pen=None,
                symbol='o', symbolBrush='g', symbolSize=12)
        p3.plot([t[event.end], t[event.end]],
                [data[event.end], data[event.end]], pen=None,
                symbol='o', symbolBrush='r', symbolSize=12)

        duration = round(event.duration, 2)
        delta = round(event.delta * BILLION, 2)
        ui.eventinfolabel.setText('Dwell Time=' + str(duration) + u' μs,   Deli=' + str(delta) + ' nA')

    #        if ui.cusumstepentry.text() != 'None':
    #
    #
    #
    #            x=data[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer]
    #            mins=signal.argrelmin(x)[0]
    #            drift=.0
    #            fitthreshold = np.float64(ui.cusumstepentry.text())
    #            eventfit=np.array((0))
    #
    #            gp, gn = np.zeros(x.size), np.zeros(x.size)
    #            ta, tai, taf = np.array([[], [], []], dtype=int)
    #            tap, tan = 0, 0
    #            # Find changes (online form)
    #            for i in range(mins[0], mins[-1]):
    #                s = x[i] - x[i-1]
    #                gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
    #                gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
    #                if gp[i] < 0:
    #                    gp[i], tap = 0, i
    #                if gn[i] < 0:
    #                    gn[i], tan = 0, i
    #                if gp[i] > fitthreshold or gn[i] > fitthreshold:  # change detected!
    #                    ta = np.append(ta, i)    # alarm index
    #                    tai = np.append(tai, tap if gp[i] > fitthreshold else tan)  # start
    #                    gp[i], gn[i] = 0, 0      # reset alarm
    #
    #            eventfit=np.repeat(np.array(baseline),ta[0])
    #            for i in range(1,ta.size):
    #                eventfit=np.concatenate((eventfit,np.repeat(np.array(np.mean(x[ta[i-1]:ta[i]])),ta[i]-ta[i-1])))
    #            eventfit=np.concatenate((eventfit,np.repeat(np.array(baseline),x.size-ta[-1])))
    #            p3.plot(t[startpoints[eventnumber]-eventbuffer:endpoints[eventnumber]+eventbuffer],eventfit
    #                ,pen=pg.mkPen(color=(255,255,0),width=3))
    #    #        pg.plot(eventfit)
    #
    #
    #            p3.plot(t[ta+startpoints[eventnumber]-eventbuffer],x[ta],pen=None,symbol='o',symbolBrush='m',symbolSize=8)
    #
    #
    #

    def next_event(self):
        element = self.ui.eventnumberentry
        num_events = len(self.events)
        # Ignore command if there are no events
        if not self.events:
            return
        input_num = num_from_text_element(element, 1, num_events, default=1)
        element.setText(str(input_num + 1))
        self.inspect_event()

    def previous_event(self):
        element = self.ui.eventnumberentry
        num_events = len(self.events)
        # Ignore command if there are no events
        if not self.events:
            return
        input_num = num_from_text_element(element, 1, num_events, default=1)
        element.setText(str(input_num - 1))
        self.inspect_event()

    def cut(self):
        """
Allows user to select region of data to be removed
        """
        data = self.data
        if data is None:
            return
        ui = self.ui
        file_type = self.file_type
        threshold = self.threshold
        baseline = self.baseline
        var = self.var
        output_sample_rate = self.output_sample_rate
        t = self.t
        batch_info = self.batch_info
        # first check to see if cutting
        cut_region = self.cut_region
        p1 = self.p1
        p3 = self.p3
        has_baseline_been_set = self.has_baseline_been_set

        if cut_region.isVisible():
            # If cut region has been set, cut region and replot remaining data
            left_bound, right_bound = cut_region.getRegion()
            cut_region.hide()
            p1.clear()
            p3.clear()
            selected_pts = np.arange(np.int(max(left_bound, 0) * output_sample_rate),
                                     np.int(right_bound * output_sample_rate))
            data = np.delete(data, selected_pts)
            t = np.arange(0, len(data)) / output_sample_rate

            if not has_baseline_been_set:
                baseline = np.median(data)
                var = np.std(data)
                has_baseline_been_set = True
                ui.eventcounterlabel.setText('Baseline=' + str(round(baseline * BILLION, 2)) + ' nA')

            update_p1(self, t, data, baseline, threshold, file_type)
            # aph_y, aph_x = np.histogram(data, bins = len(data)/1000)
            aph_y, aph_x = np.histogram(data, bins=1000)

            aph_hist = pg.BarGraphItem(height=aph_y, x0=aph_x[:-1], x1=aph_x[1:], brush='b', pen=None)
            p3.addItem(aph_hist)
            p3.setXRange(np.min(data), np.max(data))

            # cf = pd.DataFrame([cut_region], columns=list(['cutstart', 'cutend']))
            # batch_info = batch_info.append(cf, ignore_index=True)
        else:
            # detect clears and auto-position window around the clear
            # clears = np.where(np.abs(data) > baseline + 10 * var)[0]
            # if clears:
            #     clear_starts = clears[0]
            #     try:
            #         clear_ends = clear_starts + np.where((data[clear_starts:-1] > baseline) &
            #                                              (data[clear_starts:-1] < baseline + var))[0][10000]
            #     except Exception as e:
            #         print(e)
            #         clear_ends = -1
            #     clear_starts = np.where(data[0:clear_starts] > baseline)
            #     try:
            #         clear_starts = clear_starts[0][-1]
            #     except Exception as e:
            #         print(e)
            #         clear_starts = 0
            #
            #     cut_region.setRegion((t[clear_starts], t[clear_ends]))

            p1.addItem(cut_region)
            cut_region.show()

        self.has_baseline_been_set = has_baseline_been_set
        self.cut_region = cut_region
        self.baseline = baseline
        self.var = var
        self.batch_info = batch_info
        self.t = t
        self.data = data

    def set_baseline(self):
        """
        Toggle that allows a region of the graph to be selected to used as the baseline.
        """
        base_region = self.base_region
        p1 = self.p1
        ui = self.ui
        threshold = self.threshold
        data = self.data
        if data is None:
            return
        output_sample_rate = self.output_sample_rate
        t = self.t

        p1.clear()
        if base_region.isVisible():
            left_bound, right_bound = base_region.getRegion()

            selected_pts = data[np.arange(int(max(left_bound, 0) * output_sample_rate),
                                          int(right_bound * output_sample_rate))]
            baseline = np.median(selected_pts)
            var = np.std(selected_pts)
            update_p1(self, t, data, baseline, threshold)
            base_region.hide()
            baseline_text = 'Baseline=' + str(round(baseline * BILLION, 2)) + ' nA'
            ui.eventcounterlabel.setText(baseline_text)
            self.baseline = baseline
            self.has_baseline_been_set = True
            self.var = var
        else:
            # base_region = pg.LinearRegionItem()  # PyQtgraph object for selecting a region
            # base_region.hide()
            p1.addItem(base_region)
            # p1.plot(t[::100],data[::100],pen='b')
            p1.plot(t, data, pen='b')
            base_region.show()

        self.base_region = base_region

    def clear_scatter(self):
        p2 = self.p2
        ui = self.ui
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4
        w5 = self.w5
        p2.setData(x=[], y=[])
        # last_event = []
        ui.scatterplot.update()
        w2.clear()
        w3.clear()
        w4.clear()
        w5.clear()
        self.sdf = pd.DataFrame(columns=['fn', 'color', 'deli', 'frac',
                                         'dwell', 'dt', 'startpoints', 'endpoints'])

    def delete_event(self):
        events = self.events
        if not events:
            return
        deltas = [event.delta for event in events]
        durations = [event.duration for event in events]
        frac = calc_frac(events)
        dt = calc_dt(events)
        noise = [event.noise for event in events]
        p2 = self.p2
        ui = self.ui
        sdf = self.sdf
        analyze_type = self.analyze_type
        mat_file_name = self.mat_file_name

        event_number = np.int(ui.eventnumberentry.text())
        del events[event_number]
        ui.eventnumberentry.setText(str(event_number))
        first_index = sdf.fn[sdf.fn == mat_file_name].index[0]
        p2.data = np.delete(p2.data, first_index + event_number)

        ui.eventcounterlabel.setText('Events:' + str(len(events)))

        sdf = sdf.drop(first_index + event_number).reset_index(drop=True)
        self.inspect_event()

        self.w2.clear()
        self.w3.clear()
        self.w4.clear()
        self.w5.clear()
        update_histograms(sdf, ui, events, self.w2, self.w3, self.w4, self.w5)

        if analyze_type == 'coarse':
            self.save()
            self.save_target()
        if analyze_type == 'fine':
            col_stack = np.column_stack((deltas, frac, durations, dt, noise))
            header_names = "\t".join(["deli", "frac", "dwell", "dt", 'stdev'])
            np.savetxt(mat_file_name + 'llDB.txt', col_stack, delimiter='\t', header=header_names)

        self.ui = ui

    def invert_data(self):
        data = self.data
        if data is None:
            return
        p1 = self.p1
        baseline = -self.baseline
        var = self.var
        threshold = -self.threshold
        t = self.t
        has_baseline_been_set = self.has_baseline_been_set
        p1.clear()
        data = -data

        if not has_baseline_been_set:
            baseline = np.median(data)
            var = np.std(data)

        update_p1(self, t, data, baseline, threshold)

        self.var = var
        self.baseline = baseline
        self.threshold = threshold

    def clicked(self, points, sdf, p2, mat_file_name):
        for idx, pt in enumerate(p2.points()):
            if pt.pos() == points[0].pos():
                clicked_index = idx
                break
        else:
            return

        if sdf.fn[clicked_index] != mat_file_name:
            print('Event is from an earlier file, not clickable')

        else:
            # noinspection PyTypeChecker
            self.inspect_event(clicked_index)

    # move outside class
    def concatenate_text(self):
        wd = os.getcwd()
        if self.data is None:
            return
        if not wd:
            text_file_names = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open file', '*.txt')[0]
        else:
            text_file_names = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open file', wd, '*.txt')[0]
        wd = os.path.dirname(text_file_names[0])

        new_text_data = [np.loadtxt(str(text_file_name), delimiter='\t') for text_file_name in text_file_names]
        new_text_data = np.array(new_text_data)
        # for i in range(len(text_file_names)):
        #     temp_text_data = np.loadtxt(str(text_file_names[i]), delimiter='\t')
        #     if i == 0:
        #         new_text_data = temp_text_data
        #     else:
        #         new_text_data = np.concatenate((new_text_data, temp_text_data))

        new_file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'New File name', wd, '*.txt')[0]
        # noinspection PyTypeChecker
        np.savetxt(str(new_file_name), new_text_data, delimiter='\t',
                   header="\t".join(["dI", "fr", "dw", "dt", 'stdev']))
        return wd

    def next_file(self):
        file_type = self.file_type
        mat_file_name = self.mat_file_name
        if file_type == '.log':
            log_offset = 6
            next_file_name = get_file(file_type, log_offset, "next", mat_file_name)
        elif file_type == '.abf':
            abf_offset = 4
            next_file_name = get_file(file_type, abf_offset, "next", mat_file_name)
        else:
            return
        self.data_file_name = next_file_name
        self.load()

    def previous_file(self):
        file_type = self.file_type
        mat_file_name = self.mat_file_name
        if file_type == '.log':
            log_offset = 6
            next_file_name = get_file(file_type, log_offset, "prev", mat_file_name)
        elif file_type == '.abf':
            abf_offset = 4
            next_file_name = get_file(file_type, abf_offset, "prev", mat_file_name)
        else:
            return
        self.data_file_name = next_file_name
        self.load()

    def save_trace(self):
        data = self.data
        if data is None:
            return
        mat_file_name = self.mat_file_name
        data.astype('d').tofile(mat_file_name + '_trace.bin')

    def show_cat_trace(self):
        data = self.data
        if data is None:
            return
        ui = self.ui
        dt = self.dt
        start_points = self.start_points
        end_points = self.end_points
        p1 = self.p1
        event_buffer = np.int(ui.eventbufferentry.text())
        num_events = len(dt)
        output_sample_rate = self.output_sample_rate
        baseline = self.baseline
        del_i = self.del_i

        p1.clear()
        event_time = [0]
        for idx in range(num_events):
            if idx < num_events - 1:
                if end_points[idx] + event_buffer > start_points[idx + 1]:
                    print('overlapping event')
                else:
                    adj_start = start_points[idx] - event_buffer
                    adj_end = start_points[idx] - event_buffer
                    eventdata = data[adj_start:adj_end]
                    fitdata = np.concatenate(
                        (np.repeat(np.array([baseline]), event_buffer),
                         np.repeat(np.array([baseline - del_i[idx]]), adj_end - adj_start),
                         np.repeat(np.array([baseline]), event_buffer)), 0)
                    event_time = np.arange(0, len(eventdata)) + .75 * event_buffer + event_time[-1]
                    p1.plot(event_time / output_sample_rate, eventdata, pen='b')
                    p1.plot(event_time / output_sample_rate, fitdata, pen=pg.mkPen(color=(173, 27, 183), width=2))

        p1.autoRange()

    def keyPressEvent(self, event):
        key = event.key()
        qt = QtCore.Qt
        if key == qt.Key_Up:
            self.next_file()
        elif key == qt.Key_Down:
            self.previous_file()
        elif key == qt.Key_Right:
            self.next_event()
        elif key == qt.Key_Left:
            self.previous_event()
        elif key == qt.Key_Return:
            self.load()
        elif key == qt.Key_Space:
            self.analyze()
        elif key == qt.Key_Delete:
            self.delete_event()
        elif self.cut_region.isVisible and key == qt.Key_Escape:
            self.cut_region.hide()

    def save_cat_trace(self):
        if self.data is None:
            return
        save_cat_data(self.data, self.events, self.ui, self.mat_file_name)

    def save_event_fits(self):
        if self.data is None:
            return
        save_cat_data(self.data, self.events, self.ui, self.mat_file_name)

    def cusum(self):
        ui = self.ui
        output_sample_rate = self.output_sample_rate
        ui_bp = self.ui.uibp
        if self.data is None:
            return
        self.p1.clear()
        self.p1.setDownsampling(ds=False)
        dt = 1 / self.output_sample_rate
        cusum_thresh = np.float64(ui_bp.cusumthreshentry.text())
        step_size = np.float64(ui.levelthresholdentry.text())
        cusum = CUSUM.cusum(self.data, base_sd=self.var, stepsize=step_size, output_sample_rate=output_sample_rate)
        np.savetxt(self.mat_file_name + '_Levels.txt', np.abs(cusum['jumps'] * 10 ** 12), delimiter='\t')

        self.p1.plot(self.t[2:][:-2], self.data[2:][:-2], pen='b')

        self.w3.clear()
        amp = np.abs(cusum['jumps'] * 10 ** 12)
        ampy, ampx = np.histogram(amp, bins=np.linspace(float(ui.delirange0.text()),
                                                        float(ui.delirange1.text()),
                                                        int(ui.delibins.text())))
        hist = pg.BarGraphItem(height=ampy, x0=ampx[:-1], x1=ampx[1:], brush='b')
        self.w3.addItem(hist)
        # self.w3.autoRange()
        self.w3.setRange(xRange=[np.min(ampx), np.max(ampx)])

        cusum_lines = np.array([]).reshape(0, 2)
        for i, level in enumerate(cusum.get('CurrentLevels')):
            y = 2 * [level]
            x = cusum.get('EventDelay')[i:i + 2]
            self.p1.plot(y=y, x=x, pen='r')
            cusum_lines = np.concatenate((cusum_lines, np.array(list(zip(x, y)))))
            try:
                y = cusum.get('CurrentLevels')[i:i + 2]
                x = cusum.get('EventDelay')[i:i + 2]  # 2 * [cusum.get('EventDelay')[i + 1]]
                self.p1.plot(y=y, x=x, pen='r')
                cusum_lines = np.concatenate((cusum_lines, np.array(list(zip(x, y)))))
            except Exception as e:
                print(e)
                raise Exception

        cusum_lines.astype('d').tofile(self.mat_file_name + '_cusum.bin')
        self.save_trace()

        print("Cusum Params" + str(cusum[self.threshold], cusum['stepsize']))

    def save_target(self):
        self.batch_info = save_batch_info(self.events, self.batch_info, self.mat_file_name)

    def batch_info_dialog(self):
        events = self.events
        ui_bp = self.bp.uibp
        min_dwell = min([event.duration for event in events])
        min_frac = min(calc_frac(events))
        min_level_t = float(ui_bp.minleveltbox.text()) * 10 ** -6
        sample_rate = self.sample_rate
        lp_filter_cutoff = self.lp_filter_cutoff
        cusum_step = self.cusum_step
        cusum_thresh = self.cusum_thresh
        max_states = self.max_states
        p1 = self.p1
        p1.clear()
        # self.batch_processor = BatchProcessor()
        # self.batch_processor.show()
        try:
            ui_bp.mindwellbox.setText(str(min_dwell))
            ui_bp.minfracbox.setText(str(min_frac))
            ui_bp.minleveltbox.setText(str(min_level_t * 10 ** 6))
            ui_bp.sampratebox.setText(str(sample_rate))
            ui_bp.LPfilterbox.setText(str(lp_filter_cutoff / 1000))
            ui_bp.cusumstepentry.setText(str(cusum_step))
            ui_bp.cusumthreshentry.setText(str(cusum_thresh))
            ui_bp.maxLevelsBox.setText(str(max_states))
        except ValueError as e:
            print(e)
        ui_bp.okbutton.clicked.connect(self.batch_process)

    # TODO: Reorganize
    def batch_process(self):
        events = self.events
        if not events:
            return
        data = self.data
        ui = self.ui
        self.analyze_type = 'fine'
        ui_bp = self.bp.uibp
        data_file_name = self.data_file_name
        wd = os.getcwd()
        has_baseline_been_set = self.has_baseline_been_set
        baseline = self.baseline
        var = self.var
        output_sample_rate = self.output_sample_rate
        info_file_name = self.mat_file_name
        sdf = self.sdf
        p1 = self.p1
        p2 = self.p2
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4
        w5 = self.w5
        cb = self.cb

        invert_status = ui_bp.invertCheckBox.isChecked()
        ui_bp.close()
        p1.setDownsampling(ds=False)
        min_dwell = np.float64(ui_bp.mindwellbox.text())
        min_frac = np.float64(ui_bp.minfracbox.text())
        min_level_t = np.float64(ui_bp.minleveltbox.text()) * 10 ** -6
        sample_rate = ui_bp.sampratebox.text()
        lp_filter_cutoff = ui_bp.LPfilterbox.text()
        ui.outputsamplerateentry.setText(sample_rate)
        ui.LPentry.setText(lp_filter_cutoff)
        cusum_step = np.float64(ui_bp.cusumstepentry.text())
        cusum_thresh = np.float64(ui_bp.cusumthreshentry.text())
        max_states = np.int(ui_bp.maxLevelsBox.text())
        self_correct = ui_bp.selfCorrectCheckBox.isChecked()

        start_points = [event.start for event in events]
        end_points = [event.end for event in events]
        noise = [event.noise for event in events]
        deltas = [event.delta for event in events]
        durations = [event.duration for event in events]
        frac = calc_frac(events)
        dt = calc_dt(events)

        try:
            # attempt to open dialog from most recent directory
            file_list = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Files', wd, "*.pkl")[0]
            wd = os.path.dirname(file_list[0])
        except TypeError as e:
            print(e)
            # if no recent directory exists open from working directory
            file_list = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Files', os.getcwd(), "*.pkl")[0]
            print(file_list)
            # self.wd=os.path.dirname(str(self.filelist[0][0]))
        except IOError as e:
            print(e)
            # if user cancels during file selection, exit loop
            return

        event_buffer = np.int(ui.eventbufferentry.text())
        event_time = [0]
        if len(file_list) == 0:
            return
        for f in file_list:
            batch_info = pd.read_pickle(f)
            try:
                data_file_name = f[:-13] + '.opt'
                self.load(load_and_plot=False)
            except IOError:
                data_file_name = f[:-13] + '.log'
                self.load(load_and_plot=False)
            if invert_status:
                data = -data
                if not has_baseline_been_set:
                    baseline = np.median(data)
                    var = np.std(data)

            try:
                cs = batch_info.cutstart[np.isfinite(batch_info.cutstart)]
                ce = batch_info.cutend[np.isfinite(batch_info.cutend)]
                for i, cut in enumerate(cs):
                    data = np.delete(data, np.arange(np.int(cut * output_sample_rate),
                                                     np.int(ce[i] * output_sample_rate)))
            except TypeError as e:
                print(e)

            with pg.ProgressDialog("Analyzing...", 0, len(durations)) as dlg:
                for i, event in enumerate(events):
                    # t_offset = (event_time[-1] + event_buffer) / output_sample_rate
                    if not (i < len(dt) - 1 and durations > min_dwell and frac[i] > min_frac):
                        continue

                    if end_points[i] + event_buffer > start_points[i + 1]:
                        print('overlapping event')
                        frac[i] = np.NaN
                        deltas[i] = np.NaN

                    else:
                        eventdata = data[int(event.start - event_buffer):int(event.end + event_buffer)]
                        event_time = np.arange(0, len(eventdata)) + event_buffer + event_time[-1]
                        # self.p1.plot(eventtime/self.outputsamplerate, eventdata,pen='b')
                        cusum = CUSUM.cusum(eventdata, np.std(eventdata[0:event_buffer]),
                                            output_sample_rate, stepsize=cusum_step)

                        while len(cusum['CurrentLevels']) < 3:
                            cusum_thresh = cusum_thresh * .9
                            cusum_step = cusum_step * .9
                            cusum = CUSUM.cusum(eventdata, np.std(eventdata[0:event_buffer]),
                                                output_sample_rate, stepsize=cusum_step)
                            print('Not Sensitive Enough')

                        frac[i] = (np.max(cusum['CurrentLevels']) - np.min(cusum['CurrentLevels'])) / np.max(
                            cusum['CurrentLevels'])
                        deltas[i] = (np.max(cusum['CurrentLevels']) - np.min(cusum['CurrentLevels']))

                        if self_correct:
                            cusum_thresh = cusum['Threshold']
                            cusum_step = cusum['stepsize']
                    # Plotting
                    # for j, level in enumerate(cusum['CurrentLevels']):
                    #     self.p1.plot(y=2*[level], x=t_offset + cusum['EventDelay'][j:j+2], pen=pg.mkPen('r', width=5))
                    #     try:
                    #         self.p1.plot(y=cusum['CurrentLevels'][j:j+2], x=t_offset + 2*[cusum['EventDelay'][j+1]],
                    #                      pen=pg.mkPen('r', width=5))
                    #     except Exception as e:
                    #         print(e)
                    #         pass
                    dlg += 1
                # End Plotting

            durations = durations[np.isfinite(deltas)]
            dt = dt[np.isfinite(deltas)]
            noise = noise[np.isfinite(deltas)]
            frac = frac[np.isfinite(deltas)]
            start_points = start_points[np.isfinite(deltas)]
            end_points = end_points[np.isfinite(deltas)]
            deltas = deltas[np.isfinite(deltas)]

            np.savetxt(info_file_name + 'llDB.txt',
                       np.column_stack((deltas, frac, durations, dt, noise)),
                       delimiter='\t', header="\t".join(["del_i", "frac", "dwell", "dt", 'stdev']))

        p1.autoRange()
        # Plotting Histograms
        sdf = sdf[sdf.fn != info_file_name]
        num_events = len(dt)

        fn = pd.Series([info_file_name, ] * num_events)
        color = pd.Series([pg.colorTuple(cb.color()), ] * num_events)

        sdf = sdf.append(pd.DataFrame({'fn': fn, 'color': color, 'del_i': deltas,
                                       'frac': frac, 'dwell': durations,
                                       'dt': dt, 'startpoints': start_points,
                                       'endpoints': end_points}), ignore_index=True)

        p2.addPoints(x=np.log10(durations), y=frac,
                     symbol='o', brush=(cb.color()), pen=None, size=10)

        w1.addItem(p2)
        w1.setLogMode(x=True, y=False)
        p1.autoRange()
        w1.autoRange()
        ui.scatterplot.update()
        w1.setRange(yRange=[0, 1])

        update_histograms(sdf, ui, events, w2, w3, w4, w5)

        print('\007')

        self.baseline = baseline
        self.var = var
        self.sdf = sdf
        self.data_file_name = data_file_name

    @staticmethod
    def size_pore():
        pore_size = PoreSizer()
        pore_size.show()


def start():
    # global my_app
    app = QtWidgets.QApplication(sys.argv)
    resolution = app.desktop().screenGeometry()
    width, height = resolution.width(), resolution.height()
    my_app = GUIForm(width=width, height=height)
    my_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start()
