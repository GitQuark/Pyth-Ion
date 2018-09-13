from PythIon.Utility import *

from scipy import signal
from PythIon import CUSUM
from PythIon.PoreSizer import *
from PythIon.SetupUtilities import *
from PythIon.Utility import *
# plotguiuniversal works well for mac and laptops,
# for larger screens try PlotGUI
from PythIon.Widgets.PlotGUI import *


# from PythIon.plotguiuniversal import *


class GUIForm(QtWidgets.QMainWindow):
    dataset: VoltageData
    events: List[Event]

    # Setup GUI and draw elements from UI file
    def __init__(self, master=None):
        super().__init__(master)
        self.filtered_data = None
        pg.setConfigOptions(antialias=True)

        self.ui = setup_ui(self)
        self.signal_plot = setup_signal_plot(self.ui)
        self.event_plot = setup_event_plot(self.clicked)
        self.scatter_plot = setup_scatter_plot(self, self.event_plot)
        self.cb = setup_cb(self.ui)
        self.w2, self.w3, self.w4, self.w5 = setup_plots(self)
        self.logo = load_logo()
        self.voltage_hist = setup_voltage_hist(self.ui, self.logo)

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
        self.dataset = None

        self.cat_fits = None
        self.file_type = None
        self.events = None

        self.batch_info = pd.DataFrame(columns=list(['cutstart', 'cutend']))
        self.total_plot_points = len(self.event_plot.data)
        self.threshold = np.float64(self.ui.thresholdentry.text()) * 10 ** -9

    def load(self):
        sdf = self.sdf
        ui = self.ui
        data_file_name = self.data_file_name
        if not data_file_name:
            return
        self.voltage_hist.clear()
        self.voltage_hist.setLabel('bottom', text='Current', units='A', unitprefix='n')
        self.voltage_hist.setLabel('left', text='', units='Counts')
        self.voltage_hist.setAspectLocked(False)

        colors = np.array(sdf.color)
        for i in range(len(colors)):
            colors[i] = pg.Color(colors[i])

        self.event_plot.setBrush(colors, mask=None)

        ui.eventinfolabel.clear()
        ui.eventcounterlabel.clear()
        ui.meandelilabel.clear()
        ui.meandwelllabel.clear()
        ui.meandtlabel.clear()
        ui.eventnumberentry.setText(str(1))

        billionth = 10 ** -9
        threshold = float(ui.thresholdentry.text()) * billionth
        ui.filelabel.setText(data_file_name)
        print(data_file_name)
        low_pass_cutoff = float(ui.LPentry.text())  # In Hz
        # use integer multiples of 4166.67 ie 2083.33 or 1041.67
        self.dataset = VoltageData(data_file_name)

        baseline = self.dataset.data_params.get('baseline')
        ui.eventcounterlabel.setText('Baseline=' + str(round(baseline * BILLION, 2)) + ' nA')

        if self.dataset.file_type in ('.log', '.abf', '.opt'):
            self.dataset.processed_data(low_pass_cutoff)

        update_signal_plot(self.dataset, self.signal_plot, self.voltage_hist)
        self.signal_plot.autoRange()
        self.threshold = threshold
    #        if self.v != []:
    #            self.p1.plot(self.t[2:][:-2],self.v[2:][:-2],pen='r')

    #        self.w6.clear()
    #        f, Pxx_den = signal.welch(data*10**12, self.outputsamplerate, nperseg = self.outputsamplerate)
    #        self.w6.plot(x = f[1:], y = Pxx_den[1:], pen = 'b')
    #        self.w6.setXRange(0,np.log10(self.outputsamplerate))

    # Static
    def get_file(self):
        wd = os.getcwd()
        # attempt to open dialog from most recent directory
        suffix_filter = "*.log;*.opt;*.npy;*.abf"
        data_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', wd, suffix_filter)
        if data_file_name == ('', ''):  # Interaction canceled
            return

        self.data_file_name = data_file_name
        self.load()

    def analyze(self):
        dataset = self.dataset
        if dataset is None:
            return
        durations_plot = self.event_plot
        self.analyze_type = 'coarse'  # Still don't know what this was for
        self.clear_w_plots()
        threshold = np.float64(self.ui.thresholdentry.text()) * 1e-9  # Now in number of standard deviations
        dataset.detect_events(threshold)
        update_signal_plot(dataset, self.signal_plot, self.voltage_hist)

    def save(self):
        dataset = self.dataset
        event_table = dataset.generate_event_table()
        header_names = '\t'.join(['start', 'end', 'delta', 'frac', 'duration', 'event_number'])
        np.savetxt(dataset.file_name + '_EventData.txt', event_table, delimiter='\t', header=header_names)

    # Called by Go button
    # TODO: Continue cleanump and input sanitizing
    def inspect_event(self, clicked=None):
        if clicked is None:
            clicked = []
        events = self.events
        if not events:
            return
        num_events = len(events)
        baseline = events[0].baseline
        sdf = self.sdf
        ui = self.ui
        data = self.data
        p2 = self.event_plot
        p3 = self.voltage_hist
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

    def replot(self):
        if not self.dataset:
            return
        low_pass_cutoff = int(self.ui.LPentry.text())
        self.dataset.process_data(low_pass_cutoff)
        update_signal_plot(self.dataset, self.signal_plot, self.voltage_hist)

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
        dataset = self.dataset
        if dataset is None:
            return
        batch_info = self.batch_info
        # first check to see if cutting
        cut_region = self.cut_region
        signal_plot = self.signal_plot
        voltage_hist = self.voltage_hist

        if cut_region.isVisible():
            # If cut region has been set, cut region and replot remaining data
            left_bound, right_bound = cut_region.getRegion()
            cut_region.hide()
            signal_plot.clear()
            voltage_hist.clear()
            sample_rate = self.dataset.data_params.get('sample_rate')
            interval = (int(left_bound * sample_rate), int(right_bound * sample_rate) + 1)
            dataset.add_cut(interval)
            dataset.process_data()

            update_signal_plot(dataset, signal_plot, voltage_hist)
            # aph_y, aph_x = np.histogram(data, bins = len(data)/1000)
            # aph_y, aph_x = np.histogram(data, bins=1000)
            # aph_hist = pg.BarGraphItem(height=aph_y, x0=aph_x[:-1], x1=aph_x[1:], brush='b', pen=None)
            # voltage_hist.addItem(aph_hist)
            # voltage_hist.setXRange(np.min(dataset.processed_data), np.max(dataset.processed_data))

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

            signal_plot.addItem(cut_region)
            cut_region.show()

        self.cut_region = cut_region
        self.batch_info = batch_info

    def set_baseline(self):
        """
        Toggle that allows a region of the graph to be selected to used as the baseline.
        """
        dataset = self.dataset
        if dataset is None:
            return
        base_region = self.base_region
        ui = self.ui
        sample_rate = self.dataset.data_params.get('sample_rate')

        self.signal_plot.clear()
        if base_region.isVisible():
            left_bound, right_bound = base_region.getRegion()
            start, end = (int(max(left_bound, 0) * sample_rate), int(right_bound * sample_rate))
            baseline = np.mean(dataset.processed_data[start: end])
            update_signal_plot(dataset, self.signal_plot, self.voltage_plot)
            base_region.hide()

            baseline_text = 'Baseline=' + str(round(baseline * BILLION, 2)) + ' nA'
            ui.eventcounterlabel.setText(baseline_text)
            self.has_baseline_been_set = True
        else:
            # base_region = pg.LinearRegionItem()  # PyQtgraph object for selecting a region
            # base_region.hide()
            self.signal_plot.addItem(base_region)
            update_signal_plot(dataset, self.signal_plot, self.voltage_plot)
            base_region.show()
        self.base_region = base_region

    def clear_scatter(self):
        p2 = self.event_plot
        ui = self.ui
        p2.setData(x=[], y=[])
        # last_event = []
        ui.scatterplot.update()
        self.clear_w_plots()
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
        p2 = self.event_plot
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

        self.clear_w_plots()
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
        if self.dataset:
            self.dataset.data_params['inverted'] = True
        else:
            return
        update_signal_plot(self.dataset, self.signal_plot)

    def clear_w_plots(self):
        self.w2.clear()
        self.w3.clear()
        self.w4.clear()
        self.w5.clear()

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
        p1 = self.signal_plot
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
        if self.dataset is None:
            return
        print('This function is not currently in use')

    def save_event_fits(self):
        if self.dataset is None:
            return
        default_file_name = os.path.join(os.getcwd(), 'event_data.csv')
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Event Data', default_file_name, '*.csv')
        event_data = self.dataset.generate_event_table()
        column_order = ('event_number', 'start', 'end', 'duration', 'frac', 'voltage_change')
        event_data.loc[:, column_order].to_csv(file_name, index=False)

    def cusum(self):
        ui = self.ui
        output_sample_rate = self.output_sample_rate
        # ui_bp = self.ui.uibp
        if self.data is None:
            return
        self.signal_plot.clear()
        # self.signal_plot.setDownsampling(ds=False)
        # dt = 1 / self.output_sample_rate
        # cusum_thresh = np.float64(ui_bp.cusumthreshentry.text())
        # step_size = np.float64(ui.levelthresholdentry.text())
        fit_line, _, _, level_table = CUSUM.correct_cusum(self.data, 2e-10, 2.5e-11)

        file_name = "TempNameEventTable"
        extention = ".csv"
        event_indicies = level_table['Voltage Level'] < level_table['Voltage Level'][0] - 3 * np.std(self.data)
        event_table = level_table[event_indicies]
        if not event_table.empty:
            compute_dict = {
                'Start Time': event_table['Start Index'] / output_sample_rate,
                'End Time': event_table['End Index'] / output_sample_rate,
                'Duration (s)': event_table['Duration'] / output_sample_rate,
            }
            event_table = event_table.assign(**compute_dict)
            event_table = event_table.loc[:, ('Start Index', 'End Index', 'Duration',
                                              'Start Time', 'End Time', 'Duration (s)',
                                              'Voltage Level')]
            event_table.to_csv(file_name + extention)

        # np.savetxt(self.mat_file_name + '_Levels.txt', np.abs(cusum['jumps'] * 10 ** 12), delimiter='\t')

        self.signal_plot.plot(self.t[2:][:-2], self.data[2:][:-2], pen='b')
        self.signal_plot.plot(self.t[2:][:-3], fit_line[2:][:-2], pen='r')  # len(fit_line) = len(self.data) - 1
        self.w3.clear()
        # amp = np.abs(cusum['jumps'] * 10 ** 12)
        # ampy, ampx = np.histogram(amp, bins=np.linspace(float(ui.delirange0.text()),
        #                                                 float(ui.delirange1.text()),
        #                                                 int(ui.delibins.text())))
        # hist = pg.BarGraphItem(height=ampy, x0=ampx[:-1], x1=ampx[1:], brush='b')
        # self.w3.addItem(hist)
        # self.w3.autoRange()
        # self.w3.setRange(xRange=[np.min(ampx), np.max(ampx)])

        cusum_lines = np.array([]).reshape(0, 2)
        # for i, level in enumerate(cusum.get('CurrentLevels')):
        #     y = 2 * [level]
        #     x = cusum.get('EventDelay')[i:i + 2]
        #     self.signal_plot.plot(y=y, x=x, pen='r')
        #     cusum_lines = np.concatenate((cusum_lines, np.array(list(zip(x, y)))))
        #     try:
        #         y = cusum.get('CurrentLevels')[i:i + 2]
        #         x = cusum.get('EventDelay')[i:i + 2]  # 2 * [cusum.get('EventDelay')[i + 1]]
        #         self.signal_plot.plot(y=y, x=x, pen='r')
        #         cusum_lines = np.concatenate((cusum_lines, np.array(list(zip(x, y)))))
        #     except Exception as e:
        #         print(e)
        #         raise Exception

        cusum_lines.astype('d').tofile(self.mat_file_name + '_cusum.bin')
        self.save_trace()

        # print("Cusum Params" + str(cusum[self.threshold], cusum['stepsize']))

    def save_target(self):
        self.batch_info = save_batch_info(self.events, self.batch_info, self.mat_file_name)

    def batch_info_dialog(self):
        events = self.events
        ui_bp = self.bp.uibp
        min_dwell = min([event.duration for event in events])
        min_frac = min(calc_frac(events))
        min_level_t = float(ui_bp.minleveltbox.text()) * 10 ** -6
        p1 = self.signal_plot
        p1.clear()
        # self.batch_processor = BatchProcessor()
        # self.batch_processor.show()
        try:
            ui_bp.mindwellbox.setText(str(min_dwell))
            ui_bp.minfracbox.setText(str(min_frac))
            ui_bp.minleveltbox.setText(str(min_level_t * 10 ** 6))
            ui_bp.sampratebox.setText(str(self.sample_rate))
            ui_bp.LPfilterbox.setText(str(self.lp_filter_cutoff / 1000))
            ui_bp.cusumstepentry.setText(str(self.cusum_step))
            ui_bp.cusumthreshentry.setText(str(self.cusum_thresh))
            ui_bp.maxLevelsBox.setText(str(self.max_states))
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
        p1 = self.signal_plot
        p2 = self.event_plot
        w1 = self.scatter_plot
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
                self.load()
            except IOError:
                data_file_name = f[:-13] + '.log'
                self.load()
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
    # resolution = app.desktop().screenGeometry()
    # width, height = resolution.width(), resolution.height()
    my_app = GUIForm()
    my_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start()
