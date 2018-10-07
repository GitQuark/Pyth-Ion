from PythIon import CUSUM
from PythIon.PoreSizer import *
from PythIon.SetupUtilities import *
from PythIon.Utility import *
from PythIon.Widgets.PlotGUI import *
# plotguiuniversal works well for mac and laptops,
# for larger screens try PlotGUI
# from PythIon.plotguiuniversal import *


class GUIForm(QtWidgets.QMainWindow):
    dataset: CurrentData
    events: List[Event]
    ui: Ui_PythIon

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
        self.frac_plot, self.del_i_plot, self.dwell_plot, self.dt_plot = setup_plots(self)
        logo = load_logo()
        self.current_hist = setup_current_hist(self.ui, logo)

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
        self.threshold = np.float64(self.ui.threshold_entry.text()) * 10 ** -9

    def get_file(self):
        wd = os.getcwd()
        # attempt to open dialog from most recent directory
        suffix_filter = "*.log;*.opt;*.npy;*.abf"
        data_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', wd, suffix_filter)
        if data_file_name == ('', ''):  # Interaction canceled
            return

        self.data_file_name = data_file_name
        self.load()

    def load(self):
        sdf = self.sdf
        ui = self.ui
        data_file_name = self.data_file_name
        if not data_file_name:
            return
        self.current_hist.clear()
        self.current_hist.setLabel('bottom', text='Current', units='A', unitprefix='n')
        self.current_hist.setLabel('left', text='', units='Counts')
        self.current_hist.setAspectLocked(False)

        colors = np.array(sdf.color)
        for i in range(len(colors)):
            colors[i] = pg.Color(colors[i])

        self.event_plot.setBrush(colors, mask=None)

        ui.event_number_entry.setText(str(1))

        ui.status_bar.showMessage('Data file ' + data_file_name + ' loaded.')
        ui.file_label.setText(data_file_name)
        print(data_file_name)
        low_pass_cutoff = float(ui.low_pass_entry.text())  # In Hz
        # use integer multiples of 4166.67 ie 2083.33 or 1041.67
        self.dataset = CurrentData(data_file_name)

        if self.dataset.file_type in ('.log', '.abf', '.opt'):
            self.dataset.process_data(low_pass_cutoff)

        update_signal_plot(self.dataset, self.signal_plot, self.current_hist)
        self.signal_plot.autoRange()

    def analyze(self):
        dataset = self.dataset
        if dataset is None:
            return
        self.analyze_type = 'coarse'  # Still don't know what this was for
        self.clear_stat_plots()
        threshold = np.float64(self.ui.threshold_entry.text()) * 1e-9  # Now in number of standard deviations
        dataset.detect_events(threshold)
        update_signal_plot(dataset, self.signal_plot, self.current_hist)

    # Called by Go button
    # TODO: Update with new event structure
    def inspect_event(self, clicked=None):
        if clicked is None:
            clicked = []
        if not self.dataset.events:
            return

        events = self.dataset.events
        sample_rate = self.dataset.data_params.get('sample_rate')
        num_events = len(self.dataset.events)
        baseline = events[0].baseline

        ui = self.ui
        event_plot = self.event_plot
        current_hist = self.current_hist

        # Reset plot
        current_hist.setLabel('bottom', text='Time', units='s')
        current_hist.setLabel('left', text='Current', units='A')
        current_hist.clear()

        # Correct for user error if non-extistent number is entered
        event_buffer = num_from_text_element(ui.event_buffer_entry, default=1000)
        # first_index = sdf.fn[sdf.fn == mat_file_name].index[0]
        if not clicked:
            event_number = num_from_text_element(ui.event_number_entry, 1, len(events))
            ui.event_number_entry.setText(str(event_number))
            event_number -= 1
        else:
            event_number = clicked
            ui.event_number_entry.setText(str(event_number))

        # plot event trace
        event = events[event_number]
        # Just use main plot and set the view to focus on single event
        adj_start = int(event.start - event_buffer)
        adj_end = int(event.end + event_buffer)
        self.signal_plot.setXRange(adj_start / sample_rate, adj_end / sample_rate)
        self.signal_plot.setYRange(min(event.data), max(event.data), padding=0.1)
        # current_hist.plot(t[adj_start:adj_end], data[adj_start:adj_end], pen='b')

        # event_fit = event.piecewise_fit()
        # p3.plot(t[adj_start:adj_end], event_fit, pen=pg.mkPen(color=(173, 27, 183), width=3))
        # p3.autoRange()

        # Mark event that is being viewed on scatter plot

        # colors = np.array(sdf.color)
        # for i in range(len(colors)):
        #     colors[i] = pg.Color(colors[i])
        # colors[first_index + event_number] = pg.mkColor('r')
        #
        # p2.setBrush(colors, mask=None)

        # Mark event start and end points
        # p3.plot([t[event.start], t[event.start]],
        #         [data[event.start], data[event.start]], pen=None,
        #         symbol='o', symbolBrush='g', symbolSize=12)
        # p3.plot([t[event.end], t[event.end]],
        #         [data[event.end], data[event.end]], pen=None,
        #         symbol='o', symbolBrush='r', symbolSize=12)

        # duration = round(event.intervals, 2)
        # level = round(event.levels * BILLION, 2)
        # ui.status_bar.showMessage('Dwell Time=' + str(duration) + u' μs,   Deli=' + str(level) + ' nA')

    def replot(self):
        if not self.dataset:
            return
        low_pass_cutoff = int(self.ui.low_pass_entry.text())
        if low_pass_cutoff == self.dataset.data_params.get('low_pass_cutoff'):
            return
        self.dataset.process_data(low_pass_cutoff)
        update_signal_plot(self.dataset, self.signal_plot, self.current_hist)

    def next_event(self):
        # Ignore command if there are no events
        if not self.dataset.events:
            return
        element = self.ui.event_number_entry
        num_events = len(self.dataset.events)
        input_num = num_from_text_element(element, 1, num_events, default=1)
        element.setText(str(min(input_num + 1, num_events)))
        self.inspect_event()

    def previous_event(self):
        # Ignore command if there are no events
        if not self.dataset.events:
            return
        element = self.ui.event_number_entry
        num_events = len(self.dataset.events)
        input_num = num_from_text_element(element, 1, num_events, default=1)
        element.setText(str(max(input_num - 1, 1)))
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
        voltage_hist = self.current_hist

        if cut_region.isVisible():
            # If cut region has been set, cut region and replot remaining data
            left_bound, right_bound = cut_region.getRegion()
            cut_region.hide()
            signal_plot.clear()
            voltage_hist.clear()
            sample_rate = self.dataset.data_params.get('sample_rate')
            interval = (int(left_bound * sample_rate), int(right_bound * sample_rate) + 1)
            dataset.add_cut(interval)
            included_pts = np.r_[0:interval[0], interval[1]:len(dataset.processed_data)]
            dataset.processed_data = dataset.processed_data[included_pts]
            update_signal_plot(dataset, signal_plot, voltage_hist)
        else:
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

        if base_region.isVisible():
            left_bound, right_bound = base_region.getRegion()
            base_region.hide()
            start, end = (int(max(left_bound, 0) * sample_rate), int(right_bound * sample_rate))
            baseline = np.mean(dataset.processed_data[start: end])
            dataset.data_params['baseline'] = baseline
            update_signal_plot(dataset, self.signal_plot, self.current_hist)
            self.has_baseline_been_set = True
        else:
            self.signal_plot.addItem(base_region)
            base_region.show()
        self.base_region = base_region

    def clear_scatter(self):
        event_plot = self.event_plot
        ui = self.ui
        event_plot.setData(x=[], y=[])
        # last_event = []
        ui.scatter_plot.update()
        self.clear_stat_plots()
        self.sdf = pd.DataFrame(columns=['fn', 'color', 'deli', 'frac',
                                         'dwell', 'dt', 'startpoints', 'endpoints'])

    def delete_event(self):
        events = self.dataset.events
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

        event_number = np.int(ui.event_number_entry.text())
        del events[event_number]
        ui.event_number_entry.setText(str(event_number))
        first_index = sdf.fn[sdf.fn == mat_file_name].index[0]
        p2.data = np.delete(p2.data, first_index + event_number)

        # ui.eventcounterlabel.setText('Events:' + str(len(events)))

        sdf = sdf.drop(first_index + event_number).reset_index(drop=True)
        self.inspect_event()

        self.clear_stat_plots()
        update_histograms(sdf, ui, events, self.frac_plot, self.del_i_plot, self.dwell_plot, self.dt_plot)

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
            self.dataset.data_params['inverted'] = not self.dataset.data_params.get('inverted')
            self.dataset.processed_data = -self.dataset.processed_data
            # self.dataset.data_params['baseline'] = not self.dataset.data_params.get('baseline')
        else:
            return
        update_signal_plot(self.dataset, self.signal_plot, self.current_hist)

    def clear_stat_plots(self):
        self.frac_plot.clear()
        self.del_i_plot.clear()
        self.dwell_plot.clear()
        self.dt_plot.clear()

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

    def next_file(self):
        file_type = self.file_type
        if file_type == '.log':
            log_offset = 6
            next_file_name = get_file(file_type, log_offset, "next", self.data_file_name)
        elif file_type == '.abf':
            abf_offset = 4
            next_file_name = get_file(file_type, abf_offset, "next", self.data_file_name)
        else:
            return
        self.data_file_name = next_file_name
        self.load()

    def previous_file(self):
        file_type = self.file_type
        if file_type == '.log':
            log_offset = 6
            next_file_name = get_file(file_type, log_offset, "prev", self.data_file_name)
        elif file_type == '.abf':
            abf_offset = 4
            next_file_name = get_file(file_type, abf_offset, "prev", self.data_file_name)
        else:
            return
        self.data_file_name = next_file_name
        self.load()

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

    def save_event_fits(self):
        if self.dataset is None:
            return
        default_file_name = os.path.join(os.getcwd(), 'event_data.csv')
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Event Data', default_file_name, '*.csv')
        if file_name == '':
            return
        event_data = self.dataset.generate_event_table()
        column_order = ('event_number', 'start', 'end', 'duration', 'frac', 'voltage_change')
        event_data.loc[:, column_order].to_csv(file_name, index=False)

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
        w2 = self.frac_plot
        w3 = self.del_i_plot
        w4 = self.dwell_plot
        w5 = self.dt_plot
        cb = self.cb

        invert_status = ui_bp.invertCheckBox.isChecked()
        ui_bp.close()
        p1.setDownsampling(ds=False)
        min_dwell = np.float64(ui_bp.mindwellbox.text())
        min_frac = np.float64(ui_bp.minfracbox.text())
        min_level_t = np.float64(ui_bp.minleveltbox.text()) * 10 ** -6
        sample_rate = ui_bp.sampratebox.text()
        lp_filter_cutoff = ui_bp.LPfilterbox.text()
        ui.sample_rate_entry.setText(sample_rate)
        ui.low_pass_entry.setText(lp_filter_cutoff)
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

        event_buffer = np.int(ui.event_buffer_entry.text())
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
        ui.scatter_plot.update()
        w1.setRange(yRange=[0, 1])

        update_histograms(sdf, ui, events, w2, w3, w4, w5)

        print('\007')

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
