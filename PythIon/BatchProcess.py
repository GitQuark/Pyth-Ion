# from PythIon.PoreSizer import *

from PythIon.Model import *
# from PythIon.Utility import *
from PythIon.Widgets.PlotGUI import *


# plotguiuniversal works well for mac and laptops,
# for larger screens try PlotGUI
# from PythIon.plotguiuniversal import *


def batch_info_dialog(self):
    events = self.events
    ui_bp = self.bp.uibp
    min_dwell = None
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
                    # cusum = CUSUM.cusum(eventdata, np.std(eventdata[0:event_buffer]),
                    #                     output_sample_rate, stepsize=cusum_step)
                    cusum = None

                    while len(cusum['CurrentLevels']) < 3:
                        cusum_thresh = cusum_thresh * .9
                        cusum_step = cusum_step * .9
                        # cusum = CUSUM.cusum(eventdata, np.std(eventdata[0:event_buffer]),
                        #                     output_sample_rate, stepsize=cusum_step)
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

    # p2.addPoints(x=np.log10(durations), y=frac,
    #              symbol='o', brush=(cb.color()), pen=None, size=10)
    #
    # w1.addItem(p2)
    w1.setLogMode(x=True, y=False)
    p1.autoRange()
    w1.autoRange()
    ui.scatter_plot.update()
    w1.setRange(yRange=[0, 1])

    # update_histograms(sdf, ui, events, w2, w3, w4, w5)

    print('\007')

    self.sdf = sdf
    self.data_file_name = data_file_name