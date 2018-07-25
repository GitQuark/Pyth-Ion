import pandas as pd
import pandas.io.parsers
import pyqtgraph as pg
from scipy import io as spio
from scipy import ndimage
from scipy import signal

from PythIon.CUSUMV2 import detect_cusum
from PythIon.PoreSizer import *
from PythIon.abfheader import *
from PythIon.batchinfo import *
# plotguiuniversal works well for mac and laptops,
# for larger screens try PlotGUI
# from PlotGUI import *
from PythIon.plotguiuniversal import *


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


def load_log_file(mat_file_name, data_file_name, lp_filter_cutoff, output_sample_rate):
    chimera_file = np.dtype('<u2')
    data = np.fromfile(data_file_name, chimera_file)
    mat = spio.loadmat(mat_file_name)

    sample_rate = np.float64(mat['ADCSAMPLERATE'])
    tig_ain = np.int32(mat['SETUP_TIAgain'])
    pre_adc_gain = np.float64(mat['SETUP_preADCgain'])
    current_offset = np.float64(mat['SETUP_pAoffset'])
    adc_vref = np.float64(mat['SETUP_ADCVREF'])
    adc_bits = np.int32(mat['SETUP_ADCBITS'])
    closedloop_gain = tig_ain * pre_adc_gain

    if sample_rate < 4000e3:
        data = data[::round(sample_rate / output_sample_rate)]

    bitmask = (2 ** 16 - 1) - (2 ** (16 - adc_bits) - 1)
    data = -adc_vref + (2 * adc_vref) * (data & bitmask) / 2 ** 16
    data = (data / closedloop_gain + current_offset)
    data = data[0]

    # TODO: Separate loading and filtering
    # data has now been loaded
    # now filtering data

    wn = round(lp_filter_cutoff / (sample_rate / 2), 4)
    # noinspection PyTupleAssignmentBalance
    b, a = signal.bessel(4, wn, btype='low')

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
    instance.p1.clear()  # This might be unnecessary
    instance.p1.plot(t, data, pen='b')
    if file_type != '.abf':
        instance.p1.addLine(y=baseline, pen='g')
        instance.p1.addLine(y=threshold, pen='r')


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


BILLION = 10 ** 9  # Hopefully makes reading clearer


class GUIForm(QtWidgets.QMainWindow):

    def __init__(self, width, height, master=None):
        # Setup GUI and draw elements from UI file
        QtWidgets.QMainWindow.__init__(self, master)

        # Function to hide ui setup boilerplate
        def setup_ui(form):
            # TODO: Maybe move the top two lines out of function
            ui = Ui_PythIon()
            ui.setup_ui(form)

            # Linking buttons to main functions
            ui.loadbutton.clicked.connect(form.get_file)
            ui.analyzebutton.clicked.connect(form.analyze)
            ui.cutbutton.clicked.connect(form.cut)
            ui.baselinebutton.clicked.connect(form.base_line_calc)
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

        self.ui = setup_ui(self)

        def setup_p1(ui):
            p1 = ui.signalplot.addPlot()
            p1.setLabel('bottom', text='Time', units='s')
            p1.setLabel('left', text='Current', units='A')
            # p1.enableAutoRange(axis='x')
            p1.setClipToView(clip=True)  # THIS IS THE MOST IMPORTANT LINE!!!!
            p1.setDownsampling(ds=True, auto=True, mode='peak')
            return p1

        self.p1 = setup_p1(self.ui)

        def setup_p2(clicked):
            p2 = pg.ScatterPlotItem()
            p2.sigClicked.connect(clicked)
            return p2

        self.p2 = setup_p2(self.clicked)

        def setup_w1(p2):
            w1 = self.ui.scatterplot.addPlot()
            w1.addItem(p2)
            w1.setLabel('bottom', text='Time', units=u'μs')
            w1.setLabel('left', text='Fractional Current Blockage')
            w1.setLogMode(x=True, y=False)
            w1.showGrid(x=True, y=True)
            return w1

        self.w1 = setup_w1(self.p2)

        def setup_cb(ui):
            cb = pg.ColorButton(ui.scatterplot, color=(0, 0, 255, 50))
            cb.setFixedHeight(30)
            cb.setFixedWidth(30)
            cb.move(0, 210)
            cb.show()
            return cb

        self.cb = setup_cb(self.ui)

        def setup_plots():  # TODO: revisit name
            w2 = self.ui.frachistplot.addPlot()
            w2.setLabel('bottom', text='Fractional Current Blockage')
            w2.setLabel('left', text='Counts')

            w3 = self.ui.delihistplot.addPlot()
            w3.setLabel('bottom', text='ΔI', units='A')
            w3.setLabel('left', text='Counts')

            w4 = self.ui.dwellhistplot.addPlot()
            w4.setLabel('bottom', text='Log Dwell Time', units='μs')
            w4.setLabel('left', text='Counts')

            w5 = self.ui.dthistplot.addPlot()
            w5.setLabel('bottom', text='dt', units='s')
            w5.setLabel('left', text='Counts')
            return w2, w3, w4, w5

        self.w2, self.w3, self.w4, self.w5 = setup_plots()

        def load_logo():
            dir_path = os.path.dirname(os.path.realpath(__file__))
            logo = ndimage.imread(dir_path + os.sep.join(["", "Assets", "pythionlogo.png"]))
            logo = np.rot90(logo, -1)
            logo = pg.ImageItem(logo)
            return logo

        self.logo = load_logo()

        def setup_p3(ui, logo):
            p3 = ui.eventplot.addPlot()
            p3.hideAxis('bottom')
            p3.hideAxis('left')
            p3.addItem(logo)
            p3.setAspectLocked(True)
            return p3

        self.p3 = setup_p3(self.ui, self.logo)

        # TODO: See if this can be removed
        #        self.w6 = self.ui.PSDplot.addPlot()
        #        self.w6.setLogMode(x = True, y = True)
        #        self.w6.setLabel('bottom', text='Frequency (Hz)')
        #        self.w6.setLabel('left', text='PSD (pA^2/Hz)')

        # Initializing various variables used for analysis
        self.wd = os.getcwd()
        self.data_file_name = []
        self.base_region = pg.LinearRegionItem()
        self.base_region.hide()  # Needed since LinearRegionItems is loaded as visible
        self.cut_region = pg.LinearRegionItem(brush=(198, 55, 55, 75))
        self.cut_region.hide()
        self.last_event = []
        self.last_clicked = []
        self.has_baseline_been_set = False
        self.last_event = 0
        self.del_i = []
        self.frac = []
        self.dwell = []
        self.dt = []
        self.cat_data = []
        self.colors = []
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
        self.min_dwell = None
        self.min_frac = None
        self.min_level_t = None
        self.LPfiltercutoff = None
        self.max_states = 1
        self.noise = None
        self.start_points = None
        self.end_points = None
        self.t_cat = None
        self.width = width
        self.height = height
        self.file_type = None

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

        # TODO: This may break it
        self.cat_data = []

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
        wd = self.wd
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
            self.wd = wd
            self.load()

        except IOError as e:
            # if user cancels during file selection, exit loop
            print(e)

    def analyze(self):
        # start_points, end_points, mins = None, None, None  # unused
        data = self.data
        if data is None:
            return
        baseline = self.baseline
        var = self.var
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4
        w5 = self.w5
        output_sample_rate = self.output_sample_rate
        sdf = self.sdf
        t = self.t
        p1 = self.p1
        p2 = self.p2
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
        below = np.where(data < threshold)[0]

        # locate the points where the current crosses the threshold

        start_and_end = np.diff(below)
        start_points = np.insert(start_and_end, 0, 2)
        end_points = np.insert(start_and_end, -1, 2)
        start_points = np.where(start_points > 1)[0]
        end_points = np.where(end_points > 1)[0]
        start_points = below[start_points]
        end_points = below[end_points]

        # Eliminate events that start before file or end after file

        if start_points[0] == 0:
            start_points = np.delete(start_points, 0)
            end_points = np.delete(end_points, 0)
        if end_points[-1] == len(data) - 1:
            start_points = np.delete(start_points, -1)
            end_points = np.delete(end_points, -1)

        # Track points back up to baseline to find true start and end

        num_events = len(start_points)
        high_thresh = baseline - var

        def find_a_name(data, start_points, end_points, high_thresh, num_events):
            pass

        for event_idx in range(num_events):
            # Corrects the start point
            event_start = start_points[event_idx]  # mark initial guess for starting point
            while data[event_start] < high_thresh and event_start > 0:
                event_start = event_start - 1  # track back until we return to baseline
            start_points[event_idx] = event_start  # mark true start point

            # Corrects end point
            event_end = end_points[event_idx]  # repeat process for end point
            if event_end == len(data) - 1:  # sure that the current returns to baseline
                end_points[event_idx] = 0  # before file ends. If not, mark points for
                start_points[event_idx] = 0  # deletion and break from loop
                # event_end = 0 # unused
                break
            while data[event_end] < high_thresh:
                event_end = event_end + 1
                if event_end == len(data) - 1:  # sure that the current returns to baseline
                    end_points[event_idx] = 0  # before file ends. If not, mark points for
                    start_points[event_idx] = 0  # deletion and break from loop
                    # event_end = 0  # unused
                    break
                else:
                    try:
                        if event_end > start_points[event_idx + 1]:  # if we hit the next startpoint before we
                            start_points[event_idx + 1] = 0  # return to baseline, mark for deletion
                            end_points[event_idx] = 0  # and break out of loop
                            event_end = 0
                            break
                    except IndexError as e:
                        pass
                        # raise IndexError
                end_points[event_idx] = event_end

        start_points = start_points[start_points != 0]  # delete those events marked for
        end_points = end_points[end_points != 0]  # deletion earlier
        num_events = len(start_points)

        if len(start_points) > len(end_points):
            start_points = np.delete(start_points, -1)
            num_events = len(start_points)

        # Now we want to move the endpoints to be the last minimum for each
        # event so we find all minimas for each event, and set endpoint to last

        del_i = np.zeros(num_events)
        dwell = np.zeros(num_events)

        for i in range(num_events):
            event_start = start_points[i]
            event_end = end_points[i]
            mins = np.array(signal.argrelmin(data[event_start:event_end])[0] + event_start)
            # Ensures the minimum points are sufficiently far away from baseline (4 stdevs?)
            mins = mins[data[mins] < baseline - 4 * var]
            if len(mins) == 1:
                pass
                del_i[i] = baseline - min(data[event_start:event_end])  # Minimum of all data
                dwell[i] = (event_end - event_start) * 1e6 / output_sample_rate
                # event_end = mins[0] # unused
            elif len(mins) > 1:
                del_i[i] = baseline - np.mean(data[mins[0]:mins[-1]])  # Minimum to minimum for del
                event_end = mins[-1]
                dwell[i] = (event_end - event_start) * 1e6 / output_sample_rate

        start_points = start_points[del_i != 0]
        end_points = end_points[del_i != 0]
        del_i = del_i[del_i != 0]
        dwell = dwell[dwell != 0]
        frac = del_i / baseline
        dt = np.array(0)
        dt = np.append(dt, np.diff(start_points) / output_sample_rate)
        num_events = len(dt)
        noise = (10 ** 10) * np.array([np.std(data[x:end_points[i]]) for i, x in enumerate(start_points)])

        # Plotting starts after this
        p1.clear()

        # skips plotting first and last two points, there was a weird spike issue
        #        self.p1.plot(self.t[::10][2:][:-2],data[::10][2:][:-2],pen='b')
        p1.setDownsampling(ds=False)
        p1.plot(t[2:][:-2], data[2:][:-2], pen='b')
        # FIXME: Line beow is currently broken when autosize viewport of graph is set.
        if len(t[start_points]) >= 2 or len(data[start_points]) >= 2:
            p1.plot(t[start_points], data[start_points], pen=None, symbol='o', symbolBrush='g', symbolSize=10)
            p1.plot(t[end_points], data[end_points], pen=None, symbol='o', symbolBrush='r', symbolSize=10)

        ui.eventcounterlabel.setText('Events:' + str(num_events))
        # noinspection PyTypeChecker
        ui.meandelilabel.setText('Deli:' + str(round(np.mean(del_i * BILLION), 2)) + ' nA')
        # noinspection PyTypeChecker
        ui.meandwelllabel.setText('Dwell:' + str(round(np.median(dwell), 2)) + u' μs')
        ui.meandtlabel.setText('Rate:' + str(round(num_events / t[-1], 1)) + ' events/s')

        try:
            p2.data = p2.data[np.where(np.array(sdf.fn) != info_file_name)]
        except Exception as e:
            print(e)
            raise IndexError
        sdf = sdf[sdf.fn != info_file_name]

        fn = pd.Series([info_file_name, ] * num_events)
        color = pd.Series([pg.colorTuple(cb.color()), ] * num_events)

        sdf = sdf.append(pd.DataFrame({'fn': fn, 'color': color, 'deli': del_i,
                                       'frac': frac, 'dwell': dwell,
                                       'dt': dt, 'stdev': noise, 'startpoints': start_points,
                                       'endpoints': end_points}), ignore_index=True)

        p2.addPoints(x=np.log10(dwell), y=frac, symbol='o', brush=(cb.color()), pen=None, size=10)

        w1.addItem(p2)
        w1.setLogMode(x=True, y=False)
        p1.autoRange()
        w1.autoRange()
        ui.scatterplot.update()
        w1.setRange(yRange=[0, 1])

        colors = sdf.color.unique()
        for i, x in enumerate(colors):
            frac_y, frac_x = np.histogram(sdf.frac[sdf.color == x],
                                          bins=np.linspace(0, 1, int(ui.fracbins.text())))
            deli_y, deli_x = np.histogram(sdf.deli[sdf.color == x],
                                          bins=np.linspace(float(ui.delirange0.text()) * 10 ** -9,
                                                           float(ui.delirange1.text()) * 10 ** -9,
                                                           int(ui.delibins.text())))
            dwell_y, dwell_x = np.histogram(np.log10(sdf.dwell[sdf.color == x]),
                                            bins=np.linspace(float(ui.dwellrange0.text()),
                                                             float(ui.dwellrange1.text()),
                                                             int(ui.dwellbins.text())))
            dt_y, dt_x = np.histogram(sdf.dt[sdf.color == x],
                                      bins=np.linspace(float(ui.dtrange0.text()),
                                                       float(ui.dtrange1.text()),
                                                       int(ui.dtbins.text())))

            # hist = pg.PlotCurveItem(frac_y, frac_x , stepMode = True, fillLevel=0, brush = x, pen = 'k')
            hist = pg.BarGraphItem(height=frac_y, x0=frac_x[:-1], x1=frac_x[1:], brush=x)
            w2.addItem(hist)
            # hist = pg.PlotCurveItem(deli_x, deli_y , stepMode = True, fillLevel=0, brush = x, pen = 'k')
            hist = pg.BarGraphItem(height=deli_y, x0=deli_x[:-1], x1=deli_x[1:], brush=x)
            w3.addItem(hist)
            # w3.autoRange()
            w3.setRange(xRange=[float(ui.delirange0.text()) * 10 ** -9, float(ui.delirange1.text()) * 10 ** -9])
            # hist = pg.PlotCurveItem(dwell_x, dwell_y , stepMode = True, fillLevel=0, brush = x, pen = 'k')
            hist = pg.BarGraphItem(height=dwell_y, x0=dwell_x[:-1], x1=dwell_x[1:], brush=x)
            w4.addItem(hist)
            # hist = pg.PlotCurveItem(dt_x, dt_y , stepMode = True, fillLevel=0, brush = x, pen = 'k')
            hist = pg.BarGraphItem(height=dt_y, x0=dt_x[:-1], x1=dt_x[1:], brush=x)
            w5.addItem(hist)

        self.save()
        self.save_target()

        self.threshold = threshold
        self.num_events = num_events
        self.analyze_type = analyze_type
        self.dwell = dwell
        self.del_i = del_i
        self.frac = frac
        self.dt = dt
        self.noise = noise
        self.sdf = sdf

    def save(self):
        mat_file_name = self.mat_file_name
        del_i = self.del_i
        frac = self.frac
        dwell = self.dwell
        dt = self.dt
        noise = self.noise
        np.savetxt(mat_file_name + 'DB.txt',
                   np.column_stack((del_i, frac, dwell, dt, noise)), delimiter='\t',
                   header='\t'.join(["deli", "frac", "dwell", "dt", 'stdev']))

    # Called by Go button
    # TODO: Continue cleanump and input sanitizing
    def inspect_event(self, clicked=[].copy()):
        num_events = self.num_events
        if not num_events:
            return
        ui = self.ui
        start_points = self.start_points
        end_points = self.end_points
        baseline = self.baseline
        data = self.data
        p2 = self.p2
        p3 = self.p3
        sdf = self.sdf
        t = self.t
        dwell = self.dwell
        del_i = self.del_i
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
        start_pt = start_points[event_number]
        end_pt = end_points[event_number]

        adj_start = int(start_pt - event_buffer)
        adj_end = int(end_pt + event_buffer)
        p3.plot(t[adj_start:adj_end], data[adj_start:adj_end], pen='b')

        event_fit = np.concatenate((
            np.repeat(np.array([baseline]), event_buffer),
            np.repeat(np.array([baseline - del_i[event_number]]), end_pt - start_pt),
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
        event_start = int(start_pt)
        event_end = int(end_pt)
        p3.plot([t[event_start], t[event_start]],
                [data[event_start], data[event_start]], pen=None,
                symbol='o', symbolBrush='g', symbolSize=12)
        p3.plot([t[event_end], t[event_end]],
                [data[event_end], data[event_end]], pen=None,
                symbol='o', symbolBrush='r', symbolSize=12)

        ui.eventinfolabel.setText('Dwell Time=' + str(round(dwell[event_number], 2)) + u' μs,   Deli=' + str(
            round(del_i[event_number] * BILLION, 2)) + ' nA')

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
        num_events = self.num_events
        # Ignore command if there are no events
        if not num_events:
            return
        input_num = num_from_text_element(element, 1, num_events, default=1)
        element.setText(str(input_num + 1))
        self.inspect_event()

    def previous_event(self):
        element = self.ui.eventnumberentry
        num_events = self.num_events
        # Ignore command if there are no events
        if not num_events:
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

    def base_line_calc(self):
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
        # global start_points, end_points
        num_events = self.num_events
        if num_events is 0:
            return
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4
        w5 = self.w5
        p2 = self.p2
        ui = self.ui
        sdf = self.sdf
        analyze_type = self.analyze_type
        mat_file_name = self.mat_file_name
        del_i = self.del_i
        dwell = self.dwell
        dt = self.dt
        frac = self.frac
        noise = self.noise
        start_points = self.start_points
        end_points = self.end_points

        event_number = np.int(ui.eventnumberentry.text())
        first_index = sdf.fn[sdf.fn == mat_file_name].index[0]
        if event_number > num_events:
            event_number = num_events - 1
            ui.eventnumberentry.setText(str(event_number))
        del_i = np.delete(del_i, event_number)
        dwell = np.delete(dwell, event_number)
        dt = np.delete(dt, event_number)
        frac = np.delete(frac, event_number)
        try:
            noise = np.delete(noise, event_number)
        except AttributeError as e:
            print(e)
            raise AttributeError
        self.start_points = np.delete(start_points, event_number)
        self.end_points = np.delete(end_points, event_number)
        p2.data = np.delete(p2.data, first_index + event_number)

        num_events = len(dt)
        ui.eventcounterlabel.setText('Events:' + str(num_events))

        sdf = sdf.drop(first_index + event_number).reset_index(drop=True)
        self.inspect_event()

        w2.clear()
        w3.clear()
        w4.clear()
        w5.clear()
        colors = sdf.color.unique()
        for i, x in enumerate(colors):
            frac_y, frac_x = np.histogram(sdf.frac[sdf.color == x],
                                          bins=np.linspace(0, 1, int(ui.fracbins.text())))
            deli_y, deli_x = np.histogram(sdf.deli[sdf.color == x],
                                          bins=np.linspace(float(ui.delirange0.text()) * 10 ** -9,
                                                           float(ui.delirange1.text()) * 10 ** -9,
                                                           int(ui.delibins.text())))
            dwell_y, dwell_x = np.histogram(np.log10(sdf.dwell[sdf.color == x]),
                                            bins=np.linspace(float(ui.dwellrange0.text()),
                                                             float(ui.dwellrange1.text()),
                                                             int(ui.dwellbins.text())))
            dty, dtx = np.histogram(sdf.dt[sdf.color == x],
                                    bins=np.linspace(float(ui.dtrange0.text()), float(ui.dtrange1.text()),
                                                     int(ui.dtbins.text())))

            #            hist = pg.PlotCurveItem(frac_y, frac_x , stepMode = True, fillLevel=0, brush = x, pen = 'k')

            hist = pg.BarGraphItem(height=frac_y, x0=frac_x[:-1], x1=frac_x[1:], brush=x)
            w2.addItem(hist)

            #            hist = pg.PlotCurveItem(deli_x, deli_y , stepMode = True, fillLevel=0, brush = x, pen = 'k')

            hist = pg.BarGraphItem(height=deli_y, x0=deli_x[:-1], x1=deli_x[1:], brush=x)
            w3.addItem(hist)
            #            w3.autoRange()
            w3.setRange(
                xRange=[float(ui.delirange0.text()) * 10 ** -9, float(ui.delirange1.text()) * 10 ** -9])

            #            hist = pg.PlotCurveItem(dwell_x, dwell_y , stepMode = True, fillLevel=0, brush = x, pen = 'k')

            hist = pg.BarGraphItem(height=dwell_y, x0=dwell_x[:-1], x1=dwell_x[1:], brush=x)
            w4.addItem(hist)

            #            hist = pg.PlotCurveItem(dtx, dty , stepMode = True, fillLevel=0, brush = x, pen = 'k')

            hist = pg.BarGraphItem(height=dty, x0=dtx[:-1], x1=dtx[1:], brush=x)
            w5.addItem(hist)

        if analyze_type == 'coarse':
            self.save()
            self.save_target()
        if analyze_type == 'fine':
            np.savetxt(mat_file_name + 'llDB.txt',
                       np.column_stack((del_i, frac, dwell, dt, noise)),
                       delimiter='\t', header="\t".join(["deli", "frac", "dwell", "dt", 'stdev']))

        self.ui = ui
        self.dwell = dwell
        self.del_i = del_i
        self.dt = dt
        self.frac = frac
        self.noise = noise

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
        wd = self.wd
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

    def save_cat_trace(self):
        data = self.data
        if data is None:
            return
        ui = self.ui
        dt = self.dt
        start_points = self.start_points
        end_points = self.end_points
        event_buffer = np.int(ui.eventbufferentry.text())
        num_events = len(dt)
        baseline = self.baseline
        del_i = self.del_i
        output_sample_rate = self.output_sample_rate
        mat_file_name = self.mat_file_name
        cat_data = None
        cat_fits = None
        # NOTE: Below was changed from idx in range(num_events - 1) due to assumed off-by-one error
        # Section with similar code was changed as well
        for idx in range(1, num_events):
            adj_start = start_points[idx] - event_buffer
            adj_end = end_points[idx] + event_buffer
            if adj_end > start_points[idx + 1]:
                print('overlapping event')
            else:
                data_pts = generate_data_pts(data, adj_start, adj_end)
                event_pts = generate_event_pts(adj_start, adj_end, event_buffer, baseline, del_i[idx])
                if idx is 0:
                    cat_data = data_pts
                    cat_fits = event_pts
                else:
                    cat_data = np.concatenate((cat_data, data_pts), 0)
                    cat_fits = np.concatenate((cat_fits, event_pts), 0)

        t_cat = np.arange(0, len(cat_data)) / output_sample_rate
        cat_data = cat_data[::10]
        cat_data.astype('d').tofile(mat_file_name + '_cattrace.bin')

        self.cat_data = cat_data
        self.cat_fits = cat_fits
        self.t_cat = t_cat

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

    # Static ?
    def save_event_fits(self):
        data = self.data
        if data is None:
            return
        ui = self.ui
        dt = self.dt
        start_points = self.start_points
        end_points = self.end_points
        event_buffer = np.int(ui.eventbufferentry.text())
        num_events = len(dt)
        baseline = self.baseline
        del_i = self.del_i
        output_sample_rate = self.output_sample_rate
        mat_file_name = self.mat_file_name
        cat_data = None
        cat_fits = None

        for idx in range(num_events):
            adj_start = start_points[idx] - event_buffer
            adj_end = end_points[idx] + event_buffer
            if adj_end > start_points[idx + 1]:
                print('overlapping event')
            else:
                data_pts = generate_data_pts(data, adj_start, adj_end)
                event_pts = generate_event_pts(adj_start, adj_end, event_buffer, baseline, del_i[idx])
                if idx is 0:
                    cat_data = data_pts
                    cat_fits = event_pts
                else:
                    cat_data = np.concatenate((cat_data, data_pts), 0)
                    cat_fits = np.concatenate((cat_fits, event_pts), 0)

        t_cat = np.arange(0, len(cat_data)) / output_sample_rate
        cat_fits.astype('d').tofile(mat_file_name + '_cattrace.bin')

        self.cat_fits = cat_fits
        self.cat_data = cat_data
        self.t_cat = t_cat

    def cusum(self):
        ui = self.ui
        ui_bp = self.ui.uibp
        if self.data is None:
            return
        self.p1.clear()
        self.p1.setDownsampling(ds=False)
        dt = 1 / self.output_sample_rate
        cusum_thresh = np.float64(ui_bp.cusumthreshentry.text())
        step_size = np.float64(ui.levelthresholdentry.text())
        cusum = detect_cusum(self.data, base_sd=self.var, dt=dt, threshhold=cusum_thresh, stepsize=step_size,
                             minlength=10)
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
        batch_info = self.batch_info
        mat_file_name = self.mat_file_name
        del_i = self.del_i
        frac = self.frac
        dwell = self.dwell
        dt = self.dt
        noise = self.noise
        start_points = self.start_points
        end_points = self.end_points
        cut_start = batch_info["cutstart"]
        cut_end = batch_info["cutend"]
        batch_info = pd.DataFrame({'cutstart': cut_start, 'cutend': cut_end})
        batch_info = batch_info.dropna()
        batch_info = batch_info.append(pd.DataFrame({'deli': del_i, 'frac': frac, 'dwell': dwell, 'dt': dt,
                                                     'noise': noise, 'startpoints': start_points,
                                                     'endpoints': end_points}),
                                       ignore_index=True)
        batch_info.to_pickle(mat_file_name + 'batchinfo.pkl')
        self.batch_info = batch_info

    def batch_info_dialog(self):
        min_dwell = self.min_dwell
        min_frac = self.min_frac
        min_level_t = self.min_level_t
        sample_rate = self.sample_rate
        lp_filter_cutoff = self.lp_filter_cutoff
        cusum_step = self.cusum_step
        cusum_thresh = self.cusum_thresh
        max_states = self.max_states
        p1 = self.p1
        p1.clear()
        # self.batch_processor = BatchProcessor()
        # self.batch_processor.show()
        ui_bp = self.bp.uibp
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
    # def batch_process(self, data, ui):
    #     # global end_points, start_points
    #     self.analyze_type = 'fine'
    #     ui_bp = self.bp.uibp
    #     # p1 = self.p1
    #     # min_dwell = self.min_dwell
    #     # min_frac = self.min_frac
    #     # min_level_t = self.min_level_t
    #     # sample_rate = self.sample_rate
    #     # lp_filter_cutoff = self.LPfiltercutoff
    #     # max_states = self.max_states
    #     # file_list = self.file_list
    #     data_file_name = self.data_file_name
    #     wd = self.wd
    #     has_baseline_been_set = self.has_baseline_been_set
    #     baseline = self.baseline
    #     var = self.var
    #     output_sample_rate = self.output_sample_rate
    #     mat_file_name = self.mat_file_name
    #     sdf = self.sdf
    #     # num_events = self.num_events
    #     dt = self.dt
    #     del_i = self.del_i
    #     frac = self.frac
    #     dwell = self.dwell
    #     noise = self.noise
    #     p1 = self.p1
    #     p2 = self.p2
    #     w1 = self.w1
    #     w2 = self.w2
    #     w3 = self.w3
    #     w4 = self.w4
    #     w5 = self.w5
    #     cb = self.cb
    #     start_points = self.start_points
    #     end_points = self.end_points
    #
    #     invert_status = ui_bp.invertCheckBox.isChecked()
    #     ui_bp.close()
    #     p1.setDownsampling(ds=False)
    #     min_dwell = np.float64(ui_bp.mindwellbox.text())
    #     min_frac = np.float64(ui_bp.minfracbox.text())
    #     min_level_t = np.float64(ui_bp.minleveltbox.text()) * 10 ** -6
    #     sample_rate = ui_bp.sampratebox.text()
    #     lp_filter_cutoff = ui_bp.LPfilterbox.text()
    #     ui.outputsamplerateentry.setText(sample_rate)
    #     ui.LPentry.setText(lp_filter_cutoff)
    #     cusum_step = np.float64(ui_bp.cusumstepentry.text())
    #     cusum_thresh = np.float64(ui_bp.cusumthreshentry.text())
    #     max_states = np.int(ui_bp.maxLevelsBox.text())
    #     self_correct = ui_bp.selfCorrectCheckBox.isChecked()
    #
    #     try:
    #         # attempt to open dialog from most recent directory
    #         file_list = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Files', wd, "*.pkl")[0]
    #         wd = os.path.dirname(file_list[0])
    #     except TypeError as e:
    #         print(e)
    #         # if no recent directory exists open from working directory
    #         file_list = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Files', os.getcwd(), "*.pkl")[0]
    #         print(file_list)
    #         # self.wd=os.path.dirname(str(self.filelist[0][0]))
    #     except IOError as e:
    #         print(e)
    #         # if user cancels during file selection, exit loop
    #         return
    #
    #     event_buffer = np.int(ui.eventbufferentry.text())
    #     event_time = [0]
    #
    #     for f in file_list:
    #         batch_info = pd.read_pickle(f)
    #         try:
    #             data_file_name = f[:-13] + '.opt'
    #             self.load(load_and_plot=False)
    #         except IOError:
    #             data_file_name = f[:-13] + '.log'
    #             self.load(load_and_plot=False)
    #         if invert_status:
    #             data = -data
    #             if not has_baseline_been_set:
    #                 baseline = np.median(data)
    #                 var = np.std(data)
    #
    #         try:
    #             cs = batch_info.cutstart[np.isfinite(batch_info.cutstart)]
    #             ce = batch_info.cutend[np.isfinite(batch_info.cutend)]
    #             for i, cut in enumerate(cs):
    #                 data = np.delete(data, np.arange(np.int(cut * output_sample_rate),
    #                                                  np.int(ce[i] * output_sample_rate)))
    #         except TypeError as e:
    #             print(e)
    #
    #         del_i = np.array(batch_info.deli[np.isfinite(batch_info.deli)])
    #         frac = np.array(batch_info.frac[np.isfinite(batch_info.frac)])
    #         dwell = np.array(batch_info.dwell[np.isfinite(batch_info.dwell)])
    #         dt = np.array(batch_info.dt[np.isfinite(batch_info.dt)])
    #         start_points = np.array(batch_info.startpoints[np.isfinite(batch_info.startpoints)])
    #         end_points = np.array(batch_info.endpoints[np.isfinite(batch_info.endpoints)])
    #         noise = (10 ** 10) * np.array(
    #             [np.std(data[int(x):int(end_points[i])]) for i, x in enumerate(start_points)])
    #
    #         with pg.ProgressDialog("Analyzing...", 0, len(dwell)) as dlg:
    #             for i, dwell in enumerate(dwell):
    #                 # t_offset = (event_time[-1] + event_buffer) / output_sample_rate
    #                 if i < len(dt) - 1 and dwell > min_dwell and frac[i] > min_frac:
    #                     if end_points[i] + event_buffer > start_points[i + 1]:
    #                         print('overlapping event')
    #                         frac[i] = np.NaN
    #                         del_i[i] = np.NaN
    #
    #                     else:
    #                         eventdata = data[int(start_points[i] - event_buffer):int(end_points[i] + event_buffer)]
    #                         event_time = np.arange(0, len(eventdata)) + event_buffer + event_time[-1]
    #                         # self.p1.plot(eventtime/self.outputsamplerate, eventdata,pen='b')
    #                         cusum = detect_cusum(eventdata, np.std(eventdata[0:event_buffer]),
    #                                              1 / output_sample_rate, threshhold=cusum_thresh,
    #                                              stepsize=cusum_step,
    #                                              minlength=min_level_t * output_sample_rate,
    #                                              max_states=max_states)
    #
    #                         while len(cusum['CurrentLevels']) < 3:
    #                             cusum_thresh = cusum_thresh * .9
    #                             cusum_step = cusum_step * .9
    #                             cusum = detect_cusum(eventdata, base_sd=np.std(eventdata[0:event_buffer]),
    #                                                  dt=1 / output_sample_rate, threshhold=cusum_thresh,
    #                                                  stepsize=cusum_step,
    #                                                  minlength=min_level_t * output_sample_rate,
    #                                                  max_states=max_states)
    #                             print('Not Sensitive Enough')
    #
    #                         frac[i] = (np.max(cusum['CurrentLevels']) - np.min(cusum['CurrentLevels'])) / np.max(
    #                             cusum['CurrentLevels'])
    #                         del_i[i] = (np.max(cusum['CurrentLevels']) - np.min(cusum['CurrentLevels']))
    #
    #                         if self_correct:
    #                             cusum_thresh = cusum['Threshold']
    #                             cusum_step = cusum['stepsize']
    #                 # Plotting
    #                 # for j, level in enumerate(cusum['CurrentLevels']):
    #                 #     self.p1.plot(y=2*[level], x=t_offset + cusum['EventDelay'][j:j+2], pen=pg.mkPen('r', width=5))
    #                 #     try:
    #                 #         self.p1.plot(y=cusum['CurrentLevels'][j:j+2], x=t_offset + 2*[cusum['EventDelay'][j+1]],
    #                 #                      pen=pg.mkPen('r', width=5))
    #                 #     except Exception as e:
    #                 #         print(e)
    #                 #         pass
    #                 dlg += 1
    #             # End Plotting
    #
    #         dwell = dwell[np.isfinite(del_i)]
    #         dt = dt[np.isfinite(del_i)]
    #         noise = noise[np.isfinite(del_i)]
    #         frac = frac[np.isfinite(del_i)]
    #         start_points = start_points[np.isfinite(del_i)]
    #         end_points = end_points[np.isfinite(del_i)]
    #         del_i = del_i[np.isfinite(del_i)]
    #
    #         np.savetxt(mat_file_name + 'llDB.txt',
    #                    np.column_stack((del_i, frac, dwell, dt, noise)),
    #                    delimiter='\t', header="\t".join(["del_i", "frac", "dwell", "dt", 'stdev']))
    #
    #     p1.autoRange()
    #     # Plotting Histograms
    #     sdf = sdf[sdf.fn != mat_file_name]
    #     num_events = len(dt)
    #
    #     fn = pd.Series([mat_file_name, ] * num_events)
    #     color = pd.Series([pg.colorTuple(cb.color()), ] * num_events)
    #
    #     sdf = sdf.append(pd.DataFrame({'fn': fn, 'color': color, 'del_i': del_i,
    #                                    'frac': frac, 'dwell': dwell,
    #                                    'dt': dt, 'startpoints': start_points,
    #                                    'endpoints': end_points}), ignore_index=True)
    #
    #     p2.addPoints(x=np.log10(dwell), y=frac,
    #                  symbol='o', brush=(cb.color()), pen=None, size=10)
    #
    #     w1.addItem(p2)
    #     w1.setLogMode(x=True, y=False)
    #     p1.autoRange()
    #     w1.autoRange()
    #     ui.scatterplot.update()
    #     w1.setRange(yRange=[0, 1])
    #
    #     colors = sdf.color.unique()
    #     for i, x in enumerate(colors):
    #         frac_y, frac_x = np.histogram(sdf.frac[(sdf.color == x) & (not np.isnan(sdf.frac))],
    #                                       bins=np.linspace(0, 1, int(ui.fracbins.text())))
    #         deli_y, deli_x = np.histogram(sdf.deli[(sdf.color == x) & (not np.isnan(sdf.deli))],
    #                                       bins=np.linspace(float(ui.delirange0.text()) * 10 ** -9,
    #                                                        float(ui.delirange1.text()) * 10 ** -9,
    #                                                        int(ui.delibins.text())))
    #         dwell_y, dwell_x = np.histogram(np.log10(sdf.dwell[sdf.color == x]),
    #                                         bins=np.linspace(float(ui.dwellrange0.text()),
    #                                                          float(ui.dwellrange1.text()),
    #                                                          int(ui.dwellbins.text())))
    #         dt_y, dt_x = np.histogram(sdf.dt[sdf.color == x],
    #                                   bins=np.linspace(float(ui.dtrange0.text()), float(ui.dtrange1.text()),
    #                                                    int(ui.dtbins.text())))
    #         # w2.addItem(pg.PlotCurveItem(frac_y, frac_x , stepMode = True, fillLevel=0, brush = x, pen = 'k'))
    #         # w3.addItem(pg.PlotCurveItem(deli_x, deli_y , stepMode = True, fillLevel=0, brush = x, pen = 'k'))
    #         # w4.addItem(pg.PlotCurveItem(dwell_x, dwell_y , stepMode = True, fillLevel=0, brush = x, pen = 'k'))
    #         # w5.addItem(pg.PlotCurveItem(dt_x, dt_y , stepMode = True, fillLevel=0, brush = x, pen = 'k'))
    #         w2.addItem(pg.BarGraphItem(height=frac_y, x0=frac_x[:-1], x1=frac_x[1:], brush=x))
    #         w3.addItem(pg.BarGraphItem(height=deli_y, x0=deli_x[:-1], x1=deli_x[1:], brush=x))
    #         w3.setRange(xRange=[float(ui.delirange0.text()) * 10 ** -9, float(ui.delirange1.text()) * 10 ** -9])
    #         # self.w3.autoRange()
    #         w4.addItem(pg.BarGraphItem(height=dwell_y, x0=dwell_x[:-1], x1=dwell_x[1:], brush=x))
    #         w5.addItem(pg.BarGraphItem(height=dt_y, x0=dt_x[:-1], x1=dt_x[1:], brush=x))
    #
    #     print('\007')
    #
    #     self.wd = wd
    #     self.baseline = baseline
    #     self.var = var
    #     self.sdf = sdf
    #     self.dwell = dwell
    #     self.dt = dt
    #     self.noise = noise
    #     self.frac = frac
    #     self.start_points = start_points
    #     self.end_points = end_points
    #     self.del_i = del_i
    #     self.data_file_name = data_file_name

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
