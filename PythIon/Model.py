# from PythIon.plotguiuniversal import *

from PythIon import EdgeDetect
from PythIon.Utility import *
from math import log10


class Interval(object):
    def __init__(self, parent, data, start, end):
        self.start = start
        self.end = end
        self.data = data
        self.parent: Event = parent
        # dict to store data points
        self.data_points = {}

    def __len__(self):
        return len(self.data)

    def level(self):
        return np.mean(self.data)

    def noise(self):
        return np.std(self.data)

    def duration(self, sample_rate):
        return len(self.data) / sample_rate

    def segment(self, sample_rate, offset=0):
        level = self.level()
        start = (self.start + offset) / sample_rate
        end = (self.end + offset) / sample_rate
        return [(start, level), (end, level)]

    def frac(self, event_length):
        return len(self.data) / event_length

    def info_dixt(self, offset, sample_rate, event_length):
        data_dict = {
            'start': (offset + self.start) / sample_rate,
            'end': (offset + self.end) / sample_rate,
            'current_level': self.level(),
            'frac': self.frac(event_length),
            'duration': self.duration(sample_rate)
        }
        return data_dict

    def add_data_point(self, key, pos):
        point_data = {
            'x': pos[0],
            'y': pos[1],
            'data': self
        }
        self.data_points[key] = point_data

    def highlight(self, point_index):
        point = self.data_points[point_index]
        point['brush'] = 'r'
        point['size'] = 12

    def reset_style(self, point_index):
        point = self.data_points[point_index]
        reset_dict = {
            'x': point['x'],
            'y': point['y'],
            'data': self,
            'brush': 'b',
            'size': 10
        }
        self.data_points[point_index] = reset_dict


class Event(object):
    subevents: List

    def __init__(self, data, start, end, sample_rate, baseline=None, index=None):
        # Start and end are the indicies in the data
        # print(start, end)
        self.index = index
        self.sample_rate = sample_rate
        start = slope_crawl(data, start, 'backward')
        end = slope_crawl(data, end, 'forward')
        self.start, self.end = start, end  # start and end are relative to the whole dataset
        self.data = data[self.start: self.end]
        extrema = find_extrema(self.data)
        self.main_bounds = (extrema[0], extrema[-1] + 1)
        main_event = self.data[self.main_bounds[0]: self.main_bounds[1]]
        self.local_baseline, self.local_stedev = np.mean(main_event), np.std(main_event)
        # Intervals are relative to the internal data
        intervals = self.interval_detect(main_event, num_stdevs=1.75, min_length=1000)
        self.intervals = []
        for start, end in intervals:
            self.intervals.append(Interval(self, self.data[start:end], start, end))
        # True can false are converted to 1 and 0, np.argmax returns index of first ture value

        if baseline is not None:
            self.baseline = baseline
        else:
            self.baseline = np.concatenate(data[start - 1000:start], data[end:end + 1000]).mean()

        # TODO: Fix slight error in calculations; Too much past the tails is included in fall and rise
        # Rise_end seems to have consistent issues
        self.noise = np.std(self.data)

    def __repr__(self):
        num_levels = len(self.intervals)
        if num_levels == 1:
            level_text = ' level : '
        else:
            level_text = ' levels: '
        return 'Event with ' + str(num_levels) + level_text + ','.join([str(level) for level in self.levels()])

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
        # This section removes shor initial intervals
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
        fit_points = [(offset / self.sample_rate, self.data[0])]
        offset += self.main_bounds[0]
        for idx, interval in enumerate(self.intervals):
            start_pt, end_pt = interval.segment(self.sample_rate, offset)
            fit_points.append(start_pt)
            fit_points.append(end_pt)
        fit_points.append((self.end / self.sample_rate, self.data[-1]))
        # fit_points.append(fit_points[-1])  # Added since pyqtgraph does not display the last line for some reason
        # x, y = zip(*fit_points)
        return np.array(fit_points)

    def generate_info_table(self):
        data_list = []
        event_length = self.end - self.start
        for idx, interval in enumerate(self.intervals):
            data_dict = interval.info_dixt(offset=self.start, sample_rate=self.sample_rate, event_length=event_length)
            data_list.append(data_dict)
        return data_list

    def shift_by(self, shift):
        self.start += shift
        self.end += shift

    def levels(self):
        return [interval.level() for interval in self.intervals]


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
        if 'sample_rate' not in self.data_params.keys():
            self.data_params['sample_rate'] = 2500000
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
        index = 1
        for bound in bounds:
            start, end = bound
            if end - start < min_length:
                continue
            event = Event(self.processed_data, start, end, sample_rate, baseline, index=index)
            index += 1
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
            event_entries = event.generate_info_table()
            for entry in event_entries:
                entry['event_number'] = idx + 1
                event_data_list.append(entry)
        return pd.DataFrame(event_data_list)

    def remove_interval(self, interval):
        included_pts = np.r_[0:interval[0], interval[1]:len(self.processed_data)]
        self.processed_data = self.processed_data[included_pts]

    def add_cut(self, interval):
        # TODO: Handle cutting in the middle of an event
        start, end = interval
        print(interval)
        for event in self.events:
            if event.start > end:
                event.shift_by(-(end - start))
        self.remove_interval(interval)
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

    def get_event_idx_by_interval_idx(self, interval_idx):
        total = 0
        for idx, event in enumerate(self.events):
            total += len(event.intervals)
            if interval_idx == total - 1:
                break
        return idx

    def delete_event(self, event_idx):
        if not 0 <= event_idx < len(self.events):
            return
        bounds = (self.events[event_idx].start, self.events[event_idx].end)
        self.add_cut(bounds)
        del self.events[event_idx]

    def get_intervals(self):
        intervals = []
        for event in self.events:
            for interval in event.intervals:
                intervals.append(interval)
        return intervals


def update_signal_plot(dataset: CurrentData, signal_plot: pg.PlotItem, current_hist: pg.PlotItem):
    signal_plot.clear()  # This might be unnecessary
    signal_plot.setClipToView(clip=True)
    signal_plot.setDownsampling(ds=True, auto=True, mode='peak')
    signal_plot.setDownsampling()
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
        for pos_list in fits:
            # pyqtgraph doesn't plot a line to the last point given; repetition required
            x = list(pos_list[:, 0])
            y = list(pos_list[:, 1])
            x.append(x[-1])
            y.append(y[-1])
            # Doing it this way rather than using pos arg since that is broken it seems
            signal_plot.plot(x=x, y=y, pen='r')
    if dataset.file_type != '.abf':
        signal_plot.addLine(y=baseline, pen='g')
        signal_plot.addLine(y=threshold, pen='r')

    current_hist.clear()
    aph_y, aph_x = np.histogram(data, bins=1000)
    aph_hist = pg.PlotCurveItem(aph_x, aph_y, stepMode=True, fillLevel=0, brush='b')
    current_hist.addItem(aph_hist)
    current_hist.setXRange(np.min(data), np.max(data))


def update_event_stat_plots(instance, dataset: CurrentData, scatter_plot: pg.PlotItem, frac, del_i, dwell, dt):
    ui = instance.ui
    points = []
    sample_rate = dataset.data_params.get('sample_rate')
    scatter_plot.clear()
    intervals = dataset.get_intervals()
    for interval in intervals:
        x = interval.duration(sample_rate)
        y = 1 - (interval.level() / interval.parent.baseline)
        point_pos = (log10(x), y)
        interval.add_data_point('scatter', point_pos)
        points.append(interval.data_points['scatter'])
        interval.data_points['scatter']['brush'] = 'b'
        interval.data_points['scatter']['size'] = '10'
    # Using the list of dicts format
    scatter_data = pg.ScatterPlotItem(points)
    scatter_data.sigClicked.connect(instance.clicked)
    scatter_plot.addItem(scatter_data)
