import math
import os
from abc import abstractmethod

import numpy as np

import pyqtgraph as pg

from scipy import ndimage

# from PythIon.plotguiuniversal import *
from PythIon.Model import CurrentData, Event
from PythIon.Widgets.PlotGUI import *


class PGPlot(object):
    plot: pg.PlotItem

    def __init__(self, graphics_view, x_label, y_label, x_units=None, y_units=None):
        self.plot = graphics_view.addPlot()
        if x_units is None:
            self.plot.setLabel('bottom', text=x_label)
        else:
            self.plot.setLabel('bottom', text=x_label, units=x_units)

        if y_units is None:
            self.plot.setLabel('left', text=y_label)
        else:
            self.plot.setLabel('left', text=y_label, units=y_units)
        graphics_view.setBackground('w')
        self.brush = 'b'

    @abstractmethod
    def update(self, dataset: CurrentData):
        pass


class PGScatter(PGPlot):
    def __init__(self, graphics_view, x_label, y_label, x_units=None, y_units=None):
        super().__init__(graphics_view, x_label, y_label, x_units=x_units, y_units=y_units)

    @abstractmethod
    def update(self, dataset: CurrentData):
        pass

    def _basic_update(self, data):
        # Using the list of dicts format
        scatter = pg.ScatterPlotItem(data)
        self.plot.clear()
        self.plot.addItem(scatter)


class PGHist(PGPlot):
    def __init__(self, graphics_view, x_label, y_label, x_units=None, y_units=None):
        super().__init__(graphics_view, x_label, y_label, x_units=x_units, y_units=y_units)

    @abstractmethod
    def update(self, dataset: CurrentData):
        pass

    def _basic_update(self, data, bins):
        y, x = np.histogram(data, bins=bins)
        hist = pg.BarGraphItem(height=y, x0=x[:-1], x1=x[1:], brush=self.brush)
        self.plot.clear()
        self.plot.addItem(hist)


class Region(pg.LinearRegionItem):
    def __init__(self, parent_plot, sample_rate, brush=None):
        super().__init__(brush=brush)
        self.hide()  # Needed since LinearRegionItems is loaded as visible
        parent_plot.addItem(self)
        self.sample_rate = sample_rate

    def get_bounds(self):
        left_bound, right_bound = self.getRegion()
        interval = (int(left_bound * self.sample_rate), int(right_bound * self.sample_rate) + 1)
        return interval

    def toggle_region(self):
        if self.isVisible():
            self.hide()
            return self.get_bounds()
        else:
            self.show()
            return None


class SignalPlot(PGHist):
    def __init__(self, ui: Ui_PythIon):
        x_lab = 'Time'
        y_lab = 'Current'
        x_unit = 's'
        y_unit = 'A'
        super().__init__(ui.signal_plot, x_lab, y_lab, x_unit, y_unit)
        self.plot.setClipToView(clip=True)  # THIS IS THE MOST IMPORTANT LINE!!!!
        self.plot.setDownsampling(ds=True, auto=True, mode='peak')
        self.dataset = None
        sample_rate = None
        self.base_region = Region(self.plot, sample_rate)
        self.cut_region = Region(self.plot, sample_rate, brush=(198, 55, 55, 75))

    def update(self, dataset: CurrentData):
        self.plot.clear()
        data = dataset.processed_data
        sample_rate = dataset.data_params.get('sample_rate')
        t = np.arange(0, len(data)) / sample_rate
        self.plot.plot(x=t, y=data, pen='b')
        if dataset.events:
            fits = dataset.event_fits()
            for pos_list in fits:
                # pyqtgraph doesn't plot a line to the last point given; repetition required
                x = list(pos_list[:, 0])
                y = list(pos_list[:, 1])
                x.append(x[-1])
                y.append(y[-1])
                # Doing it this way rather than using pos arg since that is broken it seems
                self.plot.plot(x=x, y=y, pen='r')

    def baseline(self):
        """
        Toggle that allows a region of the graph to be selected to used as the baseline.
        """
        interval = self.base_region.toggle_region()
        if interval:
            start, end = interval
            baseline = np.mean(self.dataset.processed_data[start: end])
            self.dataset.data_params['baseline'] = baseline

    def cut(self):
        """
        Allows user to select region of data to be removed
        """
        interval = self.cut_region.toggle_region()
        if interval:
            self.dataset.add_cut(interval)

    def register_dataset(self, dataset):
        self.dataset = dataset
        sample_rate = self.dataset.data_params.get('sample_rate')
        self.cut_region.sample_rate = sample_rate
        self.base_region.sample_rate = sample_rate


class BlockageScatter(PGScatter):
    def __init__(self, ui):
        x_lab = 'Interval Duration'
        y_lab = 'Fractional Current Bloackage'
        x_unit = 's'
        super().__init__(ui.scatter_plot, x_lab, y_lab, x_unit)
        # TODO: Set limits
        self.highlighted_event = None
        self.plot.setLogMode(x=True, y=False)
        self.plot.showGrid(x=True, y=True)

    def update(self, dataset: CurrentData):
        points = []
        intervals = dataset.get_intervals()
        if not intervals:
            return
        self.plot.clear()
        for interval in intervals:
            points.append(interval.data_points['scatter'])
        data_item = pg.ScatterPlotItem(points)
        data_item.sigClicked.connect(self.clicked)
        self.plot.addItem(data_item)

    def clicked(self, _, points):
        # Plot parameter is omitted since it will always be self.plot
        if not points:
            return
        event: Event = points[0].data().parent
        if isinstance(self.highlighted_event, Event):
            for interval in self.highlighted_event.intervals:
                interval.reset_style('scatter')
        for interval in event.intervals:
            # Highlight all points on the graph in the same event
            interval.highlight('scatter')
        self.highlighted_event = event
        event.inspect()
        self.update(event.parent)

    @staticmethod
    def create_points(dataset: CurrentData):
        sample_rate = dataset.data_params.get('sample_rate')
        intervals = dataset.get_intervals()
        if not intervals:
            return
        for interval in intervals:
            x = interval.duration(sample_rate)
            y = 1 - (interval.level() / interval.parent.baseline)
            point_pos = (math.log10(x), y)
            interval.add_data_point('scatter', point_pos)
            interval.reset_style('scatter')


class FracHist(PGHist):
    def __init__(self, ui):
        x_lab = 'Fractional Current Blockage'
        y_lab = 'Counts'
        super().__init__(ui.frac_plot, x_lab, y_lab)
        self.text_inputs = {
            'frac_bins': ui.fracbins
        }

    def update(self, dataset: CurrentData):
        data = [interval.level(normalize=True) for interval in dataset.get_intervals()]
        num_bins = int(self.text_inputs['frac_bins'].text())
        bins = np.linspace(0, 1, num_bins)
        self._basic_update(data, bins)


class DeltaHist(PGHist):
    def __init__(self, ui):
        x_lab = 'ΔI'
        y_lab = 'Counts'
        x_unit = 'A'
        super().__init__(ui.del_i_plot, x_lab, y_lab, x_unit)
        self.text_inputs = {
            'delta_bins': ui.delibins,
            'delta_start': ui.delirange0,
            'delta_end': ui.delirange1
        }

    def update(self, dataset: CurrentData):
        baseline = dataset.data_params.get('baseline')
        data = [baseline - interval.level(normalize=False) for interval in dataset.get_intervals()]
        num_bins = int(self.text_inputs['delta_bins'].text())
        start = float(self.text_inputs['delta_start'].text()) * 10 ** -9
        end = float(self.text_inputs['delta_end'].text()) * 10 ** -9
        bins = np.linspace(start, end, num_bins)
        self._basic_update(data, bins)


class DurationHist(PGHist):
    def __init__(self, ui):
        x_lab = 'Log Dwell Time'
        y_lab = 'Counts'
        x_unit = 's'
        super().__init__(ui.dwell_plot, x_lab, y_lab, x_unit)
        self.text_inputs = {
            'duration_bins': ui.dwellbins,
            'duration_start': ui.dwellrange0,
            'duration_end': ui.dwellrange1
        }

    def update(self, dataset: CurrentData):
        sample_rate = dataset.data_params['sample_rate']
        data = [math.log10(interval.duration(sample_rate)) for interval in dataset.get_intervals()]
        num_bins = int(self.text_inputs['duration_bins'].text())
        start = float(self.text_inputs['duration_start'].text())
        end = float(self.text_inputs['duration_end'].text())
        bins = np.linspace(start, end, num_bins)
        self._basic_update(data, bins)


class DTHist(PGHist):
    def __init__(self, ui):
        x_lab = 'dt'
        y_lab = 'Counts'
        x_unit = 's'
        super().__init__(ui.dt_plot, x_lab, y_lab, x_unit)
        self.text_inputs = {
            'dt_bins': ui.dtbins,
            'dt_start': ui.dtrange0,
            'dt_end': ui.dtrange1
        }

    def update(self, dataset: CurrentData):
        data = dataset.get_dt()
        num_bins = int(self.text_inputs['dt_bins'].text())
        start = float(self.text_inputs['dt_start'].text())
        end = float(self.text_inputs['dt_end'].text())
        bins = np.linspace(start, end, num_bins)
        self._basic_update(data, bins)


class CurrentHist(PGHist):
    def __init__(self, ui):
        x_lab = 'Current'
        y_lab = 'Counts'
        x_unit = 'A'
        super().__init__(ui.event_plot, x_lab, y_lab, x_unit)
        self.text_inputs = {}
        self.logo = load_logo()
        self._can_plot = True

    def update(self, dataset: CurrentData):
        if not self._can_plot:
            return
        data = dataset.processed_data
        num_bins = 1000
        y, x = np.histogram(data, bins=num_bins)
        curve = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush='b')
        self.plot.clear()
        self.plot.addItem(curve)
        self.plot.setXRange(np.min(data), np.max(data))

    def chart_mode(self):
        x_lab = 'Current'
        y_lab = 'Counts'
        x_unit = 'A'
        self.plot.clear()
        plot = self.plot
        plot.setLabel('bottom', text=x_lab, units=x_unit)
        plot.setLabel('left', text=y_lab)
        plot.setMouseEnabled(x=True, y=True)
        plot.setAspectLocked(False)
        plot.setLimits(yMin=0)
        plot.brush = 'b'
        self._can_plot = True

    def logo_mode(self):
        self._can_plot = False
        plot = self.plot
        plot.clear()
        plot.hideAxis('bottom')
        plot.hideAxis('left')
        plot.setLimits(yMin=None)
        plot.setMouseEnabled(x=False, y=False)
        plot.setAspectLocked(True)
        plot.addItem(self.logo)

    def get_mode(self):
        if self._can_plot:
            return 'chart'
        else:
            return 'logo'


class PlotGroup(object):
    def __init__(self, ui):
        self.plots = []
        # self.colors = sdf.color.unique
        self.plots = {
            'frac_hist': FracHist(ui),
            'delta_hist': DeltaHist(ui),
            'duraation_hist': DurationHist(ui),
            'dt_hist': DTHist(ui),
            'current_hist': CurrentHist(ui),
            'blockage_plot': BlockageScatter(ui),
            'signal_plot': SignalPlot(ui)
        }

    def update(self, dataset):
        if self.plots['current_hist'].get_mode() is 'logo':
            self.plots['current_hist'].chart_mode()
        for key, plot in self.plots.items():
            if key in ['current_hist', 'signal_plot'] or dataset.events:
                # Every plot except the current histogram requires events
                plot.update(dataset)


# Function to hide ui setup boilerplate
def setup_connections(ui, form):
    # Linking buttons to main functions
    ui.load_button.clicked.connect(form.get_file)
    ui.analyze_button.clicked.connect(form.analyze)
    ui.cut_button.clicked.connect(form.signal_plot.cut)
    ui.baseline_button.clicked.connect(form.signal_plot.baseline)
    ui.clear_data_button.clicked.connect(form.clear_scatter)
    ui.delete_event_button.clicked.connect(form.delete_event)
    ui.next_file_button.clicked.connect(form.next_file)
    ui.previous_file_button.clicked.connect(form.previous_file)

    ui.save_event_fits_button.clicked.connect(form.save_event_fits)

    ui.pore_sizer_action.triggered.connect(form.size_pore)
    # ui.batch_process_action.triggered.connect(form.batch_info_dialog)

    ui.low_pass_entry.editingFinished.connect(form.replot)
    # TODO: Connect sample rate entry to update axis on finishing edit


def setup_cb(ui: Ui_PythIon):
    cb = pg.ColorButton(ui.scatter_plot, color=(0, 0, 255, 50))
    cb.setFixedHeight(30)
    cb.setFixedWidth(30)
    cb.move(0, 210)
    cb.show()
    return cb


def load_logo():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logo = ndimage.imread(dir_path + os.sep.join(["", "Assets", "pythionlogo.png"]))
    logo = np.rot90(logo, -1)
    logo = pg.ImageItem(logo)
    return logo


class NumberEntry(object):
    entry: QtWidgets.QLineEdit

    def __init__(self, ui_entry, min_val, max_val, entry_type, multiplier=1):
        self.entry = ui_entry  # This is the value entry object
        self.min = min_val
        self.max = max_val
        self.multiplier = multiplier
        self.type = entry_type

    def value(self):
        value = self.type(self.entry.text())
        value = min(self.max, max(value, self.min))
        return value

    def set(self, value):
        value = self.type(value)
        value = min(self.max, max(value, self.min))
        self.entry.setText(str(value))

    def increment(self):
        value = self.value() + 1
        self.set(value)

    def decrement(self):
        value = self.value() + 1
        self.set(value)


class EventSelectWidget(object):
    num_entry: NumberEntry

    # TODO: Make this a self contained widget
    def __init__(self, ui, entry_obj):
        self.go_button = ui.go_event_button
        self.num_entry = entry_obj
        ui.go_event_button.clicked.connect(self.go)
        ui.previous_event_button.clicked.connect(self.previous)
        ui.next_event_button.clicked.connect(self.next)
        self.dataset = None

    def next(self):
        self.num_entry.increment()
        self.go()

    def previous(self):
        self.num_entry.increment()
        self.go()

    def go(self):
        event_number = self.num_entry.value()
        event = self.dataset.events[event_number - 1]
        event.inspect()

    def register_dataset(self, dataset: CurrentData):
        self.dataset = dataset
