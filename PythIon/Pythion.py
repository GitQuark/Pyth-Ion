# from PythIon.PoreSizer import *
import sys

from PythIon.SetupUtilities import *
# from PythIon.Utility import *
from PythIon.Widgets.PlotGUI import *
from PythIon.Model import *
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
        pg.setConfigOptions(antialias=True)

        self.ui = Ui_PythIon()
        self.ui.setupUi(self)
        self.cb = setup_cb(self.ui)
        self.plot_group = PlotGroup(self.ui)
        self.signal_plot = self.plot_group.plots['signal_plot']
        self.plot_group.plots['current_hist'].logo_mode()
        self.event_num_entry = NumberEntry(self.ui.event_number_entry, 1, 1, int)
        self.event_selector = EventSelectWidget(self.ui, self.event_num_entry)

        # Initializing various variables used for analysis
        self.dataset = None
        self.data_file_name = None
        self.sdf = pd.DataFrame(columns=['fn', 'color', 'deli', 'frac', 'dwell', 'dt', 'startpoints', 'endpoints'])

        self.highlighted_event = None
        self.batch_info = pd.DataFrame(columns=list(['cutstart', 'cutend']))
        setup_connections(self.ui, self)

    def get_file(self):
        wd = os.getcwd()
        suffix_filter = "*.log;*.opt;*.npy;*.abf"
        # Attempt to open dialog from most recent directory
        data_file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', wd, suffix_filter)
        if data_file_name == ('', ''):  # Interaction canceled
            return

        self.data_file_name = data_file_name
        self.load()

    def load(self):
        data_file_name = self.data_file_name
        if not data_file_name:
            return
        sdf = self.sdf
        ui = self.ui

        colors = np.array(sdf.color)
        for i in range(len(colors)):
            colors[i] = pg.Color(colors[i])
        # self.scatter_plot.items[0].setBrush(colors, mask=None)

        ui.event_number_entry.setText(str(1))
        ui.status_bar.showMessage('Data file ' + data_file_name + ' loaded.')
        ui.file_label.setText(data_file_name)

        low_pass_cutoff = float(ui.low_pass_entry.text())
        self.dataset = CurrentData(self.event_num_entry, data_file_name, self.signal_plot)
        self.event_selector.register_dataset(self.dataset)
        self.ui.invert_button.clicked.connect(self.dataset.invert)
        if self.dataset.file_type in ('.log', '.abf', '.opt'):
            self.dataset.process_data(low_pass_cutoff)
        self.plot_group.update(self.dataset)

    def analyze(self):
        dataset = self.dataset
        if dataset is None:
            return
        threshold = np.float64(self.ui.threshold_entry.text()) * 1e-9  # Now in number of standard deviations
        dataset.detect_events(threshold)
        self.event_num_entry.max = len(dataset.events)
        self.plot_group.plots['blockage_plot'].create_points(dataset)
        self.plot_group.update(dataset)

    def replot(self):
        if not self.dataset:
            return
        low_pass_cutoff = int(self.ui.low_pass_entry.text())
        if low_pass_cutoff == self.dataset.data_params.get('low_pass_cutoff'):
            return
        self.dataset.process_data(low_pass_cutoff)
        self.plot_group.update(self.dataset)

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

    def clear_scatter(self):
        scatter = self.plot_group.plots['scatter_plot']
        ui = self.ui
        scatter.setData(x=[], y=[])
        # last_event = []
        ui.scatter_plot.update()
        self.clear_stat_plots()
        self.sdf = pd.DataFrame(columns=['fn', 'color', 'deli', 'frac',
                                         'dwell', 'dt', 'startpoints', 'endpoints'])

    def inspect_event(self):
        pass

    def delete_event(self):
        event_idx = int(self.ui.event_number_entry.value()) - 1
        self.dataset.delete_event(event_idx)
        self.plot_group.update(self.dataset)

    def invert_data(self):
        if self.dataset:
            self.dataset.invert()
            # self.dataset.data_params['inverted'] = not self.dataset.data_params.get('inverted')
            # self.dataset.processed_data = -self.dataset.processed_data
            # # self.dataset.data_params['baseline'] = not self.dataset.data_params.get('baseline')

    def clicked(self, plot, points):
        if not points:
            return
        selected_event = points[0].data().parent
        if isinstance(self.highlighted_event, Event):
            for interval in self.highlighted_event.intervals:
                interval.reset_style('scatter')
        # Highlight all points on the graph in the same event
        for interval in selected_event.intervals:
            interval.highlight('scatter')
        self.scatter_plot.clear()
        points = [interval.data_points['scatter'] for interval in self.dataset.get_intervals()]
        scatter_data = pg.ScatterPlotItem(points)
        scatter_data.sigClicked.connect(self.clicked)
        self.scatter_plot.addItem(scatter_data)
        # idx = [idx for idx, pt in enumerate(plot.points()) if pt.pos() == points[0].pos()]
        # if idx is None:
        #     return
        # if sdf.fn[clicked_index] != mat_file_name:
        #     print('Event is from an earlier file, not clickable')
        self.highlighted_event = selected_event
        self.inspect_event(selected_event)

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
        # TODO: convert to dict
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
        elif key == qt.Key_Escape:
            base_region = self.signal_plot.base_region
            cut_region = self.signal_plot.cut_region
            if base_region.isVisible():
                base_region.toggle_region()
            if cut_region.isVisible():
                cut_region.toggle_region()

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

    def save_trace(self):
        pass

    def save_target(self):
        self.batch_info = save_batch_info(self.events, self.batch_info, self.mat_file_name)

    @staticmethod
    def size_pore():
        pass
        # pore_size = PoreSizer()
        # pore_size.show()


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
